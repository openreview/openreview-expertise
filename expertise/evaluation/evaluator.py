'''
A script for evaluating different models
'''
import time
import json, argparse, csv, sys
import openreview, os, logging, random

import numpy as np

from expertise.execute_expertise import execute_expertise

from .goldstandard import GoldStandardEvaluator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

EVALUATOR_MAP = {
    'goldstandard': GoldStandardEvaluator
}

SUPPORTED_MODELS = ['specter2', 'scincl']

class OpenReviewExpertiseEvaluation(object):
    def __init__(self, config):
        # Validate existence of evaluation data
        self.config = config
        self.datasets = config.get('datasets')
        self.configs = config.get('configs')
        self.models = config.get('models')
        self.use_cuda = config.get('use_coda', 'False')
        self.skip_settings = config.get('skip', {})

    def run(self):
        for dataset in self.datasets:

            # Load data
            logging.info(f"Dataset: {dataset}")
            evaluator_settings = self.configs[dataset]
            evaluator = EVALUATOR_MAP[dataset](evaluator_settings)
            
            if not self.skip_settings.get('dataset', {}).get(dataset, False):
                dataset_directories = evaluator.create_dataset()
                logging.info(f"Dataset transformed")
            else:
                logging.info(f"{dataset} skipped")

            # Embed papers and calculate paper-paper similarities (submissions x publications)
            for model in self.models:
                # Skip embedding if flagged
                if self.skip_settings.get('embedding', {}).get(model, False):
                    logging.info(f"{model} skipped")
                    continue

                for id, directory in dataset_directories.items():
                    # Build a default config
                    model_config = {
                        "name": f"{model}_{id}",
                        "job_dir": directory,
                        "dataset": {
                            "directory": directory
                        },
                        "model": model,
                        "model_params": {
                            "use_title": True,
                            "use_abstract": True,
                            "average_score": False,
                            "max_score": True,
                            "skip_specter": False,
                            "specter_batch_size": 16,
                            "mfr_batch_size": 384,
                            "use_cuda": self.use_cuda,
                            "use_redis": False,
                            "dump_p2p": True,
                            "name": model,
                            "work_dir": os.path.join(directory, model),
                            "scores_path": os.path.join(directory, model),
                            "publications_path": os.path.join(directory, model),
                            "submissions_path": os.path.join(directory, model)
                        }
                    }
                    logging.info(f"Embedding {directory}-{id}")
                    execute_expertise(model_config)
            
            # Compute metrics
            eval_config = self.config['evaluate'][dataset]
            if dataset == 'goldstandard':
                logging.info(f"Running evaluations for {dataset}")
                objective = list(eval_config.get('objective', {'minimize': 'loss'}).values())[0]
                maximize = 'maximize' in eval_config.get('objective', {'minimize': 'loss'}).keys()
                base_dir = self.configs[dataset]['destination']
                hyp_config = eval_config['hyperparameters']

                fold_metrics = []
                for fold_num in range(1, eval_config.get('settings', {}).get('folds', 5) + 1):
                    logging.info(f"Running {dataset} fold {fold_num}")
                    
                    # Hyperparameters Search on Train
                    all_metrics = []
                    for hyp_idx in range(eval_config.get('settings', {}).get('num_hyperparameter_samples', 10)):
                        logging.info(f"Running {dataset} train fold {fold_num} hyp sample {hyp_idx}")
                        # Sample hyperparameters using ranges in config
                        merge = None
                        if 'merge' in hyp_config.keys():
                            merge = np.random.uniform(hyp_config['merge']['min'], hyp_config['merge']['max'])
                        select, select_type = None, None
                        if 'select' in hyp_config.keys():
                            if isinstance(hyp_config['select']['type'], list):
                                select_type = random.choice(hyp_config['select']['type'])
                            else:
                                select_type = hyp_config['select']['type']
                            select = np.random.randint(hyp_config['select']['min'], hyp_config['select']['max'])
                        hyperparameter_sample = {
                            'merge': merge,
                            'select': select,
                            'select_type': select_type
                        }

                        # Run hyperparameters on samples - Sample Loop
                        all_sample_reviewer_scores = []
                        for sample_num in range(1, eval_config.get('settings', {}).get('samples', 10) + 1):
                            logging.info(f"Running {dataset} train fold {fold_num} hyp sample {hyp_idx} subsample {sample_num}")
                            working_dir = os.path.join(base_dir, f"train_{fold_num}_{sample_num}")
                            
                            ## Aggregate models
                            p2p_affs = {}
                            final_scores = {}
                            for model in self.models:
                                model_dir = os.path.join(working_dir, model)
                                for dirpath, _, filenames in os.walk(model_dir):
                                    p2p_name = [name for name in filenames if 'p2p' in name][0]
                                    with open(os.path.join(dirpath, p2p_name), 'r') as f:
                                        p2p_affs[model] = (json.load(f))

                            for model, aff_matrix in p2p_affs.items():
                                for submission, publications in aff_matrix.items():
                                    for publication in publications.keys():
                                        if submission not in final_scores.keys():
                                            final_scores[submission] = {}
                                        if publication not in final_scores[submission]:
                                            final_scores[submission][publication] = 0
                                        
                                        if model == 'specter2' and merge is not None: ## Can generalize by reading config settings
                                            final_scores[submission][publication] += merge * aff_matrix[submission][publication]
                                        elif model == 'scincl' and merge is not None:
                                            final_scores[submission][publication] += (1 - merge) * aff_matrix[submission][publication]

                            ## Compute in-memory dict[reviewer][submission]
                            reviewer_scores = {}
                            reviewer_to_pub = {}
                            archives_dir = os.path.join(working_dir, 'archives')
                            for filename in os.listdir(archives_dir):
                                reviewer_id = filename[1:].replace('.jsonl', '')
                                with open(os.path.join(archives_dir, filename), 'r') as file:
                                    reviewer_to_pub[reviewer_id] = [json.loads(line)['id'] for line in file]

                            for reviewer, reviewer_pubs in reviewer_to_pub.items():
                                for submission, publications in final_scores.items():

                                    if reviewer not in reviewer_scores.keys():
                                        reviewer_scores[reviewer] = {}

                                    submission_scores = [(pub, final_scores[submission][pub]) for pub in reviewer_pubs]
                                    sorted_scores = [item[1] for item in sorted(submission_scores, key=lambda x: float(x[1]), reverse=True)]
                                    if select_type == 'avg':
                                        reviewer_scores[reviewer][submission] = np.mean(sorted_scores[: min(select, len(sorted_scores))])
                                    elif select_type == 'max':
                                        reviewer_scores[reviewer][submission] = max(sorted_scores[: min(select, len(sorted_scores))])
                            all_sample_reviewer_scores.append(reviewer_scores)

                        # Pass to evaluation script and store metrics
                        logging.info(f"Computing metrics...")
                        metrics = evaluator.compute_metrics('train', fold_number=fold_num, prediction_dicts=all_sample_reviewer_scores)
                        logging.info(metrics)
                        all_metrics.append({
                            'hyperparameter': hyperparameter_sample,
                            'metrics': metrics
                        })

                    # Find best sample,metrics pair and repeat on test samples
                    best_metric = sorted(all_metrics, key=lambda x: x['metrics'][objective]['point'], reverse=maximize)[0]
                    merge = best_metric['hyperparameter']['merge']
                    select = best_metric['hyperparameter']['select']
                    select_type = best_metric['hyperparameter']['select_type']

                    # Eval on Test - Sample Loop
                    all_sample_reviewer_scores = []
                    logging.info(f"Start eval on test fold")
                    for sample_num in range(1, eval_config.get('settings', {}).get('samples', 10) + 1):
                        working_dir = os.path.join(base_dir, f"test_{fold_num}_{sample_num}")
                        
                        ## Aggregate models
                        p2p_affs = {}
                        final_scores = {}
                        for model in self.models:
                            model_dir = os.path.join(working_dir, model)
                            for dirpath, _, filenames in os.walk(model_dir):
                                p2p_name = [name for name in filenames if 'p2p' in name][0]
                                with open(os.path.join(dirpath, p2p_name), 'r') as f:
                                    p2p_affs[model] = (json.load(f))

                        for model, aff_matrix in p2p_affs.items():
                            for submission, publications in aff_matrix.items():
                                for publication in publications.keys():
                                    if submission not in final_scores.keys():
                                        final_scores[submission] = {}
                                    if publication not in final_scores[submission]:
                                        final_scores[submission][publication] = 0
                                    
                                    if model == 'specter2' and merge is not None: ## Can generalize by reading config settings
                                        final_scores[submission][publication] += merge * aff_matrix[submission][publication]
                                    elif model == 'scincl' and merge is not None:
                                        final_scores[submission][publication] += (1 - merge) * aff_matrix[submission][publication]

                        ## Compute in-memory dict[reviewer][submission]
                        reviewer_scores = {}
                        reviewer_to_pub = {}
                        archives_dir = os.path.join(working_dir, 'archives')
                        for filename in os.listdir(archives_dir):
                            reviewer_id = filename[1:].replace('.jsonl', '')
                            with open(os.path.join(archives_dir, filename), 'r') as file:
                                reviewer_to_pub[reviewer_id] = [json.loads(line)['id'] for line in file]
    
                        for reviewer, reviewer_pubs in reviewer_to_pub.items():

                            if reviewer not in reviewer_scores.keys():
                                reviewer_scores[reviewer] = {}

                            for submission, publications in final_scores.items():
                                submission_scores = [(pub, final_scores[submission][pub]) for pub in reviewer_pubs]
                                sorted_scores = [item[1] for item in sorted(submission_scores, key=lambda x: float(x[1]), reverse=True)]
                                if select_type == 'avg':
                                    reviewer_scores[reviewer][submission] = np.mean(sorted_scores[: min(select, len(sorted_scores))])
                                elif select_type == 'max':
                                    reviewer_scores[reviewer][submission] = max(sorted_scores[: min(select, len(sorted_scores))])
                        all_sample_reviewer_scores.append(reviewer_scores)

                    metrics = evaluator.compute_metrics('test', fold_number=fold_num, prediction_dicts=all_sample_reviewer_scores)
                    fold_metrics.append({
                        'hyperparameter': best_metric['hyperparameter'],
                        'metrics': metrics
                    })

                overall_metric = sorted(fold_metrics, key=lambda x: x['metrics'][objective]['point'], reverse=maximize)[0]
                print(fold_metrics)
                print(overall_metric)
