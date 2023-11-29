import os, sys, json, argparse, shutil
import numpy as np
import pandas as pd

from .utils import *
from sklearn.model_selection import KFold


class GoldStandardEvaluator(object):

    def __init__(self, config):
        self.algo = config.get('algo')
        self.dataset = config.get('dataset')
        self.predictions = config.get('predictions')

        self.destination = config.get('destination')
        self.regime = config.get('regime', 'SS')
        self.hist_len = config.get('hist_len')

        self.folds = config.get('folds')

    def create_dataset(self):

        # Initialize the KFold object with 5 splits
        df = pd.read_csv(os.path.join(self.dataset, 'evaluations.csv'), sep='\t')
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        path_prefix = self.dataset

        folds = []
        for i, (train_index, test_index) in enumerate(kf.split(df)):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            
            # Save individual folds
            train_name = f'train_{i+1}.csv'
            test_name = f'test_{i+1}.csv'
            folds.append((train_name, test_name))
            with open(os.path.join(path_prefix, train_name), 'w') as f:
                train.to_csv(os.path.join(path_prefix, train_name), index=False, sep='\t')
            with open(os.path.join(path_prefix, test_name), 'w') as f:
                test.to_csv(os.path.join(path_prefix, test_name), index=False, sep='\t')

        # Clear the transformed directories if it exists, or make it if it doesn't exist
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        else:
            self._remove_contents(self.destination)

        datasets = {}
        for fold_ids in folds:
            for id in fold_ids:
                df = pd.read_csv(os.path.join(self.dataset, id), sep='\t')
                _, targets = to_dicts(df)
                for resample in range(1, 11):
                    OR_dataset_path = os.path.join(self.destination, f"{id.split('.')[0]}_{resample}")
                    datasets[f"{id.split('.')[0]}_{resample}"] = OR_dataset_path
                    self._prepare_dataset(OR_dataset_path, targets, self.dataset, self.hist_len, self.regime)

        return datasets

    def compute_metrics(self, fold_name=None, fold_number=None, prediction_dicts=None):
        df = pd.read_csv(os.path.join(self.dataset, f"{fold_name}_{fold_number}.csv"), sep='\t')
        references, _ = to_dicts(df)

        all_reviewers = list(references.keys())

        all_papers = set()
        for rev in references:
            all_papers = all_papers.union(references[rev].keys())

        # Prepare reviewer pools for computing Confidence Intervals (n=1,000 iterations)
        bootstraps = [np.random.choice(all_reviewers, len(all_reviewers), replace=True) for x in range(1000)]

        results = {'pointwise': [], 'variations': []}
        results_easy = {'pointwise': [], 'variations': []}
        results_hard = {'pointwise': [], 'variations': []}

        for prediction in prediction_dicts:
            tmp = self._score_performance(prediction, references, all_papers, all_reviewers, bootstraps)
            tmp_accs = score_resolution(prediction, references, all_papers, bootstraps)

            results['pointwise'].append(tmp[0])
            results['variations'].append(tmp[1])

            results_easy['pointwise'].append(tmp_accs[0]['score'])
            results_easy['variations'].append(tmp_accs[2])

            results_hard['pointwise'].append(tmp_accs[1]['score'])
            results_hard['variations'].append(tmp_accs[3])

        # Get pointwise estimate of performance
        point = round(np.mean(results['pointwise']), 2)
        point_easy = round(np.mean(results_easy['pointwise']), 2)
        point_hard = round(np.mean(results_hard['pointwise']), 2)

        # Get 95% confidence interval
        boot = np.matrix(results['variations']).mean(axis=0).tolist()[0]
        boot_easy = np.matrix(results_easy['variations']).mean(axis=0).tolist()[0]
        boot_hard = np.matrix(results_hard['variations']).mean(axis=0).tolist()[0]
        ci = f"[{round(np.percentile(boot, 2.5), 2)}; {round(np.percentile(boot, 97.5), 2)}]"
        ci_easy = f"[{round(np.percentile(boot_easy, 2.5), 2)}; {round(np.percentile(boot_easy, 97.5), 2)}]"
        ci_hard = f"[{round(np.percentile(boot_hard, 2.5), 2)}; {round(np.percentile(boot_hard, 97.5), 2)}]"

        return {
            'loss': {
                'point': point,
                'ci': ci
            },
            'easy': {
                'point': point_easy,
                'ci': ci_easy
            },
            'hard': {
                'point': point_hard,
                'ci': ci_hard
            }
        }

    def _remove_contents(self, dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def _get_k_recent_papers(self, participant, data_path, k, regime='SS'):
        """Get the most recent papers from a participant's publication profile.
        Only years are used for ordering and ties are broken uniformly at random.

        Args:
            participant: Semantic Scholar id of the participant
            data_path: Path to the dataset folder
            k: number of papers that should be included in the profile
            regime: whether to use SS or other representations

        Returns:
            A list of length k where each entry represents a paper
        """

        profile_coarse, profile_fine, profile = {}, {}, []

        # Read reviewer representation (contains only paper IDs)
        with open(os.path.join(data_path, 'participants', participant + '.json'), 'r') as handler:
            profile_coarse = json.load(handler)

        # For each of the reviewer's paper, read its representation (contains more paper info)
        for pid in set([pap['paperId'] for pap in profile_coarse['papers']]):
            paper = self._get_paper_representation(pid, data_path, regime)

            if paper is not None:
                profile_fine[pid] = paper

        if k is None:
            k = len(profile_fine)

        recent_papers = sorted(profile_fine.keys(),
                            key=lambda x: profile_fine[x]['year'] + np.random.uniform(0, 0.001),
                            reverse=True)[:k]

        return [{'id': tp, 'content': profile_fine[tp]} for tp in recent_papers]


    def _get_paper_representation(self, pid, data_path, regime):
        """ Read representation of the paper.

        Args:
            pid: Semantic Scholar id of the paper
            data_path: Path to the dataset folder
            regime: whether to use SS or other representations

        Returns:
            Dict representing the paper. If some important fields are missing (title, abstract, year, full text),
            return None which means that the paper should be excluded from similarity computation
        """

        # Directory with semantic scholar representations of papers
        ss_dir = os.path.join(data_path, 'papers')

        # Directory with PDFs parsed into txt files
        pdf_dir = os.path.join(data_path, 'txts')

        with open(os.path.join(ss_dir, pid + '.json'), 'r') as handler:
            paper = json.load(handler)

        # We only leave papers with non-empty title, abstract, and year of publications
        if paper['title'] is None or paper['abstract'] is None or paper['year'] is None:
            return None

        if regime == 'SS':
            return paper

        if pid + '.json' not in set(os.listdir(pdf_dir)):
            return None

        with open(os.path.join(pdf_dir, pid + '.json'), 'r') as handler:
            text = json.load(handler)

        paper['text'] = text

        return paper


    def _prepare_dataset(self, dst_path, targets, data_path, k=10, regime='SS'):
        """Transform the dataset we release into the OpenReview format required by the similarity computation methods.

        Args:
            dst_path: path to the folder where the dataset will be stored.
            The folder should not exist and will be created

            targets: dict of sets where each set corresponds to a reviewer and
            contains papers the (reviewer, paper) similarities should be computed for.

            regime: whether to use SS or other representations

            data_path: path to the dataset folder

            k: number of papers that should be included in the profile
        """

        if regime not in set(['SS', 'PDF']):
            raise ValueError("Unknown regime is provided")

        if os.path.exists(dst_path):
            print("I reuse the existing dataset. Provide a fresh path if you want to build a new one")
            return dst_path

        archives_path = os.path.join(dst_path, 'archives')
        submissions_path = os.path.join(dst_path, 'submissions.json')

        os.mkdir(dst_path)
        os.mkdir(archives_path)

        papers = {}

        for participant in targets:

            # Prepare a single jsonl with the participant's profile
            profile = self._get_k_recent_papers(participant, data_path, k, regime)

            # If a reviewer does not have papers to include in the profile, we remove them. This can happen if a reviewer
            # does not have papers with arXiv links in their SS profiles (when we construct dataset in the `PDF` regime)
            if len(profile) > 0:

                with open(os.path.join(archives_path, '~' + participant + '.jsonl'), 'w', encoding='utf-8') as handler:
                    for line in profile:
                        handler.write(json.dumps(line) + '\n')

            # Prepare a single json for all target papers

            for pid in targets[participant]:

                content = self._get_paper_representation(pid, data_path, regime)

                if content is None:
                    continue

                paper = {"id": pid, "content": content}

                papers[pid] = paper

        with open(submissions_path, 'w', encoding='utf-8') as handler:
            json.dump(papers, handler)

    def _score_performance(self, predictions, references, valid_papers, valid_reviewers, bootstraps):
        """Compute the main metric for predicted similarity together with bootstrapped values for confidence intervals

        :param pred_file: Name of the file where predicted similarities are stored (file must be in the PRED_PATH dir)
        :param references: Ground truth values of expertise
        :param valid_papers: Papers to include in evaluation
        :param valid_reviewers: Reviewers to include in evaluation
        :param bootstraps: Subsampled reviewer pools for bootstrap computations
        :return: Score of the predictions + data to compute confidence intervals (if `bootstraps` is not None)
        """

        score = compute_main_metric(predictions, references, valid_papers, valid_reviewers)
        variations = [compute_main_metric(predictions, references, valid_papers, vr) for vr in bootstraps]

        return score, variations