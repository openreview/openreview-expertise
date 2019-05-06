import os
import json
from collections import defaultdict

from expertise.utils.standard_test import test
from expertise.utils.dataset import Dataset
from tqdm import tqdm

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from . import helpers

def _setup_bert_pretrained(bert_model):
    model = BertModel.from_pretrained(bert_model)
    return model

def _write_features(lines, outfile, extraction_args):

    if not os.path.exists(outfile):
        try:
            all_lines_features = helpers.extract_features(
                lines=lines,
                model=extraction_args['model'],
                tokenizer=extraction_args['tokenizer'],
                max_seq_length=extraction_args['max_seq_length'],
                batch_size=extraction_args['batch_size'],
                no_cuda=extraction_args['no_cuda']
            )

            embeddings = extraction_args['aggregator_fn'](all_lines_features)
            np.save(outfile, embeddings)
        except RuntimeError as e:
            print('runtime error encountered for ', outfile)
            print(e)
    else:
        print('skipping {}'.format(outfile))

def setup(config, partition_id=0, num_partitions=1, local_rank=-1):
    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    feature_dirs = [
        'submissions-features',
        'archives-features'
    ]

    for d in feature_dirs:
        os.makedirs(os.path.join(setup_dir, d), exist_ok=True)

    dataset = Dataset(**config.dataset)

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model, do_lower_case=config.do_lower_case)

    model = _setup_bert_pretrained(config.bert_model)

    # convert submissions and archives to bert feature vectors

    dataset_args = {
        'partition_id': partition_id,
        'num_partitions': num_partitions,
        'progressbar': False,
        'sequential': False,
        'fields': ['title','abstract']
    }

    extraction_args = {
        'model': model,
        'tokenizer': tokenizer,
        'max_seq_length': config.max_seq_length,
        'batch_size': config.batch_size,
        'no_cuda': not config.use_cuda,
        'aggregator_fn': helpers.AGGREGATOR_MAP[config.embedding_aggregation_type]
    }

    for text_id, text_list in dataset.submissions(**dataset_args):
        feature_dir = os.path.join(setup_dir, 'submissions-features')
        outfile = os.path.join(feature_dir, '{}.npy'.format(text_id))
        _write_features(
            text_list, outfile, extraction_args)

    for text_id, text_list in dataset.archives(**dataset_args):
        feature_dir = os.path.join(setup_dir, 'archives-features')
        outfile = os.path.join(feature_dir, '{}.npy'.format(text_id))
        _write_features(
            text_list, outfile, extraction_args)


def train(config):
    pass

def infer(config):
    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')
    infer_dir = os.path.join(experiment_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    submissions_dir = os.path.join(
        setup_dir,
        'submissions-features')

    archives_dir = os.path.join(
        setup_dir,
        'archives-features')

    submission_embeddings = []
    paper_lookup = []
    for emb_file in os.listdir(submissions_dir):
        embedding_list = np.load(os.path.join(submissions_dir, emb_file))
        for emb in embedding_list:
            submission_embeddings.append(emb)
            paper_lookup.append(emb_file.replace('.npy', ''))
    submission_matrix = np.asarray(submission_embeddings)

    archive_embeddings = []
    author_lookup = []
    for emb_file in os.listdir(archives_dir):
        embedding_list = np.load(os.path.join(archives_dir, emb_file))
        for emb in embedding_list:
            archive_embeddings.append(emb)
            author_lookup.append(emb_file.replace('.npy', ''))
    archive_matrix = np.asarray(archive_embeddings)

    scores = np.dot(submission_matrix, np.transpose(archive_matrix))

    score_file_path = os.path.join(infer_dir, config.name + '-scores.jsonl')
    with open(score_file_path, 'w') as f:
        for paper_idx, row in enumerate(scores):
            paper_id = paper_lookup[paper_idx]

            author_max_scores = defaultdict(int)

            for author_idx, score in enumerate(row):
                a = author_lookup[author_idx]
                if author_max_scores[a] < score:
                    author_max_scores[a] = score

            for author_id, max_score in author_max_scores.items():
                result = {
                    'source_id': paper_id,
                    'target_id': author_id,
                    'score': float(max_score)
                }
                f.write(json.dumps(result) + '\n')
