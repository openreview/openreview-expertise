import os
import json
from expertise.utils.standard_test import test
from expertise.utils.dataset import Dataset
from tqdm import tqdm

# needed?
import gensim
from gensim.models import KeyedVectors

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from . import helpers

def _setup_bert_pretrained(bert_model):
    model = BertModel.from_pretrained(bert_model)
    return model

def _write_features(text_id, lines, feature_dir, extraction_args):
    avg_emb_file = os.path.join(feature_dir, 'avg/{}.npy'.format(text_id))
    cls_emb_file = os.path.join(feature_dir, 'cls/{}.npy'.format(text_id))

    if any([not os.path.exists(f) for f in [avg_emb_file, cls_emb_file]]):
        all_lines_features = helpers.extract_features(
            lines=lines,
            model=extraction_args['model'],
            tokenizer=extraction_args['tokenizer'],
            max_seq_length=extraction_args['max_seq_length'],
            batch_size=extraction_args['batch_size'],
            no_cuda=extraction_args['no_cuda']
        )

        avg_embeddings = helpers.get_avg_words(all_lines_features)
        class_embeddings = helpers.get_cls_vectors(all_lines_features)

        np.save(avg_emb_file, avg_embeddings)
        np.save(cls_emb_file, class_embeddings)
    else:
        print('skipping {}'.format(text_id))

def setup(config, partition_id=0, num_partitions=1, local_rank=-1):
    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    feature_dirs = [
        'submissions-features/cls',
        'submissions-features/avg',
        'archives-features/cls',
        'archives-features/avg'
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
        'sequential': False
    }

    extraction_args = {
        'model': model,
        'tokenizer': tokenizer,
        'max_seq_length': config.max_seq_length,
        'batch_size': config.batch_size,
        'no_cuda': not config.use_cuda
    }

    for text_id, text_list in dataset.submissions(**dataset_args):
        feature_dir = os.path.join(setup_dir, 'submissions-features')
        _write_features(
            text_id, text_list, feature_dir, extraction_args)

    for text_id, text_list in dataset.archives(**dataset_args):
        feature_dir = os.path.join(setup_dir, 'archives-features')
        _write_features(
            text_id, text_list, feature_dir, extraction_args)


def train(config):
    pass

def infer(config):
    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')
    infer_dir = os.path.join(experiment_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    submissions_dir = os.path.join(
        setup_dir,
        'submissions-features/{}'.format(config.embedding_aggregation_type))

    archives_dir = os.path.join(
        setup_dir,
        'archives-features/{}'.format(config.embedding_aggregation_type))

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
            max_index = row.argmax()
            max_score = row[max_index]
            author_id = author_lookup[max_index]
            paper_id = paper_lookup[paper_idx]

            result = {
                'source_id': paper_id,
                'target_id': author_id,
                'score': float(max_score)
            }
            f.write(json.dumps(result) + '\n')
