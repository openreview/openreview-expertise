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

def setup_bert_pretrained(bert_model):
    model = BertModel.from_pretrained(bert_model)
    return model

def setup(config, partition_id=0, num_partitions=1, local_rank=-1):

    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    for feature_dir in [
        'submissions-features/cls',
        'submissions-features/avg',
        'archives-features/cls',
        'archives-features/avg']:

        os.makedirs(os.path.join(setup_dir, feature_dir), exist_ok=True)

    dataset = Dataset(**config.dataset)

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model, do_lower_case=config.do_lower_case)

    model = setup_bert_pretrained(config.bert_model)

    # convert submissions and archives to bert feature vectors
    for text_id, text in dataset.submissions():
        all_lines_features = helpers.extract_features(
            lines=[text],
            model=model,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            batch_size=32
        )

        avg_embeddings = helpers.get_avg_words(all_lines_features)
        class_embeddings = helpers.get_cls_vectors(all_lines_features)

        avg_emb_file = os.path.join(setup_dir, 'submissions-features/avg/{}.npy'.format(text_id))
        np.save(avg_emb_file, avg_embeddings)

        cls_emb_file = os.path.join(setup_dir, 'submissions-features/cls/{}.npy'.format(text_id))
        np.save(cls_emb_file, class_embeddings)

    for text_id, all_text in dataset.archives(sequential=False):
        all_lines_features = helpers.extract_features(
            lines=all_text,
            model=model,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            batch_size=32
        )

        avg_embeddings = helpers.get_avg_words(all_lines_features)
        class_embeddings = helpers.get_cls_vectors(all_lines_features)

        avg_emb_file = os.path.join(setup_dir, 'archives-features/avg/{}.npy'.format(text_id))
        np.save(avg_emb_file, avg_embeddings)

        cls_emb_file = os.path.join(setup_dir, 'archives-features/cls/{}.npy'.format(text_id))
        np.save(cls_emb_file, class_embeddings)


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
