import os
import json
from collections import defaultdict

from expertise.dataset import Dataset
from tqdm import tqdm

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from .extract_features import extract_features

def get_avg_words(line_features, layer_index=-1):
    lines = []
    for line_f in line_features:
        word_embeddings = []
        for token_feature in line_f['features']:
            if not (token_feature['token'].startswith('[') and token_feature['token'].endswith(']')):

                for layer in token_feature['layers']:
                    if layer['index'] == layer_index:
                        values = np.array(layer['values'])
                        word_embeddings.append(values)

        lines.append(np.mean(word_embeddings, axis=0))
    return lines

def get_cls_vectors(line_features, layer_index=-1):
    cls_vectors = []
    for line_f in line_features:
        for token_feature in line_f['features']:
            if token_feature['token'] == '[CLS]':

                for layer in token_feature['layers']:
                    if layer['index'] == layer_index:
                        values = np.array(layer['values'])
                        cls_vectors.append(values)

    return cls_vectors

def get_all_vectors(line_features):
    all_vectors = []
    for line_f in line_features:
        for token_feature in line_f['features']:
            all_layers = []
            for layer in token_feature['layers']:
                values = np.array(layer['values'])
                all_layers.extend(values)
            all_vectors.append(all_layers)

    return all_vectors

def concatenate_layers(embeddings):
    all_line_features = []
    for line in embeddings:
        token_features = line['features']
        line_features = []
        for token_data in token_features:
            token = token_data['token']
            concatenated_vector = [
                value for layer in token_data['layers']
                for value in layer['values']
            ]
            line_features.append((token, concatenated_vector))
        all_line_features.append(line_features)

    return all_line_features

AGGREGATOR_MAP = {
    'avg': get_avg_words,
    'cls': get_cls_vectors,
    'all': get_all_vectors
}

def get_embeddings(lines, pretrained_bert, **user_args):
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_bert, do_lower_case=True)

    model = BertModel.from_pretrained(pretrained_bert)

    extraction_args = {
        'lines': lines,
        'model': model,
        'tokenizer': tokenizer
    }
    extraction_args.update(user_args)

    all_lines_features = extract_features(**extraction_args)
    embeddings = concatenate_layers(all_lines_features)

    return embeddings

# def infer(config):
#     experiment_dir = os.path.abspath(config.experiment_dir)
#     setup_dir = os.path.join(experiment_dir, 'setup')
#     infer_dir = os.path.join(experiment_dir, 'infer')
#     os.makedirs(infer_dir, exist_ok=True)

#     submissions_dir = os.path.join(
#         setup_dir,
#         'submissions-features')

#     archives_dir = os.path.join(
#         setup_dir,
#         'archives-features')

#     submission_embeddings = []
#     paper_lookup = []
#     for emb_file in os.listdir(submissions_dir):
#         embedding_list = np.load(os.path.join(submissions_dir, emb_file))
#         for emb in embedding_list:
#             submission_embeddings.append(emb)
#             paper_lookup.append(emb_file.replace('.npy', ''))
#     submission_matrix = np.asarray(submission_embeddings)

#     archive_embeddings = []
#     author_lookup = []
#     for emb_file in os.listdir(archives_dir):
#         embedding_list = np.load(os.path.join(archives_dir, emb_file))
#         for emb in embedding_list:
#             archive_embeddings.append(emb)
#             author_lookup.append(emb_file.replace('.npy', ''))
#     archive_matrix = np.asarray(archive_embeddings)

#     scores = np.dot(submission_matrix, np.transpose(archive_matrix))

#     score_file_path = os.path.join(infer_dir, config.name + '-scores.jsonl')
#     with open(score_file_path, 'w') as f:
#         for paper_idx, row in enumerate(scores):
#             paper_id = paper_lookup[paper_idx]

#             author_max_scores = defaultdict(int)

#             for author_idx, score in enumerate(row):
#                 a = author_lookup[author_idx]
#                 if author_max_scores[a] < score:
#                     author_max_scores[a] = score

#             for author_id, max_score in author_max_scores.items():
#                 result = {
#                     'source_id': paper_id,
#                     'target_id': author_id,
#                     'score': float(max_score)
#                 }
#                 f.write(json.dumps(result) + '\n')
