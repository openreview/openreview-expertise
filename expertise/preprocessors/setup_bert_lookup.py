import csv, importlib, itertools, json, math, os, pickle, random
from collections import defaultdict

import numpy as np

import openreview
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
from expertise.utils.config import Config
import expertise.utils as utils
from expertise.utils.data_to_sample import data_to_sample
import argparse
import torch

def get_values(doc_feature):
    cls = doc_feature['features'][0]
    values = cls['layers'][0]['values']
    return values

def setup(config):

    print('starting setup')
    # features_dir = './scibert_features/akbc19/setup/archives-features/'
    features_dir = config.bert_features_dir
    archive_features_dir = os.path.join(features_dir, 'archives-features')
    submission_features_dir = os.path.join(features_dir, 'submissions-features')

    bert_lookup = {}

    for target_dir in [archive_features_dir, submission_features_dir]:
        for filename in os.listdir(target_dir):
            print(filename)
            item_id = filename.replace('.npy','')
            filepath = os.path.join(target_dir, filename)
            archive_features = np.load(filepath)

            archive_values = []
            for doc_feature in archive_features:
                archive_values.append(get_values(doc_feature))

            if len(archive_values) == 0:
                archive_values = [np.zeros(768)]

            result = np.array(archive_values)
            bert_lookup[item_id] = torch.Tensor(result)

    utils.dump_pkl(os.path.join(config.setup_dir, 'bert_lookup.pkl'), bert_lookup)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(config_path)

    setup_path = os.path.join(experiment_path, 'setup')
    if not os.path.isdir(setup_path):
        os.mkdir(setup_path)

    setup(config)
