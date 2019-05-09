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


def setup(config):

    print('starting setup')
    # features_dir = './scibert_features/akbc19/setup/archives-features/'
    features_dir = config.bert_features_dir
    archive_features_dir = os.path.join(features_dir, 'archives-features')
    submission_features_dir = os.path.join(features_dir, 'submissions-features')
    textrank_kps = utils.load_pkl(os.path.join(config.setup_dir, 'textrank_kps_by_id.pkl'))

    bert_lookup = {}

    for target_dir in [archive_features_dir, submission_features_dir]:
        for filename in os.listdir(target_dir):
            print(filename)
            item_id = filename.replace('.npy','')
            filepath = os.path.join(target_dir, filename)
            archives = np.load(filepath)

            document_kps = textrank_kps[item_id]
            kps_seen = []
            kp_features = []

            for document in archives:
                features = document['features']
                for feature in features:
                    if feature['token'] in document_kps and feature['token'] not in kps_seen:
                        kps_seen.append(feature['token'])
                        kp_features.append(feature['layers'][-1]['values'])

            kp_features = kp_features[:config.max_num_keyphrases]

            while len(kp_features) < config.max_num_keyphrases:
                kp_features.append(np.zeros(config.bert_dim))

            result = np.array(kp_features)
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
