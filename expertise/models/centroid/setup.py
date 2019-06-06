import csv, importlib, itertools, json, math, os, pickle, random
from collections import defaultdict

import numpy as np

import openreview
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.dataset import Dataset
import expertise.utils as utils
from expertise.utils.data_to_sample import data_to_sample

def setup(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''
    print('starting setup')
    dataset = Dataset(**config.dataset)
    bids_by_forum = utils.get_bids_by_forum(dataset)
    vocab = utils.load_pkl(os.path.join(config.kp_setup_dir, 'textrank_vocab.pkl'))

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = utils.split_ids(list(dataset.submission_ids), seed=config.random_seed)

    def fold_reader(id):
        fold_file = f'{id}.jsonl'
        fold_path = os.path.join(config.kp_setup_dir, 'folds', fold_file)
        return utils.jsonl_reader(fold_path)

    train_folds = [fold_reader(i) for i in train_set_ids]
    dev_folds = [fold_reader(i) for i in dev_set_ids]
    test_folds = [fold_reader(i) for i in test_set_ids]

    train_samples = (data_to_sample(
        data, vocab, config.max_num_keyphrases) for data in itertools.chain(*train_folds))

    train_samples_path = os.path.join(
        config.setup_dir, 'train_samples.jsonl')

    utils.dump_jsonl(train_samples_path, train_samples)

    dev_samples = (data_to_sample(
        data, vocab, config.max_num_keyphrases) for data in itertools.chain(*dev_folds))

    dev_samples_path = os.path.join(
        config.setup_dir, 'dev_samples.jsonl')

    utils.dump_jsonl(dev_samples_path, dev_samples)

    test_samples = (data_to_sample(
        data, vocab, config.max_num_keyphrases) for data in itertools.chain(*test_folds))

    test_samples_path = os.path.join(
        config.setup_dir, 'test_samples.jsonl')

    utils.dump_jsonl(test_samples_path, test_samples)

