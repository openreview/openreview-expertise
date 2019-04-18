import csv, importlib, itertools, json, math, os, pickle, random
from collections import defaultdict

import numpy as np

import openreview
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
import expertise.utils as utils
from expertise.utils.data_to_sample import data_to_sample

def setup(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''

    dataset = Dataset(**config.dataset)
    vocab = Vocab(max_num_keyphrases = config.max_num_keyphrases)

    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    bids_by_forum = {
        forum: {
            'positive': [b['signatures'][0] for b in bids if b['tag'] in dataset.positive_bid_values],
            'negative': [b['signatures'][0] for b in bids if b['tag'] not in dataset.positive_bid_values]
        } for forum, bids in dataset.bids(sequential=False, progressbar=False)
    }

    kps_by_submission = defaultdict(list)
    for submission_id, text in dataset.submissions(sequential=True):
        kp_list = keyphrases(text[0])
        kps_by_submission[submission_id].extend(kp_list)
        vocab.load_items(kp_list)

    kps_by_reviewer = defaultdict(list)
    for reviewer_id, text in dataset.archives(sequential=True):
        kp_list = keyphrases(text[0])
        kps_by_reviewer[reviewer_id].extend(kp_list)
        vocab.load_items(kp_list)

    vocab.dump_csv(outfile=os.path.join(config.setup_dir, 'vocab'))

    train_set_ids, dev_set_ids, test_set_ids = utils.split_ids(list(bids_by_forum.keys()))

    train_set = utils.format_data_bids(
        train_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission)

    dev_set = utils.format_data_bids(
        dev_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission,
        max_num_keyphrases=10)

    test_set = utils.format_data_bids(
        test_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission,
        max_num_keyphrases=10)

    dev_labels = utils.format_bid_labels(dev_set_ids, bids_by_forum)
    dev_labels_file = config.setup_save(dev_labels, 'dev_labels.jsonl')

    test_labels = utils.format_bid_labels(test_set_ids, bids_by_forum)
    test_labels_file = config.setup_save(test_labels, 'test_labels.jsonl')

    train_set_file = config.setup_save(train_set, 'train_set.jsonl')
    dev_set_file = config.setup_save(dev_set, 'dev_set.jsonl')
    test_set_file = config.setup_save(test_set, 'test_set.jsonl')

    # for very large datasets, the training data should be shuffled in advance.
    # (you can use /expertise/utils/shuffle_big_file.py to do this)
    train_samples = (data_to_sample(data, vocab) for data in utils.jsonl_reader(train_set_file))
    config.setup_save(train_samples, 'train_samples_permuted.jsonl')

    # dev_set_permuted = np.random.permutation(list(utils.jsonl_reader(dev_set_file)))
    dev_samples = (data_to_sample(data, vocab) for data in utils.jsonl_reader(dev_set_file))
    config.setup_save(dev_samples, 'dev_samples.jsonl')

    # test_set_permuted = np.random.permutation(list(utils.jsonl_reader(test_set_file)))
    test_samples = (data_to_sample(data, vocab) for data in utils.jsonl_reader(test_set_file))
    config.setup_save(test_samples, 'test_samples.jsonl')

