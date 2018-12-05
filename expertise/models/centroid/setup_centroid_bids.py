import csv, importlib, itertools, json, math, os, pickle, random
from collections import defaultdict

import numpy as np

import openreview
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
import expertise.utils as utils


def data_to_sample(data, vocab, max_num_keyphrases=10):
    '''
    Converts one line of the training data into a training sample.

    Training samples consist of the following:

    source:
        a numpy array containing integers. Each integer corresponds to
        a token in the vocabulary. This array of tokens represents the
        source text.
    source_length:
        a list containing one element, an integer, which is the number
        of keyphrases in 'source'.
    positive:
        ...
    positive_length:
        Similar to "source_length", but applies to the "positive" list.
    negative:
        ...
    negative_length:
        Similar to "source_length", but applies to the "negative" list.

    '''


    source = vocab.to_ints(data['source'])
    source_length = [min(max_num_keyphrases, len(source))]
    positive = vocab.to_ints(data['positive'])
    positive_length = [min(max_num_keyphrases, len(positive))]
    negative = vocab.to_ints(data['negative'])
    negative_length = [min(max_num_keyphrases, len(negative))]

    sample = {
        'source': source,
        'source_length': source_length,
        'source_id': data['source_id'],
        'positive': positive,
        'positive_length': positive_length,
        'positive_id': data['positive_id'],
        'negative': negative,
        'negative_length': negative_length,
        'negative_id': data['negative_id']
    }

    return sample

def setup(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''

    dataset = Dataset(config.dataset)
    vocab = Vocab(max_num_keyphrases = config.max_num_keyphrases)
    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    bids_by_forum = utils.get_bids_by_forum(dataset)

    kps_by_submission = defaultdict(list)
    for submission_id, text in dataset.submission_records():
        kp_list = keyphrases(text)
        kps_by_submission[submission_id].extend(kp_list)
        vocab.load_items(kp_list)

    kps_by_reviewer = defaultdict(list)
    for reviewer_id, text in dataset.reviewer_archives():
        kp_list = keyphrases(text)
        kps_by_reviewer[reviewer_id].extend(kp_list)
        vocab.load_items(kp_list)

    config.setup_save(vocab, 'vocab.pkl')

    train_set_ids, dev_set_ids, test_set_ids = utils.split_ids(list(bids_by_forum.keys()))

    train_set = utils.format_bid_data(
        train_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission)

    dev_set = utils.format_bid_data(
        dev_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission,
        max_num_keyphrases=10)

    test_set = utils.format_bid_data(
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

    train_set_permuted = np.random.permutation(list(utils.jsonl_reader(train_set_file)))
    train_samples = (data_to_sample(data, vocab) for data in train_set_permuted)
    config.setup_save(train_samples, 'train_samples_permuted.jsonl')

    # dev_set_permuted = np.random.permutation(list(utils.jsonl_reader(dev_set_file)))
    dev_samples = (data_to_sample(data, vocab) for data in dev_set)
    config.setup_save(dev_samples, 'dev_samples.jsonl')

    # test_set_permuted = np.random.permutation(list(utils.jsonl_reader(test_set_file)))
    test_samples = (data_to_sample(data, vocab) for data in test_set)
    config.setup_save(test_samples, 'test_samples.jsonl')

