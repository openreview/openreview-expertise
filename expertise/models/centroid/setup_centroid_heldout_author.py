from collections import defaultdict
import json, csv
import openreview
import os
import pickle
import expertise
from expertise import preprocessors
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
import expertise.utils as utils

import numpy as np

import itertools
import math
import random

import importlib

def format_training_data(kp_lists_by_reviewer, kps_by_reviewer):
    for source_reviewer, reviewer_kp_lists in kp_lists_by_reviewer.items():
        print('processing source reviewer',source_reviewer)
        '''
        kp_lists_by_reviewer is a dict, keyed on reviewer ID, where each value is a list of lists.
            each outer list corresponds to that reviewer's papers.
            each inner list contains the keyphrases for that paper.

        kps_by_reviewer is a dict, also keyed on reviewer_id, where each value is a list of all
            keyphrases for the reviewer.
        '''

        negative_reviewers = [n for n in kps_by_reviewer if n != source_reviewer]

        for source_kps, remainder_kp_lists in utils.holdouts(reviewer_kp_lists):
            '''
            source_kps is a list of keyphrases representing one of the source_reviewer's papers.
            remainder_kp_lists is a list of lists representing the other papers.
            '''

            positive_kps = [kp for kp_list in remainder_kp_lists for kp in kp_list]

            # pick a random reviewer (who is not the same as the source/positive reviewer)
            negative_reviewer = random.sample(negative_reviewers, 1)[0]
            negative_kps = kps_by_reviewer[negative_reviewer]

            data = {
                'source': source_kps,
                'source_id': source_reviewer,
                'positive': positive_kps,
                'positive_id': source_reviewer,
                'negative': negative_kps,
                'negative_id': negative_reviewer
            }

            yield data

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
    kp_lists_by_reviewer = defaultdict(list)
    for reviewer_id, text in dataset.reviewer_archives():
        kp_list = keyphrases(text)
        kps_by_reviewer[reviewer_id].extend(kp_list)
        kp_lists_by_reviewer[reviewer_id].append(kp_list)
        vocab.load_items(kp_list)

    config.setup_save(vocab, 'vocab.pkl')

    train_set_ids, dev_set_ids, test_set_ids = utils.split_ids(list(kps_by_submission.keys()))

    train_set = format_training_data(kp_lists_by_reviewer, kps_by_reviewer)

    train_set_file = config.setup_save(train_set, 'train_set.jsonl')

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

