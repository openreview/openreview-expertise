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

def split_ids(ids):
    '''
    TODO: is "forums" an appropriate variable name? will it always be a forum ID?
    '''
    random.seed(a=3577057385653016827)

    forums = sorted(ids)
    random.shuffle(forums)

    idxs = (math.floor(0.6 * len(forums)), math.floor(0.7 * len(forums)))

    train_set_ids = forums[:idxs[0]]
    # train_set_ids = {f : sigs for f, sigs in ids.items() if f in train_set_ids}
    dev_set_ids = forums[idxs[0]:idxs[1]]
    test_set_ids = forums[idxs[1]:]

    return train_set_ids, dev_set_ids, test_set_ids

def format_data(train_set_ids, bids_by_forum, reviewer_kps, submission_kps, max_num_keyphrases=None):
    formatted_data = []
    for forum_id in train_set_ids:
        if forum_id in submission_kps:

            forum_kps = [kp for kp in submission_kps[forum_id]][:max_num_keyphrases]
            forum_pos_signatures = sorted(bids_by_forum[forum_id]['positive'])
            forum_neg_signatures = sorted(bids_by_forum[forum_id]['negative'])

            pos_neg_pairs = itertools.product(forum_pos_signatures, forum_neg_signatures)
            for pos, neg in pos_neg_pairs:
                if pos in reviewer_kps and neg in reviewer_kps:
                    # yield (forum_kps, reviewer_kps[pos], reviewer_kps[neg])
                    data = {
                        'source': forum_kps[:max_num_keyphrases],
                        'source_id': forum_id,
                        'positive': reviewer_kps[pos][:max_num_keyphrases],
                        'positive_id': pos,
                        'negative': reviewer_kps[neg][:max_num_keyphrases],
                        'negative_id': neg
                    }

                    formatted_data.append(data)
    return formatted_data

def format_labels(eval_set_ids, bids_by_forum):
    for forum_id in eval_set_ids:
        for reviewer in bids_by_forum[forum_id]['positive']:
            yield {'source_id': forum_id, 'target_id': reviewer, 'label': 1}

        for reviewer in bids_by_forum[forum_id]['negative']:
            yield {'source_id': forum_id, 'target_id': reviewer, 'label': 0}

def get_bids_by_forum(dataset):
    binned_bids = {val: [] for val in dataset.bid_values}

    positive_labels = dataset.positive_bid_values

    users_w_bids = set()
    for bid in dataset.bids():
        binned_bids[bid.tag].append(bid)
        users_w_bids.update(bid.signatures)

    bids_by_forum = defaultdict(list)
    for bid in dataset.bids():
        bids_by_forum[bid.forum].append(bid)

    pos_and_neg_signatures_by_forum = {}

    # Get pos bids for forum
    for forum_id, forum_bids in bids_by_forum.items():
        forum_bids_flat = [{"signature": bid.signatures[0], "bid": bid.tag} for bid in forum_bids]
        neg_bids = [bid for bid in forum_bids_flat if bid["bid"] not in positive_labels]
        neg_signatures = [bid['signature'] for bid in neg_bids]
        pos_bids = [bid for bid in forum_bids_flat if bid["bid"] in positive_labels]
        pos_signatures = [bid['signature'] for bid in pos_bids]
        pos_and_neg_signatures_by_forum[forum_id] = {}
        pos_and_neg_signatures_by_forum[forum_id]['positive'] = pos_signatures
        pos_and_neg_signatures_by_forum[forum_id]['negative'] = neg_signatures


    return pos_and_neg_signatures_by_forum

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

    bids_by_forum = get_bids_by_forum(dataset)

    kps_by_submission = defaultdict(list)
    for file_id, text in dataset.submission_records():
        kp_list = keyphrases(text)
        kps_by_submission[file_id].extend(kp_list)
        vocab.load_items(kp_list)

    kps_by_reviewer = defaultdict(list)
    for file_id, text in dataset.reviewer_archives():
        kp_list = keyphrases(text)
        kps_by_reviewer[file_id].extend(kp_list)
        vocab.load_items(kp_list)

    config.setup_save(vocab, 'vocab.pkl')

    train_set_ids, dev_set_ids, test_set_ids = split_ids(list(bids_by_forum.keys()))

    train_set = format_data(
        train_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission)

    dev_set = format_data(
        dev_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission,
        max_num_keyphrases=10)

    test_set = format_data(
        test_set_ids,
        bids_by_forum,
        kps_by_reviewer,
        kps_by_submission,
        max_num_keyphrases=10)

    dev_labels = format_labels(dev_set_ids, bids_by_forum)
    dev_labels_file = config.setup_save(dev_labels, 'dev_labels.jsonl')

    test_labels = format_labels(test_set_ids, bids_by_forum)
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

