from collections import defaultdict
import json, csv
import openreview
import os
import pickle
import expertise
from expertise import preprocessors
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher

import itertools
import math
import random

import importlib

def get_train_dev_test_ids(labels_by_forum):
    '''
    TODO: is "forums" an appropriate variable name? will it always be a forum ID?
    '''
    random.seed(a=3577057385653016827)

    forums = sorted(labels_by_forum.keys())
    random.shuffle(forums)

    idxs = (math.floor(0.6 * len(forums)), math.floor(0.7 * len(forums)))

    train_set_ids = forums[:idxs[0]]
    train_set_ids = {f : sigs for f, sigs in labels_by_forum.items() if f in train_set_ids}
    dev_set_ids = forums[idxs[0]:idxs[1]]
    test_set_ids = forums[idxs[1]:]

    return train_set_ids, dev_set_ids, test_set_ids

def training_data(train_set_ids, labels_by_forum, reviewer_kps, submission_kps):
    for forum_id in train_set_ids:
        if forum_id in submission_kps:
            print('forum_id', forum_id)
            forum_kps = [kp for kp in submission_kps[forum_id]]
            forum_pos_signatures = sorted(labels_by_forum[forum_id]['pos'])
            forum_neg_signatures = sorted(labels_by_forum[forum_id]['neg'])
            print('forum_pos_signatures',forum_pos_signatures)
            print('forum_neg_signatures',forum_neg_signatures)
            pos_neg_pairs = itertools.product(forum_pos_signatures, forum_neg_signatures)
            for pos, neg in pos_neg_pairs:
                if pos in reviewer_kps and neg in reviewer_kps:
                    yield (forum_kps, reviewer_kps[pos], reviewer_kps[neg])

def eval_data(eval_set_ids, labels_by_forum):
    for forum_id in eval_set_ids:
        for reviewer in labels_by_forum[forum_id]['pos']:
            yield (forum_id, reviewer, 1)

        for reviewer in labels_by_forum[forum_id]['neg']:
            yield (forum_id, reviewer, 0)

def build_labels(dataset):
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
        pos_and_neg_signatures_by_forum[forum_id]['pos'] = pos_signatures
        pos_and_neg_signatures_by_forum[forum_id]['neg'] = neg_signatures


    return pos_and_neg_signatures_by_forum

def dump_pkl(filepath, data):
    with open(filepath, 'wb') as f:
        f.write(pickle.dumps(data))

def dump_csv(filepath, data):
    '''
    Writes .csv files in a specific format preferred by some IESL students:
    tab-delimited columns, with keyphrases separated by spaces.
    '''
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for target, pos, neg in data:
            row = []
            for source in [target, pos, neg]:
                if type(source) == list:
                    row_source = ' '.join(source)
                elif type(source) in [str, int]:
                    row_source = source
                else:
                    raise TypeError('incompatible source type', type(source))
                row.append(row_source)

            writer.writerow(row)

def setup(setup_path, config, dataset):

    '''
    Processes the dataset and any other information needed by this model.
    '''

    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    # write train/dev/test labels to pickle file
    labels_by_forum = build_labels(dataset)
    labels_path = os.path.join(setup_path, 'labels.pkl')
    dump_pkl(labels_path, labels_by_forum)

    kps_by_submission = defaultdict(list)
    for file_id, text in dataset.submission_records():
        kps_by_submission[file_id].extend(keyphrases(text))

    submission_kps_path = os.path.join(setup_path, 'submission_kps.pkl')
    dump_pkl(submission_kps_path, kps_by_submission),

    # write keyphrases for reviewer archives to pickle file
    kps_by_reviewer = defaultdict(list)
    for file_id, text in dataset.reviewer_archives():
        kps_by_reviewer[file_id].extend(keyphrases(text))

    reviewer_kps_path = os.path.join(setup_path, 'reviewer_kps.pkl')
    dump_pkl(reviewer_kps_path, kps_by_reviewer)

    # define vocab and update it with keyphrases, then write to pickle file
    vocab = Vocab(max_num_keyphrases = config.max_num_keyphrases)

    for kps in kps_by_submission.values():
        vocab.load_items(kps)

    for kps in kps_by_reviewer.values():
        vocab.load_items(kps)

    vocab_file_path = os.path.join(setup_path, 'vocab.pkl')
    dump_pkl(vocab_file_path, vocab)

    train_set_ids, dev_set_ids, test_set_ids = get_train_dev_test_ids(labels_by_forum)

    train_set = training_data(train_set_ids, labels_by_forum, kps_by_reviewer, kps_by_submission)
    dev_set = eval_data(dev_set_ids, labels_by_forum)
    test_set = eval_data(test_set_ids, labels_by_forum)

    train_set_file = os.path.join(setup_path, 'train_set.tsv')
    train_samples_file = os.path.join(setup_path, 'train_samples.tsv')
    dump_csv(train_set_file, train_set)
    dump_csv(os.path.join(setup_path, 'dev_set.tsv'), dev_set)
    dump_csv(os.path.join(setup_path, 'test_set.tsv'), test_set)

    batcher = Batcher(config, vocab, input_file=train_set_file)
    batcher.dump_csv(train_samples_file)

