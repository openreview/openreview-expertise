import csv, importlib, itertools, json, math, os, pickle, random
from collections import defaultdict
from itertools import chain

import numpy as np

import openreview
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
import expertise.utils as utils
from expertise.utils.data_to_sample import data_to_sample

import ipdb

def setup(config):

    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    feature_dirs = [
        'submissions-features',
        'archives-features'
    ]

    for d in feature_dirs:
        os.makedirs(os.path.join(setup_dir, d), exist_ok=True)
        print('created', d)

    dataset = Dataset(**config.dataset)

    vocab = Vocab(max_num_keyphrases=config.max_num_keyphrases)

    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    kps_by_id = {}
    for item_id, text_list in chain(
        dataset.submissions(sequential=False),
        dataset.archives(sequential=False)):

        kp_lists = []
        for text in text_list:
            kps = keyphrases(text)
            vocab.load_items(kps)
            kp_lists.append(kps)
        kps_by_id[item_id] = kp_lists

    vocab.dump_csv(outfile=os.path.join(config.setup_dir, 'vocab'))


    submission_ids = []
    for submission_id, text_list in dataset.submissions(sequential=True):
        outfile = os.path.join(setup_dir, 'submissions-features', submission_id + '.npy')
        features = np.array([vocab.to_ints(kps) for kps in kps_by_id[submission_id]])
        np.save(outfile, features)
        submission_ids.append(submission_id)

    for reviewer_id, text_list in dataset.archives(sequential=False):
        outfile = os.path.join(setup_dir, 'archives-features', reviewer_id + '.npy')
        features = np.array([vocab.to_ints(kps) for kps in kps_by_id[reviewer_id]])
        np.save(outfile, features)

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = utils.split_ids(submission_ids)


    # we need a defaultdict here because not all submissions have bids,
    bids_by_forum = defaultdict(lambda: {'pos':[], 'neg':[]})
    for forum, bids in dataset.bids(sequential=False, progressbar=False):
        for b in bids:
            if b['tag'] in dataset.positive_bid_values:
                bids_by_forum[forum]['pos'].append(b['signatures'][0])
            else:
                bids_by_forum[forum]['neg'].append(b['signatures'][0])

    with open(os.path.join(setup_dir, 'train_samples.csv'), 'w') as f:
        for source_id in train_set_ids:
            pos_targets = bids_by_forum[source_id]['pos']
            neg_targets = bids_by_forum[source_id]['neg']
            for pos_id, neg_id in itertools.product(pos_targets, neg_targets):
                f.write('\t'.join([source_id, pos_id, neg_id]))
                f.write('\n')

    with open(os.path.join(setup_dir, 'dev_samples.csv'), 'w') as f:
        for source_id in dev_set_ids:
            pos_targets = bids_by_forum[source_id]['pos']
            neg_targets = bids_by_forum[source_id]['neg']
            targets = [(p, 1) for p in pos_targets] + [(n, 0) for n in neg_targets]
            for target_id, label in random.sample(targets, len(targets)):
                f.write('\t'.join([source_id, target_id, str(label)]))
                f.write('\n')

    with open(os.path.join(setup_dir, 'test_samples.csv'), 'w') as f:
        for source_id in test_set_ids:
            pos_targets = bids_by_forum[source_id]['pos']
            neg_targets = bids_by_forum[source_id]['neg']
            targets = [(p, 1) for p in pos_targets] + [(n, 0) for n in neg_targets]
            for target_id, label in random.sample(targets, len(targets)):
                f.write('\t'.join([source_id, target_id, str(label)]))
                f.write('\n')

