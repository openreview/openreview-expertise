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

from expertise.preprocessors.textrank import TextRank

import ipdb

def setup(config):

    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    feature_dirs = [
        'features'
    ]

    for d in feature_dirs:
        os.makedirs(os.path.join(setup_dir, d), exist_ok=True)
        print('created', d)

    dataset = Dataset(**config.dataset)

    vocab = Vocab()

    # keyphrases = importlib.import_module(config.keyphrases).keyphrases
    textrank = TextRank()

    kps_by_id = {}
    for item_id, text_list in chain(
        dataset.submissions(sequential=False, fields=['title','abstract']),
        dataset.archives(sequential=False, fields=['title','abstract'])):

        kp_lists = []
        for text in text_list:
            textrank.analyze(text)
            kps = [kp for kp, score in textrank.keyphrases()]
            vocab.load_items(kps)
            kp_lists.append(kps)
        kps_by_id[item_id] = kp_lists

    vocab.dump_csv(outfile=os.path.join(config.setup_dir, 'vocab'))

    submission_ids = list(dataset.submission_ids)
    reviewer_ids = list(dataset.reviewer_ids)
    feature_ids = []
    featureids_by_id = defaultdict(list)

    for id in submission_ids + reviewer_ids:
        for i, kps in enumerate(kps_by_id[id]):
            fid = f'{id}:{i}'
            outfile = os.path.join(
                setup_dir, 'features', f'{fid}.npy')

            features = vocab.to_ints(kps, length=config.max_num_keyphrases)
            np.save(outfile, features)
            feature_ids.append(fid)
            featureids_by_id[id].append(fid)

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = utils.split_ids(feature_ids)

    '''
    Construct label matrix.

    Make this its own function?
    '''
    label_matrix = -1 * np.ones((len(feature_ids), len(feature_ids)))
    index_to_fid = sorted(feature_ids)
    fid_to_index = {fid: i for i, fid in enumerate(index_to_fid)}

    for forum, bids in dataset.bids(sequential=False, progressbar=False):
        forum_fids = featureids_by_id[forum]

        for forum_fid in forum_fids:
            for b in bids:
                reviewer_id = b['signatures'][0]
                reviewer_fids = featureids_by_id[reviewer_id]

                for reviewer_fid in reviewer_fids:
                    reviewer_index = fid_to_index[reviewer_fid]
                    forum_index = fid_to_index[forum_fid]

                    if b['tag'] in dataset.positive_bid_values:
                        label_matrix[forum_index][reviewer_index] = 1
                        label_matrix[reviewer_index][forum_index] = 1
                    else:
                        label_matrix[forum_index][reviewer_index] = 0
                        label_matrix[reviewer_index][forum_index] = 0

    with open(os.path.join(setup_dir, 'train_samples.csv'), 'w') as f:
        for src_id in train_set_ids:
            row_num = fid_to_index[src_id]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in train_set_ids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in train_set_ids]

            for pos_fid, neg_fid in itertools.product(pos_fids, neg_fids):
                f.write('\t'.join([src_id, pos_fid, neg_fid]))
                f.write('\n')

    with open(os.path.join(setup_dir, 'dev_samples.csv'), 'w') as f:
        for src_id in dev_set_ids:
            row_num = fid_to_index[src_id]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in dev_set_ids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in dev_set_ids]
            targets = [(p, 1) for p in pos_fids] + [(n, 0) for n in neg_fids]

            for target_id, label in random.sample(targets, len(targets)):
                f.write('\t'.join([src_id, target_id, str(label)]))
                f.write('\n')

    with open(os.path.join(setup_dir, 'test_samples.csv'), 'w') as f:
        for src_id in test_set_ids:
            row_num = fid_to_index[src_id]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in test_set_ids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in test_set_ids]
            targets = [(p, 1) for p in pos_fids] + [(n, 0) for n in neg_fids]
            for target_id, label in random.sample(targets, len(targets)):
                f.write('\t'.join([src_id, target_id, str(label)]))
                f.write('\n')

