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

from expertise.utils.standard_setup import setup_kp_features

import ipdb

def setup(config):

    dataset = Dataset(**config.dataset)

    (featureids_by_id,
     train_set_ids,
     dev_set_ids,
     test_set_ids,
     vocab) = setup_kp_features(config)

    print('standard setup done')
    all_feature_ids = [fid for fids in featureids_by_id.values() for fid in fids]

    all_submission_fids = []
    all_reviewer_fids = []
    for fid in all_feature_ids:
        if fid.startswith('~'):
            all_reviewer_fids.append(fid)
        else:
            all_submission_fids.append(fid)

    '''
    Construct label matrix.

    Make this its own function?
    '''
    label_matrix = -1 * np.ones((len(all_feature_ids), len(all_feature_ids)))
    index_to_fid = sorted(all_feature_ids)
    fid_to_index = {fid: i for i, fid in enumerate(index_to_fid)}

    for forum, bids in dataset.bids(sequential=False, progressbar=True):
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

    print('label matrix', label_matrix)

    print('writing train samples')
    with open(os.path.join(config.setup_dir, 'train_samples.csv'), 'w') as f:
        for src_fid in train_set_ids:
            row_num = fid_to_index[src_fid]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in train_set_ids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in train_set_ids]

            for pos_fid, neg_fid in itertools.product(pos_fids, neg_fids):
                f.write('\t'.join([src_fid, pos_fid, neg_fid]))
                f.write('\n')

    print('writing dev samples')
    with open(os.path.join(config.setup_dir, 'dev_samples.csv'), 'w') as f:
        for src_fid in dev_set_ids:
            row_num = fid_to_index[src_fid]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in dev_set_ids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in dev_set_ids]
            targets = [(p, 1) for p in pos_fids] + [(n, 0) for n in neg_fids]

            for target_fid, label in random.sample(targets, len(targets)):
                f.write('\t'.join([src_fid, target_fid, str(label)]))
                f.write('\n')

    print('writing test samples')
    with open(os.path.join(config.setup_dir, 'test_samples.csv'), 'w') as f:
        for src_fid in test_set_ids:
            row_num = fid_to_index[src_fid]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in test_set_ids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in test_set_ids]
            targets = [(p, 1) for p in pos_fids] + [(n, 0) for n in neg_fids]
            for target_fid, label in random.sample(targets, len(targets)):
                f.write('\t'.join([src_fid, target_fid, str(label)]))
                f.write('\n')

    print('writing full labels')
    with open(os.path.join(config.setup_dir, 'full_labels.csv'), 'w') as f:
        for src_fid in all_submission_fids:
            row_num = fid_to_index[src_fid]
            pos_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==1) if val == True and index_to_fid[i] in all_reviewer_fids]
            neg_fids = [index_to_fid[i] for i, val in enumerate(label_matrix[row_num]==0) if val == True and index_to_fid[i] in all_reviewer_fids]
            targets = [(p, 1) for p in pos_fids] + [(n, 0) for n in neg_fids]
            for target_fid, label in targets:
                f.write('\t'.join([src_fid, target_fid, str(label)]))
                f.write('\n')


