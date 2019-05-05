import os
from expertise import utils
from expertise.utils.vocab import Vocab
from expertise.utils.dataset import Dataset
from expertise.preprocessors.textrank import TextRank
from itertools import chain, product
from collections import defaultdict
import numpy as np

def setup_train_dev_test(config):
    '''
    Handles train/dev/test split common to all models
    '''

    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    dataset = Dataset(**config.dataset)

    submission_ids = list(dataset.submission_ids)
    reviewer_ids = list(dataset.reviewer_ids)
    featureids_by_id = defaultdict(list)

    train_split, dev_split, test_split = utils.split_ids(submission_ids, seed=config.data_split_seed)

    utils.dump_csv(os.path.join(setup_dir, 'train_split.csv'), [[id] for id in train_split])
    utils.dump_csv(os.path.join(setup_dir, 'dev_split.csv'), [[id] for id in dev_split])
    utils.dump_csv(os.path.join(setup_dir, 'test_split.csv'), [[id] for id in test_split])

    positive_pairs = [p for p in dataset.positive_pairs()]
    negative_pairs = list(set([p for p in chain(dataset.negative_pairs(), dataset.nonpositive_pairs())]))

    positives_lookup = defaultdict(list)
    for s_id, r_id in positive_pairs:
        positives_lookup[s_id].append(r_id)
        positives_lookup[r_id].append(s_id)

    negatives_lookup = defaultdict(list)
    for s_id, r_id in negative_pairs:
        negatives_lookup[s_id].append(r_id)
        negatives_lookup[r_id].append(s_id)

    def _write_eval_data(f, data_split, pos_lookup, neg_lookup):
        for submission_id in data_split:
            for reviewer_pos_id in pos_lookup[submission_id]:
                if reviewer_pos_id:
                    f.write('\t'.join([
                        submission_id,
                        reviewer_pos_id,
                        '1'
                    ]))
                    f.write('\n')

            for reviewer_neg_id in neg_lookup[submission_id]:
                if reviewer_neg_id:
                    f.write('\t'.join([
                        submission_id,
                        reviewer_neg_id,
                        '0'
                    ]))
                    f.write('\n')


    with open(os.path.join(config.setup_dir, 'test_samples.csv'), 'w') as f:
        _write_eval_data(f, test_split, positives_lookup, negatives_lookup)

    with open(os.path.join(config.setup_dir, 'dev_samples.csv'), 'w') as f:
        _write_eval_data(f, dev_split, positives_lookup, negatives_lookup)


    return train_split, dev_split, test_split
