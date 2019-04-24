import csv, importlib, itertools, json, math, os, pickle, random
from collections import defaultdict
from itertools import chain, product, cycle

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

    features_lookup, vocab, train_dev_test = setup_kp_features(config)

    positive_pairs = [p for p in dataset.positive_pairs()]
    negative_pairs = [p for p in dataset.negative_pairs()]

    positives_lookup = defaultdict(list)
    for s_id, r_id in positive_pairs:
        positives_lookup[s_id].append(r_id)
        positives_lookup[r_id].append(s_id)

    negatives_lookup = defaultdict(list)
    for s_id, r_id in negative_pairs:
        negatives_lookup[s_id].append(r_id)
        negatives_lookup[r_id].append(s_id)

    all_ids = [id for id in set(chain(
        positives_lookup.keys(),
        negatives_lookup.keys(),
        features_lookup.keys()
    ))]

    print(sorted(all_ids))

    random.seed(config.random_seed)


    def _sample_generator(id, lookup):
        items = random.sample(
            lookup.get(id, []),
            len(lookup.get(id, [])))

        if len(items) == 0:
            '''
            If the item list is empty, the generator will stop.
            Appending None to the list lets the generator cycle None values forever,
            which can be caught and handled downstream.
            '''
            items.append(None)

        for positive in cycle(items):
            yield positive

    positive_samplers = {
        id: _sample_generator(id, positives_lookup) for id in all_ids
    }

    negative_samplers = {
        id: _sample_generator(id, negatives_lookup) for id in all_ids
    }

    feature_samplers = {
        id: _sample_generator(id, features_lookup) for id in all_ids
    }

    def _sample_feature(id):
        sampler = feature_samplers.get(id)
        if sampler:
            return next(sampler)
        else:
            return None

    def _sample_positive(id):
        sampler = positive_samplers.get(id)
        if sampler:
            return next(sampler)
        else:
            return None

    def _sample_negative(id):
        sampler = negative_samplers.get(id)
        if sampler:
            return next(sampler)
        else:
            '''
            Open question: when the given `id` has no negative samples,
            should I return a sample that has no label?

            e.g. something like this:
            >>> neutral_sampler = neutral_samplers[id]
            >>> return next(neutral_sampler)
            '''
            return None

    (train_split,
     dev_split,
     test_split) = train_dev_test

    '''
    Generate training samples.

    Hypothesis: train_samples_per_pair should be set to the median
        number of documents per archive in the dataset.
    '''
    train_samples_per_pair = 1000
    with open(os.path.join(config.setup_dir, 'train_samples.csv'), 'w') as f:
        for iteration in range(train_samples_per_pair):
            for submission_id in train_split:
                '''
                Write out two samples:
                one that finds the negative sample of the submission,
                the other that finds the negative sample of the reviewer.
                '''
                submission_feat = _sample_feature(submission_id)

                reviewer_id = _sample_positive(submission_id)
                reviewer_feat = _sample_feature(reviewer_id)

                reviewer_neg_feat = _sample_feature(_sample_negative(reviewer_id))
                submission_neg_feat = _sample_feature(_sample_negative(submission_id))

                '''
                Open question: is this the right way to be generating random training samples?
                '''
                if submission_feat and reviewer_feat and submission_neg_feat:
                    f.write('\t'.join([
                        submission_feat,
                        reviewer_feat,
                        submission_neg_feat]))
                    f.write('\n')

                if submission_feat and reviewer_feat and reviewer_neg_feat:
                    f.write('\t'.join([
                        reviewer_feat,
                        submission_feat,
                        reviewer_neg_feat]))
                    f.write('\n')

    '''
    Generate dev and test samples.

    '''
    def _write_eval_data(f, data_split):
        for submission_id in data_split:
            for reviewer_pos_id in positives_lookup[submission_id]:
                if reviewer_pos_id:
                    f.write('\t'.join([
                        submission_id,
                        reviewer_pos_id,
                        '1'
                    ]))
                    f.write('\n')

            for reviewer_neg_id in negatives_lookup[submission_id]:
                if reviewer_neg_id:
                    f.write('\t'.join([
                        submission_id,
                        reviewer_neg_id,
                        '0'
                    ]))
                    f.write('\n')

    with open(os.path.join(config.setup_dir, 'dev_samples.csv'), 'w') as f:
        _write_eval_data(f, dev_split)

    with open(os.path.join(config.setup_dir, 'test_samples.csv'), 'w') as f:
        _write_eval_data(f, test_split)
