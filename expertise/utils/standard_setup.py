import os
from expertise import utils
from expertise.utils.vocab import Vocab
from expertise.utils.dataset import Dataset
from expertise.preprocessors.textrank import TextRank
from itertools import chain, product
from collections import defaultdict
import numpy as np

def setup_kp_features(config):
    '''
    Want to end up with:
        -   A list of positive training pairs, in the format (PAPER, REVIEWER)
        -   A train/dev/test split of the list of pairs
        -   A feature file for each document in the format ID:DOC_NUM.npy
        -   An index allowing fast lookup of feature files for a given ID

    '''

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
    featureids_by_id = defaultdict(list)

    for id in submission_ids + reviewer_ids:
        for i, kps in enumerate(kps_by_id[id]):
            fid = f'{id}:{i}'
            outfile = os.path.join(
                setup_dir, 'features', f'{fid}.npy')

            features = vocab.to_ints(kps, length=config.max_num_keyphrases)
            np.save(outfile, features)
            featureids_by_id[id].append(fid)

    train_split, dev_split, test_split = utils.split_ids(submission_ids, seed=config.random_seed)

    utils.dump_csv(os.path.join(setup_dir, 'train_split.csv'), [[id] for id in train_split])
    utils.dump_csv(os.path.join(setup_dir, 'dev_split.csv'), [[id] for id in dev_split])
    utils.dump_csv(os.path.join(setup_dir, 'test_split.csv'), [[id] for id in test_split])

    utils.dump_pkl(os.path.join(setup_dir, 'featureids_lookup.pkl'), featureids_by_id)

    return featureids_by_id, vocab, (train_split, dev_split, test_split)
