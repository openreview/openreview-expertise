import os
from expertise.utils.dataset import Dataset
from expertise.utils.vocab import Vocab
from expertise import utils
from expertise.preprocessors.textrank import TextRank
from itertools import chain
from collections import defaultdict
import numpy as np


def setup_kp_features(config):
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

    all_featureids = [fid
     for fids in featureids_by_id.values()
     for fid in fids]

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = utils.split_ids(all_featureids, seed=config.random_seed)

    return featureids_by_id, train_set_ids, dev_set_ids, test_set_ids, vocab

