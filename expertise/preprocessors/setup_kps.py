import csv, importlib, itertools, json, math, os, pickle, random
import argparse
from collections import defaultdict

import numpy as np

import openreview
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
from expertise.utils.config import Config
import expertise.utils as utils
from expertise.utils.shuffle_big_file import build_folds
from expertise.utils.data_to_sample import data_to_sample

def setup(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''
    print('starting setup')
    dataset = Dataset(**config.dataset)
    vocab = Vocab()
    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    print('keyphrase extraction')
    bids_by_forum = utils.get_bids_by_forum(dataset)
    kps_by_id = {}
    all_archives = itertools.chain(
        dataset.submissions(sequential=False),
        dataset.archives(sequential=False))

    for archive_id, text_list in all_archives:
        scored_kps = []
        for text in text_list:
            kp_list = keyphrases(text, include_scores=True)
            scored_kps.extend(kp_list)
        sorted_kps = [kp for kp, _ in sorted(scored_kps, key=lambda x: x[1], reverse=True)]
        kp_list = []
        kp_count = 0
        for kp in sorted_kps:
            if kp not in kp_list:
                kp_list.append(kp)
                kp_count += 1
            if kp_count >= config.max_num_keyphrases:
                break

        vocab.load_items(kp_list)
        assert archive_id not in kps_by_id
        kps_by_id[archive_id] = kp_list

    utils.dump_pkl(os.path.join(config.setup_dir, 'kps_by_id.pkl'), kps_by_id)
    utils.dump_pkl(os.path.join(config.setup_dir, 'vocab.pkl'), vocab)

    bids_by_forum = utils.get_bids_by_forum(dataset)

    # why restrict to just bid forums? come back to this later.
    valid_ids = list(dataset.submission_ids)
    formatted_data = utils.format_data_bids(
        valid_ids,
        bids_by_forum,
        kps_by_id,
        max_num_keyphrases=config.max_num_keyphrases,
        sequential=False)

    folds_dir = os.path.join(config.setup_dir, 'folds')
    if not os.path.exists(folds_dir):
        os.mkdir(folds_dir)

    # build_folds((json.dumps(d)+'\n' for d in formatted_data), folds_dir, config.num_folds)

    for forum_id, data_rows in formatted_data:
        forum_data_path = os.path.join(folds_dir, f'{forum_id}.jsonl')
        utils.dump_jsonl(forum_data_path, data_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    parser.add_argument('additional_params', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(config_path)

    setup_path = os.path.join(experiment_path, 'setup')
    if not os.path.isdir(setup_path):
        os.mkdir(setup_path)

    setup(config, *args.additional_params)
