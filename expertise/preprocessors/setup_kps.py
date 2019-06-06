import csv, importlib, itertools, json, math, os, pickle, random
import argparse
from collections import defaultdict

import numpy as np

import openreview
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.dataset import Dataset
from expertise.config import Config
import expertise.utils as utils
from expertise.utils.shuffle_big_file import build_folds
from expertise.utils.data_to_sample import data_to_sample

def setup(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''
    print('starting setup')
    dataset = Dataset(**config.dataset)
    textrank_vocab = Vocab()
    full_vocab = Vocab()
    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    print('keyphrase extraction')
    bids_by_forum = utils.get_bids_by_forum(dataset)
    textrank_kps_by_id = {}
    full_kps_by_id = {}

    all_archives = itertools.chain(
        dataset.submissions(sequential=False),
        dataset.archives(sequential=False))

    for archive_id, text_list in all_archives:
        scored_kps = []
        full_kps = []
        for text in text_list:
            top_tokens, full_tokens = keyphrases(text, include_scores=True, include_tokenlist=True)
            scored_kps.extend(top_tokens)
            full_kps.append(full_tokens)
        sorted_kps = [kp for kp, _ in sorted(scored_kps, key=lambda x: x[1], reverse=True)]

        top_kps = []
        kp_count = 0
        for kp in sorted_kps:
            if kp not in top_kps:
                top_kps.append(kp)
                kp_count += 1
            if kp_count >= config.max_num_keyphrases:
                break

        textrank_vocab.load_items(top_kps)
        full_vocab.load_items([kp for archive in full_kps for kp in archive])
        assert archive_id not in textrank_kps_by_id
        textrank_kps_by_id[archive_id] = top_kps
        full_kps_by_id[archive_id] = full_kps

    utils.dump_pkl(os.path.join(config.setup_dir, 'textrank_kps_by_id.pkl'), textrank_kps_by_id)
    utils.dump_pkl(os.path.join(config.setup_dir, 'full_kps_by_id.pkl'), full_kps_by_id)
    utils.dump_pkl(os.path.join(config.setup_dir, 'textrank_vocab.pkl'), textrank_vocab)
    utils.dump_pkl(os.path.join(config.setup_dir, 'full_vocab.pkl'), full_vocab)

    bids_by_forum = utils.get_bids_by_forum(dataset)

    # this is only for the textrank-based models
    valid_ids = list(dataset.submission_ids)
    formatted_data = utils.format_data_bids(
        valid_ids,
        bids_by_forum,
        textrank_kps_by_id,
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
