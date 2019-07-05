import argparse
import itertools
import json
import os
from collections import OrderedDict

from expertise.utils.vocab import Vocab
from expertise.dataset import Dataset
from expertise.config import ModelConfig
import expertise.utils as utils

from .textrank_words import keyphrases

def run_textrank(config):
    '''
    First define the dataset, vocabulary, and keyphrase extractor
    '''
    print('starting setup')
    dataset = Dataset(**config.dataset)
    textrank_vocab = Vocab() # vocab used for textrank-based keyphrases
    full_vocab = Vocab() # vocab used on the full text

    print('keyphrase extraction')
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

    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'textrank_kps_by_id.pkl'), textrank_kps_by_id)
    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'full_kps_by_id.pkl'), full_kps_by_id)
    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'textrank_vocab.pkl'), textrank_vocab)
    utils.dump_pkl(os.path.join(config.kp_setup_dir, 'full_vocab.pkl'), full_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)

    with open(config_path) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    config = ModelConfig(**data)

    experiment_path = os.path.dirname(config_path)

    kps_dir = os.path.join(experiment_path, 'keyphrases')
    if not os.path.isdir(kps_dir):
        os.mkdir(kps_dir)
    config.update(kp_setup_dir=kps_dir)

    run_textrank(config)

    print('saving', config_path, config)
    config.save(config_path)
