import argparse
import os
import csv, json
from collections import defaultdict
import expertise
from expertise import utils
import itertools

from expertise.dataset import Dataset
from datetime import datetime
import multiprocessing as mp

from gensim.corpora.textcorpus import TextCorpus
from gensim.similarities.docsim import SparseMatrixSimilarity

import numpy as np
from tqdm import tqdm
import ipdb

def infer(config):
    experiment_dir = os.path.abspath(config.experiment_dir)

    model = utils.load_pkl(config.tfidf_model)

    dataset = Dataset(**config.dataset)

    paperidx_by_id = {
        paperid: index
        for index, paperid
        in enumerate(model.bow_archives_by_paperid.keys())
    }

    score_file_path = os.path.join(experiment_dir, config.name + '-scores.csv')

    bids_by_forum = expertise.utils.get_bids_by_forum(dataset)
    submission_ids = [n for n in dataset.submission_ids]
    reviewer_ids = [r for r in dataset.reviewer_ids]
    # samples = expertise.utils.format_bid_labels(submission_ids, bids_by_forum)

    scores = {}

    with open(score_file_path, 'w') as w:
        for paperid, userid in itertools.product(submission_ids, reviewer_ids):
            # label = data['label']

            if userid not in scores:
                # bow_archive is a list of BOWs.
                if userid in model.bow_archives_by_userid and len(model.bow_archives_by_userid[userid]) > 0:
                    bow_archive = model.bow_archives_by_userid[userid]
                else:
                    bow_archive = [[]]

                best_scores = np.amax(model.index[bow_archive], axis=0)
                scores[userid] = best_scores

            if paperid in paperidx_by_id:
                paper_index = paperidx_by_id[paperid]
                score = scores[userid][paper_index]

                # result = {
                #     'source_id': paperid,
                #     'target_id': userid,
                #     'score': float(score),
                #     # 'label': int(label)
                # }

                w.write('{0},{1},{2:.3f}'.format(paperid, userid, score))
                w.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    config = expertise.config.ModelConfig()
    config.update_from_file(args.config_path)

    infer(config)

