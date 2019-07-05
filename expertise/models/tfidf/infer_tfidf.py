import os
import csv, json
from collections import defaultdict
from expertise import utils

from expertise.dataset import Dataset
from datetime import datetime
import multiprocessing as mp

from gensim.corpora.textcorpus import TextCorpus
from gensim.similarities.docsim import SparseMatrixSimilarity

import numpy as np
from tqdm import tqdm

def infer(config):
    experiment_dir = os.path.abspath(config.experiment_dir)

    infer_dir = os.path.join(experiment_dir, 'infer')
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    train_dir = os.path.join(experiment_dir, 'train')
    assert os.path.isdir(train_dir), 'Train dir does not exist. Make sure that this model has been trained.'

    model = utils.load_pkl(os.path.join(train_dir, 'model.pkl'))

    dataset = Dataset(**config.dataset)

    paperid_by_index = {index: paperid \
        for index, paperid in enumerate(model.bow_by_paperid.keys())}

    score_file_path = os.path.join(infer_dir, config.name + '-scores.jsonl')
    samples_file_path = os.path.join(config.setup_dir, 'test_samples.csv')

    scores = {}

    with open(score_file_path, 'w') as w, open(samples_file_path) as r:
        sample_reader = csv.reader(r, delimiter='\t')

        for paperid, userid, label in sample_reader:

            if userid not in scores:
                bow_archive = model.bow_archives_by_userid[userid]
                best_scores = np.amax(model.index[bow_archive], axis=0)
                scores[userid] = best_scores

            for paper_index, score in enumerate(scores[userid]):
                paperid = paperid_by_index[paper_index]

                result = {
                    'source_id': paperid,
                    'target_id': userid,
                    'score': float(score),
                    'label': int(label)
                }

                w.write(json.dumps(result) + '\n')
