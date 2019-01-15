import os
import csv, json
from collections import defaultdict
from expertise import utils
from expertise.utils.config import Config
from expertise.utils.dataset import Dataset
from datetime import datetime
import multiprocessing as mp

from gensim.corpora.textcorpus import TextCorpus
from gensim.similarities.docsim import SparseMatrixSimilarity

import numpy as np



def infer(config):
    experiment_dir = os.path.abspath(config.experiment_dir)

    infer_dir = os.path.join(experiment_dir, 'infer')
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    train_dir = os.path.join(experiment_dir, 'train')
    assert os.path.isdir(train_dir), 'Train dir does not exist. Make sure that this model has been trained.'

    dataset = Dataset(**config.dataset)

    paper_ids = []
    reviewer_ids = set()

    print('loading model')
    model = utils.load_pkl(os.path.join(train_dir, 'model.pkl'))

    paper_texts = []
    for paper_id, paper_text in dataset.submissions():
        print(paper_id)
        paper_tokens = model.preprocess_content(paper_text)
        paper_bow = [(t[0], t[1]) for t in model.tfidf_model[model.tfidf_dictionary.doc2bow(paper_tokens)]]
        paper_texts.append(paper_bow)
        paper_ids.append(paper_id)

    #print(paper_texts)
    #print(len(model.tfidf_dictionary.keys()))

    index = SparseMatrixSimilarity(paper_texts, num_features=len(model.tfidf_dictionary.keys()))
    print('papers are preprocessed')

    reviewer_text_by_id = defaultdict(list)
    for reviewer_id, reviewer_text in dataset.archives():
        print(reviewer_id)
        reviewer_tokens = model.preprocess_content(reviewer_text)
        reviewer_bow = [(t[0], t[1]) for t in model.tfidf_model[model.tfidf_dictionary.doc2bow(reviewer_tokens)]]
        reviewer_text_by_id[reviewer_id].append(reviewer_bow)
        reviewer_ids.update([reviewer_id])

    print('reviewers are preprocessed')

    # appends new scores to an existing file, if possible
    score_file_path = os.path.join(infer_dir, config.name + '-scores.txt')

    with open(score_file_path, 'w') as f:
        for rev_id, reviewer_bows in reviewer_text_by_id.items():
            print('writing {}'.format(rev_id))
            scores = index[reviewer_bows]
            best_scores = np.amax(scores, axis=0)

            print(best_scores)
            for idx, paper_id in enumerate(paper_ids):
                result = {
                    'source_id': paper_id,
                    'target_id': rev_id,
                    'score': float(best_scores[idx])
                }
                f.write(json.dumps(result) + '\n')