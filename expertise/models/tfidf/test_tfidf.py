import os
import csv, json
import numpy as np
from collections import defaultdict
from expertise import utils
from expertise.utils.dataset import Dataset
from expertise.evaluators.mean_avg_precision import eval_map
from expertise.evaluators.hits_at_k import eval_hits_at_k

from tqdm import tqdm
import ipdb

def test(config):

    dataset = Dataset(**config.dataset)

    model = utils.load_pkl(os.path.join(config.train_dir, 'model.pkl'))

    paperidx_by_id = {
        paperid: index
        for index, paperid
        in enumerate(model.bow_by_paperid.keys())
    }

    score_file_path = os.path.join(config.test_dir, 'test_scores.jsonl')
    samples_file_path = os.path.join(config.setup_dir, 'test_samples.csv')

    scores = {}

    with open(score_file_path, 'w') as w, open(samples_file_path) as r:
        sample_reader = csv.reader(r, delimiter='\t')

        for paperid, userid, label in sample_reader:

            if userid not in scores:
                # bow_archive is a list of BOWs.
                bow_archive = model.bow_archives_by_userid.get(userid, [[]])
                best_scores = np.amax(model.index[bow_archive], axis=0)
                scores[userid] = best_scores

            if paperid in paperidx_by_id:
                paper_index = paperidx_by_id[paperid]
                score = scores[userid][paper_index]

                result = {
                    'source_id': paperid,
                    'target_id': userid,
                    'score': float(score),
                    'label': int(label)
                }

                w.write(json.dumps(result) + '\n')

    (list_of_list_of_labels,
     list_of_list_of_scores) = utils.load_labels(score_file_path)

    map_score = float(eval_map(list_of_list_of_labels, list_of_list_of_scores))
    hits_at_1 = float(eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=1))
    hits_at_3 = float(eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=3))
    hits_at_5 = float(eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=5))
    hits_at_10 = float(eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=10))

    score_lines = [
        [config.name, text, data] for text, data in [
            ('MAP', map_score),
            ('Hits@1', hits_at_1),
            ('Hits@3', hits_at_3),
            ('Hits@5', hits_at_5),
            ('Hits@10', hits_at_10)
        ]
    ]
    config.test_save(score_lines, 'test.scores.tsv')
