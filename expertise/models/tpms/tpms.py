import os
import csv, json
from collections import defaultdict
from expertise.evaluators.mean_avg_precision import eval_map
from expertise.evaluators.hits_at_k import eval_hits_at_k
from expertise.dataset import Dataset

from expertise import utils
import ipdb

def setup(config):
    assert os.path.exists(config.tpms_scores_file), 'This model requires a pre-computed tpms score file.'

    dataset = Dataset(**config.dataset)
    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = utils.split_ids(list(dataset.submission_ids), seed=config.random_seed)

    bids_by_forum = utils.get_bids_by_forum(dataset)

    test_labels = utils.format_bid_labels(test_set_ids, bids_by_forum)

    utils.dump_jsonl(os.path.join(config.setup_dir, 'test_labels.jsonl'), test_labels)

def train(config):
    print('Nothing to train. This model is a shell that reads in pre-computed TPMS scores.')
    assert os.path.exists(config.tpms_scores_file), 'This model requires a pre-computed tpms score file.'

def infer(config):
    print('Nothing to infer. This model is a shell that reads in pre-computed TPMS scores.')
    assert os.path.exists(config.tpms_scores_file), 'This model requires a pre-computed tpms score file.'

def test(config):

    score_file_path = os.path.join(config.test_dir, 'test_scores.jsonl')
    labels_file_path = os.path.join(config.setup_dir, 'test_labels.jsonl')
    tpms_scores_file = config.tpms_scores_file

    scores = {}
    for data in utils.jsonl_reader(tpms_scores_file):
        source_id = data['source_id']
        target_id = data['target_id']
        score = data['score']
        if source_id not in scores:
            scores[source_id] = {}

        if target_id not in scores[source_id]:
            scores[source_id][target_id] = score


    with open(score_file_path, 'w') as w:

        for data in utils.jsonl_reader(labels_file_path):
            paperid = data['source_id']
            userid = data['target_id']
            label = data['label']

            if paperid in scores:
                score = scores[paperid].get(userid, 0.0)
                if float(score) > -float('inf'):
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
