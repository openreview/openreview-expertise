import os
from collections import defaultdict
from expertise import utils
from expertise.utils.dataset import Dataset
from expertise.evaluators.mean_avg_precision import eval_map
from expertise.evaluators.hits_at_k import eval_hits_at_k
from tqdm import tqdm

def test(config):

    dataset = Dataset(**config.dataset)

    labels_by_reviewer_by_forum = defaultdict(dict)
    for bid in dataset.bids():
        label = 1 if bid.tag in dataset.positive_bid_values else 0
        labels_by_reviewer_by_forum[bid.forum][bid.signatures[0]] = label

    inferred_scores_path = os.path.join(config.infer_dir, config.name + '-scores.jsonl')

    labeled_data_list = []
    for data in utils.jsonl_reader(inferred_scores_path):
        forum = data['source_id']
        reviewer = data['target_id']
        score = float(data['score'])
        if not score >= 0.0:
            score = 0.0

        if reviewer in labels_by_reviewer_by_forum[forum]:
            label = labels_by_reviewer_by_forum[forum][reviewer]

            labeled_data = {k:v for k,v in data.items()}
            labeled_data.update({'label': label, 'score': score})
            labeled_data_list.append(labeled_data)

    config.test_save(labeled_data_list, 'score_labels.jsonl')

    labels_file = config.test_path('score_labels.jsonl')

    list_of_list_of_labels, list_of_list_of_scores = utils.load_labels(labels_file)

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
