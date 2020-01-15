# Baseline untrained ELMo model for submission-reviewer affinity for
#  paper matching

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pickle
import random
from rank_bm25 import BM25Okapi
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from IPython import embed


def read_jsonl(jsonl_file):
    lines = []
    with open(jsonl_file, encoding='utf-8') as fin:
        for line in fin:
            lines.append(json.loads(line))
    return lines


def load_test_data(data_dir):
    # Load all of the submissions
    sub_file = os.path.join(data_dir, 'test-submissions.jsonl')
    #sub_file = os.path.join(data_dir, 'submissions.jsonl')
    subs = read_jsonl(sub_file)

    # Load user publications
    reviewer_file = os.path.join(data_dir, 'user_publications.jsonl')
    reviewer_pubs = read_jsonl(reviewer_file)
    reviewer2pubs = {x['user']: x['publications'] for x in reviewer_pubs}

    # Load bids
    bids_file = os.path.join(data_dir, 'bids.jsonl')
    bids = read_jsonl(bids_file)

    return subs, bids, reviewer2pubs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()

    random.seed(42)

    # Load the submissions, bids, and reviewer test data
    submissions, bids, reviewer2pubs = load_test_data(args.data_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create submissions dict
    sub_dict = {sub['id']: sub for sub in submissions}

    # re-orgainize bids
    sub2bids = defaultdict(list)
    for b in bids:
        sub2bids[b['forum']].append(b)

    missed_papers = 0
    ranking_scores = []
    for sub_id, sub in tqdm(sub_dict.items(), desc='Ranking submissions'):
        sub_bids = sub2bids[sub_id]
        
        reviewer_inverted_index = []
        reviewer_paper_titles = []
        true_relevance = []

        for b in sub_bids:

            # if we don't have a vector for the reviewer we can't score them
            if b['signature'] not in reviewer2pubs.keys():
                continue

            # get the true score from the bid tag
            tag = b['tag']
            tr = None
            if tag == 'I want to review' or tag == 'Very High':
                tr = 1.0
            elif tag == 'I can review' or tag == 'High':
                tr = 0.5
            elif tag == 'I can probably review but am not an expert' or tag == 'Neutral':
                tr = 0.0
            elif tag == 'I cannot review' or tag == 'Low' or tag == 'Very Low':
                tr = -1.0
            else:
                assert tag == 'No bid' or tag == 'No Bid'
                continue

            _bidding_reviewer_paper_titles = list(set([p['abstract']
                for p in reviewer2pubs[b['signature']] if p['abstract'] is not None]))

            if len(_bidding_reviewer_paper_titles) == 0:
                continue

            _global_range = (len(reviewer_paper_titles), 
                len(reviewer_paper_titles) + len(_bidding_reviewer_paper_titles))
            reviewer_inverted_index.append(_global_range)
            reviewer_paper_titles.extend(_bidding_reviewer_paper_titles)

            true_relevance.append(tr)

        tokenized_reviewer_paper_titles = [p.split(' ')
                                           for p in reviewer_paper_titles]
        tokenized_sub_title = sub['abstract'].split(' ')

        bm25_reviewer_titles = BM25Okapi(tokenized_reviewer_paper_titles)
        _all_scores = bm25_reviewer_titles.get_scores(tokenized_sub_title)
        scores = [max(_all_scores[start:end]) for start, end in reviewer_inverted_index]

        if len(scores) < 2:
            missed_papers += 1
            continue

        _ranking_score = stats.kendalltau(true_relevance, scores).correlation

        if np.isnan(_ranking_score):
            missed_papers += 1
            continue

        ranking_scores.append(_ranking_score)

    embed()
    exit()
