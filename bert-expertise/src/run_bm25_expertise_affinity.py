# Baseline untrained ELMo model for submission-reviewer affinity for
#  paper matching

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pickle
import random
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from allennlp.commands.elmo import ElmoEmbedder
from sacremoses import MosesTokenizer

from IPython import embed


def read_jsonl(jsonl_file):
    lines = []
    with open(jsonl_file, encoding='utf-8') as fin:
        for line in fin:
            lines.append(json.loads(line))
    return lines


def load_test_data(data_dir):
    # Load all of the submissions
    #sub_file = os.path.join(data_dir, 'test-submissions.jsonl')
    sub_file = os.path.join(data_dir, 'submissions.jsonl')
    subs = read_jsonl(sub_file)

    # Load user publications
    reviewer_file = os.path.join(data_dir, 'user_publications.jsonl')
    reviewer_pubs = read_jsonl(reviewer_file)
    reviewer2pubs = {x['user']: x['publications'] for x in reviewer_pubs}

    # Load bids
    bids_file = os.path.join(data_dir, 'bids.jsonl')
    bids = read_jsonl(bids_file)

    return subs, bids, reviewer2pubs


def extract_elmo(papers, tokenizer, elmo):
    toks_list = []
    for p in papers:
        toks_list.append(tokenizer.tokenize(p['title'], escape=False))
    vecs = elmo.embed_batch(toks_list)
    content_vecs = []
    for vec in vecs:
        new_vec = np.transpose(vec, (1,0,2)).reshape(-1, vec.shape[0]*vec.shape[2])
        content_vecs.append(new_vec.mean(0))
    return np.vstack(content_vecs)


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

    # create tokenizer and ELMo objects
    tokenizer = MosesTokenizer()
    elmo = ElmoEmbedder()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create submission representations
    if not os.path.exists(os.path.join(args.output_dir, 'sub2vec.pkl')):
        sub2vec = {}
        for sub in tqdm(submissions, desc='Submissions'):
            sub2vec[sub['id']] = extract_elmo([sub], tokenizer, elmo)
        with open(os.path.join(args.output_dir, 'sub2vec.pkl'), 'wb') as f:
            pickle.dump(sub2vec, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading cached sub2vec...')
        with open(os.path.join(args.output_dir, 'sub2vec.pkl'), 'rb') as f:
            sub2vec = pickle.load(f)
        print('Done')

    # create author representations
    if not os.path.exists(os.path.join(args.output_dir, 'reviewer2vec.pkl')):
        reviewer2vec = {}
        for signature, pubs in tqdm(reviewer2pubs.items(), desc='Reviewers'):
            if len(pubs) == 0:
                continue
            reviewer2vec[signature] = extract_elmo(pubs, tokenizer, elmo)
        with open(os.path.join(args.output_dir, 'reviewer2vec.pkl'), 'wb') as f:
            pickle.dump(reviewer2vec, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading cached reviewer2vec...')
        with open(os.path.join(args.output_dir, 'reviewer2vec.pkl'), 'rb') as f:
            reviewer2vec = pickle.load(f)
        print('Done')

    # create submissions dict
    sub_dict = {sub['id']: sub for sub in submissions}

    # re-orgainize bids
    sub2bids = defaultdict(list)
    for b in bids:
        sub2bids[b['forum']].append(b)

    scores_csv_per_bid = []

    missed_papers = 0
    ranking_scores = []
    for sub_id, sub_rep in tqdm(sub2vec.items(), desc='Ranking submissions'):
        sub_bids = sub2bids[sub_id]

        true_relevance = []
        scores = []
        for b in sub_bids:

            # if we don't have a vector for the reviewer we can't score them
            if b['signature'] not in reviewer2vec.keys():
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

            if np.any(np.isnan(reviewer2vec[b['signature']])):
                continue

            max_score_paper_idx = np.argmax(np.matmul(normalize(reviewer2vec[b['signature']], axis=1),
                                                      normalize(sub_rep.reshape(1, -1)).reshape(-1,)))
            reviewer_rep = reviewer2vec[b['signature']][max_score_paper_idx]

            cosine_sim = 1 - cosine_dist(reviewer_rep, sub_rep)

            if np.isnan(cosine_sim):
                continue

            embed()
            exit()

            scores_csv_per_bid.append((b['forum'],
                                       b['signature'],
                                       cosine_sim,
                                       b,
                                       sub_dict[b['forum']],
                                       reviewer2pubs[b['signature']][max_score_paper_idx]))

            scores.append(cosine_sim)
            true_relevance.append(tr)

        ## Check random
        #scores = [random.random() for _ in true_relevance]

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
