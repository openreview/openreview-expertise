# Baseline untrained ELMo model for submission-reviewer affinity for
#  paper matching
#  - Submissions are represented by the ELMo vector of the abstract
#  - Reviewers are represented by the average of the ELMo vectors of all
#    their publications

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import ndcg_score
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
    sub_file = os.path.join(data_dir, 'test-submissions.jsonl')
    subs = read_jsonl(sub_file)

    # Load user publications
    reviewer_file = os.path.join(data_dir, 'user_publications.jsonl')
    reviewer_pubs = read_jsonl(reviewer_file)
    reviewer2pubs = {x['user']: x['publications'] for x in reviewer_pubs}

    # Load bids
    bids_file = os.path.join(data_dir, 'bids.jsonl')
    bids = read_jsonl(bids_file)

    return subs, bids, reviewer2pubs


def extract_elmo(paper, tokenizer, elmo):
    toks = tokenizer.tokenize(paper['abstract'], escape=False)
    vecs = elmo.embed_sentence(toks)
    new_vecs = np.transpose(vecs, (1,0,2)).reshape(-1, vecs.shape[0]*vecs.shape[2])
    content_vec = new_vecs.mean(0)
    return content_vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()

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
            sub2vec[sub['id']] = extract_elmo(sub, tokenizer, elmo)
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
            pub_reps = []
            for pub in pubs:
                pub_reps.append(extract_elmo(pub, tokenizer, elmo))
            reviewer2vec[signature] = np.vstack(pub_reps).mean(0)
        with open(os.path.join(args.output_dir, 'reviewer2vec.pkl'), 'wb') as f:
            pickle.dump(reviewer2vec, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading cached reviewer2vec...')
        with open(os.path.join(args.output_dir, 'reviewer2vec.pkl'), 'rb') as f:
            reviewer2vec = pickle.load(f)
        print('Done')

    # re-orgainize bids
    sub2bids = defaultdict(list)
    for b in bids:
        sub2bids[b['forum']].append(b)

    ndcg_scores = []
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
            if tag == 'I want to review' or tag == 'Very High':
                true_relevance.append(1.0)
            elif tag == 'I can review' or tag == 'High':
                true_relevance.append(0.5)
            elif tag == 'I can probably review but am not an expert' or tag == 'Neutral':
                true_relevance.append(0.0)
            elif tag == 'I cannot review' or tag == 'Low' or tag == 'Very Low':
                true_relevance.append(-1.0)
            else:
                assert tag == 'No bid' or tag == 'No Bid'
                continue

            reviewer_rep = reviewer2vec[b['signature']]
            if np.isnan(cosine_dist(reviewer_rep, sub_rep)):
                embed()
                exit()
            scores.append(1 - cosine_dist(reviewer_rep, sub_rep))

        embed()
        exit()
