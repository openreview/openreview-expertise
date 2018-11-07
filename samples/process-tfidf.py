'''
A file for processing ("training") the TF-IDF model


'''

import openreview
import os, sys
import json
import itertools
import multiprocessing as mp
import random
from datetime import datetime
from expertise.models import tfidf, model_utils
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import preprocess_string
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score_file', help='name of the score file')
    parser.add_argument('submission_records_dir', help='directory with .jsonl files representing submission records')
    parser.add_argument('reviewer_records_dir', help='directory with .jsonl files representing reviewer "archives" (series of records representing their expertise)')
    parser.add_argument('--num_processes', required=False, default=4, help='number of parallel processes to run')
    parser.add_argument('--baseurl', help="openreview base url")
    parser.add_argument('--username')
    parser.add_argument('--password')
    args = parser.parse_args()

    paper_ids = []
    reviewer_ids = []

    paper_content_by_id = {}
    for filename in os.listdir(args.submission_records_dir):
        paper_id = filename.replace('.jsonl','')
        with open(os.path.join(args.submission_records_dir, filename)) as f:
            paper_record = f.read()
            try:
                paper_content_by_id[paper_id] = json.loads(paper_record)
            except json.decoder.JSONDecodeError as e:
                print(paper_id)
                print(paper_record)
                raise e
        paper_ids.append(paper_id)

    reviewer_content_by_id = {}
    for filename in os.listdir(args.reviewer_records_dir):
        reviewer_id = filename.replace('.jsonl','')
        with open(os.path.join(args.reviewer_records_dir, filename)) as f:
            contents = [json.loads(r.replace('\n','')) for r in f.readlines()]
            reviewer_content_by_id[reviewer_id] = contents
        reviewer_ids.append(reviewer_id)

    # stemmer = PorterStemmer()
    def preprocess_content(content):
        text = model_utils.content_to_text(content, fields=['title', 'abstract', 'fulltext'])
        tokens = preprocess_string(text)
        return tokens

    print('fitting model')
    start_training_datetime = datetime.now()
    model = tfidf.Model(preprocess_content=preprocess_content)
    model_name = 'tfidf'

    all_content = paper_content_by_id.values()
    for content_list in reviewer_content_by_id.values():
        all_content = itertools.chain([content for content in content_list], all_content)

    model.fit(all_content)
    print('finished training in {}'.format(datetime.now() - start_training_datetime))


    def get_best_score_pool(paper_reviewer):
        paper_id, reviewer_id = paper_reviewer
        best_score = 0.0

        for reviewer_content in reviewer_content_by_id[reviewer_id]:
            score = model.score(reviewer_content, paper_content_by_id[paper_id])
            if score > best_score:
                best_score = score

        return (paper_id, reviewer_id, best_score)

    open_type = 'w'
    existing_cells = []
    if os.path.isfile(args.score_file):
        print('reading scores')
        open_type = 'a'
        with open(args.score_file) as f:
            lines = f.readlines()
            for line in lines:
                existing_cell = eval(line.replace('\n',''))
                existing_paper_reviewer = (existing_cell[0], existing_cell[1])
                existing_cells.append(existing_paper_reviewer)

    paper_reviewer_pairs = []
    for paper_id in paper_ids:
        for reviewer_id in reviewer_ids:
            if (paper_id, reviewer_id) not in existing_cells:
                paper_reviewer_pairs.append((paper_id, reviewer_id))

    start_worker_pool = datetime.now()
    print('starting pool on {} pairs at {}'.format(len(paper_reviewer_pairs), start_worker_pool))
    # start 4 worker processes
    with open(args.score_file, open_type) as f:
        pool = mp.Pool(processes=int(args.num_processes))
        for result in pool.imap(get_best_score_pool, paper_reviewer_pairs):
            f.write('{}\n'.format(result))

    print('finished job in {}'.format(datetime.now() - start_worker_pool))
