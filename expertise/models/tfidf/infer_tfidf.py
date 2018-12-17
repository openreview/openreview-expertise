import os
import csv
from collections import defaultdict
from expertise import utils
from expertise.utils.config import Config
from expertise.utils.dataset import Dataset
from datetime import datetime
import multiprocessing as mp

def get_best_score_pool(payload):
    paper_id, reviewer_id, paper_text, reviewer_archives, model = payload
    best_score = 0.0

    for reviewer_text in reviewer_archives:
        score = model.score(reviewer_text, paper_text)
        if score > best_score:
            best_score = score

    return (paper_id, reviewer_id, best_score)

def infer(config, num_processes=4):
    experiment_dir = os.path.abspath(config.experiment_dir)

    infer_dir = os.path.join(experiment_dir, 'infer')
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    train_dir = os.path.join(experiment_dir, 'train')
    assert os.path.isdir(train_dir), 'Train dir does not exist. Make sure that this model has been trained.'

    dataset = Dataset(**config.dataset)

    paper_ids = set()
    reviewer_ids = set()

    paper_text_by_id = {}
    for paper_id, paper_text in dataset.submissions():
        paper_text_by_id[paper_id] = paper_text
        paper_ids.update([paper_id])

    reviewer_text_by_id = defaultdict(list)
    for reviewer_id, reviewer_text in dataset.archives():
        reviewer_text_by_id[reviewer_id].append(reviewer_text)
        reviewer_ids.update([reviewer_id])

    print('loading model')
    model = utils.load_pkl(os.path.join(train_dir, 'model.pkl'))

    # appends new scores to an existing file, if possible
    file_mode = 'w'
    existing_cells = []
    score_file_path = os.path.join(infer_dir, config.name + '-scores.txt')

    if os.path.isfile(score_file_path):
        print('reading scores')
        file_mode = 'a'
        with open(score_file_path) as f:
            reader = csv.reader(f)
            for row in reader:
                existing_paper_reviewer = (row[0], row[1])
                existing_cells.append(existing_paper_reviewer)

    multiprocessing_payloads = []
    for paper_id in paper_ids:
        for reviewer_id in reviewer_ids:
            if (paper_id, reviewer_id) not in existing_cells:
                multiprocessing_payloads.append(
                    (
                        paper_id,
                        reviewer_id,
                        paper_text_by_id[paper_id],
                        reviewer_text_by_id[reviewer_id],
                        model
                    )
                )

    start_worker_pool = datetime.now()
    print('starting pool on {} pairs at {}'.format(len(multiprocessing_payloads), start_worker_pool))
    # start 4 worker processes
    with open(score_file_path, file_mode) as f:
        pool = mp.Pool(processes=int(num_processes))
        for result in pool.imap(get_best_score_pool, multiprocessing_payloads):
            f.write(','.join([str(r) for r in result]) + '\n')

    print('finished job in {}'.format(datetime.now() - start_worker_pool))
