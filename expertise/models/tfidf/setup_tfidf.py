import os
import importlib
import itertools
from collections import defaultdict
from expertise import utils
from expertise.utils import dump_pkl
from expertise.utils.dataset import Dataset

from tqdm import tqdm

def setup(config):
    dataset = Dataset(**config.dataset)
    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    get_keyphrases = importlib.import_module(config.keyphrases).keyphrases

    # bids_by_forum = utils.get_bids_by_forum(dataset)
    # train_set_ids, dev_set_ids, test_set_ids = utils.split_ids(list(bids_by_forum.keys()))

    # get submission contents
    # formerly "paper_content_by_id"

    kps_by_submission = defaultdict(list)
    for file_id, text in dataset.submissions():
        keyphrases = get_keyphrases(text)
        kps_by_submission[file_id].extend(keyphrases)

    submission_kps_path = os.path.join(setup_dir, 'submission_kps.pkl')
    dump_pkl(submission_kps_path, kps_by_submission)

    # write keyphrases for reviewer archives to pickle file
    # formerly "reviewer_content_by_id"
    kps_by_reviewer = defaultdict(list)

    for file_id, text in dataset.archives():
        keyphrases = get_keyphrases(text)
        kps_by_reviewer[file_id].append(keyphrases)

    reviewer_kps_path = os.path.join(setup_dir, 'reviewer_kps.pkl')
    dump_pkl(reviewer_kps_path, kps_by_reviewer)
