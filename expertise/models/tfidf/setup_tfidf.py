import os
import importlib
from collections import defaultdict
from expertise.utils import dump_pkl
from expertise.utils.dataset import Dataset
from tqdm import tqdm

def setup(config):
    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    dataset = Dataset(**config.dataset)

    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    # get submission contents
    # formerly "paper_content_by_id"

    kps_by_submission = defaultdict(list)
    for file_id, text in tqdm(dataset.submissions(), total=dataset.num_submissions, desc='parsing submission keyphrases'):
        kps_by_submission[file_id].extend(keyphrases(text))

    submission_kps_path = os.path.join(setup_dir, 'submission_kps.pkl')
    dump_pkl(submission_kps_path, kps_by_submission)

    # write keyphrases for reviewer archives to pickle file
    # formerly "reviewer_content_by_id"
    kps_by_reviewer = defaultdict(list)

    for file_id, text in tqdm(dataset.archives(), total=dataset.num_archives, desc='parsing archive keyphrases'):
        kps_by_reviewer[file_id].append(keyphrases(text))

    reviewer_kps_path = os.path.join(setup_dir, 'reviewer_kps.pkl')
    dump_pkl(reviewer_kps_path, kps_by_reviewer)
