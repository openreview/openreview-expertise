import os
import importlib
from collections import defaultdict
from expertise.utils import dump_pkl

def setup(setup_path, config, dataset):

    assert os.path.isdir(setup_path), 'setup directory must exist'

    keyphrases = importlib.import_module(config.keyphrases).keyphrases

    # get submission contents
    # formerly "paper_content_by_id"
    kps_by_submission = defaultdict(list)
    for file_id, text in dataset.submission_records():
        kps_by_submission[file_id].extend(keyphrases(text))

    submission_kps_path = os.path.join(setup_path, 'submission_kps.pkl')
    dump_pkl(submission_kps_path, kps_by_submission),

    # write keyphrases for reviewer archives to pickle file
    # formerly "reviewer_content_by_id"
    kps_by_reviewer = defaultdict(list)
    for file_id, text in dataset.reviewer_archives():
        kps_by_reviewer[file_id].extend(keyphrases(text))

    reviewer_kps_path = os.path.join(setup_path, 'reviewer_kps.pkl')
    dump_pkl(reviewer_kps_path, kps_by_reviewer)




