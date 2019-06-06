import os
import itertools
from . import tfidf
from expertise.config import Config
from expertise import utils
from datetime import datetime

import ipdb

def train(config):
    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    # submission_kps_file = os.path.join(setup_dir, 'submission_kps.pkl')
    # reviewer_kps_file = os.path.join(setup_dir, 'reviewer_kps.pkl')

    # with open(submission_kps_file, 'rb') as f:
    #     kps_by_paperid = pickle.load(f)

    # with open(reviewer_kps_file, 'rb') as f:
    #     kp_archives_by_userid = pickle.load(f)

    kps_by_id = utils.load_pkl(os.path.join(config.kp_setup_dir, 'full_kps_by_id.pkl'))
    kps_by_paperid = {k: v for k, v in kps_by_id.items() if not k.startswith('~')}
    kp_archives_by_userid = {k: v for k, v in kps_by_id.items() if k.startswith('~')}

    model = tfidf.Model(kps_by_paperid, kp_archives_by_userid)
    model.fit()

    train_dir = os.path.join(experiment_dir, 'train')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    model_out_path = os.path.join(train_dir, 'model.pkl')

    utils.dump_pkl(model_out_path, model)
