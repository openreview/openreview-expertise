import os
import itertools
from . import tfidf
from expertise.utils.config import Config
from expertise.utils import dump_pkl
from datetime import datetime
import pickle

def train(config):
    experiment_dir = os.path.abspath(config.experiment_dir)
    setup_dir = os.path.join(experiment_dir, 'setup')

    submission_kps_file = os.path.join(setup_dir, 'submission_kps.pkl')
    reviewer_kps_file = os.path.join(setup_dir, 'reviewer_kps.pkl')

    print('fitting model')
    start_training_datetime = datetime.now()

    with open(submission_kps_file, 'rb') as f:
        kps_by_paperid = pickle.load(f)

    with open(reviewer_kps_file, 'rb') as f:
        kp_archives_by_userid = pickle.load(f)

    model = tfidf.Model(kps_by_paperid, kp_archives_by_userid)
    model.fit()

    print('finished training in {}'.format(datetime.now() - start_training_datetime))

    train_dir = os.path.join(experiment_dir, 'train')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    model_out_path = os.path.join(train_dir, 'model.pkl')

    dump_pkl(model_out_path, model)
