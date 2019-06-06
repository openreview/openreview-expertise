'''
This script should "multiply" a directory's config file across K folds
'''
import argparse
import os
from expertise.config import Config
import random
import ipdb
def prepare_kfold(args, k):
    config_path = os.path.abspath(args.config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(config_path)

    old_experiment_dir = os.path.dirname(config.config_file_path)
    new_experiment_dir = os.path.join(old_experiment_dir, f'{config.name}{k}')

    if not os.path.exists(new_experiment_dir):
    	os.mkdir(new_experiment_dir)

    config.experiment_dir = new_experiment_dir
    config.config_file_path = config.config_file_path.replace(old_experiment_dir, new_experiment_dir)
    config.infer_dir = config.infer_dir.replace(old_experiment_dir, new_experiment_dir)
    config.train_dir = config.train_dir.replace(old_experiment_dir, new_experiment_dir)
    config.setup_dir = config.setup_dir.replace(old_experiment_dir, new_experiment_dir)
    config.test_dir = config.test_dir.replace(old_experiment_dir, new_experiment_dir)

    config.kp_setup_dir = os.path.join(old_experiment_dir, 'setup')
    config.random_seed = k

    config.save_config()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('num_folds', type=int)
    args = parser.parse_args()

    for k in range(args.num_folds):
    	prepare_kfold(args, k)

