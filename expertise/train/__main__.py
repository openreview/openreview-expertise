'''
Model training

'''
import argparse
import importlib
import os

from expertise.config import Config
from expertise.dataset import Dataset

def train_model(config_path):
    config_path = os.path.abspath(config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(config_path)

    model = importlib.import_module(config.model)

    setup_path = os.path.join(experiment_path, 'setup')
    assert os.path.isdir(setup_path), 'you must run expertise.setup_model first'

    train_path = os.path.join(experiment_path, 'train')
    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    model.train(config)

parser = argparse.ArgumentParser()
parser.add_argument('config_path', help="a config file for a model")
args = parser.parse_args()

train_model(args.config_path)
