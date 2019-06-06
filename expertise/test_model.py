'''
Model inference

'''
import argparse
import importlib
import os

from expertise.config import Config
from expertise.dataset import Dataset

def test_model(config_path):
    config_path = os.path.abspath(config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(config_path)

    model = importlib.import_module(config.model)

    test_path = os.path.join(experiment_path, 'test')
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    model.test(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    test_model(args.config_path)
