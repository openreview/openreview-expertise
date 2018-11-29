'''
Experimenter starts by creating an experiment directory, /exp_1, and a config file. The config file is specific to the type of model and experiment being run. It should be similar to Justin's Config class.

python -m expertise.setup_model ./experiments/exp_1/config.json

'''
import argparse
import importlib
import os

from expertise.utils.config import Config
from expertise.utils.dataset import Dataset

current_path = os.path.abspath(os.path.dirname(__file__))

def train_model(config_path):
    config_path = os.path.abspath(config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(filename=config_path)

    model = importlib.import_module(config.model)

    setup_path = os.path.join(experiment_path, 'setup')

    dataset = Dataset(config.dataset)

    model.train(config_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    train_model(args.config_path)
