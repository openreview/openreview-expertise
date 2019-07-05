'''
Generate scores from the model defined by the config.

'''

import argparse
import importlib
import os

from expertise.config import Config

def infer_model(config_path):
    config_path = os.path.abspath(config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(config_path)

    model = importlib.import_module(config.model)

    '''
    # it's not clear if this should be here or in the model's `infer` function.

    infer_path = os.path.join(experiment_path, 'infer')
    if not os.path.isdir(infer_path):
        os.mkdir(infer_path)
    '''

    model.infer(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    args = parser.parse_args()

    infer_model(args.config_path)
