'''

'''

import argparse
import importlib
import os

from expertise.config import Config

def setup_model(args):
    config_path = os.path.abspath(args.config_path)

    with open(config_path) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    config = Config(data)

    model = importlib.import_module(config.model)
    model.setup(config, *args.additional_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    parser.add_argument('additional_params', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_model(args)
