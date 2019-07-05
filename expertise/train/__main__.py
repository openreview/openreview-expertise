'''

'''

import argparse
import os
import json
from collections import OrderedDict
import expertise
from expertise.config import ModelConfig

def train_model(args):
    config_path = os.path.abspath(args.config_path)

    with open(config_path) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    config = ModelConfig(**data)

    model = expertise.load_model(config.model)
    config = model.train(config, *args.additional_params)
    config.save(config_path)
    print(config)

parser = argparse.ArgumentParser()
parser.add_argument('config_path', help="a config file for a model")
parser.add_argument('additional_params', nargs=argparse.REMAINDER)
args = parser.parse_args()

train_model(args)
