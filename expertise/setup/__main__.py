"""

"""

import argparse

import expertise
from .core import setup_model

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="a config file for a model")
parser.add_argument("additional_params", nargs=argparse.REMAINDER)
args = parser.parse_args()

config = setup_model(args)
model = expertise.load_model(config.model)
config = model.setup(config, *args.additional_params)
# config.save(config_path)
