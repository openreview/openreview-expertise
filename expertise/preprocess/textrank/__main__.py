import argparse
import json
import os
from collections import OrderedDict
from expertise.config import ModelConfig

from .core import run_textrank

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="a config file for a model")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)

    with open(config_path) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    config = ModelConfig(**data)

    run_textrank(config)

    print("saving", config_path, config)
    config.save(config_path)
