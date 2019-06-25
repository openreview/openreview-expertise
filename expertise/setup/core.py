import os
from collections import OrderedDict
import json
import expertise

def setup_model(args):
    config_path = os.path.abspath(args.config_path)

    with open(config_path) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    config = expertise.config.ModelConfig(**data)

    return config
