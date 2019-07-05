import argparse
import os
import json
from collections import OrderedDict

from expertise import utils
from expertise.config import ModelConfig

from .setup_bert_kps_lookup import setup_bert_kps_lookup
from .setup_bert_lookup import setup_bert_lookup
from .core import setup

parser = argparse.ArgumentParser()
parser.add_argument('config_path', help="a config file for a model")
parser.add_argument('--partition', type=int, default=0)
parser.add_argument('--num_partitions', type=int, default=1)
parser.add_argument('--use_kps', action='store_true', default=True)
args = parser.parse_args()

config_path = os.path.abspath(args.config_path)
experiment_path = os.path.dirname(config_path)

config = ModelConfig()
config.update_from_file(config_path)

config = setup(
	config, partition_id=args.partition, num_partitions=args.num_partitions)

if args.use_kps:
    bert_lookup = setup_bert_kps_lookup(config)
else:
    bert_lookup = setup_bert_lookup(config)

utils.dump_pkl(os.path.join(config.setup_dir, 'bert_lookup.pkl'), bert_lookup)
