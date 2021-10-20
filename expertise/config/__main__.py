"""

"""
from __future__ import absolute_import

import argparse
import os
import pkgutil
import expertise

parser = argparse.ArgumentParser()
parser.add_argument("model", help=f"select one of {expertise.available_models()}")
parser.add_argument("--outfile", "-o", help="file to write config")

args = parser.parse_args()

config = expertise.config.ModelConfig(model=args.model)

outfile = args.outfile if args.outfile else f"./{args.model}.json"

experiment_dir = os.path.dirname(os.path.abspath(outfile))

config.update(experiment_dir=experiment_dir)
config.save(outfile)
