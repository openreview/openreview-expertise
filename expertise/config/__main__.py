'''

'''
from __future__ import absolute_import

import argparse
import os
import pkgutil

from expertise import config

parser = argparse.ArgumentParser()
parser.add_argument('model', help=f'select one of {config.available_models()}')
parser.add_argument('--outfile', '-o', help='file to write config')

args = parser.parse_args()

config = config.ModelConfig(model=args.model)

outfile = args.outfile if args.outfile else f'./{args.model}.json'
config.save(outfile)


