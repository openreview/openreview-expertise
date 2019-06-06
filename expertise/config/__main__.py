'''

'''
from __future__ import absolute_import

import argparse
import os
import pkgutil

import expertise.models as models

model_importers = {m: i for i, m, _ in pkgutil.iter_modules(
    models.__path__)}

available_models = [k for k in model_importers.keys()]

parser = argparse.ArgumentParser()
parser.add_argument(
    'model', help=f'Select one of the following: {available_models}')
parser.add_argument(
    '--outfile', '-o', help='file to write config')
args = parser.parse_args()

model_default_file = os.path.join(
    model_importers[args.model].path,
    args.model,
    f'{args.model}_default.json')

if not args.outfile:
    outfile = f'./{args.model}.json'
else:
	outfile = args.outfile

with open(model_default_file) as i, open(outfile, 'w') as o:
    o.write(i.read())


