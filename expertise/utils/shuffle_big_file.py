'''
Pseudocode from:
https://blog.janestreet.com/how-to-shuffle-a-big-dataset/?utm_source=share
'''

import argparse
import random
import json
import os
from tqdm import tqdm
from . import utils


def lazy_reader(filepath):
    with open(filepath) as f:
        for line in f:
            yield line

def build_piles(big_file_path, piles_directory, num_piles):
    '''
    First pass
    create empty piles p[0], ..., p[M - 1]
    for i = 0, ..., n - 1 do
      j := uniform random draw from {0, ..., M - 1}
      append x[i] to pile p[j]

    '''

    fp_by_index = {}

    for pile_index in range(num_piles):
        pile_label = str(pile_index).zfill(len(str(num_piles)))
        pile_path = os.path.join(piles_directory, 'pile{}.jsonl'.format(pile_label))
        fp_by_index[pile_index] = open(pile_path, 'w')

    print('reading from big file')
    for line in tqdm(lazy_reader(big_file_path)):
        pile_index = random.randint(0, num_piles-1)
        fp_by_index[pile_index].write(line)

    for pile_index, fp in fp_by_index.items():
        fp.close()


def integrate_piles(piles_directory, outfile):
    '''
    Second pass (perhaps done lazily):
    for j = 0, ..., M - 1 do
      shuffle p[j] in RAM with Fisher-Yates or whatever is convenient
      append p[j] to output file

    '''

    outfile_pointer = open(outfile, 'w')

    for file in os.listdir(piles_directory):
        filepath = os.path.join(piles_directory, file)
        print('filepath', filepath)
        lines = list(utils.jsonl_reader(filepath))

        print('shuffling {}'.format(filepath))
        random.shuffle(lines)
        print('done shuffling {}'.format(filepath))

        for line in lines:
            outfile_pointer.write(json.dumps(line) + '\n')
        print('wrote {} to {}'.format(filepath, outfile))

    outfile_pointer.close()

def main(inputfile, piles_directory, outputfile, num_piles):
    build_piles(inputfile, piles_directory, num_piles)
    integrate_piles(piles_directory, outputfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('outputfile')
    parser.add_argument('--num_piles', type=int, default=20)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--integrate', action='store_true')
    args = parser.parse_args()

    filedir = os.path.dirname(os.path.abspath(args.inputfile))
    piles_directory = os.path.join(filedir, 'piles')
    if not os.path.isdir(piles_directory):
        os.mkdir(piles_directory)

    if args.build:
        build_piles(args.inputfile, piles_directory, args.num_piles)

    if args.integrate:
        integrate_piles(piles_directory, args.outputfile)
