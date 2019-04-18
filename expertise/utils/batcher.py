"""
Copyright (C) 2017-2018 University of Massachusetts Amherst.
This file is part of "learned-string-alignments"
http://github.com/iesl/learned-string-alignments
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict

import numpy as np
import time
import csv
import sys, os
import random
import ast
from . import utils

csv.field_size_limit(sys.maxsize)

class Batcher(object):
    def __init__(self, input_file, triplet=False):
        self.data = []
        self.num_examples = 0
        self.triplet = triplet
        self.input_file = input_file

        self.load_data(self.input_file)

    def reset(self):
        self.start_index = 0

    def shuffle_data(self):
        # perm = np.random.permutation(self.num_examples)
        print('shuffling {} lines via the following permutation'.format(self.num_examples))
        # data_array = np.asarray(self.data)
        # shuffled_data_array = data_array[perm]
        # self.data = shuffled_data_array.tolist()
        self.data = np.random.permutation(self.data).tolist()
        return self.data

    def load_data(self, input_file, delimiter='\t'):
        self.input_file = input_file

        self.data = []

        with open(input_file) as f:
            if any(input_file.endswith(ext) for ext in ['.tsv','.csv']):
                reader = csv.reader(f, delimiter=delimiter)

                for line in reader:
                    for column_index, item in enumerate(line):
                        self.data[column_index].append(item)

                    self.num_examples += 1

            if input_file.endswith('.jsonl'):
                for data_dict in utils.jsonl_reader(input_file):
                    self.data.append(data_dict)

        self.num_examples = len(self.data)

    def batches(self, batch_size, delimiter='\t'):
        batch = []
        self.start_index = 0
        for data in utils.jsonl_reader(self.input_file):
            batch.append(data)
            self.start_index += 1
            if self.start_index % batch_size == 0 or self.start_index == self.num_examples:
                yield batch

                batch = []

