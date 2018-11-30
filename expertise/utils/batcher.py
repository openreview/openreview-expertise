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

import numpy as np
import time
import csv
import sys, os
import random
import ast

csv.field_size_limit(sys.maxsize)

class Batcher(object):
    def __init__(self, config, vocab, input_file=None, samples_file=None,
                num_batches=False, shuffle=True, triplet=True):

        self.triplet = triplet
        if self.triplet:
            self.load_data = self.load_data_triplet
            self.shuffle_data = self.shuffle_data_triplet
            self.write_data = self.write_data_triplet
            self.get_next_batch = self.get_next_batch_triplet

        if not self.triplet:
            # deprecated.
            pass
            # self.load_data = self.load_data_pairwise
            # self.shuffle_data = self.shuffle_data_pairwise
            # self.get_next_batch = self.get_next_batch_pairwise

        self.config = config
        self.vocab = vocab

        # input_file is the (potentially ordered) training set.
        self.input_file = input_file

        if self.input_file:
            print('attempting to load data')
            self.load_data(self.input_file)

        # data_dir is the location where the (potentially randomized) batch data
        # should be written.

        # samples_file is the file name of the (potentially randomized) batch data.
        self.samples_file = samples_file
        # if self.samples_file:
        #     self.dump_csv(self.samples_file)

        self.batch_size = config.batch_size
        self.shuffle = shuffle

        # self.return_one_epoch = return_one_epoch
        # self.start_index = 0
        # self.source_lens = None
        # self.pos_lens = None
        # self.neg_lens = None
        # self.targ_lens = None




    def dump_csv(self, samples_file, delimiter='\t', shuffle=True):
        '''
        Dumps (and shuffles) the data loaded in this batcher.
        '''

        self.samples_file = samples_file

        if shuffle:
            print('shuffling data')
            self.shuffle_data()

        print('writing data')
        self.write_data()


    def reset(self):
        self.start_index = 0

    def get_next_batch_triplet(self, delimiter='\t'):
        with open(self.samples_file) as f:
            reader = csv.reader(f, delimiter=delimiter)

            source_batch = []
            positives_batch = []
            negatives_batch = []
            source_lens_batch = []
            pos_lens_batch = []
            neg_lens_batch = []

            self.start_index = 0

            for row in reader:
                sample = [ast.literal_eval(item) for item in row]

                source_batch.append(sample[0])
                positives_batch.append(sample[1])
                negatives_batch.append(sample[2])
                source_lens_batch.append(sample[3])
                pos_lens_batch.append(sample[4])
                neg_lens_batch.append(sample[5])

                self.start_index += 1

                if self.start_index % self.batch_size == 0 or self.start_index == self.num_examples:
                    batch = (
                        np.asarray(source_batch),
                        np.asarray(positives_batch),
                        np.asarray(negatives_batch),
                        np.asarray(source_lens_batch, dtype=np.float32),
                        np.asarray(pos_lens_batch, dtype=np.float32),
                        np.asarray(neg_lens_batch, dtype=np.float32)
                    )

                    yield batch

                    source_batch = []
                    positives_batch = []
                    negatives_batch = []
                    source_lens_batch = []
                    pos_lens_batch = []
                    neg_lens_batch = []


    def write_data_triplet(self, delimiter='\t'):
        print('writing data triplet to {}'.format(self.samples_file))
        with open(self.samples_file, 'w') as f:

            writer = csv.writer(f, delimiter=delimiter)
            for sample in zip(
                self.sources.tolist(),
                self.positives.tolist(),
                self.negatives.tolist(),
                self.source_lens.tolist(),
                self.pos_lens.tolist(),
                self.neg_lens.tolist()
            ):
                writer.writerow(sample)

    def shuffle_data_triplet(self):
        """
        Shuffles maintaining the same order.
        """
        perm = np.random.permutation(self.num_examples)

        for data in [
            self.sources,
            self.positives,
            self.negatives,
            self.source_lens,
            self.pos_lens,
            self.neg_lens]:
            print('type(data)',type(data))
            data = data[perm]

    def load_data_triplet(self, input_file, delimiter='\t'):
        if not self.input_file:
            self.input_file = input_file

        with open(input_file) as f:
            reader = csv.reader(f, delimiter=delimiter)

            sources = []
            sources_lengths = []
            positives = []
            pos_lengths = []
            negatives = []
            neg_lengths = []

            for ct, line in enumerate(reader):
                if len(line) < 3:

                    # TODO: log this error somewhere (don't print it!)
                    pass
                else:
                    sources.append(np.asarray(self.vocab.to_ints(line[0])))
                    sources_lengths.append([min(self.config.max_num_keyphrases,len(line[0])) ])
                    positives.append(np.asarray(self.vocab.to_ints(line[1])))
                    pos_lengths.append([min(self.config.max_num_keyphrases,len(line[1])) ])
                    negatives.append(np.asarray(self.vocab.to_ints(line[2])))
                    neg_lengths.append([min(self.config.max_num_keyphrases,len(line[2])) ])
            self.sources = np.asarray(sources)
            self.positives = np.asarray(positives)
            self.negatives = np.asarray(negatives)
            self.source_lens = np.asarray(sources_lengths, dtype=np.float32)
            self.pos_lens = np.asarray(pos_lengths, dtype=np.float32)
            self.neg_lens = np.asarray(neg_lengths, dtype=np.float32)

            self.num_examples = len(self.sources)
            print("length of data", self.num_examples)

    def get_next_batch_pairwise(self):
        """
        returns the next batch
        TODO(rajarshd): move the if-check outside the loop, so that conditioned is not checked every time. the conditions are suppose to be immutable.
        """
        print('function deprecated')

        # self.start_index = 0
        # while True:
        #     if self.start_index > self.num_examples - self.batch_size:
        #         if self.return_one_epoch:
        #             return  # stop after returning one epoch
        #         self.start_index = 0
        #         if self.shuffle:
        #             self.shuffle_data_pairwise()
        #     else:
        #         if self.input_type == "TODOIMPLEMENTDEV":
        #             current_token = self.sources[self.start_index]
        #             i = 0
        #             while self.sources[self.start_index + i] == current_token and len(self.sources) > self.start_index + i:
        #                 i += 1
        #             end_index = self.start_index + i

        #         else:
        #             num_data_returned = min(self.batch_size, self.num_examples - self.start_index)
        #             assert num_data_returned > 0
        #             end_index = self.start_index + num_data_returned

        #         yield self.sources[self.start_index:end_index], \
        #             self.targets[self.start_index:end_index], \
        #             self.labels[self.start_index:end_index], \
        #             self.source_lens[self.start_index:end_index], \
        #             self.targ_lens[self.start_index:end_index]
        #         self.start_index = end_index

    def shuffle_data_pairwise(self):
        """
        Shuffles maintaining the same order.
        """
        print('function deprecated')
        # perm = np.random.permutation(self.num_examples)  # perm of index in range(0, num_questions)
        # assert len(perm) == self.num_examples

        # for data in [self.sources, self.targets, self.labels, self.source_lens, self.targ_lens]:
        #     data = data[perm]

    def load_data_pairwise(self):
        print('function deprecated')
        # with codecs.open(self.input_file, "r", "UTF-8", errors="replace") as inp:
        #     sources = []
        #     sources_lengths = []
        #     targets = []
        #     targets_lengths = []
        #     labels = []
        #     counter = 0
        #     for line in inp:
        #         line = line.encode("UTF-8").strip()
        #         split = line.decode("UTF-8").split("\t") #source, target, label

        #         if len(split) >= 3:
        #             if split[2] == "0" or split[2] == "1":
        #                 sources.append(self.vocab.to_ints(split[0]))
        #                 sources_lengths.append([min(self.config.max_num_keyphrases,len(split[0])) ])
        #                 targets.append(self.vocab.to_ints(split[1]))
        #                 targets_lengths.append([min(self.config.max_num_keyphrases,len(split[1])) ])
        #                 labels.append(int(split[2]))
        #             else:
        #                 sources.append(self.vocab.to_ints(split[0]))
        #                 sources_lengths.append([len(split[0])])
        #                 targets.append(self.vocab.to_ints(split[1]))
        #                 targets_lengths.append([len(split[1])])
        #                 labels.append(1)
        #                 sources.append(self.vocab.to_ints(split[0]))
        #                 sources_lengths.append([len(split[0])])
        #                 targets.append(self.vocab.to_ints(split[2]))
        #                 targets_lengths.append([len(split[2])])
        #                 labels.append(0)
        #         else:
        #             print(split, len(split), counter)
        #         counter += 1
        #     self.sources = np.asarray(sources)
        #     self.targets = np.asarray(targets)
        #     self.labels = np.asarray(labels, dtype=np.int32)
        #     self.source_lens = np.asarray(sources_lengths, dtype=np.float32)
        #     self.targ_lens = np.asarray(targets_lengths, dtype=np.float32)
        #     print(self.sources.shape)
        #     print(self.targets.shape)
        #     print(self.labels.shape)
        #     print("length of data", len(sources))

