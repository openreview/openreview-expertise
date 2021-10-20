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

import codecs
import numpy as np
import pickle


class DevTestBatcher(object):
    """
    Class for Dev/Test batching
    """

    def __init__(
        self, config, vocab, input_file, submission_kps_file, reviewer_kps_file
    ):
        """Construct a DevTestBatcher

        Construct a batcher that works on the Dev / test set
        """
        self.config = config
        self.vocab = vocab

        self.filename = input_file
        self.batch_size = self.config.dev_batch_size

        with open(submission_kps_file, "rb") as f:
            self.submission_keyphrases = pickle.load(f)
        with open(reviewer_kps_file, "rb") as f:
            self.reviewer_keyphrases = pickle.load(f)

    def batches(self):
        """Provide all batches in the dev/test set

        Generator over batches in the dataset. Note that the last batch may be
         of a different size than the other batches

        :return: Generator over bathes of size self.config.dev_batch_size.
            Each element of the generator contains the following tuple:
                batch_queries,
                batch_query_lengths,
                batch_query_strings,
                batch_targets,
                batch_target_lengths,
                batch_target_strings,
                batch_labels,
                batch_size
        """
        batch_queries = []
        batch_query_lengths = []
        batch_query_strings = []
        batch_targets = []
        batch_target_lengths = []
        batch_target_strings = []
        batch_labels = []
        counter = 0
        num_oov = 0
        with codecs.open(self.filename, "r", "UTF-8") as fin:
            for line in fin:
                if counter % self.batch_size == 0 and counter > 0:
                    # print(len(batch_queries))
                    if len(batch_queries) > 0:
                        yield np.asarray(batch_queries), np.asarray(
                            batch_query_lengths
                        ), batch_query_strings, np.asarray(batch_targets), np.asarray(
                            batch_target_lengths
                        ), batch_target_strings, np.asarray(
                            batch_labels
                        ), self.batch_size
                        batch_queries = []
                        batch_query_lengths = []
                        batch_query_strings = []
                        batch_targets = []
                        batch_target_lengths = []
                        batch_target_strings = []
                        batch_labels = []

                split = line.rstrip().split("\t")

                query_string = split[0]
                target_string = split[1]

                if (
                    query_string in self.submission_keyphrases
                    and target_string in self.reviewer_keyphrases
                ):
                    if (
                        len(self.submission_keyphrases[query_string]) > 0
                        and len(self.reviewer_keyphrases[target_string]) > 0
                    ):
                        if self.config.model_name not in ["TPMS", "TFIDF", "Random"]:
                            query_string = " ".join(
                                [
                                    kp.replace(" ", "_")
                                    for kp in self.submission_keyphrases[query_string]
                                ]
                            )
                        query_len = [
                            min(
                                self.config.max_num_keyphrases,
                                len(query_string.split(" ")),
                            )
                        ]
                        query_vec = np.asarray(self.vocab.to_ints(query_string))
                        unique, counts = np.unique(query_vec, return_counts=True)
                        count_dict = dict(zip(unique, counts))
                        if 1 in count_dict:
                            num_oov_current = count_dict[1]
                        else:
                            num_oov_current = 0

                        num_oov += num_oov_current
                        batch_queries.append(query_vec)
                        batch_query_lengths.append(query_len)
                        batch_query_strings.append(query_string)

                        if self.config.model_name not in ["TPMS", "TFIDF", "Random"]:
                            target_string = " ".join(
                                [
                                    kp.replace(" ", "_")
                                    for kp in self.reviewer_keyphrases[target_string]
                                ]
                            )
                        target_len = [
                            min(
                                self.config.max_num_keyphrases,
                                len(target_string.split(" ")),
                            )
                        ]
                        target_vec = np.asarray(self.vocab.to_ints(target_string))
                        unique, counts = np.unique(target_vec, return_counts=True)
                        count_dict = dict(zip(unique, counts))
                        if 1 in count_dict:
                            num_oov_current = count_dict[1]
                        else:
                            num_oov_current = 0

                        num_oov += num_oov_current
                        label = int(split[2])

                        batch_targets.append(target_vec)
                        batch_target_lengths.append(target_len)
                        batch_target_strings.append(target_string)

                        batch_labels.append(label)

                        counter += 1

        if len(batch_queries) >= 1:
            yield np.asarray(batch_queries), np.asarray(
                batch_query_lengths
            ), batch_query_strings, np.asarray(batch_targets), np.asarray(
                batch_target_lengths
            ), batch_target_strings, np.asarray(
                batch_labels
            ), len(
                batch_queries
            )


class DevBatcher(DevTestBatcher):
    def __init__(self, config, vocab):
        super(self.__class__, self).__init__(config, vocab, use_dev=True)


class TestBatcher(DevTestBatcher):
    def __init__(self, config, vocab):
        super(self.__class__, self).__init__(config, vocab, use_dev=False)
