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

from sklearn.metrics import average_precision_score
import sys
import numpy as np

def eval_map(list_of_list_of_labels, list_of_list_of_scores, randomize=True):
    """Compute Mean Average Precision

    Given a two lists with one element per test example compute the
    mean average precision score.

    The i^th element of each list is an array of scores or labels corresponding
    to the i^th training example.

    :param list_of_list_of_labels: Binary relevance labels. One list per example.
    :param list_of_list_of_scores: Predicted relevance scores. One list per example.
    :return: the mean average precision
    """
    np.random.seed(19)
    assert len(list_of_list_of_labels) == len(list_of_list_of_scores)
    aps = []
    for i in range(len(list_of_list_of_labels)):
        if randomize == True:
            perm = np.random.permutation(len(list_of_list_of_labels[i]))
            list_of_list_of_labels[i] = np.asarray(list_of_list_of_labels[i])[perm]
            list_of_list_of_scores[i] = np.asarray(list_of_list_of_scores[i])[perm]
        # print("Labels: {}".format(list_of_list_of_labels[i]))
        # print("Scores: {}".format(list_of_list_of_scores[i]))
        # print("MAP: {}".format(average_precision_score(list_of_list_of_labels[i],
        #                                                list_of_list_of_scores[i])))
        if sum(list_of_list_of_labels[i]) > 0:
            aps.append(average_precision_score(list_of_list_of_labels[i],
                                               list_of_list_of_scores[i]))
    return sum(aps) / len(aps)

def load(filename):
    """Load the labels and scores for MAP evaluation.

    Loads labels and model predictions from files of the format:
    Query \t Example \t Label \t Score

    :param filename: Filename to load.
    :return: list_of_list_of_labels, list_of_list_of_scores
    """
    result_labels = []
    result_scores = []
    current_block_name = ""
    current_block_scores = []
    current_block_labels = []
    with open(filename) as f:
        for line in f:
            splt = line.strip().split("\t")
            if len(splt) != 4:
                print(splt)
            block_name = splt[0]
            block_example = splt[1]
            example_label = int(splt[2])
            example_score = float(splt[3])
            if block_name != current_block_name and current_block_name != "":
                 result_labels.append(current_block_labels)
                 result_scores.append(current_block_scores)
                 current_block_labels = []
                 current_block_scores = []
            current_block_labels.append(example_label)
            current_block_scores.append(example_score)
            current_block_name = block_name
    result_labels.append(current_block_labels)
    result_scores.append(current_block_scores)
    return result_labels,result_scores

def eval_map_file(filename):
    list_of_list_of_labels, list_of_list_of_scores = load(filename)
    return eval_map(list_of_list_of_labels, list_of_list_of_scores)


if __name__ == "__main__":
    print("{}\t{}".format(sys.argv[1], eval_map_file(sys.argv[1])))
