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

import sys
import numpy as np

from .. import utils

def eval_hits_at_k(list_of_list_of_labels,
                   list_of_list_of_scores,
                   k=10,
                   randomize=True,
                   oracle=False,
                   ):
    """Compute Hits at K

    Given a two lists with one element per test example compute the
    mean average precision score.

    The i^th element of each list is an array of scores or labels corresponding
    to the i^th training example.

    All scores are SIMILARITIES.

    :param list_of_list_of_labels: Binary relevance labels. One list per example.
    :param list_of_list_of_scores: Predicted relevance scores. One list per example.
    :param k: the number of elements to consider
    :param randomize: whether to randomize the ordering
    :param oracle: break ties using the labels
    :return: the mean average precision
    """
    np.random.seed(19)
    assert len(list_of_list_of_labels) == len(list_of_list_of_scores)
    aps = []
    for i in range(len(list_of_list_of_labels)):
        if randomize == True:
            perm = np.random.permutation(len(list_of_list_of_labels[i]))
            list_of_list_of_labels[i] = list(np.asarray(list_of_list_of_labels[i])[perm])
            list_of_list_of_scores[i] = list(np.asarray(list_of_list_of_scores[i])[perm])
        if oracle:
            zpd = zip(list_of_list_of_scores[i],list_of_list_of_labels[i])
            sorted_zpd =sorted(zpd, reverse=True)
            list_of_list_of_labels[i] = [x[1] for x in sorted_zpd]
            list_of_list_of_scores[i] = [x[0] for x in sorted_zpd]
        else:
            zpd = zip(list_of_list_of_scores[i],list_of_list_of_labels[i])
            sorted_zpd =sorted(zpd, key=lambda x: x[0], reverse=True)
            list_of_list_of_labels[i] = [x[1] for x in sorted_zpd]
            list_of_list_of_scores[i] = [x[0] for x in sorted_zpd]
        # print("Labels: {}".format(list_of_list_of_labels[i]))
        # print("Scores: {}".format(list_of_list_of_scores[i]))
        labels_topk = list_of_list_of_labels[i][0:k]
        # print("labels_topk: {}".format(labels_topk))
        if sum(list_of_list_of_labels[i]) > 0:
            hits_at_k = sum(labels_topk) * 1.0 / min(k, sum(list_of_list_of_labels[i]))
            # print("Hits@{}: {}".format(k,hits_at_k))
            aps.append(hits_at_k)

    return sum(aps) / len(aps)

def load(filename):
    """Load the labels and scores for Hits at K evaluation.

    Loads labels and model predictions from files of the format:
    Query \t Example \t Label \t Score

    :param filename: Filename to load.
    :return: list_of_list_of_labels, list_of_list_of_scores
    """
    print('function deprecated')

    # result_labels = []
    # result_scores = []
    # current_block_name = ""
    # current_block_scores = []
    # current_block_labels = []
    # with open(filename,'r') as fin:
    #     for line in fin:
    #         splt = line.strip().split("\t")
    #         block_name = splt[0]
    #         block_example = splt[1]
    #         example_label = int(splt[2])
    #         example_score = float(splt[3])
    #         if block_name != current_block_name and current_block_name != "":
    #             result_labels.append(current_block_labels)
    #             result_scores.append(current_block_scores)
    #             current_block_labels = []
    #             current_block_scores = []
    #         current_block_labels.append(example_label)
    #         current_block_scores.append(example_score)
    #         current_block_name = block_name
    # result_labels.append(current_block_labels)
    # result_scores.append(current_block_scores)
    # return result_labels,result_scores

if __name__ == "__main__":
    """
        Usage: filename [k=1] [oracle=False]
    """
    filename = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    oracle = sys.argv[3] == "True"  if len(sys.argv) > 3 else False
    print("{}\t{}\t{}\t{}".format(filename,k,oracle, eval_hits_at_k_file(filename,k=k,oracle=oracle)))
