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
from collections import defaultdict
from sklearn import metrics
import numpy as np
from .. import utils

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

    assert len(list_of_list_of_labels) == len(list_of_list_of_scores)
    avg_precision_scores = []

    for labels_list, scores_list in zip(list_of_list_of_labels, list_of_list_of_scores):

        if sum(labels_list) > 0:
            avg_precision = metrics.average_precision_score(
                labels_list,
                scores_list
            )

            avg_precision_scores.append(avg_precision)

    return sum(avg_precision_scores) / len(avg_precision_scores)
