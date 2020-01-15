# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
import random
import math
from collections import defaultdict
from typing import List
from transformers import PreTrainedTokenizer

from IPython import embed


logger = logging.getLogger(__name__)


class ExpertiseAffinityExample(object):

    def __init__(self, mention):
        raise NotImplementedError()


class ExpertiseAffinityTrainExample(ExpertiseAffinityExample):
    """A single training example for expertise affinity"""

    def __init__(self, paper, pos_reviewer_pubs, neg_reviewer_pubs):
        """Constructs a ExpertiseAffinityTrainExample.

        Args:
            paper: submssion object (dict) of interest
            pos_reviewer_pubs: list of paper objects that should be close to
                            paper in the representation space.
            neg_reviewer_pubs: list of mention objects that should be farther from
                            paper in the representation space.
        """
        self.paper = paper
        self.pos_reviewer_pubs = pos_reviewer_pubs
        self.neg_reviewer_pubs = neg_reviewer_pubs


class ExpertiseAffinityEvalExample(ExpertiseAffinityExample):
    """A single eval example for expertise affinity"""

    def __init__(self, paper):
        """Constructs a ExpertiseAffinityEvalExample.

        Args:
            paper: paper object (dict) of interest
        """
        self.paper = paper


class ExpertiseAffinityFeatures(object):
    """ A single example's worth of features to be fed into deep transformer """
    def __init__(self, mention_id, mention_features):
        raise NotImplementedError()


class ExpertiseAffinityTrainFeatures(ExpertiseAffinityFeatures):
    def __init__(self,
                 paper_features,
                 pos_features,
                 neg_features):
        self.paper_features = [
            {
                'or_id': or_id,
                'submission': submission,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for (or_id, submission, input_ids,
                 attention_mask, token_type_ids) in paper_features
        ]
        self.pos_features = [
            {
                'or_id': or_id,
                'submission': submission,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for (or_id, submission, input_ids,
                 attention_mask, token_type_ids) in pos_features
        ]
        self.neg_features = [
            {
                'or_id': or_id,
                'submission': submission,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for (or_id, submission, input_ids,
                 attention_mask, token_type_ids) in neg_features
        ]


class ExpertiseAffinityEvalFeatures(ExpertiseAffinityFeatures):
    def __init__(self,
                 paper_features):
        self.paper_features = [
            {
                'or_id': or_id,
                'submission': submission,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for (or_id, submission, input_ids,
                 attention_mask, token_type_ids) in paper_features
        ]


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir, split):
        raise NotImplementedError()

    def get_eval_examples(self, data_dir, split):
        raise NotImplementedError()


class ExpertiseAffinityProcessor(DataProcessor):
    """Processor for the mention affinity task."""

    def get_train_examples(self, data_dir, sample_size, split):
        """See base class."""
        assert split == 'train'

        logger.info("LOOKING AT {} train".format(data_dir))

        # Load all of the submissions
        sub_file = os.path.join(data_dir, split + '-submissions.jsonl')
        subs = self._read_jsonl(sub_file)

        # Load user publications
        reviewer_file = os.path.join(data_dir, 'user_publications.jsonl')
        reviewer_pubs = self._read_jsonl(reviewer_file)

        # Load bids
        bids_file = os.path.join(data_dir, 'bids.jsonl')
        bids = self._read_jsonl(bids_file)

        examples = self._create_train_examples(
                subs, reviewer_pubs, bids, sample_size)

        return examples

    def get_eval_examples(self, data_dir, split):
        assert split == 'train' or split == 'val' or split == 'test'

        logger.info("LOOKING AT {} | {} eval".format(data_dir, split))

        # Load all of the submissions
        sub_file = os.path.join(data_dir, split + '-submissions.jsonl')
        subs = self._read_jsonl(sub_file)

        # Load user publications
        reviewer_file = os.path.join(data_dir, 'user_publications.jsonl')
        reviewer_pubs = self._read_jsonl(reviewer_file)

        # Load bids
        bids_file = os.path.join(data_dir, 'bids.jsonl')
        bids = self._read_jsonl(bids_file)

        examples = self._create_eval_examples(subs, reviewer_pubs, bids)

        return examples

    def get_eval_resources(self, data_dir, split):
        assert split == 'train' or split == 'val' or split == 'test'

        logger.info("LOOKING AT {} | {} eval".format(data_dir, split))

        # Load all of the submissions
        sub_file = os.path.join(data_dir, split + '-submissions.jsonl')
        subs = self._read_jsonl(sub_file)

        # Load user publications
        reviewer_file = os.path.join(data_dir, 'user_publications.jsonl')
        reviewer_pubs = self._read_jsonl(reviewer_file)
        reviewer2pubs = {x['user']:x['publications'] for x in reviewer_pubs}

        # Load bids
        bids_file = os.path.join(data_dir, 'bids.jsonl')
        bids = self._read_jsonl(bids_file)
        sub2bids = defaultdict(list)
        for b in bids:
            sub2bids[b['forum']].append(b)

        return subs, reviewer2pubs, sub2bids

    def _read_jsonl(self, jsonl_file):
        lines = []
        with open(jsonl_file, encoding='utf-8') as fin:
            for line in fin:
                lines.append(json.loads(line))
        return lines

    def _read_submissions(self, sub_file):
        subs = []
        with open(sub_file, encoding='utf-8') as fin:
            for line in fin:
                subs.append(json.loads(line))
        return subs

    def _read_documents(self, document_files):
        documents = {}
        for fname in document_files:
            with open(fname, encoding='utf-8') as fin:
                for line in fin:
                    doc_dict = json.loads(line)
                    documents[doc_dict['document_id']] = doc_dict
        return documents

    def _read_candidates(self, candidate_file):
        candidates = {}
        with open(candidate_file, encoding='utf-8') as fin:
            for line in fin:
                candidate_dict = json.loads(line)
                candidates[candidate_dict['mention_id']] = candidate_dict['tfidf_candidates']
        return candidates

    def _create_train_examples(self, subs, reviewer_pubs, bids, sample_size):
        """Creates examples for the training and dev sets."""

        reviewer2pubs = {d['user']: d['publications'] for d in reviewer_pubs}
        sub2bids = defaultdict(list)

        for bid in bids:
            sub2bids[bid['forum']].append(bid)

        examples = []
        for s in subs:
            sub_bids = sub2bids[s['id']]

            pos_reviewers, neg_reviewers, neutral_reviewers = [], [], []
            for bid in sub_bids:
                if bid['signature'] in reviewer2pubs.keys():
                    tag = bid['tag']
                    if tag in ['High', 'I can review', 'Very High', 'I want to review']:
                        pos_reviewers.append(bid['signature'])
                    elif tag in ['I cannot review', 'Low', 'Very Low']:
                        neg_reviewers.append(bid['signature'])
                    else:
                        assert tag in ['I can probably review but am not an expert',
                                       'Neutral', 'No Bid', 'No bid']
                        neutral_reviewers.append(bid['signature'])

            pos_pubs = [reviewer2pubs[r] for r in pos_reviewers]
            pos_pubs = [pub for reviewer_pubs in pos_pubs
                          for pub in reviewer_pubs
                            if pub['title'] is not None]

            neg_pubs = [reviewer2pubs[r] for r in neg_reviewers]
            neg_pubs = [pub for reviewer_pubs in neg_pubs
                          for pub in reviewer_pubs
                            if pub['title'] is not None]

            if len(pos_pubs) == 0:
                continue

            if len(neg_pubs) == 0 and len(neutral_reviewers) > 0:
                neutral_pubs = [reviewer2pubs[r] for r in neutral_reviewers]
                neutral_pubs = [pub for reviewer_pubs in neutral_pubs
                                  for pub in reviewer_pubs
                                    if pub['title'] is not None]
                if len(neutral_pubs) == 0:
                    continue
                neg_pubs = neutral_pubs
            elif len(neg_pubs) == 0 and len(neutral_reviewers) == 0:
                continue

            list_len = math.ceil(max(len(pos_pubs), len(neg_pubs)) / sample_size) * sample_size

            while len(pos_pubs) < list_len:
                pos_pubs.extend(pos_pubs)
            pos_pubs = pos_pubs[:list_len]

            while len(neg_pubs) < list_len:
                neg_pubs.extend(neg_pubs)
            neg_pubs = neg_pubs[:list_len]

            chunked_pubs = [(pos_pubs[i:i+sample_size], 
                             neg_pubs[i:i+sample_size])
                                for i in range(0, list_len, sample_size)]

            # tag objects with submission flag
            s['submission'] = True
            for p in (pos_pubs + neg_pubs):
                p['submission'] = False

            for _p_pubs, _n_pubs in chunked_pubs:
                examples.append(ExpertiseAffinityTrainExample(
                                    s, _p_pubs, _n_pubs))

        return examples

    def _create_eval_examples(self, subs, reviewer_pubs, bids):
        reviewer2pubs = {d['user']: d['publications'] for d in reviewer_pubs}

        examples = []
        # add reviewer publications
        for _, pubs in reviewer2pubs.items():
            for pub in pubs:
                if pub['title'] is not None:
                    pub['submission'] = False
                    examples.append(ExpertiseAffinityEvalExample(pub))

        # add conference submissions
        for sub in subs:
            if sub['title'] is not None:
                sub['submission'] = True
                examples.append(ExpertiseAffinityEvalExample(sub))
            else:
                assert False

        return examples

def get_mention_context_tokens(context_tokens, start_index, end_index,
                               max_tokens, tokenizer):
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0

    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])
    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    mention = tokenizer.tokenize(
                ' '.join(context_tokens[start_index:end_index+1]))
    mention = ['[unused0]'] + mention + ['[unused1]']

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))

    mention_context = []

    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
    else:
        prefix_len = len(prefix)

    if prefix_len > len(prefix):
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_context = mention_context[:max_tokens]

    return mention_context


def convert_examples_to_features(
    examples: List[ExpertiseAffinityExample],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[ExpertiseAffinityFeatures]:
    """
    Loads a data file into a list of `ExpertiseAffintityFeatures`
    """

    # make space for the '[CLS]' and '[SEP]' tokens
    max_context_length = max_length - 2

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        example_pubs =  [example.paper]
        if not evaluate:
            example_pubs += example.pos_reviewer_pubs + example.neg_reviewer_pubs

        example_features = []
        for pub in example_pubs:
            or_id = pub['id']
            submission = pub['submission']

            assert 'title' in pub.keys()
            title_text = pub['title'].replace('\n', ' ')
            title_text_tokenized = tokenizer.tokenize(title_text)
            title_text_tokenized = title_text_tokenized[:max_context_length]

            input_tokens = ['[CLS]'] + title_text_tokenized + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            token_type_ids = [0] * len(input_ids) # segment_ids

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # input_mask

            assert pad_on_left == False
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length

            example_features.append((or_id,
                                     submission,
                                     input_ids,
                                     attention_mask,
                                     token_type_ids))

        if evaluate:
            features.append(ExpertiseAffinityEvalFeatures(
                                [example_features[0]]))
        else:
            features.append(ExpertiseAffinityTrainFeatures(
                            [example_features[0]],
                            example_features[1:len(example.pos_reviewer_pubs)+1],
                            example_features[len(example.pos_reviewer_pubs)+1:]))

    return features


processors = {
    "expertise_affinity": ExpertiseAffinityProcessor
}
