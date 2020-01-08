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
    """A single training example for mention affinity"""

    def __init__(self, mention, pos_coref_cand, neg_coref_cand):
        """Constructs a ExpertiseAffinityTrainExample.

        Args:
            mention: mention object (dict) of interest
            pos_coref_cand: list of mention objects that should be close to
                            mention in the representation space.
            neg_coref_cand: list of mention objects that should be close to
                            mention in the representation space.
        """
        self.mention = mention
        self.pos_coref_cand = pos_coref_cand
        self.neg_coref_cand = neg_coref_cand


class ExpertiseAffinityEvalExample(ExpertiseAffinityExample):
    """A single eval example for mention affinity"""

    def __init__(self, mention):
        """Constructs a ExpertiseAffinityEvalExample.

        Args:
            mention: mention object (dict) of interest
        """
        self.mention = mention


class ExpertiseAffinityFeatures(object):
    """ A single example's worth of features to be fed into deep transformer """
    def __init__(self, mention_id, mention_features):
        raise NotImplementedError()


class ExpertiseAffinityTrainFeatures(ExpertiseAffinityFeatures):
    def __init__(self,
                 mention_id,
                 mention_features,
                 pos_features,
                 neg_features):

        self.mention_id = mention_id
        self.mention_features = [
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in mention_features
        ]
        self.pos_features = [
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in pos_features
        ]
        self.neg_features = [
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in neg_features
        ]


class ExpertiseAffinityEvalFeatures(ExpertiseAffinityFeatures):
    def __init__(self,
                 mention_id,
                 mention_features):
        self.mention_id = mention_id
        self.mention_features = [
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in mention_features
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

        return (examples, documents)

    def get_eval_examples(self, data_dir, num_coref, split, domains):
        assert split == 'train' or split == 'val' or split == 'test'

        logger.info("LOOKING AT {} | {} eval".format(data_dir, split))

        # Load all of the mentions
        mention_file = os.path.join(data_dir, 'mentions', split + '.json')
        mentions = self._read_mentions(mention_file)

        # Load all of the documents for the mentions
        # `documents` is a dictionary with key 'document_id'
        document_files = [os.path.join(data_dir, 'documents', domain + '.json')
                            for domain in domains]
        documents = self._read_documents(document_files)

        examples = [ExpertiseAffinityEvalExample(m) for m in mentions]

        return (examples, documents)

    def get_eval_candidates_and_labels(self, data_dir, split):
        assert split == 'train' or split == 'val' or split == 'test'

        logger.info("LOOKING AT {} | {} eval".format(data_dir, split))

        # Load the precomputed candidates for each mention
        # `candidates` is a dictionary with key 'mention_id'
        candidate_file = os.path.join(data_dir, 'tfidf_candidates', split + '.json')
        candidates = self._read_candidates(candidate_file)

        # Load all of the mentions
        mention_file = os.path.join(data_dir, 'mentions', split + '.json')
        mentions = self._read_mentions(mention_file)

        labels = defaultdict(dict)
        for m in mentions:
            labels[m['corpus']][m['mention_id']] = m['label_document_id']

        return candidates, labels

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
        tags = []
        for s in subs:
            sub_bids = sub2bids[s['id']]

            pos_reviewers, neg_reviewers = [], []
            for bid in sub_bids:
                tag = bid['tag']
                tags.append(tag)
            
        embed()
        exit()

        # given entity id, store all of the mentions with that ground truth id
        entity2mention = defaultdict(list)

        # entity2mention except mention does not have candidate as ground truth
        candidate2mention = defaultdict(list)

        for m in mentions:
            uid = m['mention_id']
            label_document_id = m['label_document_id']
            entity2mention[label_document_id].append(m)
            for c in candidates[uid]:
                if c != label_document_id:
                    candidate2mention[c].append(m)

        examples = []
        for m in mentions:
            label_document_id = m['label_document_id']

            pos_coref_cand = entity2mention[label_document_id]
            pos_coref_cand += candidate2mention[label_document_id]

            # filter out `self` mention
            pos_coref_cand = [x for x in pos_coref_cand if x != m]

            if len(pos_coref_cand) == 0:
                continue

            while len(pos_coref_cand) < num_coref:
                pos_coref_cand.extend(pos_coref_cand)

            pos_coref_cand = pos_coref_cand[:num_coref]

            neg_coref_cand = random.choices(mentions, k=num_coref)
            neg_coref_cand = [x for x in neg_coref_cand
                                if (x != m and x not in pos_coref_cand)]

            assert len(neg_coref_cand) != 0

            while len(neg_coref_cand) < num_coref:
                neg_coref_cand.extend(neg_coref_cand)

            neg_coref_cand = neg_coref_cand[:num_coref]

            examples.append(ExpertiseAffinityTrainExample(mention=m,
                                                        pos_coref_cand=pos_coref_cand,
                                                        neg_coref_cand=neg_coref_cand))
            
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
    documents: dict,
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

    # account for the tokens marking the mention '[unused0]', '[unused1]'
    mention_length = max_length - 2

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        example_mentions =  [example.mention]
        if not evaluate:
            example_mentions += example.pos_coref_cand + example.neg_coref_cand
        mention_id = example.mention['mention_id']

        example_features = []
        for mention in example_mentions:
            context_document_id = mention['context_document_id']
            start_index = mention['start_index']
            end_index = mention['end_index']

            context_document = documents[context_document_id]['text']

            context_tokens = context_document.split()
            extracted_mention = context_tokens[start_index: end_index+1]
            extracted_mention = ' '.join(extracted_mention)
            assert extracted_mention == mention['text']

            mention_text_tokenized = tokenizer.tokenize(mention['text'])

            mention_context = get_mention_context_tokens(
                    context_tokens, start_index, end_index,
                    mention_length, tokenizer)
        
            input_tokens = ['[CLS]'] + mention_context + ['[SEP]']
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

            example_features.append((input_ids, attention_mask, token_type_ids))

        if evaluate:
            features.append(ExpertiseAffinityEvalFeatures(
                                mention_id, [example_features[0]]))
        else:
            features.append(ExpertiseAffinityTrainFeatures(
                            mention_id, [example_features[0]],
                            example_features[1:len(example.pos_coref_cand)+1],
                            example_features[len(example.pos_coref_cand)+1:]))

    return features


processors = {
    "expertise_affinity": ExpertiseAffinityProcessor
}
