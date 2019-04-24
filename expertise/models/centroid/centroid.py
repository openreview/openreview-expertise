import os

import numpy as np

# import fastText as ft

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from expertise import utils

from expertise.evaluators.mean_avg_precision import eval_map
from expertise.evaluators.hits_at_k import eval_hits_at_k

import ipdb

class Model(torch.nn.Module):
    def __init__(self, config, vocab):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab

        # if self.config.fasttext:
        #     self.cached_ft = ft.load_model(self.config.fasttext)
        # else:
        #     self.cached_ft = None

        # Keyword embeddings
        self.embedding = nn.Embedding(len(vocab)+1, config.embedding_dim, padding_idx=0)

        # Vector of ones (used for loss)
        if self.config.use_cuda:
            self.ones = torch.ones(config.batch_size, 1).cuda()
        else:
            self.ones = torch.ones(config.batch_size, 1)

        self._bce_loss = BCEWithLogitsLoss()

    def get_loss(self, batch_source, pos_result, neg_result, batch_lengths, pos_len, neg_len):
        """ Compute the loss (BPR) for a batch of examples
        :param batch_source: a batch of source keyphrase indices (list of lists)
        :param pos_result: True aliases of the Mentions
        :param neg_result: False aliases of the Mentions
        :param batch_lengths: a list of sample lengths, one for each sample in the batch (list of lists)
        :param pos_len: lengths of positives
        :param neg_len: lengths of negatives
        :return:
        """

        # B by dim
        source_embed = self.embed(batch_source, batch_lengths)
        # B by dim
        pos_embed = self.embed(pos_result, pos_len)
        # B by dim
        neg_embed = self.embed(neg_result, neg_len)

        loss = self._bce_loss(
            utils.row_wise_dot(source_embed, pos_embed, normalize=True)
            - utils.row_wise_dot(source_embed, neg_embed, normalize=True),
            self.ones[:len(source_embed)])

        return loss


    def embed(self, keyword_lists, keyword_lengths):
        """
        :param keyword_lists:
            np.array - B x <max number of keywords>
            A list of lists of integers corresponding to keywords in the vocabulary.

        :param keyword_lengths:
            np.array - B x 1
        :return: batch_size by embedding dim
        """

        try:
            kw_lengths = torch.from_numpy(keyword_lengths)
            kw_indices = torch.from_numpy(keyword_lists).long()
        except RuntimeError as e:
            ipdb.set_trace()
            raise e

        if self.config.use_cuda:
            kw_indices = kw_indices.cuda()
            kw_lengths = kw_lengths.cuda()

        # get all the embeddings for each keyword
        # B x L x d
        embeddings = self.embedding(kw_indices)

        # make sure that we don't divide by zero
        kw_lengths[kw_lengths == 0] = 1

        # for each sample within the batch, find the average of all of that sample's keyword embeddings
        summed_emb = torch.sum(embeddings, dim=1)
        try:
            averaged = torch.div(summed_emb, kw_lengths.float())
        except RuntimeError as e:
            ipdb.set_trace(context=30)
            raise e

        # B x 1 x d
        return averaged

    def score_pairs(self, sources, targets, source_lens, target_lens):
        """

        :param source: Batchsize by Max_String_Length
        :param target: Batchsize by Max_String_Length
        :return: Batchsize by 1
        """
        source_embeddings = self.embed(sources, source_lens)
        target_embeddings = self.embed(targets, target_lens)
        result = utils.row_wise_dot(source_embeddings, target_embeddings, normalize=True)

        if np.isnan(sum(result.detach())):
            ipdb.set_trace()

        return result

def _get_batch_lens(features_batch):
    '''
    Helper function for getting the lengths of a list of features,
    formatting it correctly for downstream use.
    '''

    return np.asarray([np.asarray([len(f)], dtype=np.float) for f in features_batch])

def format_batch(batcher, config):
    '''
    Formats the output of a batcher to produce this model's
    loss parameters.
    '''

    # ipdb.set_trace()

    for sources, positives, negatives in batcher.batches(transpose=True):

        src_features = np.asarray([utils.load_features(s, 'features', config) for s in sources])
        pos_features = np.asarray([utils.load_features(p, 'features', config) for p in positives])
        neg_features = np.asarray([utils.load_features(n, 'features', config) for n in negatives])

        src_lens = _get_batch_lens(src_features)
        pos_lens = _get_batch_lens(pos_features)
        neg_lens = _get_batch_lens(neg_features)

        max_kps = np.max(pos_lens + neg_lens)
        pos_features_pad = np.asarray([utils.fixedwidth(kps, max_kps) for kps in pos_features])
        neg_features_pad = np.asarray([utils.fixedwidth(kps, max_kps) for kps in neg_features])

        source = {
            'features': src_features,
            'lens': src_lens,
            'ids': sources
        }

        pos = {
            'features': pos_features_pad,
            'lens': pos_lens,
            'ids': positives
        }

        neg = {
            'features': neg_features_pad,
            'lens': neg_lens,
            'ids': negatives
        }

        yield (source, pos, neg)

def generate_predictions(config, model, batcher, features_lookup):
    """
    Use the model to make predictions on the data in the batcher

    :param model: Model to use to score reviewer-paper pairs
    :param batcher: Batcher containing data to evaluate (a DevTestBatcher)
    :return:
    """

    # for idx, batch in enumerate(batcher.batches(batch_size=config.dev_batch_size)):



    def _predict_max(
        sources,
        targets,
        source_lens,
        target_lens,
        source_id,
        target_id,
        label):

        score = 0.0

        if sources.size and targets.size:
            scores = model.score_pairs(
                sources=sources,
                targets=targets,
                source_lens=source_lens,
                target_lens=target_lens,
            )

            max_score = max(scores.detach())[0]

            score = float(max_score)

        return score

    predictions = []
    for source_ids, target_ids, labels in batcher.batches(transpose=True):
        for source_id, target_id, label in zip(source_ids, target_ids, labels):

            target_fids = features_lookup[target_id]
            source_fids = utils.fixedwidth(
                features_lookup[source_id],
                len(target_fids),
                pad_val=features_lookup[source_id][-1])

            prediction = {
                'source_id': source_id,
                'target_id': target_id,
                'label': label,
                'score': 0.0
            }

            if len(target_fids) > 0:
                source_features = np.asarray([utils.load_features(s, 'features', config) for s in source_fids])
                target_features = np.asarray([utils.load_features(t, 'features', config) for t in target_fids])

                source_lens = _get_batch_lens(source_features)
                target_lens = _get_batch_lens(target_features)

                max_kps = max(np.max(source_lens), np.max(target_lens))
                target_features_pad = np.asarray([utils.fixedwidth(kps, max_kps) for kps in target_features])

                prediction['score'] = _predict_max(
                    sources=source_features,
                    targets=target_features_pad,
                    source_lens=source_lens,
                    target_lens=target_lens,
                    source_id=source_id,
                    target_id=target_id,
                    label=label)

            predictions.append(prediction)

    for p in predictions:
        yield p


def load_jsonl(filename):

    labels_by_forum = defaultdict(dict)
    scores_by_forum = defaultdict(dict)

    for data in utils.jsonl_reader(filename):
        forum = data['source_id']
        reviewer = data['target_id']
        label = data['label']
        score = data['score']
        labels_by_forum[forum][reviewer] = label
        scores_by_forum[forum][reviewer] = score

    result_labels = []
    result_scores = []

    for forum, labels_by_reviewer in labels_by_forum.items():
        scores_by_reviewer = scores_by_forum[forum]

        reviewer_scores = list(scores_by_reviewer.items())
        reviewer_labels = list(labels_by_reviewer.items())

        sorted_labels = [label for _, label in sorted(reviewer_labels)]
        sorted_scores = [score for _, score in sorted(reviewer_scores)]

        result_labels.append(sorted_labels)
        result_scores.append(sorted_scores)

    return result_labels, result_scores

def eval_map_file(filename):
    list_of_list_of_labels, list_of_list_of_scores = utils.load_labels(filename)
    result = eval_map(list_of_list_of_labels, list_of_list_of_scores)
    return result

def eval_hits_at_k_file(filename, k=2, oracle=False):
    list_of_list_of_labels,list_of_list_of_scores = utils.load_labels(filename)
    return eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=k,oracle=oracle)

