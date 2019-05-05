import os

import numpy as np

# import fastText as ft

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from expertise import utils

from expertise.evaluators.mean_avg_precision import eval_map
from expertise.evaluators.hits_at_k import eval_hits_at_k

import itertools

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

    def _compare_archives(self, archive1, archive2):
        comparisons = []

        lens1_shape = list(archive1.shape)
        lens1_shape[-1] = 1
        lens1_tensor = torch.ones(lens1_shape) * self.config.max_num_keyphrases

        lens2_shape = list(archive2.shape)
        lens2_shape[-1] = 1
        lens2_tensor = torch.ones(lens2_shape) * self.config.max_num_keyphrases

        arch1_embedded = self.embed(archive1.long(), lens1_tensor)
        arch2_embedded = self.embed(archive2.long(), lens2_tensor)

        for t1 in arch1_embedded:
            for t2 in arch2_embedded:
                sim = torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
                comparisons.append(sim)

        if len(comparisons) == 0:
            ipdb.set_trace()

        comparisons = torch.stack(comparisons)
        result = torch.max(comparisons)
        return result

    def get_loss(self, batch_source, pos_result, neg_result, use_cuda=False):
        """ Compute the loss (BPR) for a batch of examples
        """

        pos_comparisons = []
        neg_comparisons = []
        for source_archive, pos_archive, neg_archive in zip(batch_source, pos_result, neg_result):
            pos_comparisons.append(self._compare_archives(source_archive, pos_archive))
            neg_comparisons.append(self._compare_archives(source_archive, neg_archive))

        pos_comparison_tensors = torch.stack(pos_comparisons, dim=0)
        neg_comparison_tensors = torch.stack(neg_comparisons, dim=0)
        if use_cuda:
            pos_comparison_tensors.cuda()
            neg_comparison_tensors.cuda()

        output = pos_comparison_tensors - neg_comparison_tensors
        target = torch.ones(pos_comparison_tensors.size())
        if use_cuda:
            target.cuda()

        # ipdb.set_trace()
        assert len(output) == len(target)
        # # B by dim
        # source_embed = self.embed(batch_source, batch_lengths)
        # # B by dim
        # pos_embed = self.embed(pos_result, pos_len)
        # # B by dim
        # neg_embed = self.embed(neg_result, neg_len)

        # loss = self._bce_loss(
        #     utils.row_wise_dot(source_embed, pos_embed, normalize=True)
        #     - utils.row_wise_dot(source_embed, neg_embed, normalize=True),
        #     self.ones[:len(source_embed)])

        loss = self._bce_loss(output, target)

        return loss


    def embed(self, kw_indices, kw_lengths):
        """
        :param keyword_lists:
            np.array - B x <max number of keywords>
            A list of lists of integers corresponding to keywords in the vocabulary.

        :param keyword_lengths:
            np.array - B x 1
        :return: batch_size by embedding dim
        """
        # try:
        #     kw_lengths = torch.from_numpy(keyword_lengths)
        #     kw_indices = torch.from_numpy(keyword_lists).long()
        # except RuntimeError as e:
        #     ipdb.set_trace()
        #     raise e

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

def _get_doc_lens(docs):
    '''
    Helper function for getting the lengths of a list of features,
    formatting it correctly for downstream use.
    '''
    return np.asarray([np.asarray([len(f)], dtype=np.float) for f in docs])

def _get_features(id, features_lookup, config):
    features = []

    # ipdb.set_trace()
    for f in features_lookup[id]:
        feature = utils.load_features(f, 'features', config)
        if feature.size > 0:
            padded_feature = torch.Tensor(
                utils.fixedwidth(feature, config.max_num_keyphrases))
            features.append(padded_feature)

    # ipdb.set_trace()
    if len(features) > 0:
        return torch.stack(features)
    else:
        return torch.Tensor([])

def format_batch(batcher, features_lookup, config):
    '''
    Formats the output of a batcher to produce this model's
    loss parameters.
    '''

    for batch in batcher.batches(transpose=False):
        # src_features = np.asarray([utils.load_features(s, 'features', config) for s in sources])
        # pos_features = np.asarray([utils.load_features(p, 'features', config) for p in positives])
        # neg_features = np.asarray([utils.load_features(n, 'features', config) for n in negatives])
        sources = []
        src_features = []
        # for s in sources:
        #     features = _get_features(s, features_lookup, config)
        #     if len(features.size()) > 0:
        #         src_features.append(features)

        positives = []
        pos_features = []
        # for p in positives:
        #     features = _get_features(p, features_lookup, config)
        #     if len(features.size()) > 0:
        #         pos_features.append(features)

        negatives = []
        neg_features = []
        # for n in negatives:
        #     features = _get_features(n, features_lookup, config)
        #     if len(features.size()) > 0:
        #         neg_features.append(features)


        for s, p, n in batch:
            s_features = _get_features(s, features_lookup, config)
            p_features = _get_features(p, features_lookup, config)
            n_features = _get_features(n, features_lookup, config)

            if len(s_features.flatten())>0 \
            and len(p_features.flatten())>0 \
            and len(n_features.flatten())>0:
                sources.append(s)
                positives.append(p)
                negatives.append(n)
                src_features.append(s_features)
                pos_features.append(p_features)
                neg_features.append(n_features)

                # src_lens = [_get_doc_lens(doc) for doc in src_features]
                # pos_lens = [_get_doc_lens(doc) for doc in pos_features]
                # neg_lens = [_get_doc_lens(doc) for doc in neg_features]

        if not len(src_features) == len(pos_features) == len(neg_features):
            ipdb.set_trace()
        if not len(sources) == len(positives) == len(negatives):
            ipdb.set_trace()

        source = {
            'features': src_features,
            # 'lens': src_lens,
            'ids': sources
        }

        pos = {
            'features': pos_features,
            # 'lens': pos_lens,
            'ids': positives
        }

        neg = {
            'features': neg_features,
            # 'lens': neg_lens,
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

    for batch in batcher.batches(transpose=False):
        for source_id, target_id, label in batch:

            source_features = _get_features(source_id, features_lookup, config)
            target_features = _get_features(target_id, features_lookup, config)

            prediction = {
                'source_id': source_id,
                'target_id': target_id,
                'label': label,
                'score': 0.0
            }

            if len(source_features.flatten())>0 and len(target_features.flatten())>0:
                score = model._compare_archives(source_features, target_features)
                prediction['score'] = float(score.detach())

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

