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
        self.linear_layer = nn.Linear(config.bert_dim, config.embedding_dim)

        # Vector of ones (used for loss)
        if self.config.use_cuda:
            self.ones = torch.ones(config.batch_size, 1).cuda()
        else:
            self.ones = torch.ones(config.batch_size, 1)

        self._bce_loss = BCEWithLogitsLoss()

    def compute_loss(self, batch_source, batch_pos, batch_neg):
        """ Compute the loss (BPR) for a batch of examples
        :param batch_source: a batch of source keyphrase indices (list of lists)
        :param batch_pos: True aliases of the Mentions
        :param batch_neg: False aliases of the Mentions
        """

        batch_size = len(batch_source)

        avg_source = torch.mean(batch_source, dim=1)
        avg_pos = torch.mean(batch_pos, dim=1)
        avg_neg = torch.mean(batch_neg, dim=1)

        # B by dim
        source_embed = self.embed(avg_source)
        # B by dim
        pos_embed = self.embed(avg_pos)
        # B by dim
        neg_embed = self.embed(avg_neg)

        loss = self._bce_loss(
            utils.row_wise_dot(source_embed, pos_embed )
            - utils.row_wise_dot(source_embed, neg_embed ),
            self.ones[:batch_size])
        return loss

    def score_pair(self, source, target, source_len, target_len):
        """

        :param source: Batchsize by Max_String_Length
        :param target: Batchsize by Max_String_Length
        :return: Batchsize by 1
        """
        source_embed = self.embed_dev(source, source_len)
        target_embed = self.embed_dev(target, target_len)
        scores = utils.row_wise_dot(source_embed, target_embed)
        return scores

    def embed(self, vector_batch):
        """
        :param vector_batch: torch.Tensor - Batch_size by bert_embedding_dim
        """

        if self.config.use_cuda:
            vector_batch = vector_batch.cuda()

        # do a linear transformation of the scibert vector into the centroid dimension
        # B x bert_embedding_dim
        try:
            embeddings = self.linear_layer(vector_batch)
        except AttributeError as e:
            ipdb.set_trace()
            raise e

        return embeddings

    def embed_dev(self, vector_batch, print_embed=False, batch_size=None):
        """
        :param keyword_lists: Batch_size by max_num_keywords
        """
        return self.embed(vector_batch)

    def score_dev_test_batch(self,
        batch_queries,
        batch_targets,
        batch_size
        ):

        avg_queries = torch.mean(batch_queries, dim=1)
        avg_targets = torch.mean(batch_targets, dim=1)

        if batch_size == self.config.dev_batch_size:
            source_embed = self.embed_dev(avg_queries)
            target_embed = self.embed_dev(avg_targets)
        else:
            source_embed = self.embed_dev(avg_queries, batch_size=batch_size)
            target_embed = self.embed_dev(avg_targets, batch_size=batch_size)

        scores = utils.row_wise_dot(source_embed, target_embed)

        # what is this?
        scores[scores != scores] = 0

        return scores


def generate_predictions(config, model, batcher, bert_lookup):
    """
    Use the model to make predictions on the data in the batcher

    :param model: Model to use to score reviewer-paper pairs
    :param batcher: Batcher containing data to evaluate (a DevTestBatcher)
    :return:
    """

    for idx, batch in enumerate(batcher.batches(batch_size=config.dev_batch_size)):
        if idx % 100 == 0:
            print('Predicted {} batches'.format(idx))

        batch_queries = []
        batch_query_lengths = []
        batch_query_ids = []
        batch_targets = []
        batch_target_lengths = []
        batch_target_ids = []
        batch_labels = []
        batch_size = len(batch)

        for data in batch:
            # append a positive sample
            batch_queries.append(bert_lookup[data['source_id']])
            batch_query_lengths.append(data['source_length'])
            batch_query_ids.append(data['source_id'])
            batch_targets.append(bert_lookup[data['positive_id']])
            batch_target_lengths.append(data['positive_length'])
            batch_target_ids.append(data['positive_id'])
            batch_labels.append(1)

            # append a negative sample
            batch_queries.append(bert_lookup[data['source_id']])
            batch_query_lengths.append(data['source_length'])
            batch_query_ids.append(data['source_id'])
            batch_targets.append(bert_lookup[data['negative_id']])
            batch_target_lengths.append(data['negative_length'])
            batch_target_ids.append(data['negative_id'])
            batch_labels.append(0)

        scores = model.score_dev_test_batch(
            torch.stack(batch_queries),
            torch.stack(batch_targets),
            np.asarray(batch_size)
        )

        if type(batch_labels) is not list:
            batch_labels = batch_labels.tolist()

        if type(scores) is not list:
            scores = list(scores.cpu().data.numpy().squeeze())

        for source, source_id, target, target_id, label, score in zip(
            batch_queries,
            batch_query_ids,
            batch_targets,
            batch_target_ids,
            batch_labels,
            scores
            ):

            # temporarily commenting out "source" and "target" because I think they are not needed.
            prediction = {
                # 'source': source,
                'source_id': source_id,
                # 'target': target,
                'target_id': target_id,
                'label': label,
                'score': float(score)
            }

            yield prediction

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
    return eval_map(list_of_list_of_labels, list_of_list_of_scores)

def eval_hits_at_k_file(filename, k=2, oracle=False):
    list_of_list_of_labels,list_of_list_of_scores = utils.load_labels(filename)
    return eval_hits_at_k(list_of_list_of_labels, list_of_list_of_scores, k=k,oracle=oracle)

