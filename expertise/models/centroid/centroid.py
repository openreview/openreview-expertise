import numpy as np

import fastText as ft

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from expertise.utils import row_wise_dot

class Model(torch.nn.Module):
    def __init__(self, config, vocab):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab

        if self.config.fasttext:
            self.cached_ft = ft.load_model(self.config.fasttext)
        else:
            self.cached_ft = None

        # Keyword embeddings
        self.embedding = nn.Embedding(len(vocab)+1, config.embedding_dim, padding_idx=0)

        # Vector of ones (used for loss)
        if self.config.use_cuda:
            self.ones = Variable(torch.ones(config.batch_size, 1).cuda())
        else:
            self.ones = Variable(torch.ones(config.batch_size, 1))

        self.loss = BCEWithLogitsLoss() # Is this the best loss function, or should we use BPR (sig(pos - neg)) instead?

    def compute_loss(self, query, pos_result, neg_result, query_len, pos_len, neg_len):
        """ Compute the loss (BPR) for a batch of examples
        :param query: Entity mentions
        :param pos_result: True aliases of the Mentions
        :param neg_result: False aliases of the Mentions
        :param query_len: lengths of mentions
        :param pos_len: lengths of positives
        :param neg_len: lengths of negatives
        :return:
        """
        # B by dim
        source_embed = self.embed(query, query_len)
        # B by dim
        pos_embed = self.embed(pos_result, pos_len)
        # B by dim
        neg_embed = self.embed(neg_result, neg_len)
        loss = self.loss(
            row_wise_dot(source_embed , pos_embed )
            - row_wise_dot(source_embed , neg_embed ),
            self.ones)
        return loss

    def score_pair(self,source, target, source_len, target_len):
        """

        :param source: Batchsize by Max_String_Length
        :param target: Batchsize by Max_String_Length
        :return: Batchsize by 1
        """
        source_embed = self.embed_dev(source, source_len)
        target_embed = self.embed_dev(target, target_len)
        scores = row_wise_dot(source_embed, target_embed)
        return scores

    def embed(self, keyword_lists, keyword_lengths):
        """
        :param keyword_lists: Numpy array - Batch_size by max_num_keywords - integers corresponding to keywords in the vocabulary.
        :param keyword_lengths: numpy array - batch_size by 1
        :return: batch_size by embedding dim
        """

        if self.cached_ft:
            print('Using fasttext pretrained embeddings')
            D = self.cached_ft.get_dimension()
            # Get the phrases
            summed_emb = np.zeros((keyword_lists.shape[0], D))
            for idx, author_kps in enumerate(keyword_lists):
                print(idx)
                print(author_kps)
                embeddings = np.zeros((len(author_kps), D))
                for phr_idx, phrase in enumerate(author_kps):
                    if phrase:
                        print(self.vocab.id2item[phrase])
                        embeddings[phr_idx, :] = self.cached_ft.get_word_vector(self.vocab.id2item[phrase])
                    else:
                        embeddings[phr_idx, :] = np.zeros((D,))
                summed_emb[idx, :] = np.sum(embeddings, axis=0)
            averaged = summed_emb / keyword_lengths
            return torch.from_numpy(averaged)

        else:
            print('Using embeddings trained on bids')
            kw_indices = torch.from_numpy(keyword_lists).long()
            kw_lengths = torch.from_numpy(keyword_lengths)
            # print(kw_lengths)
            if self.config.use_cuda:
                kw_indices = kw_indices.cuda()
                kw_lengths = kw_lengths.cuda()
            # B by L by d
            embeddings = self.embedding(kw_indices)
            if kw_indices.size()[0] == 0:
                print("kw_indices: {}".format(kw_indices))
            print("kw_indices: {}".format(kw_indices))
            kw_lengths[kw_lengths == 0] = 1
            print("kw_lengths: {}".format(kw_lengths))
            print(kw_indices.size())
            print(embeddings.size())
            print("embeddings: {}".format(embeddings))
            summed_emb = torch.sum(embeddings, dim=1)
            print("summed_emb: {}".format(summed_emb))
            averaged = torch.div(summed_emb, kw_lengths.float())
            print("averaged: {}".format(averaged))
            return averaged

    def embed_dev(self, keyword_lists, keyword_lengths, print_embed=False, batch_size=None):
        """
        :param keyword_lists: Batch_size by max_num_keywords
        :return: batch_size by embedding dim
        """
        return self.embed(keyword_lists, keyword_lengths)

    def score_dev_test_batch(
        self,batch_queries,
        batch_query_lengths,
        batch_targets,
        batch_target_lengths,
        batch_size):

        if batch_size == self.config.dev_batch_size:
            source_embed = self.embed_dev(batch_queries, batch_query_lengths)
            target_embed = self.embed_dev(batch_targets, batch_target_lengths)
        else:
            source_embed = self.embed_dev(batch_queries, batch_query_lengths,batch_size=batch_size)
            target_embed = self.embed_dev(batch_targets, batch_target_lengths,batch_size=batch_size)
        scores = row_wise_dot(source_embed, target_embed)
        scores[scores != scores] = 0
        print("scores: {}".format(scores))
        return scores
