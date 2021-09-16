import random

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from .instance import Instance
from .utils import max_pool, mean_pool


class ParaModel(nn.Module):
    def __init__(self, data, args, vocab, vocab_fr, model_dir='expertise/models/acl_scorer/'):
        super(ParaModel, self).__init__()

        self.raw_data = data
        self.args = args
        self.gpu = args.gpu
        self.model_dir = model_dir

        self.vocab = vocab
        self.vocab_fr = vocab_fr
        self.ngrams = args.ngrams
        self.seg_length = args.seg_length

        self.delta = args.delta
        self.pool = args.pool

        self.dropout = args.dropout
        self.share_encoder = args.share_encoder
        self.share_vocab = args.share_vocab
        self.zero_unk = args.zero_unk

        self.batchsize = args.batchsize
        self.max_megabatch_size = args.megabatch_size
        self.curr_megabatch_size = 1
        self.megabatch = []
        self.megabatch_anneal = args.megabatch_anneal
        self.increment = False

        self.sim_loss = nn.MarginRankingLoss(margin=self.delta)
        self.cosine = CosineSimilarity()

        self.embedding = nn.Embedding(len(self.vocab), self.args.dim)
        if self.vocab_fr is not None:
            self.embedding_fr = nn.Embedding(len(self.vocab_fr), self.args.dim)

        self.sp = None
        if args.sp_model:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(self.model_dir + args.sp_model)

    def save_params(self, epoch):
        torch.save({'state_dict': self.state_dict(),
                    'vocab': self.vocab,
                    'vocab_fr': self.vocab_fr,
                    'args': self.args,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}, "{0}_{1}.pt".format(self.args.outfile, epoch))
        return "{0}_{1}.pt".format(self.args.outfile, epoch)

    def save_final_params(self):
        print("Saving final model...")
        torch.save({'state_dict': self.state_dict(),
                    'vocab': self.vocab,
                    'vocab_fr': self.vocab_fr,
                    'args': self.args,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.args.epochs}, "{0}".format(self.args.outfile))  # .pt is in input string

    def loss_function(self, g1, g2, p1, p2):
        g1g2 = self.cosine(g1, g2)
        g1p1 = self.cosine(g1, p1)
        g2p2 = self.cosine(g2, p2)

        ones = torch.ones(g1g2.size()[0])
        if self.gpu:
            ones = ones.cuda()

        loss = self.sim_loss(g1g2, g1p1, ones) + self.sim_loss(g1g2, g2p2, ones)

        return loss

    def scoring_function(self, g_idxs1, g_lengths1, g_idxs2, g_lengths2, fr0=0, fr1=0):
        g1 = self.encode(g_idxs1, g_lengths1, fr=fr0)
        g2 = self.encode(g_idxs2, g_lengths2, fr=fr1)
        return self.cosine(g1, g2)

    def pair_up_data(self):
        idx = random.randint(0, self.seg_length)
        pairs = []
        for i in self.raw_data:
            sent = i.sentence
            sent = sent.split()
            idx = min(idx, len(sent) - 2)
            splits = []
            start = 0
            while idx < len(sent):
                seg1 = sent[start:idx]
                splits.append(seg1)
                start = idx
                idx += self.seg_length
                idx = min(idx, len(sent))
            if idx > len(sent):
                seg = sent[start:len(sent)]
                splits.append(seg)
            splits = [" ".join(i) for i in splits]
            random.shuffle(splits)
            mid = len(splits) // 2
            pairs.append((Instance(splits[0:mid]), Instance(splits[mid:])))
        return pairs


class Averaging(ParaModel):
    def __init__(self, data, args, vocab, vocab_fr, model_dir):
        super(Averaging, self).__init__(data, args, vocab, vocab_fr, model_dir)
        self.parameters = self.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=self.args.lr)

        if args.gpu:
            self.cuda()

        print(self)

    def forward(self, curr_batch):
        g_idxs1 = curr_batch.g1
        g_lengths1 = curr_batch.g1_l

        g_idxs2 = curr_batch.g2
        g_lengths2 = curr_batch.g2_l

        p_idxs1 = curr_batch.p1
        p_lengths1 = curr_batch.p1_l

        p_idxs2 = curr_batch.p2
        p_lengths2 = curr_batch.p2_l

        g1 = self.encode(g_idxs1, g_lengths1)
        g2 = self.encode(g_idxs2, g_lengths2, fr=1)
        p1 = self.encode(p_idxs1, p_lengths1, fr=1)
        p2 = self.encode(p_idxs2, p_lengths2)

        return g1, g2, p1, p2

    def encode(self, idxs, lengths, fr=0):
        if fr and not self.share_vocab:
            word_embs = self.embedding_fr(idxs)
        else:
            word_embs = self.embedding(idxs)

        if self.dropout > 0:
            F.dropout(word_embs, training=self.training)

        if self.pool == "max":
            word_embs = max_pool(word_embs, lengths, self.args.gpu)
        elif self.pool == "mean":
            word_embs = mean_pool(word_embs, lengths, self.args.gpu)

        return word_embs


class LSTM(ParaModel):
    def __init__(self, data, args, vocab, vocab_fr, model_dir):
        super(LSTM, self).__init__(data, args, vocab, vocab_fr, model_dir)

        self.hidden_dim = self.args.hidden_dim

        self.e_hidden_init = torch.zeros(2, 1, self.args.hidden_dim)
        self.e_cell_init = torch.zeros(2, 1, self.args.hidden_dim)

        if self.gpu:
            self.e_hidden_init = self.e_hidden_init.cuda()
            self.e_cell_init = self.e_cell_init.cuda()

        self.lstm = nn.LSTM(self.args.dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        if not self.share_encoder:
            self.lstm_fr = nn.LSTM(self.args.dim, self.hidden_dim, num_layers=1,
                                   bidirectional=True, batch_first=True)

        self.parameters = self.parameters()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters), self.args.lr)

        if self.gpu:
            self.cuda()

        print(self)

    def encode(self, inputs, lengths, fr=0):
        bsz, max_len = inputs.size()
        e_hidden_init = self.e_hidden_init.expand(2, bsz, self.hidden_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.hidden_dim).contiguous()
        lens, indices = torch.sort(lengths, 0, True)

        if fr and not self.share_vocab:
            in_embs = self.embedding_fr(inputs)
        else:
            in_embs = self.embedding(inputs)

        if fr and not self.share_encoder:
            if self.dropout > 0:
                F.dropout(in_embs, training=self.training)
            all_hids, (enc_last_hid, _) = self.lstm_fr(pack(in_embs[indices],
                                                            lens.tolist(), batch_first=True),
                                                       (e_hidden_init, e_cell_init))
        else:
            if self.dropout > 0:
                F.dropout(in_embs, training=self.training)
            all_hids, (enc_last_hid, _) = self.lstm(pack(in_embs[indices],
                                                         lens.tolist(), batch_first=True), (e_hidden_init, e_cell_init))

        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        if self.pool == "max":
            embs = max_pool(all_hids, lengths, self.gpu)
        elif self.pool == "mean":
            embs = mean_pool(all_hids, lengths, self.gpu)

        return embs

    def forward(self, curr_batch):
        g_idxs1 = curr_batch.g1
        g_lengths1 = curr_batch.g1_l

        g_idxs2 = curr_batch.g2
        g_lengths2 = curr_batch.g2_l

        p_idxs1 = curr_batch.p1
        p_lengths1 = curr_batch.p1_l

        p_idxs2 = curr_batch.p2
        p_lengths2 = curr_batch.p2_l

        g1 = self.encode(g_idxs1, g_lengths1)
        g2 = self.encode(g_idxs2, g_lengths2, fr=1)
        p1 = self.encode(p_idxs1, p_lengths1, fr=1)
        p2 = self.encode(p_idxs2, p_lengths2)

        return g1, g2, p1, p2
