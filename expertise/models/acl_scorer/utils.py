import sys

import numpy as np
import torch
from sacremoses import MosesTokenizer

entok = MosesTokenizer(lang="en")
unk_string = "UUUNKKK"


def max_pool(x, lengths, gpu):
    out = torch.FloatTensor(x.size(0), x.size(2)).zero_()
    if gpu:
        out = out.cuda()
    for i in range(len(lengths)):
        out[i] = torch.max(x[i][0 : lengths[i]], 0)[0]
    return out


def mean_pool(x, lengths, gpu):
    out = torch.FloatTensor(x.size(0), x.size(2)).zero_()
    if gpu:
        out = out.cuda()
    for i in range(len(lengths)):
        out[i] = torch.mean(x[i][0 : lengths[i]], 0)
    return out


def lookup(words, w, zero_unk):
    w = w.lower()
    if w in words:
        return words[w]
    else:
        if zero_unk:
            return None
        else:
            return words[unk_string]


def torchify_batch(batch, gpu=False):
    max_len = 0
    for i in batch:
        if len(i.embeddings) > max_len:
            max_len = len(i.embeddings)

    batch_len = len(batch)

    np_sents = np.zeros((batch_len, max_len), dtype="int32")
    np_lens = np.zeros((batch_len,), dtype="int32")

    for i, ex in enumerate(batch):
        np_sents[i, : len(ex.embeddings)] = ex.embeddings
        np_lens[i] = len(ex.embeddings)

    idxs, lengths = (
        torch.from_numpy(np_sents).long(),
        torch.from_numpy(np_lens).float().long(),
    )

    if gpu:
        idxs = idxs.cuda()
        lengths = lengths.cuda()

    return idxs, lengths


def print_progress(i, mod_size):
    if i != 0 and i % mod_size == 0:
        sys.stderr.write(".")
        if int(i / mod_size) % 50 == 0:
            print(i, file=sys.stderr)
        sys.stderr.flush()
