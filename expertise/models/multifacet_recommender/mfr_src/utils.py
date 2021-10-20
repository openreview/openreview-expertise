import os
import shutil
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import random
import sys
import argparse

from expertise.models.multifacet_recommender.mfr_src.model import SEQ2EMB, EMB2SEQ

NULL_IND = 0
UNK_IND = 1
EOS_IND = 2

w_d2_ind_init = {"[null]": 0, "<unk>": 1, "<eos>": 2}
ind_l2_w_freq_init = [["[null]", -1, 0], ["<unk>", 0, 1], ["<eos>", 0, 2]]
num_special_token = len(w_d2_ind_init)


class Logger(object):
    def __init__(self, logging_path, print_=True, log_=True):
        self.f_log = open(logging_path, "w")
        self.print_ = print_
        self.log_ = log_

    def logging(self, s, print_=None, log_=None):
        if print_ or (print_ is None and self.print_):
            print(s)
            sys.stdout.flush()
        if log_ or (log_ is None and self.log_):
            self.f_log.write(s + "\n")


class Dictionary(object):
    def __init__(self, byte_mode=False):
        self.w_d2_ind = w_d2_ind_init.copy()
        self.ind_l2_w_freq = ind_l2_w_freq_init.copy()
        self.num_special_token = num_special_token
        self.NULL_IND = NULL_IND
        self.UNK_IND = UNK_IND
        self.EOS_IND = EOS_IND
        self.byte_mode = byte_mode

    def dict_check(self, w, ignore_unk):
        if w not in self.w_d2_ind:
            if ignore_unk:
                w_ind = self.NULL_IND
            else:
                w_ind = self.UNK_IND
        else:
            w_ind = self.w_d2_ind[w]
        self.ind_l2_w_freq[w_ind][1] += 1
        return w_ind

    def dict_check_add(self, w):
        if w not in self.w_d2_ind:
            w_ind = len(self.w_d2_ind)
            self.w_d2_ind[w] = w_ind
            if self.byte_mode:
                self.ind_l2_w_freq.append([w.decode("utf-8"), 1, w_ind])
            else:
                self.ind_l2_w_freq.append([w, 1, w_ind])
        else:
            w_ind = self.w_d2_ind[w]
            self.ind_l2_w_freq[w_ind][1] += 1
        return w_ind

    def append_eos(self, w_ind_list):
        w_ind_list.append(self.EOS_IND)  # append <eos>
        self.ind_l2_w_freq[self.EOS_IND][1] += 1

    def densify_index(self, min_freq, ignore_unk=False):
        vocab_size = len(self.ind_l2_w_freq)
        compact_mapping = [0] * vocab_size
        for i in range(self.num_special_token):
            compact_mapping[i] = i
        # compact_mapping[1] = 1
        # compact_mapping[2] = 2

        # total_num_filtering = 0
        total_freq_filtering = 0
        current_new_idx = self.num_special_token

        # for i, (w, w_freq, w_ind_org) in enumerate(self.ind_l2_w_freq[self.num_special_token:]):
        for i in range(self.num_special_token, vocab_size):
            w, w_freq, w_ind_org = self.ind_l2_w_freq[i]
            if w_freq < min_freq:
                if not ignore_unk:
                    compact_mapping[i] = self.UNK_IND
                    self.ind_l2_w_freq[i][-1] = self.UNK_IND
                    self.ind_l2_w_freq[i].append("unk")
                else:
                    compact_mapping[i] = self.NULL_IND
                    self.ind_l2_w_freq[i][-1] = self.NULL_IND
                    self.ind_l2_w_freq[i].append("null")
                # total_num_filtering += 1
                total_freq_filtering += w_freq
            else:
                compact_mapping[i] = current_new_idx
                self.ind_l2_w_freq[i][-1] = current_new_idx
                current_new_idx += 1

        self.ind_l2_w_freq[self.UNK_IND][
            1
        ] = total_freq_filtering  # update <unk> frequency

        print(
            "{}/{} word types are filtered".format(
                vocab_size - current_new_idx, vocab_size
            )
        )

        return compact_mapping, total_freq_filtering

    def store_dict(self, f_out):
        vocab_size = len(self.ind_l2_w_freq)
        for i in range(vocab_size):
            i_l2_w_freq = [
                self.ind_l2_w_freq[i][0],
                str(self.ind_l2_w_freq[i][1]),
                str(self.ind_l2_w_freq[i][2]),
            ] + self.ind_l2_w_freq[i][3:]
            f_out.write("\t".join(i_l2_w_freq) + "\n")

    def load_dict(self, f_in):
        self.w_d2_ind = {}
        self.ind_l2_w_freq = []
        for i, line in enumerate(f_in):
            fields = line.rstrip().split("\t")
            if len(fields) == 3:
                w_idx = int(fields[2])
                self.w_d2_ind[fields[0]] = w_idx
                assert w_idx == len(self.ind_l2_w_freq)
                self.ind_l2_w_freq.append([fields[0], int(fields[1]), w_idx])


def load_word_dict(f_in):
    d = {}
    max_ind = 0  # if the dictionary is densified, max_ind is the same as len(d)
    for i, line in enumerate(f_in):
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            d[fields[0]] = [int(fields[2]), int(fields[1])]
            max_ind = int(fields[2])

    return d, max_ind


def load_idx2word_freq(f_in):
    idx2word_freq = []
    for i, line in enumerate(f_in):
        fields = line.rstrip().split("\t")
        if len(fields) == 3:
            assert len(idx2word_freq) == int(fields[2])
            idx2word_freq.append([fields[0], int(fields[1])])

    return idx2word_freq


class F2UserTagDataset(torch.utils.data.Dataset):
    # will need to handle the partial data loading if the dataset size is larger than cpu memory
    # We could also use this class to put all sentences with the same length together
    def __init__(
        self, feature, feature_type, user, tag, repeat_num, user_len, tag_len, device
    ):
        self.feature = feature
        self.feature_type = feature_type
        self.user = user
        self.tag = tag
        self.repeat_num = repeat_num
        self.user_len = user_len
        self.tag_len = tag_len
        self.output_device = device

    def __len__(self):
        return self.feature.size(0)

    def __getitem__(self, idx):
        # feature = torch.tensor(self.feature[idx, :], dtype = torch.long, device = self.output_device)
        feature = self.feature[idx, :].to(dtype=torch.long, device=self.output_device)
        if self.feature_type.size(0) > 0:
            feature_type = self.feature_type[idx, :].to(
                dtype=torch.long, device=self.output_device
            )
        else:
            feature_type = []
        if self.user is None:
            user = []
            tag = []
            repeat_num = []
            user_len = []
            tag_len = []
        else:
            user = self.user[idx, :].to(dtype=torch.long, device=self.output_device)
            tag = self.tag[idx, :].to(dtype=torch.long, device=self.output_device)
            repeat_num = self.repeat_num[idx].to(
                dtype=torch.long, device=self.output_device
            )
            user_len = self.user_len[idx].to(
                dtype=torch.long, device=self.output_device
            )
            tag_len = self.tag_len[idx].to(dtype=torch.long, device=self.output_device)
        # debug target[-1] = idx
        return [feature, feature_type, user, tag, repeat_num, user_len, tag_len, idx]
        # return [self.feature[idx, :], self.target[idx, :]]


class F2IdxDataset(torch.utils.data.Dataset):
    def __init__(self, feature, feature_type, item_idx, device):
        self.feature = feature
        self.feature_type = feature_type
        self.item_idx = item_idx
        self.output_device = device

    def __len__(self):
        return self.feature.size(0)

    def __getitem__(self, idx):
        feature = self.feature[idx, :].to(dtype=torch.long, device=self.output_device)
        if self.feature_type.size(0) > 0:
            feature_type = self.feature_type[idx, :].to(
                dtype=torch.long, device=self.output_device
            )
        else:
            feature_type = []
        item_idx = self.item_idx[idx].to(dtype=torch.long)
        return [feature, feature_type, item_idx]


def create_data_loader_split(f_in, bsz, device, split_num, copy_training):
    fields = torch.load(f_in, map_location="cpu")
    if len(fields) == 7:
        (
            feature,
            feature_type,
            user,
            tag,
            repeat_num,
            user_len,
            tag_len,
        ) = fields  # torch.load(f_in, map_location='cpu')
        bid_score = torch.zeros(0)
    else:
        (
            feature,
            feature_type,
            user,
            tag,
            repeat_num,
            user_len,
            tag_len,
            bid_score,
        ) = fields  # torch.load(f_in, map_location='cpu')

    max_sent_len = feature.size(1)
    if copy_training:
        # idx_arr= np.random.permutation(feature.size(0)).reshape(split_num,-1)
        idx_arr = np.random.permutation(feature.size(0))
        split_size = int(feature.size(0) / split_num)
        dataset_arr = []
        for i in range(split_num):
            start = i * split_size
            if i == split_num - 1:
                end = feature.size(0)
            else:
                end = (i + 1) * split_size
            if feature_type.size(0) > 0:
                feature_type_i = feature_type[start:end]
            else:
                feature_type_i = feature_type
            dataset_arr.append(
                F2UserTagDataset(
                    feature[start:end],
                    feature_type_i,
                    user[start:end],
                    tag[start:end],
                    repeat_num[start:end],
                    user_len[start:end],
                    tag_len[start:end],
                    device,
                )
            )  # assume that the dataset are randomly shuffled beforehand
        # dataset_arr = [ F2SetDataset(feature[idx_arr[i,:],:], target[idx_arr[i,:],:], device) for i in range(split_num)]
    else:
        if feature_type.size(0) > 0:
            dataset_arr = [
                F2UserTagDataset(
                    feature[i : feature.size(0) : split_num, :],
                    feature_type[i : feature_type.size(0) : split_num, :],
                    user[i : user.size(0) : split_num, :],
                    tag[i : tag.size(0) : split_num, :],
                    repeat_num[i : repeat_num.size(0) : split_num],
                    user_len[i : user_len.size(0) : split_num],
                    tag_len[i : tag_len.size(0) : split_num],
                    device,
                )
                for i in range(split_num)
            ]
        else:
            dataset_arr = [
                F2UserTagDataset(
                    feature[i : feature.size(0) : split_num, :],
                    feature_type,
                    user[i : user.size(0) : split_num, :],
                    tag[i : tag.size(0) : split_num, :],
                    repeat_num[i : repeat_num.size(0) : split_num],
                    user_len[i : user_len.size(0) : split_num],
                    tag_len[i : tag_len.size(0) : split_num],
                    device,
                )
                for i in range(split_num)
            ]

    use_cuda = False
    if device.type == "cuda":
        use_cuda = True
    # dataloader_arr = [torch.utils.data.DataLoader(dataset_arr[i], batch_size = bsz, shuffle = True, pin_memory=not use_cuda, drop_last=False) for i in range(split_num)]
    dataloader_arr = [
        torch.utils.data.DataLoader(
            dataset_arr[i],
            batch_size=bsz,
            shuffle=True,
            pin_memory=not use_cuda,
            drop_last=False,
        )
        for i in range(split_num)
    ]
    return dataloader_arr, max_sent_len


def create_uniq_paper_data(
    feature,
    feature_type,
    user,
    tag,
    device,
    user_subsample_idx,
    tag_subsample_idx,
    bid_score,
    remove_duplication=True,
):
    print("removing duplicated features")
    feature_d2_user_tag = {}
    feature_d2_type = {}
    feature_list = feature.tolist()
    feature_type_list = feature_type.tolist()
    user_list = user.tolist()
    tag_list = tag.tolist()
    bid_score_list = bid_score.tolist()
    user_idx_old_d2_new = {}
    tag_idx_old_d2_new = {}
    if len(user_subsample_idx) > 0:
        for new_idx, old_idx in enumerate(user_subsample_idx):
            user_idx_old_d2_new[old_idx] = new_idx
        for new_idx, old_idx in enumerate(tag_subsample_idx):
            tag_idx_old_d2_new[old_idx] = new_idx
    for i in range(len(feature_list)):
        if remove_duplication:
            f_tuple = tuple(feature_list[i])
        else:
            f_tuple = tuple([i] + feature_list[i])
        if len(user_subsample_idx) > 0:
            user_list_i_no_0 = [
                user_idx_old_d2_new[user_id]
                for user_id in user_list[i]
                if user_id >= num_special_token and user_id in user_idx_old_d2_new
            ]
            tag_list_i_no_0 = [
                tag_idx_old_d2_new[tag_id]
                for tag_id in tag_list[i]
                if tag_id >= num_special_token and tag_id in tag_idx_old_d2_new
            ]
        else:
            user_list_i_no_0 = [
                user_id for user_id in user_list[i] if user_id >= num_special_token
            ]
            tag_list_i_no_0 = [
                tag_id for tag_id in tag_list[i] if tag_id >= num_special_token
            ]
        user_tag_list, prev_i = feature_d2_user_tag.get(f_tuple, [[[], [], []], 0])
        user_tag_list[0] += user_list_i_no_0
        user_tag_list[1] += tag_list_i_no_0
        if len(bid_score_list) > 0:
            if len(user_subsample_idx) > 0:
                bid_score_list_i_no_0 = [
                    bid_score_list[i][j]
                    for j, user_id in enumerate(user_list[i])
                    if user_id >= num_special_token and user_id in user_idx_old_d2_new
                ]
            else:
                bid_score_list_i_no_0 = [
                    bid_score_list[i][j]
                    for j, user_id in enumerate(user_list[i])
                    if user_id >= num_special_token
                ]
            user_tag_list[2] += bid_score_list_i_no_0
        feature_d2_user_tag[f_tuple] = [user_tag_list, i]
        if len(feature_type_list) > 0:
            feature_d2_type[f_tuple] = feature_type_list[i]

    all_user_tag = []
    uniq_feature_num = len(feature_d2_user_tag)
    feature_uniq = torch.empty(uniq_feature_num, feature.size(1), dtype=torch.int32)
    if len(feature_type_list) > 0:
        feature_type_uniq = torch.empty(
            uniq_feature_num, feature.size(1), dtype=torch.int32
        )
    else:
        feature_type_uniq = torch.zeros(0)
    paper_id_tensor = torch.tensor(list(range(uniq_feature_num)), dtype=torch.int32)
    feature_user_tag_order_i = sorted(
        feature_d2_user_tag.items(), key=lambda x: x[1][1]
    )
    # feature_user_tag_order_i = feature_d2_user_tag.items()
    for paper_id, (f_tuple, (user_tag_list, order_i)) in enumerate(
        feature_user_tag_order_i
    ):
        if remove_duplication:
            feature_uniq[paper_id, :] = torch.tensor(f_tuple, dtype=torch.int32)
        else:
            feature_uniq[paper_id, :] = torch.tensor(f_tuple[1:], dtype=torch.int32)

        if len(feature_type_list) > 0:
            feature_type_uniq[paper_id, :] = torch.tensor(
                feature_d2_type[f_tuple], dtype=torch.int32
            )
        all_user_tag.append(user_tag_list)

    dataset = F2IdxDataset(feature_uniq, feature_type_uniq, paper_id_tensor, device)
    print("leading to {} unique papers".format(len(all_user_tag)))
    sys.stdout.flush()
    return dataset, all_user_tag


def create_data_loader(
    f_in,
    bsz,
    device,
    want_to_shuffle=True,
    deduplication=False,
    user_subsample_idx=[],
    tag_subsample_idx=[],
    remove_duplication=True,
):
    fields = torch.load(f_in, map_location="cpu")
    if len(fields) == 7:
        (
            feature,
            feature_type,
            user,
            tag,
            repeat_num,
            user_len,
            tag_len,
        ) = fields  # torch.load(f_in, map_location='cpu')
        bid_score = torch.zeros(0)
    else:
        (
            feature,
            feature_type,
            user,
            tag,
            repeat_num,
            user_len,
            tag_len,
            bid_score,
        ) = fields  # torch.load(f_in, map_location='cpu')
    # feature, feature_type, user, tag, repeat_num, user_len, tag_len = torch.load(f_in, map_location='cpu')
    # print(feature)
    # print(target)
    if not remove_duplication:
        assert deduplication
    if not deduplication:
        dataset = F2UserTagDataset(
            feature, feature_type, user, tag, repeat_num, user_len, tag_len, device
        )
    else:
        dataset, all_user_tag = create_uniq_paper_data(
            feature,
            feature_type,
            user,
            tag,
            device,
            user_subsample_idx,
            tag_subsample_idx,
            bid_score,
            remove_duplication=remove_duplication,
        )
    # dataset = F2SetDataset(feature[0:feature.size(0):2,:], target[0:target.size(0):2,:], device)
    use_cuda = False
    if device.type == "cuda":
        use_cuda = True
    # pin_memory should be False - VERIFY
    # return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = want_to_shuffle, pin_memory=not use_cuda, drop_last=False)
    if not deduplication:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=want_to_shuffle,
            pin_memory=not use_cuda,
            drop_last=False,
        )
    else:
        return [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=bsz,
                shuffle=want_to_shuffle,
                pin_memory=not use_cuda,
                drop_last=False,
            ),
            all_user_tag,
        ]


def convert_sent_to_tensor(proc_sent_list, max_sent_len, word2idx):
    store_type = torch.int32

    num_sent = len(proc_sent_list)
    feature_tensor = torch.zeros(num_sent, max_sent_len, dtype=store_type)
    truncate_num = 0
    for output_i, proc_sent in enumerate(proc_sent_list):
        w_ind_list = []
        w_list = proc_sent.split()
        sent_len = min(len(w_list) + 1, max_sent_len)
        if len(w_list) > max_sent_len - 1:
            truncate_num += 1
        for w in w_list[: sent_len - 1]:
            if w in word2idx:
                w_ind_list.append(word2idx[w][0])
            # elif w.lower() in word2idx:
            #    w_ind_list.append(word2idx[w.lower()][0])
            else:
                w_ind_list.append(UNK_IND)
        # w_ind_list.append(0) #buggy preprocessing
        w_ind_list.append(EOS_IND)
        # print(w_ind_list)
        feature_tensor[output_i, -sent_len:] = torch.tensor(
            w_ind_list, dtype=store_type
        )
    print("Truncation rate: ", truncate_num / float(len(proc_sent_list)))
    return feature_tensor


def load_testing_article_summ(
    word_d2_idx_freq, article, max_sent_len, eval_bsz, device
):
    feature_tensor = convert_sent_to_tensor(article, max_sent_len, word_d2_idx_freq)
    # dataset = F2SetDataset(feature_tensor, None, device)
    dataset = F2IdxDataset(feature_tensor, None, device)
    use_cuda = False
    if device.type == "cuda":
        use_cuda = True
    dataloader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_bsz,
        shuffle=False,
        pin_memory=not use_cuda,
        drop_last=False,
    )
    return dataloader_test


def load_testing_sent(dict_path, input_path, max_sent_len, eval_bsz, device):
    with open(dict_path) as f_in:
        idx2word_freq = load_idx2word_freq(f_in)

    with open(dict_path) as f_in:
        word2idx, max_idx = load_word_dict(f_in)

    org_sent_list = []
    proc_sent_list = []
    with open(input_path) as f_in:
        for line in f_in:
            org_sent, proc_sent = line.rstrip().split("\t")
            org_sent_list.append(org_sent)
            proc_sent_list.append(proc_sent)

    dataloader_test = load_testing_article_summ(
        word2idx, proc_sent_list, max_sent_len, eval_bsz, device
    )
    # feature_tensor = convert_sent_to_tensor(proc_sent_list, max_sent_len, word2idx)
    # dataset = F2SetDataset(feature_tensor, None, device)
    # use_cuda = False
    # if device.type == 'cuda':
    #    use_cuda = True
    # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size = eval_bsz, shuffle = False, pin_memory=use_cuda, drop_last=False)

    return dataloader_test, org_sent_list, idx2word_freq


def create_subsample_idx(total_w_num, subsample_ratio, num_special_token):
    used_w_num = int((total_w_num - num_special_token) * subsample_ratio)
    user_subsample_idx = (
        list(range(num_special_token))
        + (
            num_special_token
            + np.random.choice(
                total_w_num - num_special_token, used_w_num, replace=False
            )
        ).tolist()
    )
    return user_subsample_idx


def load_corpus(
    data_path,
    train_bsz,
    eval_bsz,
    device,
    tensor_folder,
    training_file="train.pt",
    split_num=1,
    copy_training=False,
    skip_training=False,
    want_to_shuffle_val=True,
    load_val=True,
    load_test=False,
    deduplication=False,
    subsample_ratio=1,
    remove_testing_duplication=True,
):

    if subsample_ratio < 1:
        assert deduplication
        assert skip_training
        assert load_test

    train_corpus_name = data_path + "/" + tensor_folder + "/" + training_file
    val_org_corpus_name = data_path + "/" + tensor_folder + "/val.pt"
    test_org_corpus_name = data_path + "/" + tensor_folder + "/test.pt"

    dictionary_input_name = data_path + "/feature/dictionary_index"
    user_dictionary_input_name = data_path + "/user/dictionary_index"
    tag_dictionary_input_name = data_path + "/tag/dictionary_index"

    if os.path.exists(dictionary_input_name):
        with open(dictionary_input_name) as f_in:
            idx2word_freq = load_idx2word_freq(f_in)
    else:
        idx2word_freq = []

    with open(user_dictionary_input_name) as f_in:
        user_idx2word_freq = load_idx2word_freq(f_in)
        if subsample_ratio < 1:
            user_subsample_idx = create_subsample_idx(
                len(user_idx2word_freq), subsample_ratio, num_special_token
            )
            user_idx2word_freq = [user_idx2word_freq[x] for x in user_subsample_idx]
        else:
            user_subsample_idx = []

    with open(tag_dictionary_input_name) as f_in:
        tag_idx2word_freq = load_idx2word_freq(f_in)
        if subsample_ratio < 1:
            tag_subsample_idx = create_subsample_idx(
                len(tag_idx2word_freq), subsample_ratio, num_special_token
            )
            tag_idx2word_freq = [tag_idx2word_freq[x] for x in tag_subsample_idx]
        else:
            tag_subsample_idx = []

    dataloader_val = None
    if load_val:
        # NOTE: Uncomment if you want evaluation on non-shuffled val data
        with open(val_org_corpus_name, "rb") as f_in:
            dataloader_val = create_data_loader(
                f_in,
                eval_bsz,
                device,
                want_to_shuffle=want_to_shuffle_val,
                deduplication=deduplication,
                user_subsample_idx=user_subsample_idx,
                tag_subsample_idx=tag_subsample_idx,
            )

    # with open(val_shuffled_corpus_name,'rb') as f_in:
    #    dataloader_val_shuffled = create_data_loader(f_in, eval_bsz, device, want_to_shuffle = want_to_shuffle_val)

    if dataloader_val is None:
        max_sent_len = 512
    elif deduplication:
        max_sent_len = dataloader_val[0].dataset.feature.size(1)
    else:
        max_sent_len = dataloader_val.dataset.feature.size(1)

    if skip_training:
        dataloader_train_arr = [0]
    else:
        with open(train_corpus_name, "rb") as f_in:
            dataloader_train_arr, max_sent_len_train = create_data_loader_split(
                f_in, train_bsz, device, split_num, copy_training
            )
        assert max_sent_len == max_sent_len_train

    if load_test:
        with open(test_org_corpus_name, "rb") as f_in:
            dataloader_test = create_data_loader(
                f_in,
                eval_bsz,
                device,
                want_to_shuffle=want_to_shuffle_val,
                deduplication=deduplication,
                user_subsample_idx=user_subsample_idx,
                tag_subsample_idx=tag_subsample_idx,
                remove_duplication=remove_testing_duplication,
            )
        if subsample_ratio == 1:
            return (
                idx2word_freq,
                user_idx2word_freq,
                tag_idx2word_freq,
                dataloader_train_arr,
                dataloader_val,
                dataloader_test,
                max_sent_len,
            )
        else:
            return (
                idx2word_freq,
                user_idx2word_freq,
                tag_idx2word_freq,
                dataloader_train_arr,
                dataloader_val,
                dataloader_test,
                max_sent_len,
                user_subsample_idx,
                tag_subsample_idx,
            )
    else:
        return (
            idx2word_freq,
            user_idx2word_freq,
            tag_idx2word_freq,
            dataloader_train_arr,
            dataloader_val,
            max_sent_len,
        )


def load_emb_file_to_dict(emb_file, lowercase_emb=False, convert_np=True):
    word2emb = {}
    emb_size = 200
    with open(emb_file) as f_in:
        for line in f_in:
            word_val = line.rstrip().split(" ")
            if len(word_val) < 3:
                continue
            word = word_val[0]
            # val = np.array([float(x) for x in  word_val[1:]])
            val = [float(x) for x in word_val[1:]]
            if convert_np:
                val = np.array(val)
            if lowercase_emb:
                word_lower = word.lower()
                if word_lower not in word2emb:
                    word2emb[word_lower] = val
                else:
                    if word == word_lower:
                        word2emb[word_lower] = val
            else:
                word2emb[word] = val
            emb_size = len(val)
    return word2emb, emb_size


def load_emb_file_to_tensor(emb_file, device, idx2word_freq):
    # with open(emb_file) as f_in:
    #    word2emb = {}
    #    for line in f_in:
    #        word_val = line.rstrip().split(' ')
    #        if len(word_val) < 3:
    #            continue
    #        word = word_val[0]
    #        val = [float(x) for x in  word_val[1:]]
    #        word2emb[word] = val
    #        emb_size = len(val)
    word2emb, emb_size = load_emb_file_to_dict(emb_file, convert_np=False)
    num_w = len(idx2word_freq)
    # emb_size = len(word2emb.values()[0])
    # external_emb = torch.empty(num_w, emb_size, device = device, requires_grad = update_target_emb)
    external_emb = torch.empty(num_w, emb_size, device=device, requires_grad=False)
    # OOV_num = 0
    OOV_freq = 0
    total_freq = 0
    oov_list = []
    for i in range(num_special_token, num_w):
        w = idx2word_freq[i][0]
        total_freq += idx2word_freq[i][1]
        if w in word2emb:
            val = torch.tensor(word2emb[w], device=device, requires_grad=False)
            # val = np.array(word2emb[w])
            external_emb[i, :] = val
        else:
            oov_list.append(i)
            external_emb[i, :] = 0
            # OOV_num += 1
            OOV_freq += idx2word_freq[i][1]

    print("OOV word type percentage: {}%".format(len(oov_list) / float(num_w) * 100))
    print("OOV token percentage: {}%".format(OOV_freq / float(total_freq) * 100))
    return external_emb, emb_size, oov_list


def output_parallel_models(use_cuda, single_gpu, encoder, decoder):
    if use_cuda:
        if single_gpu:
            parallel_encoder = encoder.cuda()
            parallel_decoder = decoder.cuda()
        else:
            parallel_encoder = nn.DataParallel(encoder, dim=0).cuda()
            parallel_decoder = nn.DataParallel(decoder, dim=0).cuda()
            # parallel_decoder = decoder.cuda()
    else:
        parallel_encoder = encoder
        parallel_decoder = decoder
    return parallel_encoder, parallel_decoder


def load_emb_from_path(emb_file_path, device, idx2word_freq):
    if emb_file_path[-3:] == ".pt":
        word_emb = torch.load(emb_file_path, map_location=device)
        output_emb_size = word_emb.size(1)
    else:
        word_emb, output_emb_size, oov_list = load_emb_file_to_tensor(
            emb_file_path, device, idx2word_freq
        )
    return word_emb, output_emb_size


def loading_all_models(
    args,
    idx2word_freq,
    user_idx2word_freq,
    tag_idx2word_freq,
    device,
    max_sent_len,
    normalize_emb=True,
    load_auto=False,
):

    # if len(args.source_emb_file) > 0:
    #    word_emb, source_emb_size = load_emb_from_path(args.source_emb_file, device, idx2word_freq)
    #    word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True))
    # if args.emb_file[-3:] == '.pt':
    #    word_emb = torch.load( args.emb_file, map_location=device )
    #    output_emb_size = word_emb.size(1)
    # else:
    #    word_emb, output_emb_size, oov_list = load_emb_file_to_tensor(args.emb_file,device,idx2word_freq)
    # elif len(args.source_emb_file) == 0:
    #    source_emb_size = args.source_emsize
    source_emb_size = args.source_emsize

    if len(args.user_emb_file) > 0:
        print(args.user_emb_file)
        user_emb, user_emb_size = load_emb_from_path(
            args.user_emb_file, device, user_idx2word_freq
        )

    if len(args.tag_emb_file) > 0 and args.tag_emb_file != "None":
        print(args.tag_emb_file)
        tag_emb, tag_emb_size = load_emb_from_path(
            args.tag_emb_file, device, tag_idx2word_freq
        )
        assert tag_emb_size == user_emb_size
        target_emb_size = tag_emb_size
    else:
        tag_emb = []
        tag_emb_size = user_emb_size
        target_emb_size = tag_emb_size
        # raise Exception("Must provide entity pair emb file when loading all models for evaluation!")

    if args.trans_nhid < 0:
        if args.target_emsize > 0:
            args.trans_nhid = args.target_emsize
        else:
            args.trans_nhid = target_emb_size

    ntokens = len(idx2word_freq)
    # external_emb = torch.tensor([0.])
    # external_emb = torch.tensor(word_emb)
    # encoder = RNNModel_simple(args.en_model, ntokens, args.emsize, args.nhid, args.nlayers, #model_old_1
    # encoder = SEQ2EMB(args.en_model, ntokens, args.emsize, args.nhid, args.nlayers, #model_old_2, model_old_3
    #               args.dropout, args.dropouti, args.dropoute, external_emb)
    # encoder = SEQ2EMB(args.en_model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouti, args.dropoute, max_sent_len,  external_emb, [], trans_layer = args.encode_trans_layer) #model_old_4
    word_norm_emb = []
    encoder = SEQ2EMB(
        args.en_model.split("+"),
        ntokens,
        source_emb_size,
        args.nhid,
        args.nlayers,
        args.dropout,
        args.dropouti,
        args.dropoute,
        max_sent_len,
        word_norm_emb,
        [],
        trans_layers=args.encode_trans_layers,
        trans_nhid=args.trans_nhid,
        num_type_feature=args.num_type_feature,
    )

    if args.nhidlast2 < 0:
        args.nhidlast2 = encoder.output_dim
    # decoder = EMB2SEQ(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = args.linear_mapping_dim, dropoutp= 0.5) #model_old_2
    # decoder = EMB2SEQ(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = args.linear_mapping_dim, dropoutp= args.dropoutp, trans_layer = args.trans_layer) #model_old_3, model_old_4
    decoder = EMB2SEQ(
        args.de_model.split("+"),
        args.de_coeff_model,
        encoder.output_dim,
        args.nhidlast2,
        target_emb_size,
        1,
        args.n_basis,
        positional_option=args.positional_option,
        dropoutp=args.dropoutp,
        trans_layers=args.trans_layers,
        using_memory=args.de_en_connection,
        dropout_prob_trans=args.dropout_prob_trans,
        dropout_prob_lstm=args.dropout_prob_lstm,
        de_output_layers=args.de_output_layers,
    )  # model_old_5
    # decoder = RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = args.linear_mapping_dim, dropoutp= 0.5) #model_old_1
    # if use_position_emb:
    #    decoder = RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = 0, dropoutp= 0.5)
    # else:
    #    decoder = RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = args.nhid, dropoutp= 0.5)

    encoder.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "encoder.pt"), map_location=device)
    )
    decoder.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "decoder.pt"), map_location=device)
    )

    # if len(args.source_emb_file) == 0:
    #    word_emb = encoder.encoder.weight.detach()
    # word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True) )
    # word_norm_emb[0,:] = 0

    if load_auto:
        with torch.no_grad():
            print(args.word_emb_file)
            word_emb, word_emb_size = load_emb_from_path(
                args.word_emb_file, device, idx2word_freq
            )
            lin_layer_path = os.path.join(args.checkpoint, "feature_linear_layer.pt")
            feature_linear_layer = torch.load(lin_layer_path, map_location=device)
            word_emb_trans = torch.mm(word_emb, feature_linear_layer)
            if normalize_emb:
                word_norm_emb_trans = word_emb_trans / (
                    0.000000000001 + word_emb_trans.norm(dim=1, keepdim=True)
                )
                # word_norm_emb_trans = word_emb_trans
            else:
                word_norm_emb_trans = word_emb_trans

    if normalize_emb:
        if len(tag_emb) > 0:
            tag_norm_emb = tag_emb / (
                0.000000000001 + tag_emb.norm(dim=1, keepdim=True)
            )
        else:
            tag_norm_emb = tag_emb
        user_norm_emb = user_emb / (0.000000000001 + user_emb.norm(dim=1, keepdim=True))
    else:
        tag_norm_emb = tag_emb
        user_norm_emb = user_emb

    if len(tag_norm_emb) > 0:
        tag_norm_emb[0, :] = 0
    user_norm_emb[0, :] = 0

    parallel_encoder, parallel_decoder = output_parallel_models(
        args.cuda, args.single_gpu, encoder, decoder
    )

    if load_auto:
        return (
            parallel_encoder,
            parallel_decoder,
            encoder,
            decoder,
            user_norm_emb,
            tag_norm_emb,
            word_norm_emb_trans,
        )
    else:
        return (
            parallel_encoder,
            parallel_decoder,
            encoder,
            decoder,
            user_norm_emb,
            tag_norm_emb,
        )


def seed_all_randomness(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not use_cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )
        else:
            torch.cuda.manual_seed(seed)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print("Experiment dir : {}".format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(
    encoder,
    decoder,
    optimizer_e,
    optimizer_d,
    optimizer_t,
    user_emb,
    tag_emb,
    feature_linear_layer,
    path,
    save_model=True,
    target_embedding_suffix="",
):
    if save_model:
        torch.save(encoder.state_dict(), os.path.join(path, "encoder.pt"))
        try:
            torch.save(decoder.state_dict(), os.path.join(path, "decoder.pt"))
        except Exception:
            pass
        torch.save(optimizer_e.state_dict(), os.path.join(path, "optimizer_e.pt"))
        torch.save(optimizer_d.state_dict(), os.path.join(path, "optimizer_d.pt"))
        torch.save(optimizer_t.state_dict(), os.path.join(path, "optimizer_t.pt"))
    if user_emb.size(0) > 1:
        torch.save(
            user_emb, os.path.join(path, "user_emb" + target_embedding_suffix + ".pt")
        )
    if tag_emb.size(0) > 1:
        torch.save(
            tag_emb, os.path.join(path, "tag_emb" + target_embedding_suffix + ".pt")
        )
    if feature_linear_layer.size(0) > 1:
        torch.save(
            feature_linear_layer,
            os.path.join(
                path, "feature_linear_layer" + target_embedding_suffix + ".pt"
            ),
        )


def str2bool(v):
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
