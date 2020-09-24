import gc
import math
import numpy as np
import os
import sys
import random
import time
import torch

from .mfr_src import model as model_code
from .mfr_src import nsd_loss
from .mfr_src.utils import seed_all_randomness, Dictionary, save_checkpoint, load_emb_file_to_dict, \
    load_emb_file_to_tensor, load_corpus, output_parallel_models, load_emb_from_path
from .mfr_src.utils_testing import compute_freq_prob_idx2word, recommend_test

from spacy.lang.en import English
from shutil import copyfile
from collections import defaultdict
from tqdm import tqdm


def logging(s, save_dir="", log_file_name="", print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(save_dir, log_file_name), 'a+') as f_log:
            f_log.write(s + '\n')


def squeeze_into_tensors(save_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus,
                         output_save_file, max_sent_len=512, max_target_num=10, shuffle_order=True, push_to_right=True):

    def save_to_tensor(w_ind_corpus_dup_j, out_tensor, idx, store_right=False):
        sent_len = len(w_ind_corpus_dup_j)
        if store_right:
            out_tensor[idx, -sent_len:] = torch.tensor(w_ind_corpus_dup_j, dtype=store_type)
        else:
            out_tensor[idx, :sent_len] = torch.tensor(w_ind_corpus_dup_j, dtype=store_type)

    def duplicate_for_long_targets(save_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus,
                                   max_target_num):
        w_ind_corpus_dup = []
        user_corpus_dup = []
        tag_corpus_dup = []
        bid_score_corpus_dup = []
        type_corpus_dup = []
        num_repeat_corpus_dup = []
        len_corpus_dup = []

        num_no_label_item = 0

        for j in save_idx:
            current_w_idx = w_ind_corpus[j]
            current_users = user_corpus[j]
            current_tags = tag_corpus[j]
            if len(bid_score_corpus) > 0:
                current_bid_score = bid_score_corpus[j]
            else:
                current_bid_score = []
            if sum(current_users) == 0 and sum(current_tags) == 0:
                num_no_label_item += 1
                continue
            num_repeat = 0
            while (len(current_users) > 0 or len(current_tags) > 0):
                w_ind_corpus_dup.append(current_w_idx)
                if len(type_corpus) > 0:
                    type_corpus_dup.append(type_corpus[j])
                user_len = min(max_target_num, len(current_users))
                tag_len = min(max_target_num, len(current_tags))
                user_corpus_dup.append(current_users[:user_len])
                tag_corpus_dup.append(current_tags[:tag_len])
                if len(bid_score_corpus) > 0:
                    bid_score_corpus_dup.append(current_bid_score[:user_len])
                    current_bid_score = current_bid_score[user_len:]
                current_users = current_users[user_len:]
                current_tags = current_tags[tag_len:]
                len_corpus_dup.append([user_len, tag_len])
                num_repeat += 1
            num_repeat_corpus_dup += [num_repeat] * num_repeat

        print("Remove {} empty items with no label".format(num_no_label_item))

        return w_ind_corpus_dup, user_corpus_dup, tag_corpus_dup, num_repeat_corpus_dup, len_corpus_dup, type_corpus_dup, bid_score_corpus_dup

    w_ind_corpus_dup, user_corpus_dup, tag_corpus_dup, num_repeat_corpus_dup, len_corpus_dup, type_corpus_dup, \
        bid_score_corpus_dup = duplicate_for_long_targets(save_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus,
                                                          bid_score_corpus, max_target_num)

    store_type = torch.int32
    corpus_size = len(w_ind_corpus_dup)
    tensor_feature = torch.zeros(corpus_size, max_sent_len, dtype=store_type)
    tensor_type = torch.zeros(corpus_size, max_sent_len, dtype=store_type)
    tensor_bid_score = torch.zeros(0, dtype=store_type)
    tensor_user = torch.zeros(corpus_size, max_target_num, dtype=store_type)
    tensor_tag = torch.zeros(corpus_size, max_target_num, dtype=store_type)
    tensor_repeat_num = torch.zeros(corpus_size, dtype=store_type)
    tensor_user_len = torch.zeros(corpus_size, dtype=store_type)
    tensor_tag_len = torch.zeros(corpus_size, dtype=store_type)
    shuffled_order = list(range(corpus_size))
    if shuffle_order:
        random.shuffle(shuffled_order)
    for i, j in enumerate(shuffled_order):
        if push_to_right:
            store_right = True
        else:
            store_right = False
        save_to_tensor(w_ind_corpus_dup[j], tensor_feature, i, store_right)
        save_to_tensor([x + 1 for x in type_corpus_dup[j]], tensor_type, i, store_right)
        save_to_tensor(user_corpus_dup[j], tensor_user, i)
        save_to_tensor(tag_corpus_dup[j], tensor_tag, i)
        tensor_repeat_num[i] = num_repeat_corpus_dup[j]
        tensor_user_len[i] = len_corpus_dup[j][0]
        tensor_tag_len[i] = len_corpus_dup[j][1]

    with open(output_save_file, 'wb') as f_out:
        torch.save([tensor_feature, tensor_type, tensor_user, tensor_tag, tensor_repeat_num, tensor_user_len,
                    tensor_tag_len, tensor_bid_score], f_out)


def counter_to_tensor(idx2word_freq,device, uniform=True, smooth_alpha=0.):
    total = len(idx2word_freq)
    w_freq = torch.zeros(total, dtype=torch.float, device = device, requires_grad=False)
    for i in range(total):
        if uniform:
            w_freq[i] = 1
        else:
            if smooth_alpha == 0:
                w_freq[i] = idx2word_freq[i][1]
            else:
                w_freq[i] = (smooth_alpha + idx2word_freq[i][2]) / smooth_alpha
    w_freq[0] = -1
    return w_freq


def load_ext_emb(emb_file, target_emb_sz, idx2word_freq, num_special_token, device):
    num_w = len(idx2word_freq)
    if len(emb_file) > 0:
        if emb_file[-3:] == '.pt':
            target_emb = torch.load(emb_file).to(device=device)
            target_emb.requires_grad = False
            target_emb_sz = target_emb.size(1)
        else:
            word2emb, emb_size = load_emb_file_to_dict(emb_file, convert_np=False)
            target_emb_sz = emb_size
            target_emb = torch.randn(num_w, target_emb_sz, device=device, requires_grad=False)
            OOV_freq = 0
            total_freq = 0
            OOV_type = 0
            for i in range(num_special_token, num_w):
                w = idx2word_freq[i][0]
                total_freq += idx2word_freq[i][1]
                if w in word2emb:
                    val = torch.tensor(word2emb[w], device=device, requires_grad=False)
                    target_emb[i, :] = val
                else:
                    OOV_type += 1
                    OOV_freq += idx2word_freq[i][1]
            print("OOV word type percentage: {}%".format(OOV_type/float(num_w)*100))
            print("OOV token percentage: {}%".format(OOV_freq/float(total_freq)*100))
    else:
        target_emb = torch.randn(num_w, target_emb_sz, device=device, requires_grad=False)
    target_emb = target_emb / (0.000000000001 + target_emb.norm(dim=1, keepdim=True))
    target_emb.requires_grad = True
    return target_emb, target_emb_sz


class MultiFacetRecommender(object):
    def __init__(self, work_dir, feature_vocab_file, model_checkpoint_dir, epochs=100, batch_size=50, small_batch_size=-1,
                 use_cuda=True, seed=111, sparse_value=None):
        if not os.path.exists(work_dir) and not os.path.isdir(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir
        self.model_checkpoint_dir = model_checkpoint_dir

        # Copy input vocabulary to work_dir for easy further handling
        if not os.path.exists(os.path.join(work_dir, "feature")):
            os.makedirs(os.path.join(work_dir, "feature"))
        copyfile(feature_vocab_file, os.path.join(work_dir, "feature", "dictionary_index"))

        self.batch_size = batch_size
        self.small_batch_size = small_batch_size
        if self.small_batch_size < 0:
            self.small_batch_size = batch_size

        self.L1_losss_B = 0.2
        self.always_save_model = True
        self.auto_avg = False
        self.auto_w = 0
        self.clip = 0.25
        self.coeff_opt = "max"
        self.coeff_opt_algo = "rmsprop"
        self.continue_train = True
        self.copy_training = True
        self.de_coeff_model = "TRANS_old"
        self.de_en_connection = True
        self.de_model = "TRANS"
        self.de_output_layers = "single_dynamic"
        self.dropout = 0.3
        self.dropout_prob_lstm = 0
        self.dropout_prob_trans = 0.3
        self.dropoute = 0
        self.dropouti = 0.3
        self.dropoutp = 0.3
        self.en_model = "TRANS"
        self.encode_trans_layers = 3
        self.epochs = epochs
        self.freeze_encoder_decoder = True
        self.inv_freq_w = True
        self.loading_target_embedding = False
        self.log_file_name = "log_train.txt"
        self.log_interval = 200
        self.loss_type = "dist"
        self.lr = 0.0002
        self.lr2_divide = 1.0
        self.lr_target = -1
        if self.lr_target < 0:
            self.lr_target = self.lr
        self.n_basis = 1
        self.neg_sample_w = 1
        self.nhid = 200
        self.nhidlast2 = -1
        self.nlayers = 2
        self.nonmono = 10
        self.norm_basis_when_freezing = False
        self.optimizer = "Adam"
        self.positional_option = "linear"
        self.rand_neg_method = "shuffle"
        self.single_gpu = False
        self.source_emb_file = "/dev/null"
        self.source_emb_source = "ext"
        self.source_emsize = 0
        self.start_training_split = 0
        self.switch_user_tag_roles = False
        self.tag_emb_file = ""
        self.tag_w = 1.0
        self.target_embedding_suffix = ""
        self.target_emsize = 100
        self.target_l2 = 0
        self.target_norm = True
        self.tensor_folder = "tensors_cold"
        self.training_file = "train.pt"
        self.training_split_num = 1
        self.trans_layers = 3
        self.trans_nhid = -1
        self.update_target_emb = True
        self.user_emb_file = ""
        self.user_w = 5.0
        self.valid_per_epoch = 1
        self.w_loss_coeff = 0.1
        self.warmup_proportion = 0
        self.wdecay = 1e-06

        self.div_eval = "openreview"
        self.most_popular_baseline = False
        self.remove_testing_duplication = False
        self.subsample_ratio = 1
        self.store_dist = "user"
        self.test_tag = False
        self.test_user = True

        self.seed = seed
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.preliminary_scores = None
        self.sparse_value = sparse_value

    def set_archives_dataset(self, archives_dataset):
        self.pub_note_id_to_author_ids = defaultdict(list)
        self.pub_author_ids_to_note_id = defaultdict(list)
        self.pub_note_id_to_abstract = {}
        self.pub_note_id_to_title = {}
        self.archive_paper_ids_list = []
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if publication['id'] not in self.pub_note_id_to_title:
                    self.archive_paper_ids_list.append(publication['id'])
                self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                self.pub_author_ids_to_note_id[profile_id].append(publication['id'])
                self.pub_note_id_to_title[publication['id']] = publication['content'].get('title')
                if self.pub_note_id_to_title[publication['id']] is None:
                    self.pub_note_id_to_title[publication['id']] = ""
                self.pub_note_id_to_abstract[publication['id']] = publication['content'].get('abstract')
                if self.pub_note_id_to_abstract[publication['id']] is None:
                    self.pub_note_id_to_abstract[publication['id']] = ""

    def set_submissions_dataset(self, submissions_dataset):
        self.sub_note_id_to_abstract = {}
        self.sub_note_id_to_title = {}
        self.submission_paper_ids_list = []
        for note_id, submission in submissions_dataset.items():
            assert submission['id'] not in self.sub_note_id_to_title
            self.submission_paper_ids_list.append(submission['id'])
            self.sub_note_id_to_title[submission['id']] = submission['content'].get('title')
            if self.sub_note_id_to_title[submission['id']] is None:
                self.sub_note_id_to_title[submission['id']] = ""
            self.sub_note_id_to_abstract[submission['id']] = submission['content'].get('abstract')
            if self.sub_note_id_to_abstract[submission['id']] is None:
                self.sub_note_id_to_abstract[submission['id']] = ""

    @staticmethod
    def _tokenize_text(corpus):
        nlp = English()
        w_list = []
        for line in corpus:
            if line == "":
                w_list.append([])
            else:
                w_list.append([w.text for w in nlp.tokenizer(line)])
        return w_list

    @staticmethod
    def _concat_title_abstract(title_token_list, abstract_token_list):
        out_list = []
        type_list = []
        for title_tokens, abstract_tokens in zip(title_token_list, abstract_token_list):
            w_list_title = ' '.join(title_tokens + ['<SEP>']).split()
            w_list_abstract = ' '.join(abstract_tokens + ['<SEP>']).split()
            out_list.append(w_list_title + w_list_abstract)
            type_list.append([0] * len(w_list_title) + [1] * len(w_list_abstract))
        return out_list, type_list

    @staticmethod
    def _map_tokens_to_indices(corpus, input_vocab, lowercase, ignore_unk, eos, save_file_name=""):
        w_ind_corpus = []

        dict_c = Dictionary(False)
        with open(input_vocab) as f_in:
            dict_c.load_dict(f_in)

        total_num_w = 0
        for w_list_org in corpus:
            w_ind_list = []
            for w in w_list_org:
                if lowercase:
                    w = w.lower()
                w_ind = dict_c.dict_check(w, ignore_unk)
                w_ind_list.append(w_ind)
                total_num_w += 1
            if eos:
                dict_c.append_eos(w_ind_list)
            w_ind_corpus.append(w_ind_list)
        print("{} tokens before filtering <null>".format(total_num_w))

        compact_mapping = list(range(len(dict_c.ind_l2_w_freq)))
        compact_w_ind_corpus = []
        for w_ind_list in w_ind_corpus:
            compact_w_ind_corpus.append([compact_mapping[x] for x in w_ind_list if compact_mapping[x] > 0])

        if save_file_name:
            with open(save_file_name, 'w') as f_out:
                for w_ind_list in w_ind_corpus:
                    f_out.write(
                        ' '.join([str(compact_mapping[x]) for x in w_ind_list if compact_mapping[x] > 0]) + '\n')

        return compact_w_ind_corpus

    @staticmethod
    def _build_dictionary(corpus, save_path, lowercase, min_freq=0, ignore_unk=True):
        dict_c = Dictionary(False)
        total_num_w = 0
        for w_list_org in corpus:
            for w in w_list_org:
                if lowercase:
                    w = w.lower()
                _ = dict_c.dict_check_add(w)
                total_num_w += 1
        compact_mapping, total_freq_filtering = dict_c.densify_index(min_freq, ignore_unk)
        print("{}/{} tokens are filtered".format(total_freq_filtering, total_num_w))

        if not os.path.exists(save_path) and not os.path.isdir(save_path):
            os.makedirs(save_path)
        dictionary_output_name = os.path.join(save_path, "dictionary_index")
        with open(dictionary_output_name, 'w') as f_out:
            dict_c.store_dict(f_out)

    @staticmethod
    def _shorten_input_features(indices, max_sent_len, push_to_right=True):
        out_indices = []
        num_too_long_sent = 0
        for fields in indices:
            end_idx = len(fields)
            if max_sent_len > 0 and end_idx > max_sent_len:
                num_too_long_sent += 1
                if push_to_right:
                    end_idx = max_sent_len - 1
                    # make sure to include eos as the last word index
                    out_indices.append([x for x in fields[:end_idx] + [fields[-1]]])
                else:
                    end_idx = max_sent_len
                    out_indices.append([x for x in fields[:end_idx]])
            else:
                out_indices.append([x for x in fields[:end_idx]])

        print("Finish loading {} sentences. While truncating {} long sentences".format(len(out_indices),
                                                                                       num_too_long_sent))
        return out_indices

    @staticmethod
    def _train_one_epoch(encoder, decoder, parallel_encoder, parallel_decoder, user_emb, tag_emb, source_emb,
                         feature_uniform, feature_freq, feature_linear_layer, dataloader_train,
                         optimizers, epoch, batch_size, small_batch_size, device, split_i, log_dir, log_file_name,
                         coeff_opt_algo='rmsprop', current_coeff_opt='max', rand_neg_method="shuffle", target_norm=True,
                         freeze_encoder_decoder=True, update_target_emb=True, user_w=5.0, tag_w=1.0, auto_w=0.0,
                         auto_avg=False, loss_type="dist", neg_sample_w=1, clip=0.25, log_interval=200, L1_losss_B=0.2,
                         norm_basis_when_freezing=False, user_uniform=None, user_freq=None,
                         tag_uniform=None, tag_freq=None, basis_pred_train_cache=None, basis_pred_tag_train_cache=None):
        start_time = time.time()
        total_loss = 0.
        total_loss_set_user = 0.
        total_loss_set_neg_user = 0.
        total_loss_set_tag = 0.
        total_loss_set_neg_tag = 0.
        total_loss_set_auto = 0.
        total_loss_set_neg_auto = 0.
        total_loss_set_reg = 0.
        total_loss_set_div = 0.
        total_loss_set_div_target_user = 0.
        total_loss_set_div_target_tag = 0.

        if freeze_encoder_decoder:
            encoder.eval()
            decoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            for p in decoder.parameters():
                p.requires_grad = False
        else:
            encoder.train()
            decoder.train()
        optimizer_e, optimizer_d, optimizer_t, optimizer_auto = optimizers

        for i_batch, sample_batched in enumerate(dataloader_train):
            feature, feature_type, user, tag, repeat_num, user_len, tag_len, sample_idx = sample_batched
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            optimizer_t.zero_grad()
            optimizer_auto.zero_grad()

            if freeze_encoder_decoder and epoch > 1:
                # load cache
                sample_idx_np = sample_idx.numpy()
                basis_pred = torch.tensor(basis_pred_train_cache[sample_idx_np, :, :], dtype=torch.float, device=device)
                basis_pred_tag = torch.tensor(basis_pred_tag_train_cache[sample_idx_np, :, :], dtype=torch.float,
                                              device=device)
            else:
                output_emb_last, output_emb = parallel_encoder(feature, feature_type)

                basis_pred, basis_pred_tag, basis_pred_auto = parallel_decoder(output_emb_last, output_emb,
                                                                               predict_coeff_sum=False)
                if freeze_encoder_decoder:
                    # store cache
                    if norm_basis_when_freezing:
                        basis_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim=2, keepdim=True))
                        basis_pred_tag = basis_pred_tag / (0.000000000001 + basis_pred_tag.norm(dim=2, keepdim=True))
                    sample_idx_np = sample_idx.numpy()
                    basis_pred_train_cache[sample_idx_np, :, :] = basis_pred.cpu().numpy()
                    basis_pred_tag_train_cache[sample_idx_np, :, :] = basis_pred_tag.cpu().numpy()

            compute_target_grad = update_target_emb

            if user_w > 0:
                input_basis = basis_pred
                loss_set_user, loss_set_neg_user, loss_set_div, loss_set_reg, loss_set_div_target_user = nsd_loss.compute_loss_set(input_basis, user_emb, user, L1_losss_B, device, user_uniform, user_freq, repeat_num, user_len, current_coeff_opt, loss_type, compute_target_grad, coeff_opt_algo, rand_neg_method, target_norm)
                if torch.isnan(loss_set_user):
                    sys.stdout.write('user nan, ')
                    continue
            else:
                loss_set_user = torch.tensor(0, device=device)
                loss_set_neg_user = torch.tensor(0, device=device)
                loss_set_div_target_user = torch.tensor(0, device=device)

            if tag_w > 0:
                input_basis = basis_pred_tag
                loss_set_tag, loss_set_neg_tag, loss_set_div, loss_set_reg, loss_set_div_target_tag = nsd_loss.compute_loss_set(input_basis, tag_emb, tag, L1_losss_B, device, tag_uniform, tag_freq, repeat_num, tag_len, current_coeff_opt, loss_type, compute_target_grad, coeff_opt_algo, rand_neg_method, target_norm)
                if torch.isnan(loss_set_tag):
                    sys.stdout.write('tag nan, ')
                    continue
            else:
                loss_set_tag = torch.tensor(0, device=device)
                loss_set_neg_tag = torch.tensor(0, device=device)
                loss_set_div_target_tag = torch.tensor(0, device=device)

            if auto_w > 0:
                feature_len = None
                if auto_avg:
                    basis_pred_auto_compressed = basis_pred_auto.mean(dim=1).unsqueeze(dim=1)
                else:
                    basis_pred_auto_compressed = basis_pred_auto
                rand_neg_method = 'rotate'
                loss_set_auto, loss_set_neg_auto = nsd_loss.compute_loss_set(basis_pred_auto_compressed, source_emb, feature, L1_losss_B, device, feature_uniform, feature_freq, repeat_num, feature_len, current_coeff_opt, loss_type, compute_target_grad, coeff_opt_algo, rand_neg_method, target_norm, compute_div_reg = False, target_linear_layer = feature_linear_layer, pre_avg = True)
                if torch.isnan(loss_set_auto):
                    sys.stdout.write('auto nan, ')
                    continue
            else:
                loss_set_auto = torch.tensor(0, device=device)
                loss_set_neg_auto = torch.tensor(0, device=device)

            total_loss_set_user += loss_set_user.item() * small_batch_size / batch_size
            total_loss_set_neg_user += loss_set_neg_user.item() * small_batch_size / batch_size
            total_loss_set_tag += loss_set_tag.item() * small_batch_size / batch_size
            total_loss_set_neg_tag += loss_set_neg_tag.item() * small_batch_size / batch_size
            total_loss_set_auto += loss_set_auto.item() * small_batch_size / batch_size
            total_loss_set_neg_auto += loss_set_neg_auto.item() * small_batch_size / batch_size

            total_loss_set_reg += loss_set_reg.item() * small_batch_size / batch_size
            total_loss_set_div += loss_set_div.item() * small_batch_size / batch_size
            total_loss_set_div_target_tag += loss_set_div_target_tag.item() * small_batch_size / batch_size
            total_loss_set_div_target_user += loss_set_div_target_user.item() * small_batch_size / batch_size

            loss = user_w * loss_set_user
            loss += tag_w * loss_set_tag
            loss += auto_w * loss_set_auto
            if loss_type == 'sim':
                loss += user_w * neg_sample_w * loss_set_neg_user
                loss += tag_w * neg_sample_w * loss_set_neg_tag
                loss += auto_w * neg_sample_w * loss_set_neg_auto
            else:
                if -loss_set_neg_user > 1:
                    loss -= user_w * neg_sample_w * loss_set_neg_user
                else:
                    loss += user_w * neg_sample_w * loss_set_neg_user
                if -loss_set_neg_tag > 1:
                    loss -= tag_w * neg_sample_w * loss_set_neg_tag
                else:
                    loss += tag_w * neg_sample_w * loss_set_neg_tag
                if -loss_set_neg_auto > 1:
                    loss -= auto_w * neg_sample_w * loss_set_neg_auto
                else:
                    loss += auto_w * neg_sample_w * loss_set_neg_auto

            loss *= small_batch_size / batch_size
            total_loss += loss.item()

            loss.backward()

            gc.collect()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            optimizer_e.step()
            optimizer_d.step()

            if update_target_emb:
                optimizer_t.step()
                optimizer_auto.step()
                if user_w > 0:
                    user_emb.data[0, :] = 0

                if tag_w > 0:
                    tag_emb.data[0, :] = 0

            if i_batch % log_interval == 0 and i_batch > 0:
                cur_loss = total_loss / log_interval
                cur_loss_set_user = total_loss_set_user / log_interval
                cur_loss_set_neg_user = total_loss_set_neg_user / log_interval
                cur_loss_set_tag = total_loss_set_tag / log_interval
                cur_loss_set_neg_tag = total_loss_set_neg_tag / log_interval
                cur_loss_set_auto = total_loss_set_auto / log_interval
                cur_loss_set_neg_auto = total_loss_set_neg_auto / log_interval
                cur_loss_set_reg = total_loss_set_reg / log_interval
                cur_loss_set_div = total_loss_set_div / log_interval
                cur_loss_set_div_target_user = total_loss_set_div_target_user / log_interval
                cur_loss_set_div_target_tag = total_loss_set_div_target_tag / log_interval
                elapsed = time.time() - start_time
                logging('| e {:3d} {:3d} | {:5d}/{:5d} b | lr-enc {:.6f} | ms/batch {:5.2f} | '
                        'l {:5.2f} | l_f_u {:5.5f} + {:2.2f}*{:5.5f} = {:5.5f} | div_u {:5.2f} | l_f_t {:5.4f} + {:2.2f}*{:5.4f} = {:5.4f} | div_t {:5.2f} | l_f_a {:5.4f} + {:2.2f}*{:5.4f} = {:5.4f} | reg {:5.2f} | div {:5.2f} '.format(
                    epoch, split_i, i_batch, len(dataloader_train.dataset) // batch_size, optimizer_e.param_groups[0]['lr'],
                    elapsed * 1000 / log_interval, cur_loss, cur_loss_set_user, neg_sample_w, cur_loss_set_neg_user, cur_loss_set_user + neg_sample_w * cur_loss_set_neg_user, cur_loss_set_div_target_user, cur_loss_set_tag, neg_sample_w, cur_loss_set_neg_tag, cur_loss_set_tag + neg_sample_w * cur_loss_set_neg_tag, cur_loss_set_div_target_tag, cur_loss_set_auto, neg_sample_w, cur_loss_set_neg_auto, cur_loss_set_auto + neg_sample_w * cur_loss_set_neg_auto, cur_loss_set_reg, cur_loss_set_div),
                    save_dir=log_dir, log_file_name=log_file_name)

                total_loss = 0.
                total_loss_set_user = 0.
                total_loss_set_neg_user = 0.
                total_loss_set_tag = 0.
                total_loss_set_neg_tag = 0.
                total_loss_set_auto = 0.
                total_loss_set_neg_auto = 0.
                total_loss_set_reg = 0.
                total_loss_set_div = 0.
                total_loss_set_div_target_tag = 0.
                total_loss_set_div_target_user = 0.
                start_time = time.time()
        return current_coeff_opt

    def embed_submissions(self, submissions_path=None):
        print('Featurizing submissions...')
        titles_corpus = [self.sub_note_id_to_title[note_id] for note_id in self.submission_paper_ids_list]
        titles_corpus_tokens = self._tokenize_text(titles_corpus)
        abstracts_corpus = [self.sub_note_id_to_abstract[note_id] for note_id in self.submission_paper_ids_list]
        abstracts_corpus_tokens = self._tokenize_text(abstracts_corpus)
        meta_corpus, type_indices = self._concat_title_abstract(titles_corpus_tokens, abstracts_corpus_tokens)
        feature_indices = self._map_tokens_to_indices(meta_corpus, os.path.join(self.work_dir, "feature",
                                                                                "dictionary_index"),
                                                      lowercase=True, ignore_unk=False, eos=True,
                                                      save_file_name=os.path.join(self.work_dir, "feature",
                                                                                  "corpus_index_test"))
        feature_indices_trunc = self._shorten_input_features(feature_indices, max_sent_len=512)
        type_indices_trunc = self._shorten_input_features(type_indices, max_sent_len=512)

        user_dictionary = os.path.join(self.work_dir, "user", "dictionary_index")
        # Create dummy user indices
        user_indices = [[1]] * len(feature_indices)

        tag_dictionary = os.path.join(self.work_dir, "tag", "dictionary_index")
        # Create dummy tag indices
        tag_indices = [[1]] * len(feature_indices)

        tensor_output_dir = os.path.join(self.work_dir, "tensors_cold/")
        all_idx = list(range(len(feature_indices_trunc)))
        squeeze_into_tensors(all_idx, feature_indices_trunc, type_indices_trunc, user_indices, tag_indices, [],
                             os.path.join(tensor_output_dir, "test.pt"), shuffle_order=False)

        print('NOTE: Currently submission embeddings cannot be precomputed. '
              'They are computed and consumed during score calculation')

    def embed_publications(self, publications_path=None):
        print('Featurizing publications...')
        titles_corpus = [self.pub_note_id_to_title[note_id] for note_id in self.archive_paper_ids_list]
        titles_corpus_tokens = self._tokenize_text(titles_corpus)
        abstracts_corpus = [self.pub_note_id_to_abstract[note_id] for note_id in self.archive_paper_ids_list]
        abstracts_corpus_tokens = self._tokenize_text(abstracts_corpus)
        meta_corpus, type_indices = self._concat_title_abstract(titles_corpus_tokens, abstracts_corpus_tokens)
        feature_indices = self._map_tokens_to_indices(meta_corpus, os.path.join(self.work_dir, "feature",
                                                                                "dictionary_index"),
                                                      lowercase=True, ignore_unk=False, eos=True,
                                                      save_file_name=os.path.join(self.work_dir, "feature",
                                                                                  "corpus_index"))
        feature_indices_trunc = self._shorten_input_features(feature_indices, max_sent_len=512)
        type_indices_trunc = self._shorten_input_features(type_indices, max_sent_len=512)

        user_dictionary = os.path.join(self.work_dir, "user", "dictionary_index")
        user_corpus = [self.pub_note_id_to_author_ids[note_id] for note_id in self.archive_paper_ids_list]
        if not os.path.exists(user_dictionary):
            print('Building user dictionary...')
            self._build_dictionary(corpus=user_corpus, save_path=os.path.join(self.work_dir, "user"), lowercase=False)
        user_indices = self._map_tokens_to_indices(user_corpus, user_dictionary, lowercase=False, ignore_unk=True,
                                                   eos=False, save_file_name=os.path.join(self.work_dir, "user",
                                                                                          "corpus_index"))

        tag_dictionary = os.path.join(self.work_dir, "tag", "dictionary_index")
        tag_corpus = [self.pub_note_id_to_author_ids[note_id] for note_id in self.archive_paper_ids_list]
        if not os.path.exists(tag_dictionary):
            print('Building tag dictionary...')
            self._build_dictionary(corpus=tag_corpus, save_path=os.path.join(self.work_dir, "tag"), lowercase=False)
        tag_indices = self._map_tokens_to_indices(tag_corpus, tag_dictionary, lowercase=False, ignore_unk=True,
                                                  eos=False, save_file_name=os.path.join(self.work_dir, "tag",
                                                                                         "corpus_index"))

        tensor_output_dir = os.path.join(self.work_dir, "tensors_cold/")
        if not os.path.exists(tensor_output_dir):
            os.makedirs(tensor_output_dir)

        all_idx = list(range(len(feature_indices_trunc)))
        squeeze_into_tensors(all_idx, feature_indices_trunc, type_indices_trunc, user_indices, tag_indices, [],
                             os.path.join(tensor_output_dir, "train.pt"), shuffle_order=False)

        seed_all_randomness(self.seed, self.use_cuda)
        print("Loading data")
        idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, _, max_sent_len = \
            load_corpus(self.work_dir, self.batch_size, self.batch_size, self.device, "tensors_cold", "train.pt",
                        1, copy_training=True, load_val=False)

        print("Initializing model")
        num_special_token = 3

        # Initialize or load source embeddings
        source_emb = torch.tensor([0.])
        extra_init_idx = []
        source_emb, source_emb_size, extra_init_idx = load_emb_file_to_tensor("/dev/null", self.device,
                                                                              idx2word_freq)
        source_emb = source_emb / (0.000000000001 + source_emb.norm(dim=1, keepdim=True))
        source_emb.requires_grad = False

        user_emb, target_emb_sz = load_ext_emb(self.user_emb_file, self.target_emsize, user_idx2word_freq,
                                               num_special_token, self.device)
        if self.tag_w > 0:
            tag_emb, target_emb_sz_tag = load_ext_emb(self.tag_emb_file, self.target_emsize, tag_idx2word_freq,
                                                      num_special_token, self.device)
        else:
            tag_emb = torch.zeros(0)
            target_emb_sz_tag = target_emb_sz

        if self.trans_nhid < 0:
            if self.target_emsize > 0:
                self.trans_nhid = self.target_emsize
            else:
                self.trans_nhid = target_emb_sz

        if not self.inv_freq_w:
            user_uniform = counter_to_tensor(user_idx2word_freq, self.device, uniform=True)
            tag_uniform = counter_to_tensor(tag_idx2word_freq, self.device, uniform=True)
        else:
            user_uniform = counter_to_tensor(user_idx2word_freq, self.device, uniform=False)
            tag_uniform = counter_to_tensor(tag_idx2word_freq, self.device, uniform=False)
        user_freq = counter_to_tensor(user_idx2word_freq, self.device, uniform=False)
        tag_freq = counter_to_tensor(tag_idx2word_freq, self.device, uniform=False)
        # When do the categorical sampling, do not include <null>, <eos> and <unk> (just gives 0 probability)
        user_freq[:num_special_token] = 0
        tag_freq[:num_special_token] = 0

        if self.auto_w > 0:
            compute_freq_prob_idx2word(idx2word_freq)
            feature_uniform = counter_to_tensor(idx2word_freq, self.device, uniform=False, smooth_alpha=1e-4)
            feature_freq = counter_to_tensor(idx2word_freq, self.device, uniform=False)
            feature_linear_layer = torch.randn(source_emb_size, target_emb_sz, device=self.device, requires_grad=True)
        else:
            feature_uniform = None
            feature_freq = None
            feature_linear_layer = torch.zeros(0)

        ntokens = len(idx2word_freq)
        encoder = model_code.SEQ2EMB(self.en_model.split('+'), ntokens, self.source_emsize, self.nhid, self.nlayers,
                                     self.dropout, self.dropouti, self.dropoute, max_sent_len, source_emb,
                                     extra_init_idx, self.encode_trans_layers, self.trans_nhid)

        if self.auto_w == 0:
            del source_emb
            source_emb = None

        if self.nhidlast2 < 0:
            self.nhidlast2 = encoder.output_dim

        decoder = model_code.EMB2SEQ(self.de_model.split('+'), self.de_coeff_model, encoder.output_dim, self.nhidlast2,
                                     target_emb_sz, 1, self.n_basis, positional_option=self.positional_option,
                                     dropoutp=self.dropoutp, trans_layers=self.trans_layers,
                                     using_memory=self.de_en_connection, dropout_prob_trans=self.dropout_prob_trans,
                                     dropout_prob_lstm=self.dropout_prob_lstm, de_output_layers=self.de_output_layers)

        if self.de_en_connection and decoder.trans_dim is not None and encoder.output_dim != decoder.trans_dim:
            print("dimension mismatch. The encoder output dimension is ", encoder.output_dim,
                  " and the transformer dimension in decoder is ", decoder.trans_dim)
            sys.exit(1)

        if self.continue_train:
            encoder.load_state_dict(torch.load(os.path.join(self.model_checkpoint_dir, 'encoder.pt')))
            decoder.load_state_dict(torch.load(os.path.join(self.model_checkpoint_dir, 'decoder.pt')))
            if self.loading_target_embedding:
                user_emb_load = torch.load(os.path.join(self.model_checkpoint_dir, 'user_emb.pt'))
                user_emb = user_emb.new_tensor(user_emb_load)
                if self.tag_w > 0:
                    tag_emb_load = torch.load(os.path.join(self.model_checkpoint_dir, 'tag_emb.pt'))
                    tag_emb = tag_emb.new_tensor(tag_emb_load)

        parallel_encoder, parallel_decoder = output_parallel_models(self.use_cuda, self.single_gpu, encoder, decoder)

        total_params = sum(x.data.nelement() for x in encoder.parameters())
        logging('Encoder total parameters: {}'.format(total_params), self.work_dir, self.log_file_name)
        total_params = sum(x.data.nelement() for x in decoder.parameters())
        logging('Decoder total parameters: {}'.format(total_params), self.work_dir, self.log_file_name)

        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=self.lr, weight_decay=self.wdecay)
        optimizer_d = torch.optim.Adam(decoder.parameters(), lr=self.lr / self.lr2_divide, weight_decay=self.wdecay)
        optimizer_t = torch.optim.Adam([user_emb, tag_emb], lr=self.lr_target, weight_decay=self.target_l2)
        optimizer_auto = torch.optim.SGD([feature_linear_layer], lr=self.lr_target)

        num_sample_train = dataloader_train_arr[0].dataset.feature.size(0)
        basis_pred_train_cache = np.empty((num_sample_train, self.n_basis, target_emb_sz))
        basis_pred_tag_train_cache = np.empty((num_sample_train, self.n_basis, target_emb_sz))

        if self.coeff_opt == 'maxlc':
            current_coeff_opt = 'max'
        else:
            current_coeff_opt = self.coeff_opt

        print('Training user embeddings...')
        steps = 0
        saving_freq = int(math.floor(self.training_split_num / self.valid_per_epoch))
        for epoch in range(1, self.epochs + 1):
            for i in range(len(dataloader_train_arr)):
                if epoch == 1 and i < self.start_training_split:
                    print("Skipping epoch " + str(epoch) + ' split ' + str(i))
                    continue
                current_coeff_opt = self._train_one_epoch(encoder, decoder, parallel_encoder, parallel_decoder,
                                                          user_emb, tag_emb, source_emb, feature_uniform, feature_freq,
                                                          feature_linear_layer, dataloader_train_arr[i],
                                                          (optimizer_e, optimizer_d, optimizer_t, optimizer_auto),
                                                          epoch, self.batch_size, self.small_batch_size, self.device, i,
                                                          self.work_dir, self.log_file_name, self.coeff_opt_algo,
                                                          current_coeff_opt, self.rand_neg_method, self.target_norm,
                                                          self.freeze_encoder_decoder, self.update_target_emb,
                                                          self.user_w, self.tag_w, self.auto_w, self.auto_avg,
                                                          self.loss_type, self.neg_sample_w, self.clip,
                                                          self.log_interval, self.L1_losss_B,
                                                          self.norm_basis_when_freezing, user_uniform, user_freq,
                                                          tag_uniform, tag_freq, basis_pred_train_cache,
                                                          basis_pred_tag_train_cache)
                steps += len(dataloader_train_arr[i])
                if i != self.training_split_num - 1 and (i + 1) % saving_freq != 0:
                    continue

                # self.always_save_model
                # save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, source_emb, target_emb, args.save)
                if self.freeze_encoder_decoder:
                    save_model = False
                else:
                    save_model = True
                target_embedding_suffix = self.target_embedding_suffix
                if self.always_save_model:
                    target_embedding_suffix += '_always'

                save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, optimizer_t, user_emb, tag_emb,
                                feature_linear_layer, self.work_dir, save_model=save_model,
                                target_embedding_suffix=target_embedding_suffix)
                logging('Models Saved', self.work_dir, self.log_file_name)

        self.user_emb_file = os.path.join(self.work_dir, "user_emb" + self.target_embedding_suffix + '_always.pt')
        self.tag_emb_file = os.path.join(self.work_dir, "tag_emb" + self.target_embedding_suffix + '_always.pt')
        self.source_emsize = 200

        print('Reviewer embeddings are saved at {}'.format(self.user_emb_file))
        print('NOTE: Archive publication embeddings are not explicitly computed. '
              'Reviewer embeddings are computed and consumed with submission embeddings during score calculation')

    def all_scores(self, publications_path=None, submissions_path=None, scores_path=None):
        print("Loading data...")
        seed_all_randomness(self.seed, self.use_cuda)
        idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, _,\
            dataloader_test_info, max_sent_len = \
            load_corpus(self.work_dir, self.batch_size, self.batch_size, self.device, tensor_folder=self.tensor_folder,
                        skip_training=True, want_to_shuffle_val=False, load_val=False, load_test=True,
                        deduplication=True, subsample_ratio=self.subsample_ratio,
                        remove_testing_duplication=self.remove_testing_duplication)

        print("Loading Model")
        normalize_emb = True
        if self.loss_type != 'dist':
            normalize_emb = False

        source_emb_size = self.source_emsize
        user_emb, user_emb_size = load_emb_from_path(self.user_emb_file, self.device, user_idx2word_freq)

        tag_emb, tag_emb_size = load_emb_from_path(self.tag_emb_file, self.device, tag_idx2word_freq)
        assert tag_emb_size == user_emb_size
        target_emb_size = tag_emb_size

        if self.trans_nhid < 0:
            if self.target_emsize > 0:
                self.trans_nhid = self.target_emsize
            else:
                self.trans_nhid = target_emb_size

        ntokens = len(idx2word_freq)
        encoder = model_code.SEQ2EMB(self.en_model.split('+'), ntokens, self.source_emsize, self.nhid, self.nlayers,
                                     dropout=0, dropouti=0, dropoute=0, max_sent_len=max_sent_len, external_emb=[],
                                     init_idx=[], trans_layers=self.encode_trans_layers, trans_nhid=self.trans_nhid)

        if self.nhidlast2 < 0:
            self.nhidlast2 = encoder.output_dim

        decoder = model_code.EMB2SEQ(self.de_model.split('+'), self.de_coeff_model, encoder.output_dim, self.nhidlast2,
                                     target_emb_size, 1, self.n_basis, positional_option=self.positional_option,
                                     dropoutp=0, trans_layers=self.trans_layers, using_memory=self.de_en_connection,
                                     dropout_prob_trans=0, dropout_prob_lstm=0, de_output_layers=self.de_output_layers)

        encoder.load_state_dict(torch.load(os.path.join(self.model_checkpoint_dir, 'encoder.pt'),
                                           map_location=self.device))
        decoder.load_state_dict(torch.load(os.path.join(self.model_checkpoint_dir, 'decoder.pt'),
                                           map_location=self.device))

        if normalize_emb:
            if len(tag_emb) > 0:
                tag_norm_emb = tag_emb / (0.000000000001 + tag_emb.norm(dim=1, keepdim=True))
            else:
                tag_norm_emb = tag_emb
            user_norm_emb = user_emb / (0.000000000001 + user_emb.norm(dim=1, keepdim=True))
        else:
            tag_norm_emb = tag_emb
            user_norm_emb = user_emb
        if len(tag_norm_emb) > 0:
            tag_norm_emb[0, :] = 0
        user_norm_emb[0, :] = 0

        parallel_encoder, parallel_decoder = output_parallel_models(self.use_cuda, self.single_gpu, encoder, decoder)

        encoder.eval()
        decoder.eval()

        print('Computing all scores...')
        # Compute distances
        with open(os.path.join(self.work_dir, "reviewer_submission_dist_arr.txt"), 'w') as dist_file:
            recommend_test(dataloader_test_info, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb,
                           idx2word_freq, user_idx2word_freq, tag_idx2word_freq, self.coeff_opt, self.loss_type,
                           self.test_user, self.test_tag, dist_file, self.device, self.most_popular_baseline,
                           self.div_eval, self.switch_user_tag_roles, self.store_dist, figure_name='')

        # Convert distances to scores
        num_special_token = 3
        csv_scores = []
        self.preliminary_scores = []
        dist_arr = np.loadtxt(os.path.join(self.work_dir, "reviewer_submission_dist_arr.txt"))
        assert dist_arr.shape[0] == len(self.submission_paper_ids_list)
        assert dist_arr.shape[1] == len(user_idx2word_freq)
        sim_arr = 1. - dist_arr
        for j in range(num_special_token, len(user_idx2word_freq)):
            user_raw = user_idx2word_freq[j][0]
            user_name = user_raw
            if '|' in user_raw:
                suffix_start = user_raw.index('|')
                user_name = user_raw[:suffix_start]
            for i, paper_id in enumerate(self.submission_paper_ids_list):
                csv_line = '{note_id},{reviewer},{score}'.format(note_id=paper_id, reviewer=user_name,
                                                                 score=sim_arr[i, j])
                csv_scores.append(csv_line)
                self.preliminary_scores.append((paper_id, user_name, sim_arr[i, j]))

        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')

        return self.preliminary_scores

    def _sparse_scores_helper(self, all_scores, id_index):
        counter = 0
        # Get the first note_id or profile_id
        current_id = self.preliminary_scores[0][id_index]
        if id_index == 0:
            desc = 'Note IDs'
        else:
            desc = 'Profiles IDs'
        for note_id, profile_id, score in tqdm(self.preliminary_scores, total=len(self.preliminary_scores), desc=desc):
            if counter < self.sparse_value:
                all_scores.add((note_id, profile_id, score))
            elif (note_id, profile_id)[id_index] != current_id:
                counter = 0
                all_scores.add((note_id, profile_id, score))
                current_id = (note_id, profile_id)[id_index]
            counter += 1
        return all_scores

    def sparse_scores(self, scores_path=None):
        if self.preliminary_scores is None:
            raise RuntimeError("Call all_scores before calling sparse_scores")

        print('Sorting...')
        self.preliminary_scores.sort(key=lambda x: (x[0], x[2]), reverse=True)
        print('preliminary', self.preliminary_scores, len(self.preliminary_scores))
        all_scores = set()
        # They are first sorted by note_id
        all_scores = self._sparse_scores_helper(all_scores, 0)

        # Sort by profile_id
        print('Sorting...')
        self.preliminary_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        all_scores = self._sparse_scores_helper(all_scores, 1)

        print('Final Sort...')
        all_scores = sorted(list(all_scores), key=lambda x: (x[0], x[2]), reverse=True)
        if scores_path:
            with open(scores_path, 'w') as f:
                for note_id, profile_id, score in all_scores:
                    f.write('{0},{1},{2}\n'.format(note_id, profile_id, score))

        print('ALL SCORES', all_scores)
        return all_scores
