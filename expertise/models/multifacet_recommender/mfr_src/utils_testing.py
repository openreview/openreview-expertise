import torch
from .nsd_loss import estimate_prod_coeff_mat_batch_opt, estimate_coeff_mat_batch_opt, estimate_coeff_mat_batch_max
import numpy as np
from scipy.spatial import distance
import scipy.stats as ss
import gc
import sys
import torch.utils.data
from .utils import str2bool

import math
from collections import Counter
from sklearn.metrics import ndcg_score
sys.path.insert(0, sys.path[0] + '/testing/old/sim')


def add_model_arguments(parser):
    ###encoder
    # parser.add_argument('--en_model', type=str, default='LSTM',
    parser.add_argument('--en_model', type=str, default='TRANS',
                        help='type of encoder model (LSTM)')
    # parser.add_argument('--emsize', type=int, default=300,
    #                    help='size of word embeddings')
    parser.add_argument('--source_emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--target_emsize', type=int, default=0,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--encode_trans_layers', type=int, default=3,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--trans_nhid', type=int, default=-1,
                        help='number of hidden units per layer in transformer')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to the output layer (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--num_type_feature', type=int, default=6,
                        help='number of type featres for encoder')
    ###decoder
    # parser.add_argument('--de_model', type=str, default='LSTM',
    parser.add_argument('--de_model', type=str, default='TRANS',
                        help='type of decoder model (LSTM, LSTM+TRANS, TRANS+LSTM, TRANS)')
    # parser.add_argument('--de_coeff_model', type=str, default='LSTM',
    parser.add_argument('--de_coeff_model', type=str, default='TRANS',
                        help='type of decoder model to predict coefficients (LSTM, TRANS)')
    parser.add_argument('--de_output_layers', type=str, default='no_dynamic',
                        help='could be no_dynamic, single dynamic, double dynamic')
    parser.add_argument('--trans_layers', type=int, default=3,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--de_en_connection', type=str2bool, nargs='?', default=True,
                        help='If True, using Transformer decoder in our decoder. Otherwise, using Transformer encoder')
    parser.add_argument('--nhidlast2', type=int, default=-1,
                        help='hidden embedding size of the second LSTM')
    parser.add_argument('--n_basis', type=int, default=3,
                        help='number of basis we want to predict')
    parser.add_argument('--positional_option', type=str, default='linear',
                        help='options of encode positional embedding into models (linear, cat, add)')
    parser.add_argument('--linear_mapping_dim', type=int, default=0,
                        help='map the input embedding by linear transformation')
    # parser.add_argument('--postional_option', type=str, default='linear',
    #                help='options of encode positional embedding into models (linear, cat, add)')
    parser.add_argument('--dropoutp', type=float, default=0,
                        help='dropout of positional embedding or input embedding after linear transformation (when linear_mapping_dim != 0)')
    parser.add_argument('--dropout_prob_trans', type=float, default=0,
                        help='hidden_dropout_prob and attention_probs_dropout_prob in Transformer')
    parser.add_argument('--dropout_prob_lstm', type=float, default=0,
                        help='dropout_prob in LSTM')

    ##scoring
    parser.add_argument('--coeff_opt', type=str, default='max',
                        help='Could be prod or lc or max')
    parser.add_argument('--loss_type', type=str, default='sim',
                        help='Could be sim or dist')


def predict_batch_simple(feature, feature_type, parallel_encoder, parallel_decoder, normalize_emb=True):
    # output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
    # output_emb_last = parallel_encoder(feature)
    output_emb_last, output_emb = parallel_encoder(feature, feature_type)
    basis_pred, basis_pred_tag, basis_pred_auto = parallel_decoder(output_emb_last, output_emb)
    # basis_pred, coeff_pred = predict_basis(parallel_decoder, n_basis, output_emb_last, predict_coeff_sum = True )

    if normalize_emb:
        basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim=2, keepdim=True))
        basis_norm_pred_tag = basis_pred_tag / (0.000000000001 + basis_pred_tag.norm(dim=2, keepdim=True))
        basis_norm_pred_auto = basis_pred_auto / (0.000000000001 + basis_pred_auto.norm(dim=2, keepdim=True))
    else:
        basis_norm_pred = basis_pred
        basis_norm_pred_tag = basis_pred_tag
        basis_norm_pred_auto = basis_pred_auto
    return basis_norm_pred, basis_norm_pred_tag, basis_norm_pred_auto, output_emb_last, output_emb


def predict_batch(feature, feature_type, parallel_encoder, parallel_decoder, word_norm_emb, top_k):
    basis_norm_pred, basis_norm_pred_tag, basis_norm_pred_auto, output_emb_last, output_emb = predict_batch_simple(
        feature, feature_type, parallel_encoder, parallel_decoder)
    # output_emb_last, output_emb = parallel_encoder(feature)
    # basis_pred, coeff_pred =  parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = True)

    # coeff_sum = coeff_pred.cpu().numpy()

    # coeff_sum_diff = coeff_pred[:,:,0] - coeff_pred[:,:,1]
    # coeff_sum_diff_pos = coeff_sum_diff.clamp(min = 0)
    # coeff_sum_diff_cpu = coeff_sum_diff.cpu().numpy()
    # coeff_order = np.argsort(coeff_sum_diff_cpu, axis = 1)
    # coeff_order = np.flip( coeff_order, axis = 1 )

    # basis_pred = basis_pred.permute(0,2,1)
    # basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 1, keepdim=True) )
    basis_norm_pred = basis_norm_pred.permute(0, 2, 1)
    # basis_norm_pred should have dimension (n_batch, emb_size, n_basis)
    # word_norm_emb should have dimension (ntokens, emb_size)
    sim_pairwise = torch.matmul(word_norm_emb.unsqueeze(dim=0), basis_norm_pred)
    # print(sim_pairwise.size())
    # sim_pairwise should have dimension (n_batch, ntokens, emb_size)
    top_value, top_index = torch.topk(sim_pairwise, top_k, dim=1, sorted=True)

    bsz, max_sent_len, emb_size = output_emb.size()
    avg_out_emb = torch.empty(bsz, emb_size)
    for i in range(bsz):
        # print(feature[i,:])
        sent_len = (feature[i, :] != 0).sum()
        avg_out_emb[i, :] = output_emb[i, -sent_len:, :].mean(dim=0)

    # return basis_norm_pred, coeff_order, coeff_sum, top_value, top_index, output_emb_last, avg_out_emb
    return basis_norm_pred, top_value, top_index, output_emb_last, avg_out_emb


def convert_feature_to_text(feature, idx2word_freq):
    if type(feature) is list:
        feature_list = feature
    else:
        feature_list = feature.tolist()
    feature_text = []
    for i in range(len(feature_list)):
        current_sent = []
        for w_ind in feature_list[i]:
            if w_ind != 0:
                if type(idx2word_freq) is dict and w_ind not in idx2word_freq:
                    current_sent.append('<UNK>')
                else:
                    w = idx2word_freq[w_ind][0]
                    current_sent.append(w)
        feature_text.append(current_sent)
    return feature_text


def print_basis_text(feature, idx2word_freq, tag_idx2word_freq, top_value, top_index, i_batch, outf, word_d2_vis,
                     word_imp_sim=None):
    # n_basis = coeff_order.shape[1]
    n_basis = top_index.shape[2]
    top_k = top_index.size(1)
    feature_text = convert_feature_to_text(feature, idx2word_freq)
    for i_sent in range(len(feature_text)):
        # outf.write('{} batch, {}th sent: '.format(i_batch, i_sent)+' '.join(feature_text[i_sent])+'\n')
        if word_imp_sim is not None:
            # outf.write(' '.join('{0:.2f}'.format(x/4.0) for x in word_imp_sim[i_sent]])+'\n')
            outf.write(' '.join(['\\colorbox{{c{0:02}}}{{{1}}}'.format(int(100 - y * 100 / 4.0), x) for x, y in
                                 zip(feature_text[i_sent], word_imp_sim[i_sent])]) + '\n')
        else:
            outf.write(' '.join(feature_text[i_sent]) + '\n\n')

        for j in range(n_basis):
            # org_ind = coeff_order[i_sent, j]
            org_ind = j
            # outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')

            for k in range(top_k):
                word_nn = tag_idx2word_freq[top_index[i_sent, k, org_ind].item()][0]
                if len(word_d2_vis) > 0 and word_nn in word_d2_vis:
                    outf.write(word_nn + ',' + word_d2_vis[word_nn] + ' {:5.3f}'.format(
                        top_value[i_sent, k, org_ind].item()) + ' ')
                else:
                    outf.write(word_nn + ' {:5.3f}'.format(top_value[i_sent, k, org_ind].item()) + ' ')
            outf.write('\n')
        outf.write('\n\n')


def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq,
                         tag_idx2word_freq, outf, max_batch_num, word_d2_vis={}):
    # topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, feature_type, user, tag, repeat_num, user_len, tag_len, sample_idx = sample_batched

            basis_norm_pred, top_value, top_index, encoded_emb, avg_encoded_emb = predict_batch(feature, feature_type,
                                                                                                parallel_encoder,
                                                                                                parallel_decoder,
                                                                                                word_norm_emb, top_k)
            # print_basis_text(feature, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf, word_imp_sim)
            print_basis_text(feature, idx2word_freq, tag_idx2word_freq, top_value, top_index, i_batch, outf,
                             word_d2_vis)

            if i_batch >= max_batch_num:
                break


num_special_token = 3


def compute_all_dist(feature, feature_type, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, coeff_opt,
                     loss_type, test_user, test_tag, switch_user_tag_roles, device):
    def compute_dist(user_norm_emb, basis_norm_pred, coeff_opt, loss_type, device):
        lr_coeff = 0.05
        iter_coeff = 60
        coeff_opt_algo = "rmsprop"
        L1_losss_B = 0.2
        bsz = basis_norm_pred.size(0)
        vocab_size, emb_size = user_norm_emb.size()
        target_embeddings = user_norm_emb.expand((bsz, vocab_size, emb_size))
        with torch.enable_grad():
            if coeff_opt == 'prod':
                coeff_mat_user = estimate_prod_coeff_mat_batch_opt(target_embeddings.detach(),
                                                                            basis_norm_pred.detach(), L1_losss_B,
                                                                            device, coeff_opt_algo, lr_coeff,
                                                                            iter_coeff)
            elif coeff_opt == 'lc':
                coeff_mat_user = estimate_coeff_mat_batch_opt(target_embeddings.detach(),
                                                                       basis_norm_pred.detach(), L1_losss_B, device,
                                                                       coeff_opt_algo, lr_coeff, iter_coeff)
            elif coeff_opt == 'max':
                coeff_mat_user = estimate_coeff_mat_batch_max(target_embeddings.detach(),
                                                                       basis_norm_pred.detach(), device, loss_type)

        pred_embeddings_user = torch.bmm(coeff_mat_user, basis_norm_pred)
        if loss_type == 'sim' or loss_type == 'sim_norm':
            dist_all_user_tensor = - torch.sum(pred_embeddings_user * target_embeddings, dim=2)
        elif loss_type == 'dist':
            dist_all_user_tensor = torch.pow(torch.norm(pred_embeddings_user - target_embeddings, dim=2), 2)
        # To exclude <null>, <eos>, <unk>, just give it a super high distance
        dist_all_user_tensor[:, :num_special_token] = 100
        return dist_all_user_tensor.tolist()

    normalize_emb = True
    if loss_type != 'dist' and loss_type != 'sim_norm':
        normalize_emb = False

    basis_norm_pred, basis_norm_pred_tag, basis_norm_pred_auto, output_emb_last, output_emb = predict_batch_simple(
        feature, feature_type, parallel_encoder, parallel_decoder, normalize_emb)
    if test_user:
        if switch_user_tag_roles:
            input_basis = basis_norm_pred_tag
        else:
            input_basis = basis_norm_pred
        all_dist_user = compute_dist(user_norm_emb, input_basis, coeff_opt, loss_type, device)
    else:
        all_dist_user = []

    if test_tag:
        if switch_user_tag_roles:
            input_basis = basis_norm_pred
        else:
            input_basis = basis_norm_pred_tag

        all_dist_tag = compute_dist(tag_norm_emb, input_basis, coeff_opt, loss_type, device)
    else:
        all_dist_tag = []

    return all_dist_user, all_dist_tag


def print_rank(feature_text_j, user_text_j, tag_text_j, gt_rank_user_j, gt_rank_tag_j, top_user_text_j, top_tag_text_j,
               top_values_user_j, top_values_tag_j, paper_id, outf):
    outf.write(str(paper_id) + ', ' + ' '.join(feature_text_j) + '\n')

    if len(user_text_j) > 0:
        for k in range(len(user_text_j)):
            outf.write(user_text_j[k] + ':r' + str(gt_rank_user_j[k]) + ' ')
        outf.write('\n')

        for k in range(len(tag_text_j)):
            outf.write(tag_text_j[k] + ':r' + str(gt_rank_tag_j[k]) + ' ')
        outf.write('\n\n')

    if len(tag_text_j) > 0:
        for k in range(len(top_user_text_j)):
            outf.write(top_user_text_j[k] + ':v' + str(top_values_user_j[k]) + ' ')
        outf.write('\n')

        for k in range(len(top_tag_text_j)):
            outf.write(top_tag_text_j[k] + ':v' + str(top_values_tag_j[k]) + ' ')
        outf.write('\n\n\n')


def update_user_dict(user_batch_list, user_d2_paper_id, paper_id):
    for k in range(len(user_batch_list)):
        user_id = user_batch_list[k]
        if user_id < num_special_token:
            # break
            continue
        paper_id_list = user_d2_paper_id.get(user_id, [])
        paper_id_list.append(paper_id)
        user_d2_paper_id[user_id] = paper_id_list


def update_user_score_dict(user_batch_list, bid_score_batch_list, user_d2_paper_id_score, paper_id):
    if len(bid_score_batch_list) == 0:
        return
    for k in range(len(user_batch_list)):
        user_id = user_batch_list[k]
        if user_id < num_special_token:
            continue
        paper_id_score_list = user_d2_paper_id_score.get(user_id, [])
        paper_id_score_list.append([paper_id, bid_score_batch_list[k]])
        user_d2_paper_id_score[user_id] = paper_id_score_list


def gt_rank_from_list(all_dist_user_j, user_batch_list_j, num_special_token):
    user_rank = ss.rankdata(all_dist_user_j)  # handle the tie cases by average the ranks
    gt_rank_j = []
    for user_id in user_batch_list_j:
        if user_id < num_special_token:
            # break
            continue
        else:
            gt_rank_j.append(user_rank[user_id])
    return gt_rank_j


def compute_AUC_ROC(gt_rank_j_sorted, total_size):
    AP_list = []
    correct_num = len(gt_rank_j_sorted)
    for i, x in enumerate(gt_rank_j_sorted):
        left_correct = i
        left_incorrect = x - left_correct
        right_correct = correct_num - i - 1
        right_incorrect = total_size - x - right_correct
        if left_incorrect + right_incorrect > 0:
            AP_list.append(right_incorrect / float(left_incorrect + right_incorrect))
    # print(AP_list)
    return np.mean(AP_list)


def compute_AP_best_F1(gt_rank_j_sorted):
    AP_list = []
    best_F1 = 0
    for i, x in enumerate(gt_rank_j_sorted):
        precision = (i + 1) / float(x)
        AP_list.append(precision)
        recall = (i + 1) / float(len(gt_rank_j_sorted))
        F1 = 2 * precision * recall / (precision + recall)
        if F1 > best_F1:
            best_F1 = F1

    return np.mean(AP_list), best_F1


def pred_per_paper(all_dist_user, user_batch_list, recall_at_th, recall_all_user, weight_all_user, batch_dist=True):
    bsz = len(user_batch_list)
    gt_rank_user = []
    top_prediction_user = []
    top_values_user = []
    # recall_list_user = []
    MAP_list_user = []
    F1_list_user = []
    AUC_list_user = []
    NDCG_list_user = []
    for j in range(bsz):
        # pred_user_rank = np.argsort(all_dist_user[j])
        # gt_rank_j = [user_rank[x] for x in user_batch_list[j]]
        if batch_dist:
            dist_j = all_dist_user[j]
        else:
            dist_j = all_dist_user
        gt_rank_j = gt_rank_from_list(dist_j, user_batch_list[j], num_special_token=num_special_token)
        top_pred_j = np.argsort(dist_j)[:recall_at_th[0]]
        top_prediction_user.append(top_pred_j)
        top_values_user.append([dist_j[x] for x in top_pred_j])
        gt_rank_user.append(gt_rank_j)
        if len(gt_rank_j) > 0:
            gt_rel = np.zeros(len(dist_j))
            gt_rel[user_batch_list[j]] = 1
            pred_score = -np.array(dist_j).reshape((1, len(dist_j)))
            ndcg = ndcg_score(gt_rel.reshape(1, len(dist_j)), pred_score)
            for m in range(len(recall_at_th)):
                weight = min(len(gt_rank_j), recall_at_th[m])
                recall_all_user[m].append(np.sum([int(x <= recall_at_th[m]) for x in gt_rank_j]) / weight)
                weight_all_user[m].append(weight)
            gt_rank_j_sorted = sorted(gt_rank_j)
            AP, best_F1 = compute_AP_best_F1(gt_rank_j_sorted)
            F1_list_user.append(best_F1)
            if len(gt_rank_j_sorted) < len(dist_j):
                AUC = compute_AUC_ROC(gt_rank_j_sorted, len(dist_j))
                AUC_list_user.append(AUC)
            NDCG_list_user.append(ndcg)
            MAP_list_user.append(AP)

    return gt_rank_user, top_prediction_user, top_values_user, MAP_list_user, AUC_list_user, NDCG_list_user, F1_list_user


def div_by_tags(paper_user_dist, paper_id_d2_tags, recall_at_th):
    # user_paper_dist = list(zip(*paper_user_dist))
    div_list = [[] for m in range(len(recall_at_th))]
    # for user_id in range(len(user_paper_dist)):
    for user_id in range(paper_user_dist.shape[1]):
        paper_dist = paper_user_dist[:, user_id]
        paper_dist_sorted = np.argsort(paper_dist)
        for m, recall_at_th_m in enumerate(recall_at_th):
            top_pred_j = paper_dist_sorted[:recall_at_th_m]
            tag_list_all = []
            for k in range(recall_at_th_m):
                paper_id = top_pred_j[k]
                tag_list = paper_id_d2_tags[paper_id]
                tag_list_all += tag_list
            tag_num = len(tag_list_all)
            tag_counter = Counter(tag_list_all)
            uniq_tag_num = len(tag_counter)
            div = uniq_tag_num / float(tag_num)
            div_list[m].append(div)
    return [np.mean(div_list[m]) for m in range(len(div_list))]


def inbalance_by_top_choice(paper_user_dist, reviewer_num_per_paper):
    user_num = paper_user_dist.shape[1]
    paper_num = paper_user_dist.shape[0]
    paper_num_per_reviewer = math.ceil(reviewer_num_per_paper * paper_num / float(user_num))
    avg_paper_count = paper_num_per_reviewer * user_num / float(paper_num)
    paper_count_top_choice = np.zeros(paper_num)
    for user_id in range(user_num):
        paper_dist = paper_user_dist[:, user_id]
        paper_dist_sorted = np.argsort(paper_dist)
        for top_pred in paper_dist_sorted[:paper_num_per_reviewer]:
            paper_count_top_choice[top_pred] += 1
    var_avg = np.mean(np.abs(paper_count_top_choice - avg_paper_count))

    return var_avg, paper_num_per_reviewer


def div_by_categories(paper_user_dist, paper_id_d2_categories, recall_at_th):
    def compute_div(cat_list, div_list, ent_list):
        have_cat_num = len(cat_list)
        cat_counter = Counter(cat_list)
        # mode_freq_cat = max(cat_counter)
        mode_freq_cat = max(cat_counter, key=cat_counter.get)
        div = 1 - cat_counter[mode_freq_cat] / float(have_cat_num)
        div_list.append(div)
        ent = ss.entropy(list(cat_counter.values()))
        ent_list.append(ent)

    # user_paper_dist = list(zip(*paper_user_dist))
    div_list = [[] for m in range(len(recall_at_th))]
    div_course_list = [[] for m in range(len(recall_at_th))]
    ent_list = [[] for m in range(len(recall_at_th))]
    ent_course_list = [[] for m in range(len(recall_at_th))]
    for user_id in range(paper_user_dist.shape[1]):
        # paper_dist = user_paper_dist[user_id]
        paper_dist = paper_user_dist[:, user_id]
        paper_dist_sorted = np.argsort(paper_dist)
        for m, recall_at_th_m in enumerate(recall_at_th):
            top_pred_j = paper_dist_sorted[:recall_at_th_m]
            cat_list = []
            cat_course_list = []
            for k in range(recall_at_th_m):
                paper_id = top_pred_j[k]
                categories = paper_id_d2_categories[paper_id][0]
                categories_course = paper_id_d2_categories[paper_id][1]
                if len(categories) > 0:
                    cat_list.append(categories)
                    cat_course_list.append(categories_course)
            if len(cat_list) > 0:
                compute_div(cat_list, div_list[m], ent_list[m])
                compute_div(cat_course_list, div_course_list[m], ent_course_list[m])
    div_avg = [np.mean(div_list[m]) for m in range(len(div_list))]
    div_course_avg = [np.mean(div_course_list[m]) for m in range(len(div_course_list))]
    ent_avg = [np.mean(ent_list[m]) for m in range(len(ent_list))]
    ent_course_avg = [np.mean(ent_course_list[m]) for m in range(len(ent_course_list))]
    return div_avg, div_course_avg, ent_avg, ent_course_avg


def bid_score_rank_eval(user_d2_paper_id_score, paper_user_dist):
    spearman_list = []
    for user_id in user_d2_paper_id_score:
        paper_dist = paper_user_dist[:, user_id]
        paper_id_list, score_list = list(zip(*user_d2_paper_id_score[user_id]))
        # print(paper_dist)
        # print(paper_id_list)
        # print(score_list)
        if np.var(score_list) > 0:
            paper_sim = -paper_dist[list(paper_id_list)]
            if np.var(paper_sim) > 0:
                spearman, sig = ss.spearmanr(score_list, -paper_dist[list(paper_id_list)])
                spearman_list.append(spearman)
            else:
                spearman_list.append(0)
    spearman_score_avg = np.mean(spearman_list)
    return spearman_score_avg


def paper_recall_per_user(user_d2_paper_id, paper_user_dist, recall_at_th, dist_matrix=True):
    # if dist_matrix:
    #    user_paper_dist = list(zip(*paper_user_dist))
    recall_list_paper = [[] for k in range(len(recall_at_th))]
    weight_list_paper = [[] for k in range(len(recall_at_th))]
    MAP_list_paper = []
    F1_list_paper = []
    AUC_list_paper = []
    NDCG_list_paper = []
    for user_id in user_d2_paper_id:
        if dist_matrix:
            # paper_dist = user_paper_dist[user_id]
            paper_dist = paper_user_dist[:, user_id]
        else:
            paper_dist = paper_user_dist
        paper_id_list = user_d2_paper_id[user_id]
        gt_rel = np.zeros(len(paper_dist))
        gt_rel[paper_id_list] = 1
        pred_score = -np.array(paper_dist).reshape(1, len(paper_dist))
        ndcg = ndcg_score(gt_rel.reshape(1, len(paper_dist)), pred_score)
        NDCG_list_paper.append(ndcg)
        gt_rank_paper = gt_rank_from_list(paper_dist, paper_id_list, num_special_token=-1)
        for k in range(len(recall_at_th)):
            weight = min(len(gt_rank_paper), recall_at_th[k])
            recall_list_paper[k].append(np.sum([int(x <= recall_at_th[k]) for x in gt_rank_paper]) / weight)
            weight_list_paper[k].append(weight)
        gt_rank_paper_sorted = sorted(gt_rank_paper)
        AP, best_F1 = compute_AP_best_F1(gt_rank_paper_sorted)
        F1_list_paper.append(best_F1)
        if len(gt_rank_paper_sorted) < len(paper_dist):
            AUC = compute_AUC_ROC(gt_rank_paper_sorted, len(paper_dist))
            AUC_list_paper.append(AUC)
        MAP_list_paper.append(AP)

    # print(recall_list_paper)
    recall_avg = [np.average(recall_list_paper[k]) for k in range(len(recall_at_th))]
    recall_avg_w = [np.average(recall_list_paper[k], weights=weight_list_paper[k]) for k in range(len(recall_at_th))]
    F1 = np.mean(F1_list_paper)
    MAP = np.mean(MAP_list_paper)
    AUC = np.mean(AUC_list_paper)
    NDCG = np.mean(NDCG_list_paper)
    return recall_avg, recall_avg_w, MAP, AUC, NDCG, F1


def extract_category(feature, feature_type, paper_id_list, paper_id_d2_categories):
    bsz = feature.size(0)
    for i in range(bsz):
        paper_id = paper_id_list[i]
        feature_category = feature[i, :][feature_type[i, :] == 3].tolist()
        if feature_type[i, -1] == 3 or len(feature_category) <= 3:
            paper_id_d2_categories[paper_id] = ['', '']
            continue
        f_sep_id = feature_category[-2]
        # up_one_layer = feature_category[:-2].rfind(f_sep_id)
        try:
            up_one_layer = len(feature_category[:-2]) - 1 - feature_category[:-2][::-1].index(f_sep_id)
        except:
            up_one_layer = 0
        # if up_one_layer == -1:
        #    up_one_layer = 0
        paper_id_d2_categories[paper_id] = [" ".join(map(str, feature_category)),
                                            " ".join(map(str, feature_category[:up_one_layer]))]


def extract_tag(paper_id_list, tag_batch_list, paper_id_d2_tags):
    for i, paper_id in enumerate(paper_id_list):
        tag_list = tag_batch_list[i]
        paper_id_d2_tags[paper_id] = tag_list


def all_dist_from_recommend(dataloader_info, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, coeff_opt,
                            loss_type, user_idx2word_freq, tag_idx2word_freq, test_user, test_tag,
                            switch_user_tag_roles, device):
    with torch.no_grad():
        dataloader, all_user_tag = dataloader_info
        paper_user_dist = []
        paper_tag_dist = []
        if test_user:
            paper_user_dist = np.empty((len(all_user_tag), len(user_idx2word_freq)), dtype=np.float16)
        if test_tag:
            paper_tag_dist = np.empty((len(all_user_tag), len(tag_idx2word_freq)), dtype=np.float16)

        for i_batch, sample_batched in enumerate(dataloader):
            # feature, user, tag = sample_batched
            sys.stdout.write(str(i_batch) + ' ')
            sys.stdout.flush()

            feature, feature_type, paper_id_tensor = sample_batched
            paper_id_list = paper_id_tensor.tolist()

            # feature_text = convert_feature_to_text(feature, idx2word_freq)

            all_dist_user, all_dist_tag = compute_all_dist(feature, feature_type, parallel_encoder, parallel_decoder,
                                                           user_norm_emb, tag_norm_emb, coeff_opt, loss_type, test_user,
                                                           test_tag, switch_user_tag_roles, device)

            bsz = feature.size(0)
            for j in range(bsz):
                paper_id = paper_id_list[j]
                if test_user:
                    paper_user_dist[paper_id, :] = all_dist_user[j]
                if test_tag:
                    paper_tag_dist[paper_id, :] = all_dist_tag[j]
    return paper_user_dist, paper_tag_dist


def recommend_test_from_all_dist(dataloader_info, paper_user_dist, paper_tag_dist, idx2word_freq, user_idx2word_freq,
                                 tag_idx2word_freq, test_user, test_tag, outf, device, most_popular_baseline=True,
                                 div_eval='openreview', figure_name=''):
    with torch.no_grad():
        dataloader, all_user_tag = dataloader_info
        user_d2_paper_id = {}
        user_d2_paper_id_score = {}
        tag_d2_paper_id = {}
        recall_at_th = [5, 20, 50, 200, 1000]
        recall_at_th_str = ' '.join(map(str, recall_at_th))
        # recall_all_user = []
        # recall_all_tag = []
        recall_all_user = [[] for k in range(len(recall_at_th))]
        recall_all_tag = [[] for k in range(len(recall_at_th))]
        weight_all_user = [[] for k in range(len(recall_at_th))]
        weight_all_tag = [[] for k in range(len(recall_at_th))]
        F1_all_user = []
        F1_all_tag = []
        MAP_all_user = []
        MAP_all_tag = []
        AUC_all_user = []
        AUC_all_tag = []
        NDCG_all_user = []
        NDCG_all_tag = []

        paper_id_d2_categories = {}
        paper_id_d2_tags = {}
        paper_id_l2_neg_user_freq = [-1] * len(all_user_tag)
        paper_id_l2_pred_user_dist = [-1] * len(all_user_tag)
        paper_id_l2_neg_tag_freq = [-1] * len(all_user_tag)
        paper_id_l2_pred_tag_dist = [-1] * len(all_user_tag)
        if most_popular_baseline:
            if test_user:
                user_neg_freq = [-x[1] for x in user_idx2word_freq]
                # p_recall_all_user = []
                p_recall_all_user = [[] for k in range(len(recall_at_th))]
                p_weight_all_user = [[] for k in range(len(recall_at_th))]
                p_F1_all_user = []
                p_MAP_all_user = []
                p_AUC_all_user = []
                p_NDCG_all_user = []
            if test_tag:
                tag_neg_freq = [-x[1] for x in tag_idx2word_freq]
                # p_recall_all_tag = []
                p_recall_all_tag = [[] for k in range(len(recall_at_th))]
                p_weight_all_tag = [[] for k in range(len(recall_at_th))]
                p_F1_all_tag = []
                p_MAP_all_tag = []
                p_AUC_all_tag = []
                p_NDCG_all_tag = []
        for i_batch, sample_batched in enumerate(dataloader):
            # feature, user, tag = sample_batched
            sys.stdout.write(str(i_batch) + ' ')
            sys.stdout.flush()

            feature, feature_type, paper_id_tensor = sample_batched
            paper_id_list = paper_id_tensor.tolist()

            feature_text = convert_feature_to_text(feature, idx2word_freq)

            bsz = feature.size(0)
            user_batch_list = [all_user_tag[paper_id][0] for paper_id in paper_id_list]
            tag_batch_list = [all_user_tag[paper_id][1] for paper_id in paper_id_list]
            bid_score_batch_list = [all_user_tag[paper_id][2] for paper_id in paper_id_list]

            if div_eval == 'amazon':
                extract_category(feature, feature_type, paper_id_list, paper_id_d2_categories)
            elif div_eval == 'citeulike':
                extract_tag(paper_id_list, tag_batch_list, paper_id_d2_tags)

            all_dist_user = []
            all_dist_tag = []
            for j in range(bsz):
                paper_id = paper_id_list[j]
                if test_user:
                    all_dist_user.append(paper_user_dist[paper_id, :])
                if test_tag:
                    all_dist_tag.append(paper_tag_dist[paper_id, :])
            # all_dist_user, all_dist_tag = compute_all_dist(feature, feature_type, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, coeff_opt, loss_type, test_user, test_tag, device)
            # user_batch_list = user.tolist()
            # tag_batch_list = tag.tolist()
            if test_user:
                gt_rank_user, top_prediction_user, top_values_user, MAP_list_user, AUC_list_user, NDCG_list_user, F1_list_user = pred_per_paper(
                    all_dist_user, user_batch_list, recall_at_th, recall_all_user, weight_all_user)
                # recall_all_user += recall_list_user
                F1_all_user += F1_list_user
                MAP_all_user += MAP_list_user
                AUC_all_user += AUC_list_user
                NDCG_all_user += NDCG_list_user
                user_text = convert_feature_to_text(user_batch_list, user_idx2word_freq)
                top_user_text = convert_feature_to_text(top_prediction_user, user_idx2word_freq)
                if most_popular_baseline:
                    p_gt_rank_user, p_top_prediction_user, p_top_values_user, p_MAP_list_user, p_AUC_list_user, p_NDCG_list_user, p_F1_list_user = pred_per_paper(
                        user_neg_freq, user_batch_list, recall_at_th, p_recall_all_user, p_weight_all_user,
                        batch_dist=False)
                    # p_recall_all_user += p_recall_list_user
                    p_F1_all_user += p_F1_list_user
                    p_MAP_all_user += p_MAP_list_user
                    p_AUC_all_user += p_AUC_list_user
                    p_NDCG_all_user += p_NDCG_list_user

            if test_tag:
                gt_rank_tag, top_prediction_tag, top_values_tag, MAP_list_tag, AUC_list_tag, NDCG_list_tag, F1_list_tag = pred_per_paper(
                    all_dist_tag, tag_batch_list, recall_at_th, recall_all_tag, weight_all_tag)
                # recall_all_tag += recall_list_tag
                F1_all_tag += F1_list_tag
                MAP_all_tag += MAP_list_tag
                AUC_all_tag += AUC_list_tag
                NDCG_all_tag += NDCG_list_tag
                tag_text = convert_feature_to_text(tag_batch_list, tag_idx2word_freq)
                top_tag_text = convert_feature_to_text(top_prediction_tag, tag_idx2word_freq)
                if most_popular_baseline:
                    p_gt_rank_tag, p_top_prediction_tag, p_top_values_tag, p_MAP_list_tag, p_AUC_list_tag, p_NDCG_list_tag, p_F1_list_tag = pred_per_paper(
                        tag_neg_freq, tag_batch_list, recall_at_th, p_recall_all_tag, p_weight_all_tag,
                        batch_dist=False)
                    # p_recall_all_tag += p_recall_list_tag
                    p_F1_all_tag += p_F1_list_tag
                    p_MAP_all_tag += p_MAP_list_tag
                    p_AUC_all_tag += p_AUC_list_tag
                    p_NDCG_all_tag += p_NDCG_list_tag

            # user_max = user.size(1)
            # tag_max = tag.size(1)

            for j in range(bsz):
                # paper_id = len(paper_user_dist)
                paper_id = paper_id_list[j]
                user_text_j = []
                tag_text_j = []
                gt_rank_user_j = []
                gt_rank_tag_j = []
                top_user_text_j = []
                top_tag_text_j = []
                top_values_user_j = []
                top_values_tag_j = []

                if test_user:
                    update_user_dict(user_batch_list[j], user_d2_paper_id, paper_id)
                    update_user_score_dict(user_batch_list[j], bid_score_batch_list[j], user_d2_paper_id_score,
                                           paper_id)
                    # paper_user_dist.append(all_dist_user[j])
                    # paper_user_dist[paper_id,:] = all_dist_user[j]
                    user_text_j = user_text[j]
                    gt_rank_user_j = gt_rank_user[j]
                    top_user_text_j = top_user_text[j]
                    top_values_user_j = top_values_user[j]
                    # if most_popular_baseline:
                    paper_id_l2_neg_user_freq[paper_id] = - len(user_batch_list[j])
                    paper_id_l2_pred_user_dist[paper_id] = sum(all_dist_user[j])

                if test_tag:
                    update_user_dict(tag_batch_list[j], tag_d2_paper_id, paper_id)
                    # paper_tag_dist.append(all_dist_tag[j])
                    # paper_tag_dist[paper_id,:] = all_dist_tag[j]
                    tag_text_j = tag_text[j]
                    gt_rank_tag_j = gt_rank_tag[j]
                    top_tag_text_j = top_tag_text[j]
                    top_values_tag_j = top_values_tag[j]
                    paper_id_l2_neg_tag_freq[paper_id] = - len(tag_batch_list[j])
                    paper_id_l2_pred_tag_dist[paper_id] = sum(all_dist_tag[j])

                print_rank(feature_text[j], user_text_j, tag_text_j, gt_rank_user_j, gt_rank_tag_j, top_user_text_j,
                           top_tag_text_j, top_values_user_j, top_values_tag_j, paper_id, outf)
            # if i_batch > 3:
            #    break
            # print_basis_text(feature, idx2word_freq, tag_idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf)
        div_th = recall_at_th
        div_th_str = recall_at_th_str
        if test_user:
            print(
                "\nUser recall per paper at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}".format(
                    recall_at_th_str, [np.mean(recall_all_user[m]) for m in range(len(recall_all_user))],
                    [np.average(recall_all_user[m], weights=weight_all_user[m]) for m in range(len(recall_all_user))],
                    np.mean(MAP_all_user), np.mean(F1_all_user), np.mean(AUC_all_user), np.mean(NDCG_all_user)))
            user_id_l2_pred_paper_dist = np.sum(paper_user_dist[:, num_special_token:], axis=0)
            user_id_l2_pred_paper_dist_mean = np.mean(user_id_l2_pred_paper_dist)
            print("Average absolute deviation on predicted distance {} per paper".format(
                np.mean(np.absolute(user_id_l2_pred_paper_dist - user_id_l2_pred_paper_dist_mean)) / np.absolute(
                    user_id_l2_pred_paper_dist_mean)))
            recall_avg_user, recall_w_avg_user, MAP_user, AUC_user, NDCG_user, F1_user = paper_recall_per_user(
                user_d2_paper_id, paper_user_dist, recall_at_th)
            print(
                "Paper recall per user at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}, popularity correlation is {}".format(
                    recall_at_th_str, recall_avg_user, recall_w_avg_user, MAP_user, F1_user, AUC_user, NDCG_user,
                    ss.pearsonr(paper_id_l2_neg_user_freq, paper_id_l2_pred_user_dist)[0]))
            if len(user_d2_paper_id_score) > 0:
                spearman_score_avg = bid_score_rank_eval(user_d2_paper_id_score, paper_user_dist)
                print("Bid score Spearman correlation coefficient {}".format(spearman_score_avg))
            paper_id_l2_pred_user_dist_mean = np.mean(paper_id_l2_pred_user_dist)
            print("Average absolute deviation on predicted distance {} per user".format(
                np.mean(np.absolute(paper_id_l2_pred_user_dist - paper_id_l2_pred_user_dist_mean)) / np.absolute(
                    paper_id_l2_pred_user_dist_mean)))
            if len(figure_name) > 0:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.bar(range(len(paper_id_l2_pred_user_dist)), np.sort(paper_id_l2_pred_user_dist))
                plt.savefig(figure_name + '_user_dist_sum_per_paper.png')
                plt.figure()
                plt.bar(range(user_id_l2_pred_paper_dist.shape[0]), np.sort(user_id_l2_pred_paper_dist))
                plt.savefig(figure_name + '_paper_dist_sum_per_user.png')

            if div_eval == 'openreview':
                reviewer_num_per_paper = 4
                var_avg, paper_num_per_reviewer = inbalance_by_top_choice(paper_user_dist, reviewer_num_per_paper)
                print(
                    "When reviewer top {} papers, each paper has at least {} reviewers, average L1 dist from paper count to average paper count is {}".format(
                        paper_num_per_reviewer, reviewer_num_per_paper, var_avg))

            if len(paper_id_d2_categories) > 0:
                div_avg, div_course_avg, ent_avg, ent_course_avg = div_by_categories(paper_user_dist,
                                                                                     paper_id_d2_categories, div_th)
                print(
                    "Diversification metric of user by category at {} is {}. Entropy is {}. Div of course layer is {}. Entorpy is {}".format(
                        div_th_str, div_avg, ent_avg, div_course_avg, ent_course_avg))
            if len(paper_id_d2_tags) > 0:
                div_avg = div_by_tags(paper_user_dist, paper_id_d2_tags, div_th)
                print("Diversification metric of user by tag difference at {} is {}".format(div_th_str, div_avg))
            if most_popular_baseline:
                print(
                    "Popularity GT baseline. User recall per paper at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}".format(
                        recall_at_th_str, [np.mean(p_recall_all_user[m]) for m in range(len(p_recall_all_user))],
                        [np.average(p_recall_all_user[m], weights=p_weight_all_user[m]) for m in
                         range(len(p_recall_all_user))], np.mean(p_MAP_all_user), np.mean(p_F1_all_user),
                        np.mean(p_AUC_all_user), np.mean(p_NDCG_all_user)))
                p_recall_avg_user, p_weight_recall_avg_user, p_MAP_user, p_AUC_user, p_NDCG_user, p_F1_user = paper_recall_per_user(
                    user_d2_paper_id, paper_id_l2_neg_user_freq, recall_at_th, dist_matrix=False)
                print(
                    "Popularity GT baseline. Paper recall per user at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}".format(
                        recall_at_th_str, p_recall_avg_user, p_weight_recall_avg_user, p_MAP_user, p_F1_user,
                        p_AUC_user, p_NDCG_user))

        if test_tag:
            print("\nTag recall per paper at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}".format(
                recall_at_th_str, [np.mean(recall_all_tag[m]) for m in range(len(recall_all_tag))],
                [np.average(recall_all_tag[m], weights=weight_all_tag[m]) for m in range(len(recall_all_tag))],
                np.mean(MAP_all_tag), np.mean(F1_all_tag), np.mean(AUC_all_tag), np.mean(NDCG_all_tag)))
            recall_avg_tag, recall_w_avg_tag, MAP_tag, AUC_tag, NDCG_tag, F1_tag = paper_recall_per_user(
                tag_d2_paper_id, paper_tag_dist, recall_at_th)
            print(
                "Paper recall per tag at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}, popularity correlation is {}".format(
                    recall_at_th_str, recall_avg_tag, recall_w_avg_tag, MAP_tag, F1_tag, AUC_tag, NDCG_tag,
                    ss.pearsonr(paper_id_l2_neg_tag_freq, paper_id_l2_pred_tag_dist)[0]))
            if len(paper_id_d2_categories) > 0:
                div_avg, div_course_avg, ent_avg, ent_course_avg = div_by_categories(paper_tag_dist,
                                                                                     paper_id_d2_categories, div_th)
                print(
                    "Diversification metric of tag by category at {} is {}. Entropy is {}. Div of course layer is {}. Entorpy is {}".format(
                        div_th_str, div_avg, ent_avg, div_course_avg, ent_course_avg))
            if most_popular_baseline:
                print(
                    "Popularity baseline. Tag recall per paper at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}".format(
                        recall_at_th_str, [np.mean(p_recall_all_user[m]) for m in range(len(p_recall_all_tag))],
                        [np.average(p_recall_all_tag[m], weights=p_weight_all_tag[m]) for m in
                         range(len(p_recall_all_tag))], np.mean(p_MAP_all_tag), np.mean(p_F1_all_tag),
                        np.mean(p_AUC_all_tag), np.mean(p_NDCG_all_tag)))
                p_recall_avg_tag, p_recall_w_avg_tag, p_MAP_tag, p_AUC_tag, p_NDCG_tag, p_F1_tag = paper_recall_per_user(
                    tag_d2_paper_id, paper_id_l2_neg_tag_freq, recall_at_th, dist_matrix=False)
                print(
                    "Popularity GT baseline. Paper recall per tag at {} is {}, weighted recall is {}, MAP is {}, F1 is {}, AUC is {}, NDCG is {}".format(
                        recall_at_th_str, p_recall_avg_tag, p_recall_w_avg_tag, p_MAP_tag, p_F1_tag, p_AUC_tag,
                        p_NDCG_tag))


def recommend_test(dataloader_info, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, idx2word_freq,
                   user_idx2word_freq, tag_idx2word_freq, coeff_opt, loss_type, test_user, test_tag, outf, device,
                   most_popular_baseline=True, div_eval='openreview', switch_user_tag_roles=False, store_dist='',
                   figure_name=''):
    paper_user_dist, paper_tag_dist = all_dist_from_recommend(dataloader_info, parallel_encoder, parallel_decoder,
                                                              user_norm_emb, tag_norm_emb, coeff_opt, loss_type,
                                                              user_idx2word_freq, tag_idx2word_freq, test_user,
                                                              test_tag, switch_user_tag_roles, device)
    if store_dist == 'user':
        np.savetxt(outf, paper_user_dist)
    elif store_dist == 'tag':
        np.savetxt(outf, paper_tag_dist)
    else:
        recommend_test_from_all_dist(dataloader_info, paper_user_dist, paper_tag_dist, idx2word_freq,
                                     user_idx2word_freq, tag_idx2word_freq, test_user, test_tag, outf, device,
                                     most_popular_baseline, div_eval, figure_name)


class Set2SetDataset(torch.utils.data.Dataset):
    # def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent):
    def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb,
                 target_avg_word_emb):
        self.source = source
        self.source_w = source_w
        self.source_sent_emb = source_sent_emb


class Set2SetDataset(torch.utils.data.Dataset):
    # def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent):
    def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb,
                 target_avg_word_emb):
        self.source = source
        self.source_w = source_w
        self.source_sent_emb = source_sent_emb
        self.source_avg_word_emb = source_avg_word_emb
        # self.source_w_imp_list = source_w_imp_list
        # self.source_proc_sent = source_proc_sent
        self.target = target
        self.target_w = target_w
        self.target_sent_emb = target_sent_emb
        self.target_avg_word_emb = target_avg_word_emb
        # self.target_w_imp_list = target_w_imp_list
        # self.target_proc_sent = target_proc_sent

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        source = self.source[idx, :, :]
        source_w = self.source_w[idx, :]
        source_sent_emb = self.source_sent_emb[idx, :]
        source_avg_word_emb = self.source_avg_word_emb[idx, :]
        # source_w_imp_list = self.source_w_imp_list[idx]
        # source_proc_sent = self.source_proc_sent[idx]
        target = self.target[idx, :, :]
        target_w = self.target_w[idx, :]
        target_sent_emb = self.target_sent_emb[idx, :]
        target_avg_word_emb = self.target_avg_word_emb[idx, :]
        # target_w_imp_list = self.target_w_imp_list[idx]
        # target_proc_sent = self.target_proc_sent[idx]
        # debug target[-1] = idx
        # return [source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent]
        return [source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb,
                target_avg_word_emb, idx]


def build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device):
    def store_topics(sent, sent_d2_topics, topic_v_tensor, topic_w_tensor, sent_emb_tensor, avg_word_emb_tensor,
                     w_imp_list, proc_sent_list, i_pairs, device):
        topic_v, topic_w, sent_emb, avg_word_emb, w_imp_arr, proc_sent = sent_d2_topics[sent]
        topic_v_tensor[i_pairs, :, :] = torch.tensor(topic_v, device=device)
        topic_w_tensor[i_pairs, :] = torch.tensor(topic_w, device=device)
        sent_emb_tensor[i_pairs, :] = torch.tensor(sent_emb, device=device)
        avg_word_emb_tensor[i_pairs, :] = torch.tensor(avg_word_emb, device=device)
        w_imp_list[i_pairs] = w_imp_arr
        proc_sent_list[i_pairs] = proc_sent

    corpus_size = len(testing_list)
    first_sent_info = list(sent_d2_topics.values())[0]
    first_topic = first_sent_info[0]
    n_basis = len(first_topic)
    emb_size = len(first_topic[0])
    first_sent_emb = first_sent_info[2]
    encoder_emsize = len(first_sent_emb)

    topic_v_tensor_1 = torch.empty(corpus_size, n_basis, emb_size, device=device)
    topic_w_tensor_1 = torch.empty(corpus_size, n_basis, device=device)
    sent_emb_tensor_1 = torch.empty(corpus_size, encoder_emsize, device=device)
    avg_word_emb_tensor_1 = torch.empty(corpus_size, encoder_emsize, device=device)
    w_imp_list_1 = [0] * corpus_size
    proc_sent_list_1 = [0] * corpus_size
    topic_v_tensor_2 = torch.empty(corpus_size, n_basis, emb_size, device=device)
    topic_w_tensor_2 = torch.empty(corpus_size, n_basis, device=device)
    sent_emb_tensor_2 = torch.empty(corpus_size, encoder_emsize, device=device)
    avg_word_emb_tensor_2 = torch.empty(corpus_size, encoder_emsize, device=device)
    w_imp_list_2 = [0] * corpus_size
    proc_sent_list_2 = [0] * corpus_size

    for i_pairs, fields in enumerate(testing_list):
        sent_1 = fields[0]
        sent_2 = fields[1]
        store_topics(sent_1, sent_d2_topics, topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1,
                     avg_word_emb_tensor_1, w_imp_list_1, proc_sent_list_1, i_pairs, device)
        store_topics(sent_2, sent_d2_topics, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2,
                     avg_word_emb_tensor_2, w_imp_list_2, proc_sent_list_2, i_pairs, device)
    # print(w_imp_list_1)
    # dataset = Set2SetDataset(topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, avg_word_emb_tensor_1, w_imp_list_1, proc_sent_list_1, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2, avg_word_emb_tensor_2, w_imp_list_2, proc_sent_list_2)
    dataset = Set2SetDataset(topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, avg_word_emb_tensor_1,
                             topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2, avg_word_emb_tensor_2)
    use_cuda = False
    if device == 'cude':
        use_cuda = True
    testing_pair_loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=False, pin_memory=use_cuda,
                                                      drop_last=False)
    return testing_pair_loader, (w_imp_list_1, proc_sent_list_1, w_imp_list_2, proc_sent_list_2)


def compute_freq_prob(word_d2_idx_freq):
    all_idx, all_freq = list(zip(*word_d2_idx_freq.values()))
    freq_sum = float(sum(all_freq))
    for w in word_d2_idx_freq:
        idx, freq = word_d2_idx_freq[w]
        word_d2_idx_freq[w].append(freq / freq_sum)


def compute_freq_prob_idx2word(idx2word_freq):
    all_word, all_freq = list(zip(*idx2word_freq))
    freq_sum = float(sum(all_freq))
    for i, (w, freq) in enumerate(idx2word_freq):
        idx2word_freq[i].append(freq / freq_sum)


def safe_cosine_sim(emb_1, emb_2):
    dist = distance.cosine(emb_1, emb_2)
    if math.isnan(dist):
        return 0
    else:
        return 1 - dist


def compute_AP_best_F1_acc(score_list, gt_list, correct_label=1):
    sorted_idx = np.argsort(score_list)
    sorted_idx = sorted_idx.tolist()[::-1]
    # print(sorted_idx)
    # total_correct = sum(gt_list)
    total_correct = sum([1 if x == correct_label else 0 for x in gt_list])
    correct_count = 0
    total_count = 0
    precision_list = []
    F1_list = []
    acc_list = []
    false_neg_num = total_correct
    for idx in sorted_idx:
        total_count += 1
        if gt_list[idx] == correct_label:
            correct_count += 1
            precision = correct_count / float(total_count)
            precision_list.append(precision)
            recall = correct_count / float(total_correct)
            F1_list.append(2 * (precision * recall) / (recall + precision))
            false_neg_num = total_correct - correct_count
        rest_num = len(sorted_idx) - total_count
        true_neg_num = rest_num - false_neg_num
        acc_list.append((correct_count + true_neg_num) / float(len(sorted_idx)))
    return np.mean(precision_list), np.max(F1_list), np.max(acc_list)


def safe_normalization(weight):
    weight_sum = torch.sum(weight, dim=1, keepdim=True)
    # assert weight_sum>0
    return weight / (weight_sum + 0.000000000001)
