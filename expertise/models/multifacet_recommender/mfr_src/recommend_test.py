import argparse
import os
import torch

from .utils import seed_all_randomness, load_corpus, loading_all_models, str2bool
from .utils_testing import add_model_arguments, recommend_test

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--data', type=str, default='./data/processed/citeulike-a_lower/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_cold_0',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--tag_emb_file', type=str, default='tag_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--user_emb_file', type=str, default='user_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')
parser.add_argument('--store_dist', type=str, default='',
                    help='if set as user/tag, store user/tag dist. Otherwise, run the evaluation')

###system
parser.add_argument('--test_user', type=str2bool, nargs='?', default=True,
                    help='Whether we want to test user embeddings')
parser.add_argument('--test_tag', type=str2bool, nargs='?', default=True,
                    help='Whether we want to test tag embeddings')
parser.add_argument('--switch_user_tag_roles', type=str2bool, nargs='?', default=False,
                    help='If true and use TRANS_two_heads as de_coeff_model, switch the magnitude of two models')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--most_popular_baseline', type=str2bool, nargs='?', default=False,
                    help='Whether to test most popular baseline')
parser.add_argument('--subsample_ratio', type=float, default=1,
                    help='ratio of subsampling the user or tag')
parser.add_argument('--div_eval', type=str, default='openreview',
                    help='Could be citeulike, amazon, and openreview')
parser.add_argument('--remove_testing_duplication', type=str2bool, nargs='?', default=True,
                    help='Whether we want to remove the duplicated record in testing data')

add_model_arguments(parser)

args = parser.parse_args()

if args.switch_user_tag_roles:
    assert args.de_coeff_model == 'TRANS_two_heads'

if args.tag_emb_file[:7] == "tag_emb":
    args.tag_emb_file = os.path.join(args.checkpoint, args.tag_emb_file)
if args.user_emb_file[:8] == "user_emb":
    args.user_emb_file = os.path.join(args.checkpoint, args.user_emb_file)

seed_all_randomness(args.seed, args.cuda)

########################
print("Loading data")
########################

device = torch.device("cuda" if args.cuda else "cpu")

all_corpus = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training=True,
                         want_to_shuffle_val=False, load_test=True, deduplication=True,
                         tensor_folder=args.tensor_folder, subsample_ratio=args.subsample_ratio,
                         remove_testing_duplication=args.remove_testing_duplication)
if args.subsample_ratio < 1:
    idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, dataloader_val_info, dataloader_test_info, max_sent_len, user_subsample_idx, tag_subsample_idx = all_corpus
else:
    idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, dataloader_val_info, dataloader_test_info, max_sent_len = all_corpus
dataloader_train = dataloader_train_arr[0]

########################
print("Loading Models")
########################

normalize_emb = True
if args.loss_type != 'dist':
    normalize_emb = False
parallel_encoder, parallel_decoder, encoder, decoder, user_norm_emb, tag_norm_emb = \
    loading_all_models(args, idx2word_freq, user_idx2word_freq, tag_idx2word_freq, device, max_sent_len, normalize_emb)

if args.subsample_ratio < 1:
    user_norm_emb = user_norm_emb[user_subsample_idx, :]
    if len(tag_norm_emb) > 0:
        tag_norm_emb = tag_norm_emb[tag_subsample_idx, :]

encoder.eval()
decoder.eval()

with open(args.outf, 'w') as outf:
    recommend_test(dataloader_test_info, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, idx2word_freq,
                   user_idx2word_freq, tag_idx2word_freq, args.coeff_opt, args.loss_type, args.test_user, args.test_tag,
                   outf, device, args.most_popular_baseline, args.div_eval, args.switch_user_tag_roles, args.store_dist,
                   figure_name=args.outf + '_fig')
