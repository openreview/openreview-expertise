import os
from expertise.utils.dataset import Dataset
from tqdm import tqdm

# needed?
import gensim
from gensim.models import KeyedVectors

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from . import helpers

def setup_bert_pretrained(bert_model):
    model = BertModel.from_pretrained(bert_model)

    return model

def setup(config, partition_id=0, num_partitions=1, local_rank=-1):

    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    submissions_features_dir = os.path.join(setup_dir, 'submissions-features')
    if not os.path.exists(submissions_features_dir):
        os.mkdir(submissions_features_dir)

    archives_features_dir = os.path.join(setup_dir, 'archives-features')
    if not os.path.exists(archives_features_dir):
        os.mkdir(archives_features_dir)

    dataset = Dataset(**config.dataset)

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model, do_lower_case=config.do_lower_case)

    model = setup_bert_pretrained(config.bert_model)

    # convert submissions and archives to bert feature vectors
    for paper_generator in [dataset.submissions(), dataset.archives()]:
        for filename, text in paper_generator:
            all_lines_features = helpers.extract_features(
                lines=[text],
                model=model,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
                batch_size=32
            )

            avg_embeddings = helpers.get_avg_words(all_lines_features)
            class_embeddings = helpers.get_cls_vectors(all_lines_features)

            avg_emb_file = os.path.join(submissions_features_dir, '{}-avg_emb.npy'.format(filename))
            np.save(avg_emb_file, avg_embeddings)

            cls_emb_file = os.path.join(submissions_features_dir, '{}-cls_emb.npy'.format(filename))
            np.save(cls_emb_file, class_embeddings)


    # for file in tqdm(submission_files, total=len(submission_files), desc='extracting submission features'):
    #     output_file = os.path.join(submissions_features_dir, file)
    #     # write output_file

    # for file in tqdm(archives_files, total=len(archives_files), desc='extracting archive features'):
    #     output_file = os.path.join(archives_features_dir, file)
    #     # write output_file


def train(config):
    pass

def infer(config):
    pass

def test(config):
    pass
