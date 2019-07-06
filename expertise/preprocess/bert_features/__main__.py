import argparse
import os
import json
from collections import OrderedDict

from expertise import utils
from expertise.config import ModelConfig

from .setup_bert_kps_lookup import setup_bert_kps_lookup
from .setup_bert_lookup import setup_bert_lookup
from .core import get_embeddings

def main(config, partition_id=0, num_partitions=1, local_rank=-1):
    experiment_dir = os.path.abspath(config.experiment_dir)
    bert_dir = os.path.join(experiment_dir, 'bert')

    feature_dirs = [
        'submissions-features',
        'archives-features'
    ]

    for d in feature_dirs:
        os.makedirs(os.path.join(bert_dir, d), exist_ok=True)

    dataset = Dataset(**config.dataset)

    extraction_args = {
        'max_seq_length': config.max_seq_length,
        'batch_size': config.batch_size,
        'no_cuda': not config.use_cuda
    }

    # convert submissions and archives to bert feature vectors
    dataset_args = {
        'partition_id': partition_id,
        'num_partitions': num_partitions,
        'progressbar': True,
        'sequential': False
    }

    for text_id, text_list in dataset.submissions(**dataset_args):
        feature_dir = os.path.join(bert_dir, 'submissions-features')
        outfile = os.path.join(feature_dir, '{}.npy'.format(text_id))

        if not os.path.exists(outfile):
            embeddings = get_embeddings(
                text_list[:config.max_lines], config.bert_model)
            np.save(outfile, embeddings)
        else:
            print('skipping {}'.format(outfile))

    for text_id, text_list in dataset.archives(**dataset_args):
        feature_dir = os.path.join(bert_dir, 'archives-features')
        outfile = os.path.join(feature_dir, '{}.npy'.format(text_id))

        if not os.path.exists(outfile):
            embeddings = get_embeddings(
                text_list[:config.max_lines], config.bert_model)
            np.save(outfile, embeddings)
        else:
            print('skipping {}'.format(outfile))

    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help="a config file for a model")
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--num_partitions', type=int, default=1)
    parser.add_argument('--use_kps', action='store_true', default=True)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    experiment_path = os.path.dirname(config_path)

    config = ModelConfig()
    config.update_from_file(config_path)

    config = main(
        config,
        partition_id=args.partition,
        num_partitions=args.num_partitions)

    # if args.use_kps:
    #     bert_lookup = setup_bert_kps_lookup(config)
    # else:
    #     bert_lookup = setup_bert_lookup(config)

    # utils.dump_pkl(os.path.join(config.experiment_dir, 'setup', 'bert_lookup.pkl'), bert_lookup)
