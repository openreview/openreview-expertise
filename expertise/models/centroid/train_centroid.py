import datetime, os, sys, csv
from shutil import copyfile, copytree
import json

import torch
import torch.optim as optim
import numpy as np

from expertise import utils

from expertise.models import centroid
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.config import Config

import ipdb

current_path = os.path.abspath(os.path.dirname(__file__))

KPS_PER_USER = 100

def train(config):
    for train_subdir in ['dev_scores', 'dev_predictions']:
        train_subdir_path = os.path.join(config.train_dir, train_subdir)
        if not os.path.exists(train_subdir_path):
            os.mkdir(train_subdir_path)

    vocabfile = os.path.join(config.setup_dir, 'vocab')
    vocab = Vocab(vocabfile=vocabfile, max_num_keyphrases=config.max_num_keyphrases)

    torch.manual_seed(config.random_seed)

    batcher = Batcher(
        input_file=os.path.join(config.setup_dir, 'train_samples.csv'),
        batch_size=config.batch_size,
        max_num_batches=config.num_minibatches)

    model = centroid.Model(config, vocab)
    if config.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2penalty)

    # Stats
    best_map = 0
    sum_loss = 0.0

    print('Begin Training')

    for batch_idx, (src, pos, neg) in enumerate(
        centroid.format_batch(batcher, config)):

        optimizer.zero_grad()

        loss_parameters = {
            'batch_source': src['features'],
            'pos_result': pos['features'],
            'neg_result': neg['features'],
            'batch_lengths': src['lens'],
            'pos_len': pos['lens'],
            'neg_len': neg['lens']
        }

        # ipdb.set_trace()

        loss = model.get_loss(**loss_parameters)
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        optimizer.step()

        if batch_idx % config.eval_every == 0:

            # stats & monitoring
            this_loss = loss.cpu().data.numpy()
            sum_loss += this_loss

            print(
                f'Loss of batch {batch_idx:04}: {this_loss}. '
                f'Average loss: {sum_loss / (batch_idx / 100)}'
            )

            # write out scores and predictions
            scores_file = os.path.join(
                config.train_dir, 'dev_scores', f'batch{batch_idx:04}.tsv')

            predictions_file = os.path.join(
                config.train_dir, 'dev_predictions', f'batch{batch_idx:04}.jsonl')

            dev_batcher = Batcher(
                input_file=os.path.join(config.setup_dir, 'dev_samples.csv'),
                batch_size=config.dev_batch_size,
                max_num_batches=config.num_minibatches)

            predictions = centroid.generate_predictions(config, model, dev_batcher)
            utils.dump_jsonl(predictions_file, predictions)

            map_score = float(centroid.eval_map_file(predictions_file))
            hits_at_1 = float(centroid.eval_hits_at_k_file(predictions_file, 1))
            hits_at_3 = float(centroid.eval_hits_at_k_file(predictions_file, 3))
            hits_at_5 = float(centroid.eval_hits_at_k_file(predictions_file, 5))
            hits_at_10 = float(centroid.eval_hits_at_k_file(predictions_file, 10))
            hits_at_1 = 0
            hits_at_3 = 0
            hits_at_5 = 0
            hits_at_10 = 0

            score_lines = [
                [config.name, batch_idx, text, data] for text, data in [
                    ('MAP', map_score),
                    ('Hits@1', hits_at_1),
                    ('Hits@3', hits_at_3),
                    ('Hits@5', hits_at_5),
                    ('Hits@10', hits_at_10)
                ]
            ]

            utils.dump_csv(scores_file, score_lines)

            if map_score > best_map:
                best_map = map_score

                best_model_path = os.path.join(
                    config.train_dir, f'best_model_{config.name}.torch')

                torch.save(model, best_model_path)
                config.best_model_path = best_model_path
                config.best_map_score = best_map
                config.hits_at_1 = hits_at_1
                config.hits_at_3 = hits_at_3
                config.hits_at_5 = hits_at_5
                config.hits_at_10 = hits_at_10
                config.save_config()

                config.train_save(score_lines, 'dev.scores.best.tsv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='a config file for a model')
    args = parser.parse_args()

    train_model(args.config_path)
