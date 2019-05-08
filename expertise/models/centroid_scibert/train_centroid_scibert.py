import datetime, os, sys, csv
from shutil import copyfile, copytree
import pickle

import torch
import torch.optim as optim
import numpy as np

from expertise.models import centroid_scibert

from expertise import utils
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.config import Config

current_path = os.path.abspath(os.path.dirname(__file__))

#def train(setup_path, train_path, config, dataset):
def train(config):

    for train_subdir in ['dev_scores', 'dev_predictions']:
        train_subdir_path = os.path.join(config.train_dir, train_subdir)
        if not os.path.exists(train_subdir_path):
            os.mkdir(train_subdir_path)

    vocab_path = os.path.join(
        config.kp_setup_dir, 'vocab.pkl')
    vocab = utils.load_pkl(vocab_path)

    torch.manual_seed(config.random_seed)

    train_samples_path = os.path.join(
        config.setup_dir, 'train_samples.jsonl')

    dev_samples_path = os.path.join(
        config.setup_dir, 'dev_samples.jsonl')

    print('reading train samples from ', train_samples_path)
    batcher = Batcher(input_file=train_samples_path)
    batcher_dev = Batcher(input_file=dev_samples_path)

    model = centroid_scibert.Model(config, vocab)
    if config.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2penalty)

    # Stats
    best_map = 0
    sum_loss = 0.0

    # a lookup table of torch.Tensor objects, keyed by user/paper ID.
    bert_lookup = utils.load_pkl(os.path.join(config.kp_setup_dir, 'bert_lookup.pkl'))

    print('Begin Training')

    # Training loop
    for counter, batch in enumerate(batcher.batches(batch_size=config.batch_size)):

        batch_source = []
        batch_pos = []
        batch_neg = []

        for data in batch:
            batch_source.append(bert_lookup[data['source_id']])
            batch_pos.append(bert_lookup[data['positive_id']])
            batch_neg.append(bert_lookup[data['negative_id']])

        print('num_batches: {}'.format(counter))
        optimizer.zero_grad()

        loss_parameters = (
            torch.stack(batch_source),
            torch.stack(batch_pos),
            torch.stack(batch_neg)
        )

        loss = model.compute_loss(*loss_parameters)
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        optimizer.step()

        # Question: is this if block just for monitoring?
        if counter % 100 == 0:

            this_loss = loss.cpu().data.numpy()
            sum_loss += this_loss

            print('Processed {} batches, Loss of batch {}: {}. Average loss: {}'.format(
                counter, counter, this_loss, sum_loss / (counter / 100)))

        if counter % config.eval_every == 0:

            # is this reset needed?
            batcher_dev.reset()

            predictions = centroid_scibert.generate_predictions(config, model, batcher_dev, bert_lookup)

            prediction_filename = config.train_save(predictions,
                'dev_predictions/dev.predictions.{}.jsonl'.format(counter))

            print('prediction filename', prediction_filename)
            map_score = float(centroid_scibert.eval_map_file(prediction_filename))
            hits_at_1 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 1))
            hits_at_3 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 3))
            hits_at_5 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 5))
            hits_at_10 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 10))

            score_lines = [
                [config.name, counter, text, data] for text, data in [
                    ('MAP', map_score),
                    ('Hits@1', hits_at_1),
                    ('Hits@3', hits_at_3),
                    ('Hits@5', hits_at_5),
                    ('Hits@10', hits_at_10)
                ]
            ]
            config.train_save(score_lines, 'dev_scores/dev.scores.{}.tsv'.format(counter))

            if map_score > best_map:
                best_map = map_score

                best_model_path = os.path.join(
                    config.train_dir, 'model_{}_{}.torch'.format(config.name, 'best'))

                torch.save(model, best_model_path)
                config.best_model_path = best_model_path
                config.best_map_score = best_map
                config.hits_at_1 = hits_at_1
                config.hits_at_3 = hits_at_3
                config.hits_at_5 = hits_at_5
                config.hits_at_10 = hits_at_10
                config.save_config()

                config.train_save(score_lines, 'dev.scores.best.tsv')

        if counter == config.num_minibatches:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='a config file for a model')
    args = parser.parse_args()

    train_model(args.config_path)
