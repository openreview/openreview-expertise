import datetime, os, sys, csv
from shutil import copyfile, copytree
import pickle

import torch
import torch.optim as optim
import numpy as np

from expertise.models import centroid

from expertise.utils import save_dict_to_json
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise import utils

current_path = os.path.abspath(os.path.dirname(__file__))

def train(config):

    train_dir = os.path.join(config.experiment_dir, 'train')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    for train_subdir in ['dev_scores', 'dev_predictions']:
        train_subdir_path = os.path.join(train_dir, train_subdir)
        if not os.path.exists(train_subdir_path):
            os.mkdir(train_subdir_path)

    vocab_file = os.path.join(config.kp_setup_dir, 'textrank_vocab.pkl')
    vocab = utils.load_pkl(vocab_file)

    torch.manual_seed(config.random_seed)

    batcher = Batcher(
        input_file=os.path.join(config.experiment_dir, 'setup', 'train_samples.jsonl'))
    batcher_dev = Batcher(
        input_file=os.path.join(config.experiment_dir, 'setup', 'dev_samples.jsonl'))

    model = centroid.Model(config, vocab)
    if config.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2penalty)

    # Stats
    best_map = 0
    sum_loss = 0.0

    print('Begin Training')

    # Training loop
    for counter, batch in enumerate(batcher.batches(batch_size=config.batch_size)):
        batch_source = []
        batch_pos = []
        batch_neg = []
        batch_source_lens = []
        batch_pos_lens = []
        batch_neg_lens = []

        for data in batch:
            batch_source.append(np.asarray(data['source']))
            batch_pos.append(np.asarray(data['positive']))
            batch_neg.append(np.asarray(data['negative']))
            batch_source_lens.append(np.asarray(data['source_length'], dtype=np.float32))
            batch_pos_lens.append(np.asarray(data['positive_length'], dtype=np.float32))
            batch_neg_lens.append(np.asarray(data['negative_length'], dtype=np.float32))

        print('num_batches: {}'.format(counter))
        optimizer.zero_grad()

        loss_parameters = (
            np.asarray(batch_source),
            np.asarray(batch_pos),
            np.asarray(batch_neg),
            np.asarray(batch_source_lens, dtype=np.float32),
            np.asarray(batch_pos_lens, dtype=np.float32),
            np.asarray(batch_neg_lens, dtype=np.float32)
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

            predictions = centroid.generate_predictions(config, model, batcher_dev)

            prediction_filename = os.path.join(
                train_dir,
                'dev_predictions/dev.predictions.{}.jsonl'.format(counter))

            utils.dump_jsonl(prediction_filename, predictions)

            print('prediction filename', prediction_filename)
            map_score = float(centroid.eval_map_file(prediction_filename))
            hits_at_1 = float(centroid.eval_hits_at_k_file(prediction_filename, 1))
            hits_at_3 = float(centroid.eval_hits_at_k_file(prediction_filename, 3))
            hits_at_5 = float(centroid.eval_hits_at_k_file(prediction_filename, 5))
            hits_at_10 = float(centroid.eval_hits_at_k_file(prediction_filename, 10))

            score_lines = [
                [config.name, counter, text, data] for text, data in [
                    ('MAP', map_score),
                    ('Hits@1', hits_at_1),
                    ('Hits@3', hits_at_3),
                    ('Hits@5', hits_at_5),
                    ('Hits@10', hits_at_10)
                ]
            ]
            dev_scores_file = os.path.join(
                train_dir,
                'dev_scores/dev.scores.{}.tsv'.format(counter))
            utils.dump_csv(dev_scores_file, score_lines)

            if map_score > best_map:
                best_map = map_score

                best_model_path = os.path.join(
                    train_dir, 'model_{}_{}.torch'.format(config.name, 'best'))

                torch.save(model, best_model_path)
                config.update(best_model_path=best_model_path)

                best_scores_file = os.path.join(
                    train_dir,
                    'dev.scores.best.tsv')

                utils.dump_csv(best_scores_file, score_lines)

        if counter == config.num_minibatches:
            return config
    return config
