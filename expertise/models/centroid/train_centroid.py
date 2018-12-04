import datetime, os, sys, csv
from shutil import copyfile, copytree
import pickle

import torch
import torch.optim as optim

from expertise.models import centroid

from expertise.utils import save_dict_to_json
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.config import Config

from expertise.evaluators.hits_at_k import eval_hits_at_k_file
from expertise.evaluators.mean_avg_precision import eval_map_file
import numpy as np

current_path = os.path.abspath(os.path.dirname(__file__))

#def train(setup_path, train_path, config, dataset):
def train(config):

    vocab = config.setup_load('vocab.pkl')

    torch.manual_seed(config.random_seed)

    batcher = Batcher(input_file=config.setup_path('train_samples_permuted.jsonl'))
    batcher_dev = Batcher(input_file=config.setup_path('dev_samples.jsonl'))

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

            prediction_filename = config.train_save(predictions,
                'dev_predictions/dev.predictions.{}.jsonl'.format(counter))

            print('prediction filename', prediction_filename)
            map_score = float(eval_map_file(prediction_filename))
            hits_at_1 = float(eval_hits_at_k_file(prediction_filename, 1))
            hits_at_3 = float(eval_hits_at_k_file(prediction_filename, 3))
            hits_at_5 = float(eval_hits_at_k_file(prediction_filename, 5))
            hits_at_10 = float(eval_hits_at_k_file(prediction_filename, 10))

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
