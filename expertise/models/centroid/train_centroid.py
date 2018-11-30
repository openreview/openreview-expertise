import datetime, os, sys, csv
from shutil import copyfile, copytree
import pickle

import torch
import torch.optim as optim

from expertise.models import centroid

from expertise.utils import save_dict_to_json
from expertise.utils.vocab import Vocab
from expertise.utils.batcher import Batcher
from expertise.utils.batcher_devtest import DevTestBatcher
from expertise.utils.config import Config

from expertise.evaluators.hits_at_k import eval_hits_at_k_file
from expertise.evaluators.mean_avg_precision import eval_map_file

current_path = os.path.abspath(os.path.dirname(__file__))

def generate_predictions(model, batcher, outfilename):
    """ Use the model to make predictions on the data in the batcher

    :param model: Model to use to score reviewer-paper pairs
    :param batcher: Batcher containing data to evaluate (a DevTestBatcher)
    :param outfilename: Where to write the predictions to a file for evaluation (tsv) (overwrites)
    :return:
    """

    for idx, batch in enumerate(batcher.batches()):
        if idx % 100 == 0:
            print('Predicted {} batches'.format(idx))
            sys.stdout.flush()

        batch_queries, batch_query_lengths, batch_query_strings,\
        batch_targets, batch_target_lengths, batch_target_strings,\
        batch_labels, batch_size = batch
        # print(batch_query_strings)
        # print(batch_target_strings)

        score_types = ['tpms_scores', 'random_scores', 'tfidf_scores']
        if any([ hasattr(model, s) for s in score_types ]):
            print('scoring based on score file')
            scores = model.score_dev_test_batch(batch_query_strings,
                                                batch_query_lengths,
                                                batch_target_strings,
                                                batch_target_lengths,
                                                batch_size)
        else:
            scores = model.score_dev_test_batch(batch_queries,
                                                batch_query_lengths,
                                                batch_targets,
                                                batch_target_lengths,
                                                batch_size)
        if type(batch_labels) is not list:
            batch_labels = batch_labels.tolist()

        if type(scores) is not list:
            scores = list(scores.cpu().data.numpy().squeeze())

        for source, target, label, score in zip(batch_query_strings,batch_target_strings,batch_labels,scores):
            yield source, target, label, score


def train(config_path):
    """ Train based on the given config, model / dataset

    :param config: config object
    :param dataset_name: name of dataset
    :param model_name: name of model
    :return:
    """

    config_path = os.path.abspath(config_path)
    experiment_path = os.path.dirname(config_path)

    config = Config(filename=config_path)
    setup_path = os.path.join(experiment_path, 'setup')

    assert os.path.isdir(setup_path), 'setup directory must exist'

    vocab_file = os.path.join(setup_path, 'vocab.pkl')
    submission_kps_file = os.path.join(setup_path, 'submission_kps.pkl')
    reviewer_kps_file = os.path.join(setup_path, 'reviewer_kps.pkl')
    train_set_file = os.path.join(setup_path, 'train_set.tsv')

    '''
    This train_sample_file is a string
    '''
    train_sample_file = os.path.join(setup_path, 'train_samples.tsv')
    if not os.exists(train_sample_file):
        train_sample_file = None

    dev_set_file = os.path.join(setup_path, 'dev_set.tsv')

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    train_path = os.path.join(experiment_path, 'train')
    dev_scores_path = os.path.join(train_path, 'dev_scores')
    dev_predictions_path = os.path.join(train_path, 'dev_predictions')

    for path in [train_path, dev_scores_path, dev_predictions_path]:
        if not os.path.isdir(path):
            os.mkdir(path)

    # save the vocab to out dir
    vocab.dump_csv(os.path.join(train_path, 'vocab.tsv'))

    # Why save the source code??
    # copytree(os.path.join(os.environ['EXPERT_MODEL_ROOT'], 'src'), os.path.join(train_path, 'src'))

    torch.manual_seed(config.random_seed)

    # Set up batcher
    batcher = Batcher(config, vocab, samples_file=train_sample_file)

    model = centroid.Model(config, vocab)

    if config.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2penalty)

    # Stats
    best_map = 0
    sum_loss = 0.0

    print('Begin Training')
    sys.stdout.flush()

    # Training loop
    for counter, (source, pos, neg, source_len, pos_len, neg_len) in enumerate(batcher.get_next_batch()):
        print('num_batches: {}'.format(counter))
        optimizer.zero_grad()
        loss = model.compute_loss(source, pos, neg, source_len, pos_len, neg_len)
        print('now backward pass')
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
        optimizer.step()

        # Question: is this if block just for monitoring?
        if counter % 100 == 0:
            # print("p-n:{}".format(model.print_loss(source,pos,neg,source_len,pos_len,neg_len)))

            # `this_loss` originally defined as follows:
            # >>> this_loss = loss.cpu().data.numpy()[0]
            # changed it according to this thread:
            # https://github.com/pytorch/pytorch/issues/6921
            this_loss = loss.cpu().data.numpy()

            sum_loss += this_loss
            print('Processed {} batches, Loss of batch {}: {}. Average loss: {}'.format(counter, counter, this_loss,
                                                                                        sum_loss / (counter / 100)))
            sys.stdout.flush()

        if counter % config.eval_every == 0:
            dev_batcher = DevTestBatcher(
                config, vocab, dev_set_file,
                submission_kps_file, reviewer_kps_file)

            prediction_filename = os.path.join(dev_predictions_path, 'dev.predictions.{}.tsv').format(counter)
            with open(prediction_filename, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                for pred_source, target, label, score in generate_predictions(model, dev_batcher, prediction_filename):
                    writer.writerow([pred_source, target, label, score])

            map_score = float(eval_map_file(prediction_filename))
            hits_at_1 = float(eval_hits_at_k_file(prediction_filename, 1))
            hits_at_3 = float(eval_hits_at_k_file(prediction_filename, 3))
            hits_at_5 = float(eval_hits_at_k_file(prediction_filename, 5))
            hits_at_10 = float(eval_hits_at_k_file(prediction_filename, 10))

            score_obj = {
                'samples': counter,
                'map': map_score,
                'hits_at_1': hits_at_1,
                'hits_at_3': hits_at_3,
                'hits_at_5': hits_at_5,
                'hits_at_10': hits_at_10,
                'config': config.__dict__
            }

            dev_scores_json_file = os.path.join(
                dev_scores_path, 'dev.scores.{}.json'.format(counter))

            dev_scores_tsv_file = os.path.join(
                dev_scores_path, 'dev.scores.{}.tsv'.format(counter))

            save_dict_to_json(
                score_obj, dev_scores_json_file)

            with open(dev_scores_tsv_file, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                score_lines = [
                    [config.name, counter, text, data] for text, data in [
                        ('MAP', map_score),
                        ('Hits@1', hits_at_1),
                        ('Hits@3', hits_at_3),
                        ('Hits@5', hits_at_5),
                        ('Hits@10', hits_at_10)
                    ]
                ]
                for line in score_lines:
                    writer.writerow(line)

            if map_score > best_map:
                best_map = map_score
                torch.save(model, os.path.join(
                    train_path, 'model_{}_{}.torch'.format(config.name, 'best')))

                best_dev_scores_file = os.path.join(train_path, 'dev.scores.best.tsv')
                with open(best_dev_scores_file, 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for line in score_lines:
                        writer.writerow(line)

        if counter == config.num_minibatches:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='a config file for a model')
    args = parser.parse_args()

    train_model(args.config_path)
