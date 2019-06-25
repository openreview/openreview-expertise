import torch
import os
from expertise import utils
from expertise.utils.batcher import Batcher
from expertise.models import centroid

def test(config):
    print('config.best_model_path', config.best_model_path)
    test_dir = os.path.join(config.experiment_dir, 'test')
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    model = torch.load(config.best_model_path)

    batcher = Batcher(
        input_file=os.path.join(config.experiment_dir, 'setup', 'test_samples.jsonl'))

    predictions = centroid.generate_predictions(config, model, batcher)

    prediction_filename = os.path.join(config.experiment_dir, 'test', 'test.predictions.jsonl')

    utils.dump_jsonl(prediction_filename, predictions)

    print('prediction filename', prediction_filename)
    map_score = float(centroid.eval_map_file(prediction_filename))
    hits_at_1 = float(centroid.eval_hits_at_k_file(prediction_filename, 1))
    hits_at_3 = float(centroid.eval_hits_at_k_file(prediction_filename, 3))
    hits_at_5 = float(centroid.eval_hits_at_k_file(prediction_filename, 5))
    hits_at_10 = float(centroid.eval_hits_at_k_file(prediction_filename, 10))

    score_lines = [
        [config.name, text, data] for text, data in [
            ('MAP', map_score),
            ('Hits@1', hits_at_1),
            ('Hits@3', hits_at_3),
            ('Hits@5', hits_at_5),
            ('Hits@10', hits_at_10)
        ]
    ]
    utils.dump_csv(
        os.path.join(config.experiment_dir, 'test', 'test.scores.tsv'), score_lines)

    report = {
        'MAP': map_score,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@5': hits_at_5,
        'Hits@10': hits_at_10
    }

    config.update(test_scores=report)

    return config
