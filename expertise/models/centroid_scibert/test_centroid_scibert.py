import torch
import os

from expertise.utils.batcher import Batcher
from expertise.models import centroid_scibert
from expertise import utils

def test(config):
    print('config.best_model_path', config.best_model_path)
    model = torch.load(config.best_model_path)

    test_samples_path = os.path.join(
        config.setup_dir, 'test_samples.jsonl')


    batcher = Batcher(input_file=config.setup_path(test_samples_path))
    # a lookup table of torch.Tensor objects, keyed by user/paper ID.
    bert_lookup = utils.load_pkl(os.path.join(config.setup_dir, 'bert_lookup.pkl'))

    predictions = centroid_scibert.generate_predictions(config, model, batcher, bert_lookup)

    prediction_filename = config.test_save(predictions,
        'test.predictions.jsonl')

    print('prediction filename', prediction_filename)
    map_score = float(centroid_scibert.eval_map_file(prediction_filename))
    hits_at_1 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 1))
    hits_at_3 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 3))
    hits_at_5 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 5))
    hits_at_10 = float(centroid_scibert.eval_hits_at_k_file(prediction_filename, 10))

    score_lines = [
        [config.name, text, data] for text, data in [
            ('MAP', map_score),
            ('Hits@1', hits_at_1),
            ('Hits@3', hits_at_3),
            ('Hits@5', hits_at_5),
            ('Hits@10', hits_at_10)
        ]
    ]
    config.test_save(score_lines, 'test.scores.tsv')
