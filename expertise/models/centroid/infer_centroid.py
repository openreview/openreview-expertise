import os
import itertools
import torch
from expertise.models import centroid
from expertise import utils
from expertise.utils.batcher import Batcher
from expertise.utils.config import Config

def infer(config):
    experiment_dir = os.path.abspath(config.experiment_dir)
    infer_dir = os.path.join(experiment_dir, 'infer')
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    '''
	1. write full_samples.jsonl, containing every paper-reviewer pair

    '''

    model = torch.load(config.best_model_path)

    batcher = Batcher(input_file=os.path.join(config.setup_dir, 'full_labels.csv'))

    predictions = centroid.generate_predictions(config, model, batcher)

    max_predictions = {}
    for p in predictions:
    	new_src_id, _ = p['source_id'].split(':')
    	new_target_id, _ = p['target_id'].split(':')
    	key = (new_src_id, new_target_id)
    	score = p['score']

    	if key in max_predictions:
    		if score > max_predictions[key]['score']:
    			max_predictions[key]['score'] = score
    		assert max_predictions[key]['label'] == p['label']
    	else:
	    	max_predictions[key] = {
	    		'source_id': new_src_id,
	    		'target_id': new_target_id,
	    		'score': score,
	    		'label': p['label']
			}



    prediction_filename = os.path.join(infer_dir, 'inferred_scores.jsonl')
    utils.dump_jsonl(prediction_filename, [p for p in max_predictions.values()])

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
    scores_filename = os.path.join(infer_dir, 'inferred_performance.tsv')
    utils.dump_csv(scores_filename, score_lines)









