import argparse
import json
from collections import OrderedDict
from expertise.config import ModelConfig

from .preprocess.textrank import run_textrank
from .models.tfidf.train_tfidf import train
from .models.tfidf.infer_tfidf import infer

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('config_path', help="a config file for a model")
	args = parser.parse_args()

	config = ModelConfig(config_file_path=args.config_path)

	textrank_config = run_textrank(config)
	textrank_config.save(args.config_path)

	trained_config = train(config)
	trained_config.save(args.config_path)

	inferred_config = infer(config)
	inferred_config.save(args.config_path)
