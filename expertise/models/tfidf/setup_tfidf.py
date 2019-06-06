import os
import importlib
import itertools
from collections import defaultdict
from expertise import utils
from expertise.utils import dump_pkl
from expertise.dataset import Dataset

from tqdm import tqdm

def setup(config):
    dataset = Dataset(**config.dataset)
    experiment_dir = os.path.abspath(config.experiment_dir)

    setup_dir = os.path.join(experiment_dir, 'setup')
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    (train_set_ids,
     dev_set_ids,
     test_set_ids) = utils.split_ids(list(dataset.submission_ids), seed=config.random_seed)

    bids_by_forum = utils.get_bids_by_forum(dataset)

    test_labels = utils.format_bid_labels(test_set_ids, bids_by_forum)

    utils.dump_jsonl(os.path.join(config.setup_dir, 'test_labels.jsonl'), test_labels)

