import json
import random
import os

class Dataset(object):
    def __init__(self, directory):

        assert os.path.isdir(directory), 'Directory does not exist.'

        self.reviewer_bids_file = os.path.join(
            directory, 'reviewer_bids.jsonl')

        self.reviewer_archives_path = os.path.join(
            directory, 'reviewer_archives')

        self.submission_records_path = os.path.join(
            directory, 'submission_records_fulltext')

        self.train_set_path = os.path.join(
            directory, 'train_set.tsv')

        self.dev_set_path = os.path.join(
            directory, 'dev_set.tsv')

        self.test_set_path = os.path.join(
            directory, 'test_set.tsv')

        self.bid_values = [
            'I want to review',
            'I can review',
            'I can probably review but am not an expert',
            'I cannot review',
            'No bid'
        ]

        self.positive_bid_values = ["I want to review", "I can review"]


