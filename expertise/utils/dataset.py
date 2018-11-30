import json
import random
import os
import openreview

from . import utils

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

        # TODO: Important! Need to make sure that different bid values get handled properly
        # across different kinds of datasets.
        self.bid_values = [
            'I want to review',
            'I can review',
            'I can probably review but am not an expert',
            'I cannot review',
            'No bid'
        ]

        self.positive_bid_values = ["I want to review", "I can review"]

    def bids(self):
        for json_line in utils.jsonl_reader(self.reviewer_bids_file):
            yield openreview.Tag.from_json(json_line)

    def _read_json_records(self, data_dir):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            file_id = filename.replace('.jsonl', '')

            for json_line in utils.jsonl_reader(filepath):
                content = json_line['content']

                # preprocessing
                record_text_unfiltered = utils.content_to_text(content, fields=['title', 'abstract', 'fulltext'])
                record_text_filtered = utils.strip_nonalpha(record_text_unfiltered)

            yield file_id, record_text_unfiltered

    def submission_records(self):
        for submission_id, submission_text in self._read_json_records(self.submission_records_path):
            yield submission_id, submission_text

    def reviewer_archives(self):
        for reviewer_id, paper_text in self._read_json_records(self.reviewer_archives_path):
            yield reviewer_id, paper_text

