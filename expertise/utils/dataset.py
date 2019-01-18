import json
import random
import os
import openreview

from . import utils

class Dataset(object):
    '''
    A class representing an OpenReview paper-reviewer affinity dataset.

    '''

    def __init__(
        self,
        directory=None,
        archive_dirname='archives',
        submissions_dirname='submissions',
        bids_filename = 'bids.jsonl',
        bid_values=[
            'Very High',
            'High',
            'Neutral',
            'Low',
            'Very Low',
            'No Bid'
        ],
        positive_bid_values=[
            'Very High',
            'High'
        ]):

        assert directory and os.path.isdir(directory), 'Directory <{}> does not exist.'.format(directory)

        self.reviewer_bids_file = os.path.join(
            directory, bids_filename)

        self.archives_path = os.path.join(
            directory,  archive_dirname)

        self.submission_records_path = os.path.join(
            directory, submissions_dirname)

        self.train_set_path = os.path.join(
            directory, 'train_set.tsv')

        self.dev_set_path = os.path.join(
            directory, 'dev_set.tsv')

        self.test_set_path = os.path.join(
            directory, 'test_set.tsv')

        # TODO: Important! Need to make sure that different bid values get handled properly
        # across different kinds of datasets.
        self.bid_values = bid_values

        self.positive_bid_values = positive_bid_values

    def bids(self):
        for json_line in utils.jsonl_reader(self.reviewer_bids_file):
            yield openreview.Tag.from_json(json_line)

    def _read_json_records(self, data_dir):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            file_id = filename.replace('.jsonl', '')

            for content in utils.jsonl_reader(filepath):
                # preprocessing
                record_text_unfiltered = utils.content_to_text(content, fields=['title', 'abstract', 'fulltext'])
                record_text_filtered = utils.strip_nonalpha(record_text_unfiltered)

                yield file_id, record_text_unfiltered
    '''
    WARNING: This generator used to be called "submission_records".
    '''
    # def submission_records(self):
    def submissions(self):
        for submission_id, submission_text in self._read_json_records(self.submission_records_path):
            yield submission_id, submission_text

    '''
    WARNING: This generator used to be called "reviewer_archives".
    '''
    # def reviewer_archives(self):
    def archives(self):
        for reviewer_id, paper_text in self._read_json_records(self.archives_path):
            yield reviewer_id, paper_text

