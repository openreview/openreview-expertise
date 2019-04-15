import json
import random
import os
import openreview
from tqdm import tqdm

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

        self.num_submissions = len(list(self._read_json_records(self.submission_records_path)))

        self.num_archives = 0
        self.reviewer_ids = set()
        for userid, archive in self._read_json_records(self.archives_path):
            self.reviewer_ids.add(userid)
            self.num_archives += 1

        # TODO: Important! Need to make sure that different bid values get handled properly
        # across different kinds of datasets.
        self.bid_values = bid_values

        self.positive_bid_values = positive_bid_values

    def bids(self):
        for json_line in utils.jsonl_reader(self.reviewer_bids_file):
            yield openreview.Tag.from_json(json_line)

    def _read_json_records(self, data_dir, fields=['title','abstract','fulltext'], sequential=True):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            file_id = filename.replace('.jsonl', '')

            if not sequential:
                all_text = []

            for content in utils.jsonl_reader(filepath):
                # preprocessing
                record_text_unfiltered = utils.content_to_text(content, fields)
                record_text_filtered = utils.strip_nonalpha(record_text_unfiltered)

                if sequential:
                    yield file_id, record_text_unfiltered
                else:
                    all_text.append(record_text_unfiltered)

            if not sequential:
                yield file_id, all_text

    # def submission_records(self):
    def submissions(self, fields=['title', 'abstract', 'fulltext']):
        for submission_id, submission_text in tqdm(
            self._read_json_records(self.submission_records_path, fields),
            total=self.num_submissions,
            desc='dataset submissions'):

            yield submission_id, submission_text

    # def reviewer_archives(self):
    def archives(self, fields=['title', 'abstract', 'fulltext'], sequential=True):
        for reviewer_id, paper_text in tqdm(
            self._read_json_records(self.archives_path, fields, sequential=sequential),
            total=self.num_archives if sequential else len(self.reviewer_ids),
            desc='dataset archives'):

            yield reviewer_id, paper_text

