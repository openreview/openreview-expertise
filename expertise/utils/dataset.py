import json
import random
import os
import openreview
from tqdm import tqdm
import ipdb
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
        bids_dirname = 'bids',
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

        self.bids_path = os.path.join(
            directory, bids_dirname)

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

    # def bids(self):
    #     for json_line in utils.jsonl_reader(self.reviewer_bids_file):
    #         yield openreview.Tag.from_json(json_line)

    def bids(self):
        for filename in os.listdir(self.bids_path):
            filepath = os.path.join(self.bids_path, filename)
            file_id = filename.replace('.jsonl','')
            for json_line in utils.jsonl_reader(filepath):
                yield openreview.Tag.from_json(json_line)

    def _read_json_records(self, data_dir, fields=['title','abstract'], sequential=True):
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


    def _items(self, path, num_items, desc='', fields=['title', 'abstract'], sequential=True, progressbar=True, partition_id=0, num_partitions=1):
        item_generator = self._read_json_records(path, fields, sequential=sequential)

        if num_partitions > 1:
            item_generator = utils.partition(
                item_generator,
                partition_id=partition_id, num_partitions=num_partitions)
            num_items = num_items / num_partitions
            desc = '{} (partition {})'.format(desc, partition_id)

        if progressbar:
            item_generator = tqdm(
                item_generator,
                total=num_items,
                desc=desc)

        return item_generator

    def submissions(self, fields=['title', 'abstract'], sequential=True, progressbar=True, partition_id=0, num_partitions=1):

        submission_generator = self._items(
            path=self.submission_records_path,
            num_items=self.num_submissions,
            desc='submissions',
            fields=fields,
            sequential=sequential,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for submission_id, submission_items in submission_generator:
            yield submission_id, submission_items

    # def reviewer_archives(self):
    def archives(self, fields=['title', 'abstract'], sequential=True, progressbar=True, partition_id=0, num_partitions=1):

        archive_generator = self._items(
            path=self.archives_path,
            num_items=self.num_archives,
            desc='archives',
            fields=fields,
            sequential=sequential,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for archive_id, archive_items in archive_generator:
            yield archive_id, archive_items
