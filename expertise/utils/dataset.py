import json
import random
import os
import itertools
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

        self.submissions_path = os.path.join(
            directory, submissions_dirname)

        self.bid_values = bid_values
        self.positive_bid_values = positive_bid_values

        self.num_bids = len(list(self._read_json_records(self.bids_path)))

        self.num_archives = 0 # counts the number of reviewer *documents*, not ids
        reviewer_id_set = set()
        for userid, _ in self._read_json_records(self.archives_path):
            reviewer_id_set.add(userid)
            self.num_archives += 1
        self.reviewer_ids = list(reviewer_id_set)

        submission_id_set = set()
        for submission_id, _ in self._read_json_records(self.submissions_path):
            submission_id_set.add(submission_id)

        self.submission_ids = list(submission_id_set)
        self.num_submissions = len(self.submission_ids)

    def _read_json_records(self, data_dir, sequential=True):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            file_id = filename.replace('.jsonl', '')

            if not sequential:
                all_data = []

            for data in utils.jsonl_reader(filepath):

                if sequential:
                    yield file_id, [data]
                else:
                    all_data.append(data)

            if not sequential:
                yield file_id, all_data


    def _items(self, path, num_items, desc='', sequential=True, progressbar=True, partition_id=0, num_partitions=1):
        item_generator = self._read_json_records(path, sequential=sequential)

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

    def submissions(self, fields=['title', 'abstract', 'fulltext'], sequential=True, progressbar=True, partition_id=0, num_partitions=1):

        submission_generator = self._items(
            path=self.submissions_path,
            num_items=self.num_submissions,
            desc='submissions',
            sequential=sequential,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for submission_id, submission_items in submission_generator:
            yield submission_id, [utils.content_to_text(i, fields) for i in submission_items]

    # def reviewer_archives(self):
    def archives(self, fields=['title', 'abstract', 'fulltext'], sequential=True, progressbar=True, partition_id=0, num_partitions=1):

        archive_generator = self._items(
            path=self.archives_path,
            num_items=self.num_archives if sequential else len(self.reviewer_ids),
            desc='archives',
            sequential=sequential,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for archive_id, archive_items in archive_generator:
            yield archive_id, [utils.content_to_text(i, fields) for i in archive_items]


    def bids(self, sequential=True, progressbar=True, partition_id=0, num_partitions=1):

        bid_generator = self._items(
            path=self.bids_path,
            num_items=self.num_bids,
            desc='bids',
            sequential=sequential,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for forum_id, bid_items in bid_generator:
            yield forum_id, bid_items


    def positive_pairs(self):
        for forum, bids in self.bids(sequential=False, progressbar=False):
            for b in bids:
                reviewer_id = b['signatures'][0]

                if b['tag'] in self.positive_bid_values:
                    yield (forum, reviewer_id)

    def negative_pairs(self):
        for forum, bids in self.bids(sequential=False, progressbar=False):
            for b in bids:
                reviewer_id = b['signatures'][0]

                if b['tag'] not in self.positive_bid_values:
                    yield (forum, reviewer_id)

    def nonpositive_pairs(self):
        positive_pairs = [p for p in self.positive_pairs()]
        for pair in itertools.product(self.submission_ids, self.reviewer_ids):
            if pair not in positive_pairs:
                yield pair
