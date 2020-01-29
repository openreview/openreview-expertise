import os
import json
import torch
from pathlib import Path
from collections import defaultdict

from openreview import Tag
from expertise import utils

from .helpers import filter_by_fields, read_json_records, get_items_generator
from .helpers import get_bids_generator

default_fields = [
    'title',
    'abstract',
    'fulltext',
    'keywords',
    'subject_areas'
]

default_labels = [
    'Very High',
    'High',
    'Neutral',
    'Low',
    'Very Low',
    'No Bid'
]

default_positive_labels = [
    'Very High',
    'High'
]

class ArchivesDataset(torch.utils.data.Dataset):
    '''
    This class maps a tilde id to its list of publications
    '''
    def __init__(self, archives_path):
        print('Loading Archives dataset...')
        self.author_archives = defaultdict(list)
        for author_file in Path(archives_path).iterdir():
            dot_location = str(author_file.name).rindex('.')
            # author_id is the tilde id of the people that will review papers
            author_id = str(author_file.name)[:dot_location]
            with open(author_file) as file_handle:
                for line in file_handle:
                    self.author_archives[author_id].append(json.loads(line.rstrip()))

    def __len__(self):
        return len(self.author_archives)

    def __getitem__(self, author_id):
        return self.author_archives[author_id]

class SubmissionsDataset(torch.utils.data.Dataset):
    '''
    This class maps a Note id to its Note
    '''
    def __init__(self, submissions_path):
        print('Loading Submissions dataset...')
        self.submissions = {}
        for submission_file in Path(submissions_path).iterdir():
            dot_location = str(submission_file.name).rindex('.')
            note_id = str(submission_file.name)[:dot_location]
            with open(submission_file) as file_handle:
                for line in file_handle:
                    self.submissions[note_id] = json.loads(line.rstrip())

    def __len__(self):
        return len(self.submissions)

    def __getitem__(self, note_id):
        return self.submissions[note_id]

class BidsDataset(torch.utils.data.Dataset):
    '''
    This class maps a Note id to its Bids
    '''
    def __init__(self, bids_path):
        print('Loading Bids dataset...')
        self.submission_bids = defaultdict(list)
        for submission_file in Path(bids_path).iterdir():
            dot_location = str(submission_file.name).rindex('.')
            note_id = str(submission_file.name)[:dot_location]
            with open(submission_file) as file_handle:
                for line in file_handle:
                    self.submission_bids[note_id].append(json.loads(line.rstrip()))

    def __len__(self):
        return len(self.submission_bids)

    def __getitem__(self, note_id):
        return self.submission_bids[note_id]

class Dataset(object):
    '''
    A class representing an OpenReview paper-reviewer affinity dataset.

    '''

    def __init__(
        self,
        directory=None,
        archive_dirname='archives',
        submissions_dirname='submissions',
        bids_dirname='bids',
        bid_labels=default_labels,
        positive_bid_labels=default_positive_labels
        ):

        if not directory or not os.path.isdir(directory):
            raise ValueError('Directory <{}> does not exist.'.format(directory))

        self.bid_labels = bid_labels
        self.positive_bid_labels = positive_bid_labels

        self.bids_dir = os.path.join(
            directory, bids_dirname)

        self.archives_dir = os.path.join(
            directory,  archive_dirname)

        self.submissions_dir = os.path.join(
            directory, submissions_dirname)

        with open(os.path.join(directory, 'metadata.json')) as f:
            self.metadata = json.load(f)
            self.submission_count = self.metadata['submission_count']
            self.reviewer_count = self.metadata['reviewer_count']
            self.archive_counts = self.metadata['archive_counts']
            self.bid_counts = self.metadata['bid_counts']

        self.total_bid_count = sum(self.bid_counts.values())
        self.total_archive_count = sum([v['arx'] for v in self.archive_counts.values()])

        self.reviewer_ids = sorted(self.archive_counts.keys())
        self.submission_ids = sorted(self.bid_counts.keys())

    def get_stats(self):
        return self.metadata

    def _read_bids(self):
        for filename in os.listdir(self.bids_dir):
            filepath = os.path.join(self.bids_dir, filename)
            file_id = filename.replace('.jsonl','')
            for json_line in utils.jsonl_reader(filepath):
                yield Tag.from_json(json_line)

    def bids(
            self,
            return_batches=False,
            progressbar='',
            partition_id=0,
            num_partitions=1
        ):

        bids_generator = get_bids_generator(
            path=self.bids_dir,
            num_items=self.submission_count if return_batches else self.total_bid_count,
            return_batches=return_batches,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for submission_id, bids in bids_generator:
            yield submission_id, bids

    def submissions(self,
        fields=default_fields,
        return_batches=False,
        progressbar='',
        partition_id=0,
        num_partitions=1
        ):

        submission_generator = get_items_generator(
            path=self.submissions_dir,
            num_items=self.submission_count,
            return_batches=return_batches,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for submission_id, result in submission_generator:
            if type(result) == list:
                yield submission_id, [filter_by_fields(i['content'], fields) for i in result]
            if type(result) == dict:
                yield submission_id, filter_by_fields(result['content'], fields)

    def archives(self,
        fields=default_fields,
        return_batches=False,
        progressbar='',
        partition_id=0,
        num_partitions=1
        ):

        archive_generator = get_items_generator(
            path=self.archives_dir,
            num_items=self.reviewer_count if return_batches else self.total_archive_count,
            return_batches=return_batches,
            progressbar=progressbar,
            partition_id=int(partition_id),
            num_partitions=int(num_partitions)
        )

        for archive_id, result in archive_generator:
            if type(result) == list:
                yield archive_id, [filter_by_fields(i['content'], fields) for i in result]
            if type(result) == dict:
                yield archive_id, filter_by_fields(result['content'], fields)

