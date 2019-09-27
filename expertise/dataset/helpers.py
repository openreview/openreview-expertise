import os
import json
from openreview import Tag
from expertise import utils
from tqdm import tqdm

def filter_by_fields(content, fields):
    filtered_record = {field: value for field, value in content.items() if field in fields}
    return filtered_record

def read_json_records(data_dir, return_batches):
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        file_id = filename.replace('.jsonl', '')

        if return_batches:
            batch = []

        for content in utils.jsonl_reader(filepath):
            if not return_batches:
                yield file_id, content

            else:
                batch.append(content)

        if return_batches:
            yield file_id, batch

def get_items_generator(path, num_items, return_batches, progressbar='', partition_id=0, num_partitions=1):
    items_generator = read_json_records(
        path, return_batches=return_batches)

    if num_partitions > 1:
        items_generator = utils.partition(
            items_generator,
            partition_id=partition_id, num_partitions=num_partitions)
        num_items = num_items / num_partitions
        desc = '{} (partition {})'.format(progressbar, partition_id)

    if progressbar:
        items_generator = tqdm(
            items_generator,
            total=num_items,
            desc=progressbar)

    return items_generator


def read_bid_records(data_dir, return_batches):
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        file_id = filename.replace('.jsonl', '')

        if return_batches:
            batch = []

        for record in utils.jsonl_reader(filepath):

            if not return_batches:
                yield file_id, record
            else:
                batch.append(record)

        if return_batches:
            yield file_id, batch

def get_bids_generator(path, num_items, return_batches, progressbar='', partition_id=0, num_partitions=1):
    items_generator = read_bid_records(
        path, return_batches=return_batches)

    if num_partitions > 1:
        items_generator = utils.partition(
            items_generator, partition_id=partition_id, num_partitions=num_partitions)

        num_items = num_items / num_partitions
        desc = '{} (partition {})'.format(progressbar, partition_id)

    if progressbar:
        items_generator = tqdm(
            items_generator,
            total=num_items,
            desc=progressbar)

    return items_generator
