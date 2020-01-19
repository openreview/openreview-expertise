'''
A script for generating a dataset.

Assumes that the necessary evidence-collecting steps have been done,
and that papers have been submitted.

'''

import json, argparse
from datetime import datetime
from pathlib import Path, PurePath
# from expertise.config import ModelConfig

import openreview
from tqdm import tqdm

from collections import defaultdict, OrderedDict

def get_publications(openreview_client, author_id):
    content = {
        'authorids': author_id
    }
    publications = openreview.tools.iterget_notes(openreview_client, content=content)
    return [publication for publication in publications]

def exclude(openreview_client, config):
    excluded_ids_by_user = defaultdict(list)
    user_grouped_edges = openreview.tools.iterget_grouped_edges(
        openreview_client,
        invitation=config['exclusion_inv'],
        groupby='tail',
        select='id,head,label,weight'
    )

    for edges in user_grouped_edges:
        for edge in edges:
            excluded_ids_by_user[edge.tail].append(edge.head)

    return excluded_ids_by_user

def retrieve_expertise(openreview_client, config, excluded_ids_by_user):
    # if group ID is supplied, collect archives for every member
    # (except those whose archives already exist)
    group_id = config['match_group']
    group = openreview_client.get_group(group_id)

    profile_members = [member for member in group.members if '~' in member]
    email_members = [member for member in group.members if '@' in member]
    profile_search_results = openreview_client.search_profiles(
        emails=email_members, ids=None, term=None)

    valid_members = []
    if profile_members:
        valid_members.extend(profile_members)
    if profile_search_results and type(profile_search_results) == dict:
        valid_members.extend([p.id for p in profile_search_results.values()])

    print('finding archives for {} valid members'.format(len(valid_members)))

    archive_direct_uploads = openreview.tools.iterget_notes(
        openreview_client, invitation='OpenReview.net/Archive/-/Direct_Upload')

    direct_uploads_by_signature = defaultdict(list)

    for direct_upload in archive_direct_uploads:
        direct_uploads_by_signature[direct_upload.signatures[0]].append(direct_upload)

    for member in tqdm(valid_members, total=len(valid_members)):
        file_path = PurePath.joinpath(archive_dir, member + '.jsonl')
        if args.overwrite or not Path.exists(file_path):
            member_papers = get_publications(openreview_client, member)

            member_papers.extend(direct_uploads_by_signature[member])

            filtered_papers = [
                n for n in member_papers \
                if n.id not in excluded_ids_by_user[member] \
            ]

            seen_keys = []
            filtered_papers = []
            for n in member_papers:
                paperhash = openreview.tools.get_paperhash('', n.content['title'])

                timestamp = n.cdate if n.cdate else n.tcdate

                if n.id not in excluded_ids_by_user[member] \
                and timestamp > minimum_timestamp \
                and paperhash not in seen_keys:
                    filtered_papers.append(n)
                    seen_keys.append(paperhash)

            metadata['archive_counts'][member]['arx'] = len(filtered_papers)

            with open(file_path, 'w') as f:
                for paper in filtered_papers:
                    f.write(json.dumps(paper.to_json()) + '\n')

def get_submissions(openreview_client, config):
    invitation_id = config['paper_invitation']

    # (1) get submissions from OpenReview
    submissions = list(openreview.tools.iterget_notes(
        openreview_client, invitation=invitation_id))

    print('finding records of {} submissions'.format(len(submissions)))
    for paper in tqdm(submissions, total=len(submissions)):
        file_path = PurePath.joinpath(submission_dir, paper.id + '.jsonl')
        if args.overwrite or not Path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(json.dumps(paper.to_json()) + '\n')

        metadata['bid_counts'][paper.forum] = 0

    return submissions

def get_bids(openreview_client, config):
    invitation_id = config['bid_inv']

    bids = openreview.tools.iterget_tags(
        openreview_client, invitation=invitation_id)

    for bid in tqdm(bids, desc='writing bids'):
        file_path = PurePath.joinpath(bids_dir, bid.forum + '.jsonl')

        if bid.forum in metadata['bid_counts']:
            metadata['bid_counts'][bid.forum] += 1
            metadata['archive_counts'][bid.signatures[0]]['bid'] += 1

            with open(file_path, 'a') as f:
                f.write(json.dumps(bid.to_json()) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a JSON file containing all other arguments')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--username')
    parser.add_argument('--password')
    parser.add_argument('--baseurl')
    args = parser.parse_args()

    with open(args.config) as file_handle:
        config = json.load(file_handle)
    # with open(config_path) as f:
	#     data = json.load(f, object_pairs_hook=OrderedDict)
	# config = ModelConfig(**data)

    print(config)

    dataset_dir = config['dataset']['directory'] if 'dataset' in config else './'
    dataset_dir = Path(dataset_dir)

    if not Path.is_dir(dataset_dir):
        Path.mkdir(dataset_dir)

    archive_dir = PurePath.joinpath(dataset_dir, 'archives')
    if not Path.is_dir(archive_dir):
        Path.mkdir(archive_dir)

    submission_dir = PurePath.joinpath(dataset_dir, 'submissions')
    if not Path.is_dir(submission_dir):
        Path.mkdir(submission_dir)

    bids_dir = PurePath.joinpath(dataset_dir, 'bids')
    if not Path.is_dir(bids_dir):
        Path.mkdir(bids_dir)

    openreview_client = openreview.Client(
        username=args.username,
        password=args.password,
        baseurl=args.baseurl
    )

    metadata = {
        "reviewer_count": 0,
        "submission_count": 0,
        "archive_counts": defaultdict(lambda: {'arx': 0, 'bid': 0}),
        "bid_counts": {},
    }

    minimum_timestamp = 0
    if 'oldest_year' in config:
        epoch = datetime.fromtimestamp(0)
        date = datetime.strptime(config['oldest_year'], '%Y')
        minimum_timestamp = (date - epoch).total_seconds() * 1000.0

    print('minimum_timestamp', minimum_timestamp)

    if 'exclusion_inv' in config:
        excluded_ids_by_user = exclude(openreview_client, config)

    if 'match_group' in config:
        retrieve_expertise(openreview_client, config, excluded_ids_by_user)

    # if invitation ID is supplied, collect records for each submission
    if 'paper_invitation' in config:
        submissions = get_submissions(openreview_client, config)

    if 'bid_inv' in config:
        get_bids(openreview_client, config)

    metadata['bid_counts'] = OrderedDict(
        sorted(metadata['bid_counts'].items(), key=lambda t: t[0]))

    metadata['archive_counts'] = OrderedDict(
        sorted(metadata['archive_counts'].items(), key=lambda t: t[0]))

    metadata['reviewer_count'] = len(metadata['archive_counts'])
    metadata['submission_count'] = len(submissions)

    metadata_file = PurePath.joinpath(dataset_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
