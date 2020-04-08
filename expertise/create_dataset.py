'''
A script for generating a dataset.

Assumes that the necessary evidence-collecting steps have been done,
and that papers have been submitted.

'''

import json, argparse
from datetime import datetime
from pathlib import Path
from itertools import chain

import openreview
from tqdm import tqdm

from collections import defaultdict, OrderedDict

from .config import ModelConfig

def convert_to_list(config_invitations):
    if (isinstance(config_invitations, str)):
        invitations = [config_invitations]
    else:
        invitations = config_invitations
    assert isinstance(invitations, list), 'Input should be a str or a list'
    return invitations

def get_publications(openreview_client, author_id):
    content = {
        'authorids': author_id
    }
    publications = openreview.tools.iterget_notes(openreview_client, content=content)
    return [publication for publication in publications]

def get_profile_ids(openreview_client, group_ids):
    """
    Returns a list of all the tilde id members from a list of groups.

    Example:

    >>> get_profiles(openreview_client, ['ICLR.cc/2018/Conference/Reviewers'])

    :param client: OpenReview Client
    :type client: Client
    :param group_ids: List of group ids
    :type group_ids: list[str]

    :return: List of tuples containing (tilde_id, email)
    :rtype: list
    """
    members = []
    for group_id in group_ids:
        group = openreview_client.get_group(group_id)

        profile_members = [member for member in group.members if '~' in member]
        profile_search_results = openreview_client.search_profiles(emails=None, ids=profile_members, term=None)
        tilde_members = []
        for profile in profile_search_results:
            preferredEmail = profile.content.get('preferredEmail') or profile.content.get('emails')[0]
            tilde_members.append((profile.id, preferredEmail))
        members.extend(tilde_members)

        email_members = [member for member in group.members if '@' in member]
        email_set = set(email_members)
        profile_search_results = openreview_client.search_profiles(emails=email_members, ids=None, term=None)
        email_profiles = []
        for email, profile in profile_search_results.items():
            email_profiles.append((profile.id, email))
        members.extend(email_profiles)

    return members

def exclude(openreview_client, config):
    exclusion_invitations = convert_to_list(config['exclusion_inv'])

    for invitation in exclusion_invitations:
        excluded_ids_by_user = defaultdict(list)
        user_grouped_edges = openreview.tools.iterget_grouped_edges(
            openreview_client,
            invitation=invitation,
            groupby='tail',
            select='id,head,label,weight'
        )

        for edges in user_grouped_edges:
            for edge in edges:
                excluded_ids_by_user[edge.tail].append(edge.head)

    return excluded_ids_by_user

def retrieve_expertise(openreview_client, config, excluded_ids_by_user, archive_dir, metadata):
    # if group ID is supplied, collect archives for every member
    # (except those whose archives already exist)
    use_email_ids = config.get('use_email_ids', False)
    group_ids = convert_to_list(config['match_group'])
    valid_members = get_profile_ids(openreview_client, group_ids)

    print('finding archives for {} valid members'.format(len(valid_members)))

    for (member, email) in tqdm(valid_members, total=len(valid_members)):
        file_path = Path(archive_dir).joinpath((email if use_email_ids else member) + '.jsonl')

        if Path(file_path).exists() and not args.overwrite:
            continue

        member_papers = get_publications(openreview_client, member)

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
            and timestamp > config.get('minimum_timestamp', 0) \
            and paperhash not in seen_keys:
                filtered_papers.append(n)
                seen_keys.append(paperhash)

        metadata['archive_counts'][member]['arx'] = len(filtered_papers)
        if len(filtered_papers) == 0:
            metadata['no_publications_count'] += 1
            metadata['no_publications'].append(email if use_email_ids else member)

        with open(file_path, 'w') as f:
            for paper in filtered_papers:
                f.write(json.dumps(paper.to_json()) + '\n')

def get_submissions(openreview_client, config):

    invitation_ids = convert_to_list(config['paper_invitation'])
    submissions = []

    for invitation_id in invitation_ids:
        submissions.extend(list(openreview.tools.iterget_notes(
            openreview_client, invitation=invitation_id)))

    print('finding records of {} submissions'.format(len(submissions)))
    for paper in tqdm(submissions, total=len(submissions)):
        file_path = Path(submission_dir).joinpath(paper.id + '.jsonl')
        if args.overwrite or not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(json.dumps(paper.to_json()) + '\n')

        metadata['bid_counts'][paper.forum] = 0

    return submissions

# Gets bids, no matter if the invitations are for tags or edges
def get_bids(openreview_client, config):
    # Get edge and/or tag invitations
    invitation_ids = convert_to_list(config['bid_inv'])

    # Gather all bid iterators
    bid_iterators = []
    for invitation_id in invitation_ids:
        # try to find the invitation in both collections
        bid_iterators.append(openreview.tools.iterget_tags(
            openreview_client, invitation=invitation_id))
        bid_iterators.append(openreview.tools.iterget_edges(
            openreview_client, invitation=invitation_id))

    # Use chain to put together bid iterators. Once one is done continue with the next one.
    for bid in tqdm(chain(*bid_iterators), desc='writing bids'):
        reduced_bid = {
            'id': bid.id,
            'invitation': bid.invitation,
            'forum': getattr(bid, 'forum', None) or getattr(bid, 'head'),
            'tag': getattr(bid, 'tag', None) or getattr(bid, 'label'),
            'signature': getattr(bid, 'tail', None) or getattr(bid, 'signatures')[0],
        }
        file_path = Path(bids_dir).joinpath(reduced_bid['forum'] + '.jsonl')

        if reduced_bid['forum'] in metadata['bid_counts']:
            metadata['bid_counts'][reduced_bid['forum']] += 1
            metadata['archive_counts'][reduced_bid['signature']]['bid'] += 1

            with open(file_path, 'a') as f:
                f.write(json.dumps(reduced_bid) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a JSON file containing all other arguments')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--username')
    parser.add_argument('--password')
    parser.add_argument('--baseurl')
    args = parser.parse_args()

    config = ModelConfig(config_file_path=args.config)

    print(config)

    dataset_dir = config['dataset']['directory'] if 'dataset' in config else './'
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        dataset_dir.mkdir()

    openreview_client = openreview.Client(
        username=args.username,
        password=args.password,
        baseurl=args.baseurl
    )

    metadata = {
        "reviewer_count": 0,
        "submission_count": 0,
        "no_publications_count": 0,
        "no_publications": [],
        "archive_counts": defaultdict(lambda: {'arx': 0, 'bid': 0}),
        "bid_counts": {},
    }

    if 'exclusion_inv' in config:
        excluded_ids_by_user = exclude(openreview_client, config)
    else:
        excluded_ids_by_user = defaultdict(list)

    if 'match_group' in config:
        archive_dir = dataset_dir.joinpath('archives')
        if not archive_dir.is_dir():
            archive_dir.mkdir()
        retrieve_expertise(openreview_client, config, excluded_ids_by_user, archive_dir, metadata)

    # if invitation ID is supplied, collect records for each submission
    if 'paper_invitation' in config:
        submission_dir = dataset_dir.joinpath('submissions')
        if not submission_dir.is_dir():
            submission_dir.mkdir()
        submissions = get_submissions(openreview_client, config)
        metadata['submission_count'] = len(submissions)

    if 'bid_inv' in config:
        bids_dir = dataset_dir.joinpath('bids')
        if not bids_dir.is_dir():
            bids_dir.mkdir()
        get_bids(openreview_client, config)

    metadata['bid_counts'] = OrderedDict(
        sorted(metadata['bid_counts'].items(), key=lambda t: t[0]))

    metadata['archive_counts'] = OrderedDict(
        sorted(metadata['archive_counts'].items(), key=lambda t: t[0]))

    metadata['reviewer_count'] = len(metadata['archive_counts'])

    metadata_file = dataset_dir.joinpath('metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
