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
    if (isinstance(config_invitations, str) or isinstance(config_invitations, dict)):
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

def get_profile_ids(client, group_ids):
    """
    Returns a list of all the tilde id members from a list of groups.

    Example:

    >>> get_profiles(openreview_client, ['ICLR.cc/2018/Conference/Reviewers'])

    :param client: OpenReview Client
    :type client: Client
    :param group_ids: List of group ids
    :type group_ids: list[str]

    :return: List of tilde ids
    :rtype: list
    """
    tilde_members = []
    for group_id in group_ids:
        group = openreview_client.get_group(group_id)
        profile_members = [member for member in group.members if '~' in member]
        email_members = [member for member in group.members if '@' in member]
        profile_search_results = openreview_client.search_profiles(emails=email_members, ids=None, term=None)

        if profile_members:
            tilde_members.extend(profile_members)
        if profile_search_results and type(profile_search_results) == dict:
            tilde_members.extend([p.id for p in profile_search_results.values()])

    return tilde_members

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

def retrieve_expertise(openreview_client, config, excluded_ids_by_user):
    # if group ID is supplied, collect archives for every member
    # (except those whose archives already exist)
    group_ids = convert_to_list(config['match_group'])
    valid_members = get_profile_ids(openreview_client, group_ids)

    print('finding archives for {} valid members'.format(len(valid_members)))

    archive_direct_uploads = openreview.tools.iterget_notes(
        openreview_client, invitation='OpenReview.net/Archive/-/Direct_Upload')

    direct_uploads_by_signature = defaultdict(list)

    for direct_upload in archive_direct_uploads:
        direct_uploads_by_signature[direct_upload.signatures[0]].append(direct_upload)

    for member in tqdm(valid_members, total=len(valid_members)):
        file_path = Path(archive_dir).joinpath(member + '.jsonl')

        if Path(file_path).exists() and not args.overwrite:
            continue

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
        if len(filtered_papers) == 0:
            metadata['no_publications_count'] += 1
            metadata['no_publications'].append(member)

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

    all_bids = defaultdict(list)

    # Use chain to put together bid iterators. Once one is done continue with the next one.
    for bid in tqdm(chain(*bid_iterators), desc='writing bids'):
        reduced_bid = {
            'id': bid.id,
            'forum': getattr(bid, 'forum', None) or getattr(bid, 'head'),
            'tag': getattr(bid, 'tag', None) or getattr(bid, 'label'),
            'signature': getattr(bid, 'tail', None) or getattr(bid, 'signatures')[0],
        }
        all_bids[reduced_bid['signature']].append(reduced_bid)

        if reduced_bid['forum'] in metadata['bid_counts']:
            metadata['bid_counts'][reduced_bid['forum']] += 1
            metadata['archive_counts'][reduced_bid['signature']]['bid'] += 1

    file_path = Path(bids_dir).joinpath('bids.json')
    with open(file_path, 'w') as f:
        json.dump(all_bids, f, indent=4)

def get_assignments(openreview_client, config):
    assignments_params = convert_to_list(config['assignments'])

    assignment_iterators = []
    for params in assignments_params:
        if isinstance(params, dict):
            try:
                assignment_iterators.append(openreview.tools.iterget_edges(openreview_client, **params))
                assignment_iterators.append(openreview.tools.iterget_notes(openreview_client, **params))
            except:
                assignment_iterators.append(openreview.tools.iterget_notes(openreview_client, **params))
        else:
            invitation_id = params
            assignment_iterators.append(openreview.tools.iterget_edges(
                openreview_client, invitation=invitation_id))
            assignment_iterators.append(openreview.tools.iterget_notes(
                openreview_client, invitation=invitation_id))

    all_assignments = defaultdict(list)
    # Use chain to put together bid iterators. Once one is done continue with the next one.
    for assignment in tqdm(chain(*assignment_iterators), desc='writing assignments'):
        # This is for old conferences like ICLR 2018
        if getattr(assignment, 'content', None) and assignment.content.get('assignments'):
            papers = assignment.content.get('assignments')
            for paper in papers.values():
                if len(paper.get('assigned', [])) > 0:
                    for userId in paper.get('assigned'):
                        all_assignments[userId].append({
                            'id': None,
                            'head': paper['forum'],
                            'tail': userId,
                            'weight': None
                        })
            continue

        # This is for venues like ICLR 2019
        if getattr(assignment, 'content', None) and assignment.content.get('assignedGroups'):
            for group in assignment.content.get('assignedGroups'):
                all_assignments[rgroup['userId']].append({
                    'id': assignment.id,
                    'head': assignment.forum,
                    'tail' = group['userId']
                    'weight' = group['finalScore']
                })
            continue

        reduced_assignment = {
            'id': assignment.id,
            'head': assignment.forum,
            'tail' = assignment.tail,
            'weight' = assignment.weight
        }
        all_assignments[reduced_assignment['tail']].append(reduced_assignment)

    file_path = Path(assignments_dir).joinpath('assignments.json')

    with open(file_path, 'w') as f:
        json.dump(all_assignments, f, indent=4)

def get_profile_expertise(openreview_client, config):
    group_ids = convert_to_list(config['match_group'])
    valid_members = get_profile_ids(openreview_client, group_ids)

    profiles = openreview_client.search_profiles(ids=valid_members, term=None)

    profiles_expertise = {}
    for profile in profiles:
        profiles_expertise[profile.id] = profile.content.get('expertise', None)

    file_path = Path(profiles_expertise_dir).joinpath('profiles_expertise.json')

    with open(file_path, 'w') as f:
        json.dump(profiles_expertise, f, indent=4)

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

    archive_dir = dataset_dir.joinpath('archives')
    if not archive_dir.is_dir():
        archive_dir.mkdir()

    submission_dir = dataset_dir.joinpath('submissions')
    if not submission_dir.is_dir():
        submission_dir.mkdir()

    bids_dir = dataset_dir.joinpath('bids')
    if not bids_dir.is_dir():
        bids_dir.mkdir()

    assignments_dir = dataset_dir.joinpath('assignments')
    if not assignments_dir.is_dir():
        assignments_dir.mkdir()

    profiles_expertise_dir = dataset_dir.joinpath('profiles_expertise')
    if not profiles_expertise_dir.is_dir():
        profiles_expertise_dir.mkdir()

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

    minimum_timestamp = 0
    if 'oldest_year' in config:
        epoch = datetime.fromtimestamp(0)
        date = datetime.strptime(config['oldest_year'], '%Y')
        minimum_timestamp = (date - epoch).total_seconds() * 1000.0

    print('minimum_timestamp', minimum_timestamp)

    if 'exclusion_inv' in config:
        excluded_ids_by_user = exclude(openreview_client, config)
    else:
        excluded_ids_by_user = defaultdict(list)

    if 'match_group' in config:
        retrieve_expertise(openreview_client, config, excluded_ids_by_user)
        get_profile_expertise(openreview_client, config)

    # if invitation ID is supplied, collect records for each submission
    if 'paper_invitation' in config:
        submissions = get_submissions(openreview_client, config)

    if 'bid_inv' in config:
        get_bids(openreview_client, config)

    if 'assignments' in config:
        get_assignments(openreview_client, config)

    metadata['bid_counts'] = OrderedDict(
        sorted(metadata['bid_counts'].items(), key=lambda t: t[0]))

    metadata['archive_counts'] = OrderedDict(
        sorted(metadata['archive_counts'].items(), key=lambda t: t[0]))

    metadata['reviewer_count'] = len(metadata['archive_counts'])
    metadata['submission_count'] = len(submissions)

    metadata_file = dataset_dir.joinpath('metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
