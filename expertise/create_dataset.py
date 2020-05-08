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

def get_publications(openreview_client, config, author_id):
    content = {
        'authorids': author_id
    }
    publications = openreview.tools.iterget_notes(openreview_client, content=content)

    # Get all publications and assign tcdate to cdate in case cdate is None. If tcdate is also None
    # assign cdate := 0
    unsorted_publications = []
    for publication in publications:
        if getattr(publication, 'cdate') is None:
            publication.cdate = getattr(publication, 'tcdate', 0)
        unsorted_publications.append(publication)

    # If the author does not have publications, then return early
    if not unsorted_publications:
        return unsorted_publications

    dataset_params = config.get('dataset', {})
    minimum_pub_date = dataset_params.get('minimum_pub_date') or dataset_params.get('or', {}).get('minimum_pub_date', 0)
    top_recent_pubs = dataset_params.get('top_recent_pubs') or dataset_params.get('or', {}).get('top_recent_pubs', False)

    # If there is no minimum publication date and no recent publications constraints we return
    # all the publications in any order
    if not top_recent_pubs and not minimum_pub_date:
        return unsorted_publications

    # Sort publications in descending order based on the cdate
    sorted_publications = sorted(unsorted_publications, key=lambda pub: getattr(pub, 'cdate'), reverse=True)

    if not top_recent_pubs:
        return [publication for publication in sorted_publications if publication.cdate >= minimum_pub_date]

    paper_percent = 0
    if isinstance(top_recent_pubs, str) and top_recent_pubs[-1] == '%':
        paper_percent = int(top_recent_pubs[:-1]) / 100
    elif isinstance(top_recent_pubs, int):
        paper_num = top_recent_pubs

    if paper_percent:
        non_int_value = len(sorted_publications) * paper_percent
        # Use remainder to always round up decimals, then convert to int.
        # This is useful if the percentage is 10%, but the user only has 3 publications, for example.
        # 3 * 0.1 = 0.3. So we want to round up to 1 in this case.
        paper_num = int(non_int_value) + (non_int_value % 1 > 0)

    top = sorted_publications[:paper_num]
    if not minimum_pub_date:
        return top

    minimum = [publication for publication in sorted_publications if publication.cdate >= minimum_pub_date]

    # We need to figure out if the constraints have an OR or AND relation
    if dataset_params.get('or', False):
        # minimum and top will have the same pubs because they are sorted, however, one of them
        # is larger than the other. In OR relation we want the largest one as it would be the
        # union between the two sets
        if len(minimum) > len(top):
            return minimum
        return top

    # AND relation
    if len(minimum) > len(top):
        return top
    return minimum

def get_profile_ids(openreview_client, group_ids=None, reviewer_ids=None):
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
    tilde_members = set()
    email_members = set()

    if group_ids:
        for group_id in group_ids:
            group = openreview_client.get_group(group_id)
            for member in group.members:
                if '~' in member:
                    tilde_members.add(member)
                elif '@' in member:
                    email_members.add(member)

    if reviewer_ids:
        for reviewer_id in reviewer_ids:
            if '~' in reviewer_id:
                tilde_members.add(reviewer_id)
            elif '@' in reviewer_id:
                email_members.add(reviewer_id)

    members = []
    tilde_members_list = list(tilde_members)
    profile_search_results = openreview_client.search_profiles(emails=None, ids=tilde_members_list, term=None) if tilde_members_list else []
    tilde_members_list = []
    for profile in profile_search_results:
        preferredEmail = profile.content.get('preferredEmail')
        # If user does not have preferred email, use first email in the emailsConfirmed list
        preferredEmail = preferredEmail or profile.content.get('emailsConfirmed') and len(profile.content.get('emailsConfirmed')) and profile.content.get('emailsConfirmed')[0]
        # If the user does not have emails confirmed, use the first email in the emails list
        preferredEmail = preferredEmail or profile.content.get('emails') and len(profile.content.get('emails')) and profile.content.get('emails')[0]
        # If the user Profile does not have an email, use its Profile ID
        tilde_members_list.append((profile.id, preferredEmail or profile.id))
    members.extend(tilde_members_list)

    email_members_list = list(email_members)
    profile_search_results = openreview_client.search_profiles(emails=email_members_list, ids=None, term=None) if email_members_list else {}
    email_profiles = []
    for email, profile in profile_search_results.items():
        email_profiles.append((profile.id, email))
    members.extend(email_profiles)

    invalid_members = []
    valid_members = list(set(members))
    if len(email_members):
        valid_emails = set([email_id for _, email_id in valid_members])
        for member in email_members:
            if member not in valid_emails:
                invalid_members.append(member)

    return valid_members, invalid_members

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
    group_ids = convert_to_list(config.get('match_group', []))
    reviewer_ids = convert_to_list(config.get('reviewer_ids', []))
    valid_members, invalid_members = get_profile_ids(openreview_client, group_ids=group_ids, reviewer_ids=reviewer_ids)

    metadata['no_profile'] = invalid_members

    print('finding archives for {} valid members'.format(len(valid_members)))

    for (member, email) in tqdm(valid_members, total=len(valid_members)):
        file_path = Path(archive_dir).joinpath((email if use_email_ids else member) + '.jsonl')

        if Path(file_path).exists() and not args.overwrite:
            continue

        member_papers = get_publications(openreview_client, config, member)

        filtered_papers = [
            n for n in member_papers \
            if n.id not in excluded_ids_by_user[member] \
        ]

        seen_keys = []
        filtered_papers = []
        for n in member_papers:
            # Check if paper has abstract or title, otherwise continue
            if config.get('dataset', {}).get('with_abstract', False):
                if not 'abstract' in n.content or not n.content.get('abstract'):
                    continue
            if config.get('dataset', {}).get('with_title', False):
                if not 'title' in n.content or not n.content.get('title'):
                    continue

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

    if 'match_group' in config or 'reviewer_ids' in config:
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
