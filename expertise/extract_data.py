import os
import json
import argparse
import openreview

def get_submissions(client, invitation_ids):
  """
    Returns a dictionary with the forum id of the submissions as keys. Each key maps to another dictionary containing the keys title and abstract.

    Example:

    >>> get_submissions(client, invitations=['ICLR.cc/2018/Conference/-/Blind_Submission', 'ICLR.cc/2019/Conference/-/Blind_Submission'])

    :param client: OpenReview Client
    :type client: Client
    :param invitation_ids: List of invitation ids of Submission Invitations
    :type invitation_ids: list[str]

    :return: Dictionary with the forum id of the submission as keys
    :rtype: dict
  """
  reduced_submissions = []
  for invitation_id in invitation_ids:
    submissions = openreview.tools.iterget_notes(client, invitation=invitation_id)
    for submission in submissions:
      reduced_submissions.append({
        'id': submission.id,
        'invitation': submission.invitation,
        'title': submission.content['title'],
        'abstract': submission.content['abstract'],
        'keywords': submission.content.get('keywords'),
        'TL;DR': submission.content.get('TL;DR'),
        'authors': submission.content['authors'],
        'authorids': submission.content['authorids']
      })
  return reduced_submissions

def get_bids(client, invitation_ids):
  """
    Returns a dictionary with the signature of the bids as keys. Each key maps to another dictionary containing the keys forum and tag.

    Example:

    >>> get_bids(openreview_client, ['ICLR.cc/2018/Conference/-/Add_Bid', 'ICLR.cc/2019/Conference/-/Add_Bid'])

    :param client: OpenReview Client
    :type client: Client
    :param invitation_ids: List of invitation ids of the Bids
    :type invitation_ids: list[str]

    :return: Dictionary with the signature of the bids as keys
    :rtype: dict
  """
  reduced_bids = []
  for invitation_id in invitation_ids:
    bids = openreview.tools.iterget_tags(client, invitation=invitation_id)
    for bid in bids:
      reduced_bids.append({
        'id': bid.id,
        'invitation': bid.invitation,
        'forum': bid.forum,
        'tag': bid.tag,
        'signature': bid.signatures[0]
      })
  return reduced_bids

def get_edge_bids(client, invitation_ids):
  """
    Returns a dictionary with the signature of the bids as keys. Each key maps to another dictionary containing the keys forum and tag.

    Example:

    >>> get_edge_bids(openreview_client, ['ICLR.cc/2018/Conference/-/Add_Bid', 'ICLR.cc/2019/Conference/-/Add_Bid'])

    :param client: OpenReview Client
    :type client: Client
    :param invitation_ids: List of invitation ids of the Bids
    :type invitation_ids: list[str]

    :return: Dictionary with the signature of the bids as keys
    :rtype: dict
  """
  reduced_bids = []
  for invitation_id in invitation_ids:
    bids = openreview.tools.iterget_edges(client, invitation=invitation_id)
    for bid in bids:
      reduced_bids.append({
        'id': bid.id,
        'invitation': bid.invitation,
        'forum': bid.head,
        'tag': bid.label,
        'signature': bid.tail
      })
  return reduced_bids


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

def get_publications(client, author_id):
  """
    Returns a list with the containing all the publications of the passed authorid

    Example:

    >>> get_publications(openreview_client, '~Andrew_McCallum1')

    :param client: OpenReview Client
    :type client: Client
    :param author_id: Tilde id of an author
    :type author_id: str

    :return: List with Notes (Publications)
    :rtype: list[Note]
  """
  content = {
    'authorids': author_id
  }
  publications = openreview.tools.iterget_notes(client, content=content)
  reduced_publications = []
  for publication in publications:
    reduced_publications.append({
      'id': publication.id,
      'invitation': publication.invitation,
      'title': publication.content['title'],
      'abstract': publication.content.get('abstract'),
      'keywords': publication.content.get('keywords'),
      'TL;DR': publication.content.get('TL;DR'),
      'authors': publication.content['authors'],
      'authorids': publication.content['authorids']
    })
  return reduced_publications

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--username')
  parser.add_argument('--password')
  parser.add_argument('--baseurl')
  args = parser.parse_args()

  openreview_client = openreview.Client(
    username=args.username,
    password=args.password,
    baseurl=args.baseurl
  )

  print('Get submissions...')
  submissions = get_submissions(openreview_client, ['ICLR.cc/2018/Conference/-/Blind_Submission', 'ICLR.cc/2019/Conference/-/Blind_Submission', 'ICLR.cc/2020/Conference/-/Blind_Submission'])


  with open('./submissions.jsonl', 'w') as f:
    for submission in submissions:
        f.write(json.dumps(submission) + '\n')

  print('Get bids...')
  bids = get_bids(openreview_client, ['ICLR.cc/2018/Conference/-/Add_Bid', 'ICLR.cc/2019/Conference/-/Add_Bid'])
  edge_bids = get_edge_bids(openreview_client, ['ICLR.cc/2020/Conference/Reviewers/-/Bid', 'ICLR.cc/2020/Conference/Area_Chairs/-/Bid'])

  with open('./bids.jsonl', 'w') as f:
    for bid in bids:
        f.write(json.dumps(bid) + '\n')
    for bid in edge_bids:
        f.write(json.dumps(bid) + '\n')

  print('Get reviewer publications...')
  profile_ids = get_profile_ids(openreview_client, ['ICLR.cc/2018/Conference/Reviewers',
  'ICLR.cc/2018/Conference/Area_Chairs',
  'ICLR.cc/2019/Conference/Reviewers',
  'ICLR.cc/2019/Conference/Area_Chairs',
  'ICLR.cc/2020/Conference/Reviewers',
  'ICLR.cc/2020/Conference/Area_Chairs',
  ])

  profiles_with_publications = []
  for profile_id in profile_ids:
    profiles_with_publications.append({
      'user': profile_id,
      'publications': get_publications(openreview_client, profile_id)
    })

  with open('./user_publications.jsonl', 'w') as f:
    for user_publications in profiles_with_publications:
        f.write(json.dumps(user_publications) + '\n')
