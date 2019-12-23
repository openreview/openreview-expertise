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
  reduced_submissions = {}
  for invitation_id in invitation_ids:
    submissions = openreview.tools.iterget_notes(client, invitation=invitation_id)
    for submission in submissions:
      reduced_submission = {
        'title': submission.content['title'],
        'abstract': submission.content['abstract']
      }
      reduced_submissions[submission.forum] = reduced_submission
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
  reduced_bids = {}
  for invitation_id in invitation_ids:
    bids = openreview.tools.iterget_tags(client, invitation=invitation_id)
    for bid in bids:
      reduced_bid = {
        'forum': bid.forum,
        'tag': bid.tag
      }
      if (bid.signatures[0] in reduced_bids):
        reduced_bids[bid.signatures[0]].append(reduced_bid)
      else:
        reduced_bids[bid.signatures[0]] = [reduced_bid]
  return reduced_bids
      

def get_profiles(client, group_ids):
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
  for group_id in group_ids:
    group = openreview_client.get_group(group_id)
    profile_members = [member for member in group.members if '~' in member]
    email_members = [member for member in group.members if '@' in member]
    profile_search_results = openreview_client.search_profiles(emails=email_members, ids=None, term=None)

    tilde_members = []
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
    reduced_publication = {
      'forum': publication.forum,
      'title': publication.content['title'],
      'abstract': publication.content.get('abstract', None)
    }
    reduced_publications.append(reduced_publication)
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

  submissions = get_submissions(openreview_client, ['ICLR.cc/2018/Conference/-/Blind_Submission', 'ICLR.cc/2019/Conference/-/Blind_Submission', 'ICLR.cc/2020/Conference/-/Blind_Submission'])
  bids = get_bids(openreview_client, ['ICLR.cc/2018/Conference/-/Add_Bid', 'ICLR.cc/2019/Conference/-/Add_Bid'])
  profiles = get_profiles(openreview_client, ['ICLR.cc/2018/Conference/Reviewers', 'ICLR.cc/2019/Conference/Reviewers'])
  profiles_with_publications = {}
  for profile in profiles:
    profiles_with_publications[profile] = get_publications(openreview_client, profile)
  print(profiles_with_publications)
  