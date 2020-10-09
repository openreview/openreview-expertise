'''
A script for generating a dataset.

Assumes that the necessary evidence-collecting steps have been done,
and that papers have been submitted.

'''
from .config import ModelConfig

import json, argparse, csv
from pathlib import Path
from itertools import chain
import openreview
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

class OpenReviewExpertise(object):
    def __init__(self, openreview_client, config):
        self.openreview_client = openreview_client
        self.config = config
        self.root = Path(config.get('dataset', {}).get('directory', './'))
        self.excluded_ids_by_user = defaultdict(list)

        self.metadata = {
            'submission_count': 0,
            'no_publications_count': 0,
            'no_publications': []
        }

    def convert_to_list(self, config_invitations):
        if (isinstance(config_invitations, str)):
            invitations = [config_invitations]
        else:
            invitations = config_invitations
        assert isinstance(invitations, list), 'Input should be a str or a list'
        return invitations

    def get_paper_notes(self, author_id, dataset_params):

        use_bids_as_expertise = dataset_params.get('bid_as_expertise', False)

        if use_bids_as_expertise:
            print('get bids')
            bid_invitation = dataset_params['bid_invitation']
            bids = openreview.tools.iterget_edges(self.openreview_client, invitation=bid_invitation, tail=author_id)
            note_ids = [e.head for e in bids]

            notes = self.openreview_client.get_notes_by_ids(ids=note_ids)
            difference = list(set(note_ids) - set([n.id for n in notes]))
            if difference:
                print('difference', difference)
            return notes

        return openreview.tools.iterget_notes(self.openreview_client, content={'authorids': author_id})


    def get_publications(self, author_id):

        dataset_params = self.config.get('dataset', {})
        minimum_pub_date = dataset_params.get('minimum_pub_date') or dataset_params.get('or', {}).get('minimum_pub_date', 0)
        top_recent_pubs = dataset_params.get('top_recent_pubs') or dataset_params.get('or', {}).get('top_recent_pubs', False)

        publications = self.get_paper_notes(author_id, dataset_params)

        # Get all publications and assign tcdate to cdate in case cdate is None. If tcdate is also None
        # assign cdate := 0
        unsorted_publications = []
        for publication in publications:
            # Check if paper has abstract or title, otherwise continue
            if self.config.get('dataset', {}).get('with_abstract', False):
                if not 'abstract' in publication.content or not publication.content.get('abstract'):
                    continue
            if self.config.get('dataset', {}).get('with_title', False):
                if not 'title' in publication.content or not publication.content.get('title'):
                    continue
            if getattr(publication, 'cdate') is None:
                publication.cdate = getattr(publication, 'tcdate', 0)
            reduced_publication = {
                'id': publication.id,
                'cdate': publication.cdate,
                'content': {
                    'title': publication.content.get('title'),
                    'abstract': publication.content.get('abstract')
                }
            }
            unsorted_publications.append(reduced_publication)

        # If the author does not have publications, then return early
        if not unsorted_publications:
            return unsorted_publications

        # If there is no minimum publication date and no recent publications constraints we return
        # all the publications in any order
        if not top_recent_pubs and not minimum_pub_date:
            return unsorted_publications

        # Sort publications in descending order based on the cdate
        sorted_publications = sorted(unsorted_publications, key=lambda pub: pub.get('cdate'), reverse=True)

        if not top_recent_pubs:
            return [publication for publication in sorted_publications if publication['cdate'] >= minimum_pub_date]

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

        minimum = [publication for publication in sorted_publications if publication['cdate'] >= minimum_pub_date]

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

    def get_profile_ids(self, group_ids=None, reviewer_ids=None):
        """
        Returns a list of all the tilde id members from a list of groups.

        Example:

        >>> get_profiles(['ICLR.cc/2018/Conference/Reviewers'])

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
                group = self.openreview_client.get_group(group_id)
                for member in group.members:
                    if '~' in member:
                        tilde_members.add(member)
                    elif '@' in member:
                        email_members.add(member.lower())

        if reviewer_ids:
            for reviewer_id in reviewer_ids:
                if '~' in reviewer_id:
                    tilde_members.add(reviewer_id)
                elif '@' in reviewer_id:
                    email_members.add(reviewer_id.lower())

        members = []
        tilde_members_list = list(tilde_members)
        profile_search_results = self.openreview_client.search_profiles(emails=None, ids=tilde_members_list, term=None) if tilde_members_list else []
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
        profile_search_results = self.openreview_client.search_profiles(emails=email_members_list, ids=None, term=None) if email_members_list else {}
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

    def exclude(self):
        exclusion_invitations = self.convert_to_list(self.config['exclusion_inv'])
        excluded_ids_by_user = defaultdict(list)
        for invitation in exclusion_invitations:
            user_grouped_edges = openreview.tools.iterget_grouped_edges(
                self.openreview_client,
                invitation=invitation,
                groupby='tail',
                select='id,head,label,weight'
            )

            for edges in user_grouped_edges:
                for edge in edges:
                    excluded_ids_by_user[edge.tail].append(edge.head)

        return excluded_ids_by_user

    def retrieve_expertise_helper(self, member, email):
        self.pbar.update(1)
        member_papers = self.get_publications(member)

        filtered_papers = [
            n for n in member_papers \
            if n['id'] not in self.excluded_ids_by_user[member] \
        ]

        seen_keys = set()
        filtered_papers = []
        for n in member_papers:

            paperhash = openreview.tools.get_paperhash('', n['content']['title'])

            if n['id'] not in self.excluded_ids_by_user[member] and paperhash not in seen_keys:
                filtered_papers.append(n)
                seen_keys.add(paperhash)

        return member, email, filtered_papers


    def retrieve_expertise(self):
        # if group ID is supplied, collect archives for every member
        # (except those whose archives already exist)
        use_email_ids = self.config.get('use_email_ids', False)
        group_ids = self.convert_to_list(self.config.get('match_group', []))
        reviewer_ids = self.convert_to_list(self.config.get('reviewer_ids', []))
        valid_members, invalid_members = self.get_profile_ids(group_ids=group_ids, reviewer_ids=reviewer_ids)

        self.metadata['no_profile'] = invalid_members

        print('finding archives for {} valid members'.format(len(valid_members)))

        expertise = defaultdict(list)
        futures = []
        self.pbar = tqdm(total=len(valid_members), desc='Retrieving expertise...')
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers')) as executor:
            for (member, email) in valid_members:
                futures.append(executor.submit(self.retrieve_expertise_helper, member, email))
        self.pbar.close()

        for future in futures:
            member, email, filtered_papers = future.result()
            member_id = email if use_email_ids else member
            if len(filtered_papers) == 0:
                self.metadata['no_publications_count'] += 1
                self.metadata['no_publications'].append(member_id)
            else:
                for paper in filtered_papers:
                    expertise[member_id].append(paper)

        csv_expertise = self.config.get('csv_expertise')
        if csv_expertise:
            print('adding expertise from csv file ')
            with open(self.root.joinpath(csv_expertise)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for publication in tqdm(csv_reader):
                    member_id = publication[0]
                    _id = publication[1]
                    title = publication[2]
                    abstract = publication[3]
                    expertise[member_id].append({
                        'id': _id,
                        'content': {
                            'title': title,
                            'abstract': abstract
                        }
                    })

        return expertise


    def get_submissions(self):
        invitation_ids = self.convert_to_list(self.config.get('paper_invitation', []))
        submissions = []

        for invitation_id in invitation_ids:
            submissions.extend(list(openreview.tools.iterget_notes(
                self.openreview_client, invitation=invitation_id)))

        print('finding records of {} submissions'.format(len(submissions)))
        reduced_submissions = {}
        for paper in tqdm(submissions, total=len(submissions)):
            paper_id = paper.id
            reduced_submissions[paper_id] = {
                'id': paper_id,
                'content': {
                    'title': paper.content.get('title'),
                    'abstract': paper.content.get('abstract')
                }
            }

        csv_submissions = self.config.get('csv_submissions')
        if csv_submissions:
            print('adding records from csv file ')
            with open(self.root.joinpath(csv_submissions)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for submission in tqdm(csv_reader):
                    paper_id = submission[0]
                    title = submission[1]
                    abstract = submission[2]
                    reduced_submissions[paper_id] = {
                        'id': paper_id,
                        'content': {
                            'title': title,
                            'abstract': abstract
                        }
                    }

        return reduced_submissions

    def run(self):
        dataset_dir = self.config.get('dataset', {}).get('directory')
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path('./')
        if not self.dataset_dir.is_dir():
            self.dataset_dir.mkdir()

        if 'exclusion_inv' in self.config:
            self.excluded_ids_by_user = self.exclude()

        if 'match_group' in self.config or 'reviewer_ids' in self.config:
            self.archive_dir = self.dataset_dir.joinpath('archives')
            if not self.archive_dir.is_dir():
                self.archive_dir.mkdir()
            expertise = self.retrieve_expertise()
            for reviewer_id, pubs in expertise.items():
                with open(self.archive_dir.joinpath(reviewer_id + '.jsonl'), 'w') as f:
                    for paper in pubs:
                        f.write(json.dumps(paper) + '\n')

        # if invitation ID is supplied, collect records for each submission
        if 'paper_invitation' in self.config or 'csv_submissions' in self.config:
            submissions = self.get_submissions()
            with open(self.root.joinpath('submissions.json'), 'w') as f:
                json.dump(submissions, f, indent=2)
            self.metadata['submission_count'] = len(submissions.keys())

        with open(self.root.joinpath('metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)

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

    client = openreview.Client(
        username=args.username,
        password=args.password,
        baseurl=args.baseurl
    )

    expertise = OpenReviewExpertise(client, config)
    expertise.run()
