'''
A script for generating a dataset.

Assumes that the necessary evidence-collecting steps have been done,
and that papers have been submitted.

'''
import time

from .config import ModelConfig

import json, argparse, csv
from openreview import Note
from pathlib import Path
from itertools import chain
import openreview
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

class OpenReviewExpertise(object):
    def __init__(self, openreview_client, openreview_client_v2, config):
        self.openreview_client = openreview_client
        self.openreview_client_v2 = openreview_client_v2
        self.config = config
        self.root = Path(config.get('dataset', {}).get('directory', './'))
        self.excluded_ids_by_user = defaultdict(list)
        self.included_ids_by_user = defaultdict(list)
        self.alternate_excluded_ids_by_user = defaultdict(list)
        self.alternate_included_ids_by_user = defaultdict(list)

        self.metadata = {
            'submission_count': 0,
            'no_publications_count': 0,
            'no_publications': []
        }

        self.venue_list = openreview_client_v2.get_group('venues').members

        ModelConfig.validate_weight_specification(self.config)

    def convert_to_list(self, config_invitations):
        if (isinstance(config_invitations, str)):
            invitations = [config_invitations]
        else:
            invitations = config_invitations
        assert isinstance(invitations, list), 'Input should be a str or a list'
        return invitations
    
    def match_content(self, content, match_content):
        if not match_content:
            return True
        for name, value in match_content.items():

            paper_value = content.get(name)
            if isinstance(paper_value, dict):
                paper_value = paper_value.get('value')

            if paper_value != value:
                return False
        return True

    def get_paper_notes(self, author_id, dataset_params):

        use_bids_as_expertise = dataset_params.get('bid_as_expertise', False)

        if use_bids_as_expertise:
            bid_invitation = dataset_params['bid_invitation']
            paper_invitation = self.config['paper_invitation']
            bids = openreview.tools.iterget_edges(self.openreview_client, invitation=bid_invitation, tail=author_id)
            note_ids = [e.head for e in bids if e.label in ['Very High', 'High']]
            return [n for n in self.openreview_client.get_notes_by_ids(ids=note_ids) if n.invitation == paper_invitation]

        notes_v1 = list(openreview.tools.iterget_notes(self.openreview_client, content={'authorids': author_id}))
        notes_v1 = [note for note in notes_v1 if note.readers == ['everyone']]
        notes_v2 = list(openreview.tools.iterget_notes(self.openreview_client_v2, content={'authorids': author_id}))
        notes_v2 = [note for note in notes_v2 if note.readers == ['everyone']]
        return notes_v1 + notes_v2

    def deduplicate_publications(self, publications):
        deduplicated = []

        # Build index of pub.original
        original_ids = {getattr(pub, 'original', None) for pub in publications}

        for pub in publications:
            # Keep all blind notes, and keep originals that do not have a corresponding blind
            if pub.id not in original_ids:
                deduplicated.append(pub)

        return deduplicated
    
    def get_pub_weight(self, pub=None, weight_specification=[]):
        def _matches(venue_spec, venueid, domain):
            in_openreview = venueid == domain and domain in self.venue_list
            ## Papers allowed: accepted papers from an OpenReview venue
            ## not in_openreview: DBLP papers (venueid =/= domain) and non-accepted papers (domain not in venue_list)

            return (
                ('prefix' in venue_spec and in_openreview and venueid.startswith(venue_spec['prefix'])) or
                ('value' in venue_spec and in_openreview and venueid == venue_spec['value']) or
                ('inOpenReview' in venue_spec and in_openreview and venue_spec['inOpenReview']) or
                ('inOpenReview' in venue_spec and not in_openreview and not venue_spec['inOpenReview'])
            )
        
        venueid = pub.content.get('venueid', '')
        if isinstance(venueid, dict):
            venueid = venueid.get('value')
        
        ## If weight specification but no venueid, return default weight
        if not venueid:
            return 1

        # Get domain from either domain field or invitation prefix
        domain = getattr(pub, 'domain', None)
        if domain is None:
            domain = pub.invitation.split('/-/')[0]
            
        # Find matching weight specification
        current_weight, current_order = 1, None ## Default weight one
        for idx, venue_spec in enumerate(weight_specification):
            if _matches(venue_spec, venueid, domain):
                order = venue_spec.get('order', idx) ## Support optional order, fallback to index
                if current_order is None or order <= current_order:
                    current_weight = venue_spec['weight']
                    current_order = order

        return current_weight

    def get_publications(self, author_id):

        dataset_params = self.config.get('dataset', {})
        weight_specification = dataset_params.get('weight_specification', [])
        minimum_pub_date = dataset_params.get('minimum_pub_date') or dataset_params.get('or', {}).get('minimum_pub_date', 0)
        top_recent_pubs = dataset_params.get('top_recent_pubs') or dataset_params.get('or', {}).get('top_recent_pubs', False)
        publications = self.deduplicate_publications(
            self.get_paper_notes(author_id, dataset_params)
        )

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
            if getattr(publication, 'pdate') is not None:
                publication.cdate = publication.pdate
            if getattr(publication, 'cdate') is None:
                publication.cdate = getattr(publication, 'tcdate', 0)

            publication.mdate = getattr(publication, 'tmdate', int(time.time()))

            # Get title + abstract depending on API version
            pub_title = publication.content.get('title', '')
            if isinstance(pub_title, dict):
                pub_title = pub_title.get('value')

            pub_abstr = publication.content.get('abstract', '')
            if isinstance(pub_abstr, dict):
                pub_abstr = pub_abstr.get('value')

            # Compare venueid to domain/invitation prefix to determine acceptance
            pub_weight = self.get_pub_weight(pub=publication, weight_specification=weight_specification)

            reduced_publication = {
                'id': publication.id,
                'cdate': publication.cdate,
                'mdate': publication.mdate,
                'content': {
                    'title': pub_title,
                    'abstract': pub_abstr
                }
            }
            if weight_specification:
                reduced_publication['content']['weight'] = pub_weight
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

        emails_confirmed_set = set()
        members = []
        tilde_members_list = list(tilde_members)
        profile_search_results = self.openreview_client.search_profiles(confirmedEmails=None, ids=tilde_members_list, term=None) if tilde_members_list else []
        tilde_members_list = []
        for profile in profile_search_results:
            preferredEmail = profile.content.get('preferredEmail')
            # If user does not have preferred email, use first email in the emailsConfirmed list
            preferredEmail = preferredEmail or profile.content.get('emailsConfirmed') and len(profile.content.get('emailsConfirmed')) and profile.content.get('emailsConfirmed')[0]
            # If the user does not have emails confirmed, use the first email in the emails list
            preferredEmail = preferredEmail or profile.content.get('emails') and len(profile.content.get('emails')) and profile.content.get('emails')[0]
            # If the user Profile does not have an email, use its Profile ID
            tilde_members_list.append((profile.id, preferredEmail or profile.id))
            if profile.content.get('emailsConfirmed'):
                for email in profile.content.get('emailsConfirmed'):
                    emails_confirmed_set.add(email)
        members.extend(tilde_members_list)

        email_members_list = list(email_members)
        profile_search_results = self.openreview_client.search_profiles(confirmedEmails=email_members_list, ids=None, term=None) if email_members_list else {}
        email_profiles = []
        for email, profile in profile_search_results.items():
            email_profiles.append((profile.id, email))
            if profile.content.get('emailsConfirmed'):
                for email in profile.content.get('emailsConfirmed'):
                    emails_confirmed_set.add(email)
        members.extend(email_profiles)

        invalid_members = []
        valid_members = list(set(members))
        if len(email_members):
            for member in email_members:
                if member not in emails_confirmed_set:
                    invalid_members.append(member)

        return valid_members, invalid_members

    def get_expertise_selection_edges(self, invitation_key, label=None):
        edge_invitations = self.convert_to_list(self.config[invitation_key])
        selected_ids_by_user = defaultdict(list)
        for invitation in edge_invitations:
            
            user_grouped_edges = self.openreview_client.get_grouped_edges(
                invitation=invitation,
                groupby='tail',
                select='id,head,label,weight,tail'
            )

            for edges in user_grouped_edges:
                for edge in edges['values']:
                    if not label or (label and edge.get('label') == label):
                        selected_ids_by_user[edge.get('tail')].append(edge.get('head'))

        return selected_ids_by_user

    def exclude(self):
        return self.get_expertise_selection_edges('exclusion_inv', 'Exclude')
    
    def alternate_exclude(self):
        return self.get_expertise_selection_edges('alternate_exclusion_inv', 'Exclude')
    
    def include(self):
        return self.get_expertise_selection_edges('inclusion_inv', 'Include')
    
    def alternate_include(self):
        return self.get_expertise_selection_edges('alternate_inclusion_inv', 'Include')

    def retrieve_expertise_helper(self, member, email):
        self.pbar.update(1)
        member_papers = self.get_publications(member)

        include_all_papers = False
        if 'inclusion_inv' in self.config and len(self.included_ids_by_user[member]) > 0 and not any(paper['id'] in self.included_ids_by_user[member] for paper in member_papers):
            include_all_papers = True

        seen_keys = set()
        filtered_papers = []

        print(f"{member} include/exclude")
        print(self.included_ids_by_user.get(member))
        print(self.excluded_ids_by_user.get(member))

        for n in member_papers:
            paper_title = openreview.tools.get_paperhash('', n['content']['title'])
            if 'inclusion_inv' in self.config:
                # The paper must be included or the user has included no papers
                if paper_title and ((n['id'] in self.included_ids_by_user[member] or len(self.included_ids_by_user[member]) == 0) or include_all_papers):
                    filtered_papers.append(n)
            else:
                if paper_title and n['id'] not in self.excluded_ids_by_user[member] and paper_title not in seen_keys:
                    filtered_papers.append(n)
            seen_keys.add(paper_title)

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

        # Raise an error if none of the values in expertise have papers
        if not any(expertise.values()):
            raise ValueError('Not Found Error: No publications found, please ensure members have public and published papers attached to their profiles')

        return expertise

    def get_papers_from_group(self, submission_groups):
        submission_groups = self.convert_to_list(submission_groups)
        client = self.openreview_client

        # Cast from [(tilde_id, email)] -> [tilde_id]
        group_a_members, invalid_members = self.get_profile_ids(group_ids=submission_groups)
        group_a_members = [member_pair[0] for member_pair in group_a_members]
        self.metadata['no_profile_submission'] = invalid_members 

        pbar = tqdm(total=len(group_a_members), desc='submissions status...')
        publications_by_profile_id = {}
        all_papers = []

        def get_status(profile_id):
            pbar.update(1)
            profile = openreview.tools.get_profile(client, profile_id)
            if profile:
                publications = [
                    openreview.Note.from_json(n) for n in self.get_publications(profile.id)
                ]
                return { 'profile_id': profile_id, 'papers': publications }
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers')) as executor:
            for profile_id in group_a_members:
                futures.append(executor.submit(get_status, profile_id))
        pbar.close()

        for future in futures:
            result = future.result()
            # Convert publications to json
            papers_json_list = []
            filtered_papers = []
            seen_keys = set()
            for paper_note in result['papers']:
                paper_hash = openreview.tools.get_paperhash('', paper_note.content['title'])
                if 'alternate_inclusion_inv' in self.config:
                    if paper_hash and (paper_note.id in self.alternate_included_ids_by_user[result['profile_id']] or len(self.alternate_included_ids_by_user[result['profile_id']]) == 0):
                        filtered_papers.append(paper_note)
                        papers_json_list.append(paper_note.to_json())
                else:
                    if paper_hash and paper_note.id not in self.alternate_excluded_ids_by_user[result['profile_id']] and paper_hash not in seen_keys:
                        filtered_papers.append(paper_note)
                        papers_json_list.append(paper_note.to_json())
                seen_keys.add(paper_hash)
            publications_by_profile_id[result['profile_id']] = papers_json_list
            all_papers = all_papers + filtered_papers

        # Dump publications by profile id
        with open(self.root.joinpath('publications_by_profile_id.json'), 'w') as f:
            json.dump(publications_by_profile_id, f, indent=2)
        
        return all_papers
    
    def _validate_paper_data(self,
        reduced_submissions,
        invitation_ids=None,
        paper_id=None,
        paper_venueid=None,
        paper_content=None,
        submission_groups=None,
    ):
        err_string = 'Not Found Error: No papers found for: '

        if sum(len(v) for v in reduced_submissions.values()) == 0:
            args_strings = []
            if invitation_ids:
                args_strings.append(f'invitation_ids: {invitation_ids}')
            if paper_id:
                args_strings.append(f'paper_id: {paper_id}')
            if paper_venueid:
                args_strings.append(f'paper_venueid: {paper_venueid}')
            if paper_content:
                args_strings.append(f'paper_content: {paper_content}')
            if submission_groups:
                args_strings.append(f'submission_groups: {submission_groups}')
                err_string = 'Not Found Error: No publications found for: '

            err_string += ', '.join(args_strings)
            
            raise ValueError(err_string)

    def get_submissions_helper(self, 
        invitation_ids=None,
        paper_id=None,
        paper_venueid=None,
        paper_content=None,
        submission_groups=None,
    ):
        submissions = []
        # Fetch papers from alternate match group
        # If no alternate match group provided, aggregate papers from all other sources
        if submission_groups:
            aggregate_papers = self.get_papers_from_group(submission_groups)
            submissions.extend(aggregate_papers)
        else:
            if invitation_ids:
                for invitation_id in invitation_ids:
                    # Assume invitation is valid for both APIs, but only 1
                    # will have the associated notes
                    submissions_v1 = self.openreview_client.get_all_notes(invitation=invitation_id, content={'venueid': paper_venueid})

                    submissions.extend(submissions_v1)
                    submissions.extend(self.openreview_client_v2.get_all_notes(invitation=invitation_id, content={'venueid': paper_venueid}))
            elif paper_venueid:
                submissions_v1 = self.openreview_client.get_all_notes(content={'venueid': paper_venueid})

                submissions.extend(submissions_v1)
                submissions.extend(self.openreview_client_v2.get_all_notes(content={'venueid': paper_venueid}))

            if paper_id:
                # If note not found, keep executing and raise an overall exception later
                # Otherwise if the exception is anything else, raise it again
                note_v1, note_v2 = None, None
                try:
                    note_v1 = self.openreview_client.get_note(paper_id)
                    submissions.append(note_v1)
                except openreview.OpenReviewException as e:
                    err_name = e.args[0].get('name').lower()
                    if err_name != 'notfounderror':
                        raise e

                try:
                    note_v2 = self.openreview_client_v2.get_note(paper_id)
                    submissions.append(note_v2)
                except openreview.OpenReviewException as e:
                    err_name = e.args[0].get('name').lower()
                    if err_name != 'notfounderror':
                        raise e

                if not note_v1 and not note_v2:
                    raise openreview.OpenReviewException(f"Note {paper_id} not found")

        print('finding records of {} submissions'.format(len(submissions)))
        reduced_submissions = {}
        for paper in tqdm(submissions, total=len(submissions)):
            paper_id = paper.id

            if self.match_content(paper.content, paper_content):

                # Get title + abstract depending on API version
                paper_title = paper.content.get('title')
                if isinstance(paper_title, dict):
                    paper_title = paper_title.get('value')

                paper_abstr = paper.content.get('abstract')
                if isinstance(paper_abstr, dict):
                    paper_abstr = paper_abstr.get('value')

                reduced_submissions[paper_id] = {
                    'id': paper_id,
                    'content': {
                        'title': paper_title,
                        'abstract': paper_abstr
                    }
                }

        return reduced_submissions

    def get_match_submissions(self):
        invitation_ids = self.convert_to_list(self.config.get('match_paper_invitation', []))
        paper_id = self.config.get('match_paper_id')
        paper_venueid = self.config.get('match_paper_venueid', None)
        paper_content = self.config.get('match_paper_content', None)

        reduced_submissions = self.get_submissions_helper(
            invitation_ids=invitation_ids,
            paper_id=paper_id,
            paper_venueid=paper_venueid,
            paper_content=paper_content
        )

        self._validate_paper_data(
            reduced_submissions,
            invitation_ids=invitation_ids,
            paper_id=paper_id,
            paper_venueid=paper_venueid,
            paper_content=paper_content
        )

        return reduced_submissions
    

    def get_submissions(self):
        invitation_ids = self.convert_to_list(self.config.get('paper_invitation', []))
        paper_id = self.config.get('paper_id')
        paper_venueid = self.config.get('paper_venueid', None)
        paper_content = self.config.get('paper_content', None)
        submission_groups = self.convert_to_list(self.config.get('alternate_match_group', []))

        reduced_submissions = self.get_submissions_helper(
            invitation_ids=invitation_ids,
            paper_id=paper_id,
            paper_venueid=paper_venueid,
            paper_content=paper_content,
            submission_groups=submission_groups
        )

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

        self._validate_paper_data(
            reduced_submissions,
            invitation_ids=invitation_ids,
            paper_id=paper_id,
            paper_venueid=paper_venueid,
            paper_content=paper_content,
        )

        return reduced_submissions

    def run(self):
        dataset_dir = self.config.get('dataset', {}).get('directory')
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path('./')
        if not self.dataset_dir.is_dir():
            self.dataset_dir.mkdir()

        if 'exclusion_inv' in self.config:
            self.excluded_ids_by_user = self.exclude()

        if 'inclusion_inv' in self.config:
            self.included_ids_by_user = self.include()

        if 'alternate_exclusion_inv' in self.config:
            self.alternate_excluded_ids_by_user = self.alternate_exclude()

        if 'alternate_inclusion_inv' in self.config:
            self.alternate_included_ids_by_user = self.alternate_include()


        if 'match_group' in self.config or 'reviewer_ids' in self.config:
            self.archive_dir = self.dataset_dir.joinpath('archives')
            if not self.archive_dir.is_dir():
                self.archive_dir.mkdir()
            expertise = self.retrieve_expertise()
            for reviewer_id, pubs in expertise.items():
                with open(self.archive_dir.joinpath(reviewer_id + '.jsonl'), 'w') as f:
                    for paper in pubs:
                        f.write(json.dumps(paper) + '\n')

        if 'match_paper_invitation' in self.config or 'match_paper_id' in self.config or 'match_paper_venueid' in self.config:
            self.archive_dir = self.dataset_dir.joinpath('archives')
            if not self.archive_dir.is_dir():
                self.archive_dir.mkdir()
            papers = self.get_match_submissions()
            with open(self.archive_dir.joinpath('match_submissions.jsonl'), 'w') as f:
                for paper in papers.values():
                    f.write(json.dumps(paper) + '\n')

        # Retrieve match groups to detect group-group matching
        group_group_matching = 'alternate_match_group' in self.config.keys()

        # if invitation ID is supplied, collect records for each submission
        if 'paper_invitation' in self.config or 'csv_submissions' in self.config or 'paper_id' in self.config or 'paper_venueid' in self.config or group_group_matching:
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
    parser.add_argument('--baseurl_v2')
    args = parser.parse_args()

    config = ModelConfig(config_file_path=args.config)

    print(config)

    client = openreview.Client(
        username=args.username,
        password=args.password,
        baseurl=args.baseurl
    )

    client_v2 = openreview.api.OpenReviewClient(
        username=args.username,
        password=args.password,
        baseurl=args.baseurl_v2
    )

    expertise = OpenReviewExpertise(client, client_v2, config)
    expertise.run()
