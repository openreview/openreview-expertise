#!/usr/bin/env python3
"""
Standalone script to set up mock data for container tests.
Extracts and uses the fixture logic from conftest.py without requiring pytest.

This script sets up the required conferences and test data that the container tests expect:
- ABC.cc conference with reviewers and submissions
- DEF.cc conference for dataset tests

Usage:
    python tests/container_tests/setup_container_test_data.py
"""

import openreview
import datetime
import sys
import os
import json
import time
from pathlib import Path

# Add parent directories to path to import test modules
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
project_root = tests_dir.parent

sys.path.insert(0, str(tests_dir))
sys.path.insert(0, str(project_root))

# Import conference locks from tests directory
from tests.conference_locks import conference_lock


class ContainerTestHelpers:
    """Helper class containing extracted logic from conftest.py Helpers class"""
    
    strong_password = 'Or$3cur3P@ssw0rd'

    @staticmethod
    def create_user(email, first, last, alternates=[], institution=None):
        """Create a user profile - extracted from conftest.py"""
        client = openreview.Client(baseurl='http://localhost:3000')
        assert client is not None, "Client is none"
        if openreview.tools.get_profile(client, email) is not None:
            return openreview.Client(baseurl='http://localhost:3000', username=email, password=ContainerTestHelpers.strong_password)
        fullname = f'{first} {last}'
        res = client.register_user(email=email, fullname=fullname, password=ContainerTestHelpers.strong_password)
        username = res.get('id')
        assert res, "Res is none"
        profile_content = {
            'names': [
                {
                    'first': first,
                    'last': last,
                    'username': username
                }
            ],
            'emails': [email] + alternates,
            'preferredEmail': 'info@openreview.net' if email == 'openreview.net' else email
        }
        if institution:
            profile_content['history'] = [{
                'position': 'PhD Student',
                'start': 2017,
                'end': None,
                'institution': {
                    'domain': institution
                }
            }]
        res = client.activate_user(email, profile_content)
        assert res, "Res is none"
        return client

    @staticmethod
    def await_queue(super_client=None):
        """Wait for OpenReview processing queue to complete - extracted from conftest.py"""
        if super_client is None:
            super_client = openreview.Client(baseurl='http://localhost:3000', username='openreview.net', password=ContainerTestHelpers.strong_password)
            assert super_client is not None, 'Super Client is none'

        while True:
            jobs = super_client.get_jobs_status()
            jobCount = 0
            for jobName, job in jobs.items():
                jobCount += job.get('waiting', 0) + job.get('active', 0) + job.get('delayed', 0)

            if jobCount == 0:
                break

            time.sleep(0.5)

        assert not super_client.get_process_logs(status='error')

    @staticmethod
    def post_profiles(client, data):
        """Post user profiles - extracted from conftest.py"""
        for profile_json in data['profiles']:
            if not client.search_profiles(ids=[profile_json['id']]):
                # If the profile hasn't already been posted get the data and create the user
                email = profile_json.get('content').get('preferredEmail') or profile_json.get('content').get('emails')[0]
                first_name = profile_json['id'][1:-1].split('_')[0]
                last_name = profile_json['id'][1:-1].split('_')[-1]
                ContainerTestHelpers.create_user(email, first_name, last_name)

    @staticmethod
    def post_publications(client_v1, client_v2, data, conference_members):
        """Post publications for users - extracted from conftest.py"""
        EXCEPTIONS = ['~Raia_Hadsell1', '~Kyunghyun_Cho1']  # Handled separately
        for profile_json in data['profiles']:
            authorid = profile_json['id']
            emails = client_v2.get_profile(authorid).content.get('emails', [])
            names = [
                o['username'] for o in client_v2.get_profile(authorid).content.get('names', [])
            ]
            if authorid in EXCEPTIONS or (not any(email in conference_members for email in emails) and not any(name in conference_members for name in names)):
                continue
            for idx, pub_json in enumerate(profile_json['publications']):
                content = pub_json['content']
                content['authorids'] = [authorid]
                cdate = pub_json.get('cdate')

                existing_pubs_v2 = list(openreview.tools.iterget_notes(client_v2, content={'authorids': authorid}))
                existing_pubs_v1 = list(openreview.tools.iterget_notes(client_v1, content={'authorids': authorid}))

                existing_titles_v2 = [pub.content.get('title', {}).get('value') for pub in existing_pubs_v2]
                existing_titles_v1 = [pub.content.get('title') for pub in existing_pubs_v1]

                existing_titles = existing_titles_v2 + existing_titles_v1

                # Distribute publications evenly between APIs
                if content.get('title') not in existing_titles:
                    if idx % 2 == 0:
                        note = openreview.Note(
                            invitation='openreview.net/-/paper',
                            readers=['everyone'],
                            writers=['~SomeTest_User1'],
                            signatures=['~SomeTest_User1'],
                            content=content,
                            cdate=cdate
                        )
                        note = client_v1.post_note(note)
                    else:
                        edit = client_v2.post_note_edit(
                            invitation='openreview.net/Archive/-/Direct_Upload',
                            signatures=['~SomeTest_User1'],
                            note=openreview.api.Note(
                                pdate=cdate,
                                content={
                                    'title': {'value': content['title']},
                                    'abstract': {'value': content['abstract']},
                                    'authors': {'value': content['authorids']},
                                    'authorids': {'value': content['authorids']},
                                    'venue': {'value': 'Other Venue 2024 Main'}
                                },
                                license='CC BY-SA 4.0'
                            ))

    @staticmethod
    def post_submissions(data, invitation, api_version=1, datasource_invitation=None):
        """Post submissions for a conference - extracted from conftest.py"""
        if datasource_invitation is None:
            datasource_invitation = invitation

        test_user_client = openreview.Client(username='test@google.com', password=ContainerTestHelpers.strong_password)

        notes = test_user_client.get_all_notes(invitation=invitation)
        if api_version == 1:
            existing_titles = [note.content.get('title') for note in notes]
        elif api_version == 2:
            existing_titles = [note.content.get('title', {}).get('value') for note in notes]

        # All mock data is in API1 format
        if api_version == 1:
            for note_json in data['notes'][datasource_invitation]:
                content = note_json['content']
                content['authors'] = ['SomeTest User']
                content['authorids'] = ['~SomeTest_User1']
                cdate = note_json.get('cdate')

                if content.get('title') not in existing_titles:
                    note = openreview.Note(
                        invitation=invitation,
                        readers=[invitation.split('/')[0], '~SomeTest_User1'],
                        writers=[invitation.split('/')[0], '~SomeTest_User1'],
                        signatures=['~SomeTest_User1'],
                        content=content,
                        cdate=cdate
                    )
                    note = test_user_client.post_note(note)


def setup_openreview_clients():
    """Initialize OpenReview clients"""
    print("Initializing OpenReview clients...")
    client_v1 = openreview.Client(
        baseurl='http://localhost:3000', 
        username='openreview.net', 
        password=ContainerTestHelpers.strong_password
    )
    client_v2 = openreview.api.OpenReviewClient(
        baseurl='http://localhost:3001', 
        username='openreview.net', 
        password=ContainerTestHelpers.strong_password
    )
    
    # Create required test users
    ContainerTestHelpers.create_user('test@mail.com', 'SomeFirstName', 'User')
    ContainerTestHelpers.create_user('test@google.com', 'SomeTest', 'User')
    
    return client_v1, client_v2


def setup_archival_invitation(client_v1):
    """Set up the archival/publication invitation if it doesn't exist"""
    if openreview.tools.get_invitation(client_v1, 'openreview.net/-/paper') is None:
        print("Creating archival invitation...")
        invitation = openreview.Invitation(
            id='openreview.net/-/paper',
            writers=['openreview.net'],
            signatures=['openreview.net'],
            readers=['everyone'],
            invitees=['everyone'],
            reply={
                "forum": None,
                "replyto": None,
                "writers": {
                    "values": ["~SomeTest_User1"]
                },
                "signatures": {
                    "description": "How your identity will be displayed with the above content.",
                    "values": ["~SomeTest_User1"]
                },
                "readers": {
                    "description": "The users who will be allowed to read the above content.",
                    "values": ["everyone"]
                },
                "content": {
                    "title": {
                        "required": False,
                        "order": 1,
                        "description": "Title of paper.",
                        "value-regex": ".{0,100}"
                    },
                    "abstract": {
                        "required": False,
                        "order": 2,
                        "description": "Abstract of paper.",
                        "value-regex": "[\\S\\s]{0,5000}"
                    },
                    "authors": {
                        "required": False,
                        "order": 3,
                        "description": "Comma separated list of author names, as they appear in the paper.",
                        "values-regex": "[^,\\n]*(,[^,\\n]+)*"
                    },
                    "authorids": {
                        "required": False,
                        "order": 4,
                        "description": "Comma separated list of author email addresses, in the same order as above.",
                        "values-regex": "[^,\\n]*(,[^,\\n]+)*"
                    }
                }
            }
        )
        client_v1.post_invitation(invitation)


def create_conference(client_v1, client_v2, conference_id, post_reviewers=True, post_submissions=True, post_publications=True):
    """Create a conference with specified settings - extracted from clean_start_conference logic"""
    print(f"Setting up conference: {conference_id}")
    
    with conference_lock(conference_id, timeout=60) as acquired:
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for conference {conference_id} within 60 seconds")
        
        # Load fake data
        fake_data_path = tests_dir / 'data' / 'fakeData.json'
        with open(fake_data_path) as json_file:
            data = json.load(json_file)
        
        def _populate_groups(committee_name):
            group = openreview.tools.get_group(client_v1, f'{conference_id}/{committee_name}')
            if len(group.members) == 0:
                ContainerTestHelpers.post_profiles(client_v1, data)
                # Use ABC.cc as the source for member data since that's what the fake data contains
                source_id = 'ABC.cc' if 'ABC.cc' in data['groups'] else conference_id.split('/')[0] + '.cc'
                members = data['groups'][f'{source_id}/{committee_name}']['members']
                client_v1.add_members_to_group(f'{conference_id}/{committee_name}', members)
        
        def _post_publications(group_members):
            ContainerTestHelpers.post_publications(client_v1, client_v2, data, group_members)

        def _post_submissions():
            # Use ABC.cc as the source for submission data
            source_invitation = 'ABC.cc/-/Submission'
            ContainerTestHelpers.post_submissions(data, f'{conference_id}/-/Submission', datasource_invitation=source_invitation)

        # Check if conference already exists
        if openreview.tools.get_group(client_v1, conference_id) is not None:
            print(f"Conference {conference_id} already exists, skipping creation...")
        else:
            # Create new conference
            first_element = conference_id.split('/')[0]
            conf_prefix = first_element.split('.')[0]

            pc_email = f'pc@{first_element.lower()}'
            pc_name = f'{conf_prefix.upper()}Chair'
            pc_id = f'~Program_{pc_name}1'
            
            now = datetime.datetime.utcnow()
            due_date = now + datetime.timedelta(days=3)
            first_date = now + datetime.timedelta(days=1)

            pc_client = ContainerTestHelpers.create_user(pc_email, 'Program', pc_name)

            print(f"Creating conference request form for {conference_id}...")
            request_form_note = pc_client.post_note(openreview.Note(
                invitation='openreview.net/Support/-/Request_Form',
                signatures=[pc_id],
                readers=['openreview.net/Support', pc_id],
                writers=[],
                content={
                    'title': conference_id.split('/')[0].replace('/', ' '),
                    'Official Venue Name': conference_id.split('/')[0].replace('/', ' '),
                    'Abbreviated Venue Name': conference_id.split('/')[0].replace('/', ' '),
                    'Official Website URL': 'https://neurips.cc',
                    'program_chair_emails': [pc_email],
                    'contact_email': pc_email,
                    'Area Chairs (Metareviewers)': 'Yes, our venue has Area Chairs',
                    'senior_area_chairs': 'Yes, our venue has Senior Area Chairs',
                    'publication_chairs': 'No, our venue does not have Publication Chairs',
                    'Venue Start Date': '2021/12/01',
                    'Submission Deadline': due_date.strftime('%Y/%m/%d'),
                    'abstract_registration_deadline': first_date.strftime('%Y/%m/%d'),
                    'Location': 'Virtual',
                    'Author and Reviewer Anonymity': 'Double-blind',
                    'reviewer_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                    'area_chair_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                    'senior_area_chair_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                    'Open Reviewing Policy': 'Submissions and reviews should both be private.',
                    'submission_readers': 'Program chairs and paper authors only',
                    'How did you hear about us?': 'ML conferences',
                    'Expected Submissions': '100',
                    'include_expertise_selection': 'No',
                    'submission_reviewer_assignment': 'Automatic',
                    'submission_license': ['CC BY-SA 4.0'],
                    'venue_organizer_agreement': [
                        'OpenReview natively supports a wide variety of reviewing workflow configurations. However, if we want significant reviewing process customizations or experiments, we will detail these requests to the OpenReview staff at least three months in advance.',
                        'We will ask authors and reviewers to create an OpenReview Profile at least two weeks in advance of the paper submission deadlines.',
                        'When assembling our group of reviewers and meta-reviewers, we will only include email addresses or OpenReview Profile IDs of people we know to have authored publications relevant to our venue.  (We will not solicit new reviewers using an open web form, because unfortunately some malicious actors sometimes try to create "fake ids" aiming to be assigned to review their own paper submissions.)',
                        'We acknowledge that, if our venue\'s reviewing workflow is non-standard, or if our venue is expecting more than a few hundred submissions for any one deadline, we should designate our own Workflow Chair, who will read the OpenReview documentation and manage our workflow configurations throughout the reviewing process.',
                        'We acknowledge that OpenReview staff work Monday-Friday during standard business hours US Eastern time, and we cannot expect support responses outside those times.  For this reason, we recommend setting submission and reviewing deadlines Monday through Thursday.',
                        'We will treat the OpenReview staff with kindness and consideration.'
                    ]
                }))

            ContainerTestHelpers.await_queue()

            # Post a deploy note
            print(f"Deploying conference {conference_id}...")
            client_v1.post_note(openreview.Note(
                content={'venue_id': conference_id},
                forum=request_form_note.forum,
                invitation='openreview.net/Support/-/Request{}/Deploy'.format(request_form_note.number),
                readers=['openreview.net/Support'],
                referent=request_form_note.forum,
                replyto=request_form_note.forum,
                signatures=['openreview.net/Support'],
                writers=['openreview.net/Support']
            ))
            ContainerTestHelpers.await_queue()

            # Set up reviewers and other committees
            if post_reviewers:
                print(f"Setting up reviewers for {conference_id}...")
                _populate_groups('Reviewers')
                reviewers = client_v1.get_group(f'{conference_id}/Reviewers')
                if post_publications:
                    print(f"Posting publications for reviewers...")
                    _post_publications(reviewers.members)

            if post_submissions:
                print(f"Setting up submissions for {conference_id}...")
                _post_submissions()
                pc_client = openreview.Client(
                    baseurl='http://localhost:3000', username=pc_email, password=ContainerTestHelpers.strong_password
                )
                post_submission_note = pc_client.post_note(openreview.Note(
                    content={
                        'force': 'Yes',
                        'hide_fields': ['keywords'],
                        'submission_readers': 'All program committee (all reviewers, all area chairs, all senior area chairs if applicable)'
                    },
                    forum=request_form_note.id,
                    invitation=f'openreview.net/Support/-/Request{request_form_note.number}/Post_Submission',
                    readers=[f'{conference_id}/Program_Chairs', 'openreview.net/Support'],
                    referent=request_form_note.id,
                    replyto=request_form_note.id,
                    signatures=[pc_id],
                    writers=[],
                ))
                ContainerTestHelpers.await_queue()

    print(f"Successfully set up conference: {conference_id}")


def setup_abc_conference(client_v1, client_v2):
    """Set up ABC.cc conference with reviewers, submissions, and publications - matches test_expertise_service.py"""
    create_conference(
        client_v1, 
        client_v2, 
        'ABC.cc',
        post_reviewers=True,
        post_submissions=True, 
        post_publications=True
    )


def setup_def_conference(client_v1, client_v2):
    """Set up DEF.cc conference for dataset tests - matches test_create_dataset.py"""
    create_conference(
        client_v1, 
        client_v2, 
        'DEF.cc',
        post_reviewers=True,
        post_submissions=True, 
        post_publications=True
    )


def main():
    """Main entry point"""
    try:
        print("Starting container test data setup...")
        print("=" * 50)
        
        client_v1, client_v2 = setup_openreview_clients()
        
        # Set up archival invitation first
        setup_archival_invitation(client_v1)
        
        print("\nSetting up conferences required by container tests...")
        
        # Set up conferences required by container tests
        setup_abc_conference(client_v1, client_v2)
        setup_def_conference(client_v1, client_v2)
        
        print("\n" + "=" * 50)
        print("Container test data setup completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError setting up container test data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())