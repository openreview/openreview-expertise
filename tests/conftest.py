import openreview
import datetime
import pytest
import requests
import time
import json
from tests.conference_locks import conference_lock

from openreview.api import OpenReviewClient
from openreview.api import Note
from openreview.api import Group
from openreview.api import Invitation
from openreview.api import Edge
from openreview.venue import Venue
from openreview.journal import Journal

class Helpers:
    strong_password = 'Or$3cur3P@ssw0rd'

    @staticmethod
    def create_user(email, first, last, alternates=[], institution=None):
        client = openreview.Client(baseurl = 'http://localhost:3000')
        assert client is not None, "Client is none"
        if openreview.tools.get_profile(client, email) is not None:
            return openreview.Client(baseurl = 'http://localhost:3000', username = email, password = Helpers.strong_password)
        fullname = f'{first} {last}'
        res = client.register_user(email=email, fullname=fullname, password=Helpers.strong_password)
        username = res.get('id')
        assert res, "Res i none"
        profile_content={
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
        assert res, "Res i none"
        return client

    @staticmethod
    def get_user(email):
        return openreview.Client(baseurl = 'http://localhost:3000', username = email, password = Helpers.strong_password)

    @staticmethod
    def await_queue(super_client=None):
        if super_client is None:
            super_client = openreview.Client(baseurl='http://localhost:3000', username='openreview.net', password=Helpers.strong_password)
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
    def await_queue_edit(super_client, edit_id=None, invitation=None, count=1, error=False):
        expected_status = 'error' if error else 'ok'
        while True:
            process_logs = super_client.get_process_logs(id=edit_id, invitation=invitation)
            if len(process_logs) >= count and all(process_log['status'] == expected_status for process_log in process_logs):
                break

            time.sleep(0.5)

        assert process_logs[0]['status'] == (expected_status), process_logs[0]['log']


    @staticmethod
    def create_reviewer_edge(client, conference, name, note, reviewer, label=None, weight=None):
        conference_id = conference.id
        sac = [conference.get_senior_area_chairs_id(number=note.number)] if conference.use_senior_area_chairs else []
        return client.post_edge(openreview.Edge(
            invitation=f'{conference.id}/Reviewers/-/{name}',
            readers = [conference_id] + sac + [conference.get_area_chairs_id(number=note.number), reviewer] ,
            nonreaders = [conference.get_authors_id(number=note.number)],
            writers = [conference_id] + sac + [conference.get_area_chairs_id(number=note.number)],
            signatures = [conference_id],
            head = note.id,
            tail = reviewer,
            label = label,
            weight = weight
        ))
    
    @staticmethod
    def post_profiles(client, data):
        for profile_json in data['profiles']:
            if not client.search_profiles(ids=[profile_json['id']]):
                # If the profile hasn't already been posted get the data and create the user
                email = profile_json.get('content').get('preferredEmail') or profile_json.get('content').get('emails')[0]
                first_name = profile_json['id'][1:-1].split('_')[0]
                last_name = profile_json['id'][1:-1].split('_')[-1]
                Helpers.create_user(email, first_name, last_name)
    
    @staticmethod
    def post_publications(client_v1, client, data, conference_members):
        EXCEPTIONS = ['~Raia_Hadsell1', '~Kyunghyun_Cho1'] ## Handled separately
        for profile_json in data['profiles']:
            authorid = profile_json['id']
            emails = client.get_profile(authorid).content.get('emails', [])
            names = [
                o['username'] for o in client.get_profile(authorid).content.get('names', [])
            ]
            if authorid in EXCEPTIONS or (not any(email in conference_members for email in emails) and not any(name in conference_members for name in names)):
                continue
            for idx, pub_json in enumerate(profile_json['publications']):
                content = pub_json['content']
                content['authorids'] = [authorid]
                cdate = pub_json.get('cdate')

                existing_pubs_v2 = list(openreview.tools.iterget_notes(client, content={'authorids': authorid}))
                existing_pubs_v1 = list(openreview.tools.iterget_notes(client_v1, content={'authorids': authorid}))

                existing_titles_v2 = [pub.content.get('title', {}).get('value') for pub in existing_pubs_v2]
                existing_titles_v1 = [pub.content.get('title') for pub in existing_pubs_v1]

                existing_titles = existing_titles_v2 + existing_titles_v1

                # Distribute publications evenly between APIs
                if content.get('title') not in existing_titles:
                    if idx % 2 == 0:
                        note = openreview.Note(
                            invitation = 'openreview.net/-/paper',
                            readers = ['everyone'],
                            writers = ['~SomeTest_User1'],
                            signatures = ['~SomeTest_User1'],
                            content = content,
                            cdate = cdate
                        )
                        note = client_v1.post_note(note)
                    else:
                        edit = client.post_note_edit(
                            invitation='openreview.net/Archive/-/Direct_Upload',
                            signatures = ['~SomeTest_User1'],
                            note = openreview.api.Note(
                                pdate = cdate,
                                content = {
                                    'title': { 'value': content['title'] },
                                    'abstract': { 'value': content['abstract'] },
                                    'authors': { 'value': content['authorids'] },
                                    'authorids': { 'value': content['authorids'] },
                                    'venue': { 'value': 'Other Venue 2024 Main' }
                                },
                                license = 'CC BY-SA 4.0'
                        ))

    @staticmethod
    def post_submissions(data, invitation, api_version=1, datasource_invitation=None):
        if datasource_invitation is None:
            datasource_invitation = invitation

        test_user_client = openreview.Client(username='test@google.com', password=Helpers.strong_password)

        notes = test_user_client.get_all_notes(invitation=invitation)
        if api_version == 1:
            existing_titles = [note.content.get('title') for note in notes]
        elif api_version == 2:
            existing_titles = [note.content.get('title', {}).get('value') for note in notes]

        ## All mock data is in API1 format
        if api_version == 1:
            for note_json in data['notes'][datasource_invitation]:
                content = note_json['content']
                content['authors'] = ['SomeTest User']
                content['authorids'] = ['~SomeTest_User1']
                cdate = note_json.get('cdate')

                if content.get('title') not in existing_titles:
                    note = openreview.Note(
                        invitation = invitation,
                        readers = [invitation.split('/')[0], '~SomeTest_User1'],
                        writers = [invitation.split('/')[0], '~SomeTest_User1'],
                        signatures = ['~SomeTest_User1'],
                        content = content,
                        cdate = cdate
                    )
                    note = test_user_client.post_note(note)
        elif api_version == 2:
            test_client_v2 = openreview.api.OpenReviewClient(username='test@mail.com', password=Helpers.strong_password)

            for note_json in data['notes'][datasource_invitation]:
                content = note_json['content']
                cdate = note_json.get('cdate')

                if content.get('title') not in existing_titles:
                    submission_note = test_client_v2.post_note_edit(
                        invitation = invitation,
                        signatures = ['~SomeFirstName_User1'],
                        note = Note(
                            content = {
                                'title': { 'value': content.get('title').get('value') },
                                'abstract': { 'value': content.get('abstract').get('value') },
                                'venueid': { 'value': content.get('venueid', {}).get('value')},
                                'authors': { 'value': ['Test User']},
                                'authorids': { 'value': ['~SomeFirstName_User1']},
                                'pdf': {'value': '/pdf/' + 'p' * 40 +'.pdf' },
                                'supplementary_material': { 'value': '/attachment/' + 's' * 40 +'.zip'},
                                'competing_interests': { 'value': 'None beyond the authors normal conflict of interests'},
                                'human_subjects_reporting': { 'value': 'Not applicable'}
                            }
                        )
                    )
                
    @staticmethod
    def post_expertise_publication(client, user, conference_id, committee_name=None, edge_label='Include', api_version=2):

        test_title = f"{user} {conference_id}"
        test_abstract = f"This is a test abstract for {test_title}"

        if api_version == 2:
            content = {
                'title': { 'value': test_title },
                'abstract': { 'value': test_abstract },
                'authors': { 'value': [user] },
                'authorids': { 'value': [user] }
            }
            note_edit = client.post_note_edit(
                invitation='openreview.net/Archive/-/Direct_Upload',
                signatures = [user],
                note = Note(
                    pdate = 1554819115,
                    content = content,
                    license = 'CC BY-SA 4.0'
                )
            )
            edge = openreview.api.Edge(
                invitation=f'{conference_id}/{committee_name}/-/Expertise_Selection',
                head=note_edit['note']['id'],
                tail=user,
                label=edge_label,
                readers=[conference_id, user],
                writers=[user],
                signatures=[user]
            )
            edge = client.post_edge(edge)
        elif api_version == 1:
            content = {
                'title': test_title,
                'abstract': test_abstract,
                'authors': [user],
                'authorids': [user]
            }
            note = client.post_note(
                openreview.Note(
                    invitation='openreview.net/-/paper',
                    readers = ['everyone'],
                    writers = ['~SomeTest_User1'],
                    signatures = ['~SomeTest_User1'],
                    content = content,
                    cdate = 1554819115
                )
            )
            edge = openreview.Edge(
                invitation=f'{conference_id}/-/Expertise_Selection',
                head=note.id,
                tail=user,
                label=edge_label,
                readers=[conference_id, user],
                writers=[user],
                signatures=[user]
            )
            edge = client.post_edge(edge)
                
    @staticmethod
    def post_editor_data(client, data, editors):
        for profile_json in data['profiles']:
            if profile_json['id'] not in editors:
                continue

            authorid = profile_json['id']
            name = ' '.join(authorid[1:-1].split('_'))
            for pub_json in profile_json['publications']:
                content = pub_json['content']
                cdate = pub_json.get('cdate')

                existing_pubs = list(openreview.tools.iterget_notes(client, content={'authorids': authorid}))
                existing_titles = [pub.content.get('title') for pub in existing_pubs]

                if content.get('title') not in existing_titles:
                    submission_note = client.post_note_edit(
                        invitation = 'TMLR/-/Submission',
                        signatures = ['~Super_User1'],
                        note = Note(
                            #cdate = cdate,
                            content = {
                                'title': { 'value': content.get('title').get('value') },
                                'abstract': { 'value': content.get('abstract').get('value') },
                                'authors': { 'value': [name]},
                                'authorids': { 'value': [authorid]},
                                'pdf': {'value': '/pdf/' + 'p' * 40 +'.pdf' },
                                'supplementary_material': { 'value': '/attachment/' + 's' * 40 +'.zip'},
                                'competing_interests': { 'value': 'None beyond the authors normal conflict of interests'},
                                'human_subjects_reporting': { 'value': 'Not applicable'}
                            }
                        ))
                    publication_note = client.post_note_edit(
                        invitation = 'openreview.net/Archive/-/Direct_Upload',
                        signatures = [authorid],
                        note = Note(
                            pdate = cdate,
                            content = {
                                'title': { 'value': content.get('title').get('value') },
                                'abstract': { 'value': content.get('abstract').get('value') },
                                'authors': { 'value': [name]},
                                'authorids': { 'value': [authorid]},
                            },
                            license = 'CC BY-SA 4.0'
                        ))

@pytest.fixture(scope="class")
def helpers():
    return Helpers

@pytest.fixture(scope="session")
def test_client():
    Helpers.create_user('test@mail.com', 'SomeFirstName', 'User')
    yield openreview.Client(baseurl = 'http://localhost:3000', username='test@mail.com', password=Helpers.strong_password)

@pytest.fixture(scope="session")
def test_google_user():
    Helpers.create_user('test@google.com', 'SomeTest', 'User')
    yield openreview.Client(baseurl = 'http://localhost:3000', username='test@google.com', password=Helpers.strong_password)

@pytest.fixture(scope="session")
def client():
    yield openreview.Client(baseurl = 'http://localhost:3000', username='openreview.net', password=Helpers.strong_password)

@pytest.fixture(scope="session")
def openreview_client():
    yield openreview.api.OpenReviewClient(baseurl = 'http://localhost:3001', username='openreview.net', password=Helpers.strong_password)

@pytest.fixture(scope="session")
def clean_start_journal(client, openreview_client, test_google_user, test_client):
    def _start_journal(
        openreview_client,
        journal_id,
        editors=[],
        additional_editors=[],
        post_submissions=False,
        post_publications=False,
        post_editor_data=False
    ):
        # Use the conference lock to ensure only one thread can write to this journal at a time
        with conference_lock(journal_id, timeout=60) as acquired:
            if not acquired:
                raise TimeoutError(f"Could not acquire lock for journal {journal_id} within 60 seconds")
            
            def _post_submissions():
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_submissions(data, f'{journal_id}/-/Submission', api_version=2)

            def _post_publications(committee_name):
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                group = openreview.tools.get_group(client, f'{journal_id}/{committee_name}')
                Helpers.post_publications(client, openreview_client, data, group.members)
            
            def _post_profiles():
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_profiles(client, data)

            def _handle_editor_data():
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_editor_data(openreview_client, data, editors)

            _post_profiles()
            
            first_element = journal_id.split('/')[0]
            conf_prefix = first_element.split('.')[0]

            eic_email = f'eic@{first_element.lower()}.org'
            eic_name = f'{conf_prefix.upper()}Chair'
            eic_id = f'~Editor_{eic_name}1'

            eic_client=Helpers.create_user(
                eic_email,
                'Editor',
                eic_name
            )
            journal=Journal(
                openreview_client,
                journal_id,
                '1234',
                contact_info=f'{eic_email}',
                full_name=f'{conf_prefix.upper()} Journal',
                short_name=f'{conf_prefix.lower()}',
                submission_name='Submission'
            )

            journal.setup(support_role=eic_id, editors=editors)

            openreview_client.add_members_to_group(f'{journal_id}/Action_Editors', editors + additional_editors)
            openreview_client.add_members_to_group(f'{journal_id}/Reviewers', editors + additional_editors)

            if post_submissions:
                _post_submissions()

            if post_publications:
                _post_publications('Action_Editors')
                _post_publications('Reviewers')

            if post_editor_data:
                _handle_editor_data()

            return journal
    return _start_journal

@pytest.fixture(scope="session")
def clean_start_conference(client, openreview_client, test_google_user):
    def _start_conference(
        client,
        conference_id,
        fake_data_source_id = None,
        exclude_expertise=True,
        post_reviewers = False,
        post_area_chairs = False,
        post_senior_area_chairs = False,
        post_submissions = False,
        post_publications = False,
        post_expertise_selection = None ## Posts a new publication and an edge to it
    ):
        # Use the conference lock to ensure only one thread can write to this conference at a time
        with conference_lock(conference_id, timeout=60) as acquired:
            if not acquired:
                raise TimeoutError(f"Could not acquire lock for conference {conference_id} within 60 seconds")
            
            # Post expertise selection
            # { '~Harold_Rice1': 'Include' }
            
            def _populate_groups(committee_name):
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                group = openreview.tools.get_group(client, f'{conference_id}/{committee_name}')
                if len(group.members) == 0:
                    Helpers.post_profiles(client, data)
                    members = data['groups'][f'{fake_data_source_id}/{committee_name}']['members']
                    client.add_members_to_group(f'{conference_id}/{committee_name}', members)
            
            def _post_publications(group_members):
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_publications(client, client_v2, data, group_members)

            def _post_submissions():
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_submissions(data, f'{conference_id}/-/Submission', datasource_invitation=f'{fake_data_source_id}/-/Submission')

            def _post_expertise_selection():
                for user, label in post_expertise_selection.items():
                    Helpers.post_expertise_publication(
                        client,
                        user,
                        conference_id,
                        edge_label=label,
                        api_version=1
                    )

            client_v2 = openreview.api.OpenReviewClient(
                baseurl = 'http://localhost:3001', username='openreview.net', password=Helpers.strong_password
            )

            if openreview.tools.get_invitation(client, f'openreview.net/-/paper') is None:
                # Archival/publication invitation
                invitation = openreview.Invitation(
                    id = 'openreview.net/-/paper',
                    writers = ['openreview.net'],
                    signatures = ['openreview.net'],
                    readers = ['everyone'],
                    invitees = ['everyone'],
                    reply={
                        "forum": None,
                        "replyto": None,
                        "writers": {
                            "values": [
                                "~SomeTest_User1"
                            ]
                        },
                        "signatures": {
                            "description": "How your identity will be displayed with the above content.",
                            "values": [
                                "~SomeTest_User1"
                            ]
                        },
                        "readers": {
                            "description": "The users who will be allowed to read the above content.",
                            "values": [
                                "everyone"
                            ]
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
                client.post_invitation(invitation)

            # If no fake data source id is provided, use the conference id
            if fake_data_source_id is None:
                fake_data_source_id = conference_id

            # If conference exists, select it
            conference, request_form_note = None, None
            if openreview.tools.get_group(client, conference_id) is not None:
                request_forms = client.get_all_notes(invitation='openreview.net/Support/-/Request_Form')
                for note in request_forms:
                    if note.content['venue_id'] == conference_id:
                        conference = openreview.conference.helpers.get_conference(client, note.id, support_user='openreview.net/Support')
                        request_form_note = note

            first_element = conference_id.split('/')[0]
            conf_prefix = first_element.split('.')[0]

            pc_email = f'pc@{first_element.lower()}'
            pc_name = f'{conf_prefix.upper()}Chair'
            pc_id = f'~Program_{pc_name}1'
            
            if conference is None:
                now = datetime.datetime.utcnow()
                due_date = now + datetime.timedelta(days=3)
                first_date = now + datetime.timedelta(days=1)

                pc_client=Helpers.create_user(
                    pc_email,
                    'Program',
                    pc_name
                )

                request_form_note = pc_client.post_note(openreview.Note(
                    invitation='openreview.net/Support/-/Request_Form',
                    signatures=[pc_id],
                    readers=[
                        'openreview.net/Support',
                        pc_id
                    ],
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
                        'publication_chairs':'No, our venue does not have Publication Chairs',
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
                        'include_expertise_selection': 'No' if exclude_expertise else 'Yes',
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

                Helpers.await_queue()

                # Post a deploy note
                client.post_note(openreview.Note(
                    content={'venue_id': conference_id},
                    forum=request_form_note.forum,
                    invitation='openreview.net/Support/-/Request{}/Deploy'.format(request_form_note.number),
                    readers=['openreview.net/Support'],
                    referent=request_form_note.forum,
                    replyto=request_form_note.forum,
                    signatures=['openreview.net/Support'],
                    writers=['openreview.net/Support']
                ))
                Helpers.await_queue()

                print(request_form_note)

            pc_client = openreview.Client(
                baseurl = 'http://localhost:3000', username=pc_email, password=Helpers.strong_password
            )

            if post_reviewers:
                _populate_groups('Reviewers')
                reviewers = pc_client.get_group(f'{conference_id}/Reviewers')
                if post_publications:
                    _post_publications(reviewers.members)
            if post_area_chairs:
                _populate_groups('Area_Chairs')
                area_chairs = pc_client.get_group(f'{conference_id}/Area_Chairs')
                if post_publications:
                    _post_publications(area_chairs.members)
            if post_senior_area_chairs:
                _populate_groups('Senior_Area_Chairs')
                senior_area_chairs = pc_client.get_group(f'{conference_id}/Senior_Area_Chairs')
                if post_publications:
                    _post_publications(senior_area_chairs.members)

            if post_submissions:
                _post_submissions()
                post_submission_note=pc_client.post_note(openreview.Note(
                    content= {
                        'force': 'Yes',
                        'hide_fields': ['keywords'],
                        'submission_readers': 'All program committee (all reviewers, all area chairs, all senior area chairs if applicable)'
                    },
                    forum= request_form_note.id,
                    invitation= f'openreview.net/Support/-/Request{request_form_note.number}/Post_Submission',
                    readers= [f'{conference_id}/Program_Chairs', 'openreview.net/Support'],
                    referent= request_form_note.id,
                    replyto= request_form_note.id,
                    signatures= [pc_id],
                    writers= [],
                ))
                Helpers.await_queue()

            if post_expertise_selection:
                _post_expertise_selection()
        
    return _start_conference


@pytest.fixture(scope="session")
def clean_start_conference_v2(client, openreview_client, test_google_user):
    def _start_conference_v2(
        openreview_client,
        conference_id,
        fake_data_source_id = None,
        post_reviewers = False,
        post_area_chairs = False,
        post_senior_area_chairs = False,
        post_submissions = False,
        post_publications = False,
        post_expertise_selection = None ## Posts a new publication and an edge to it
    ):
        # Use the conference lock to ensure only one thread can write to this conference at a time
        with conference_lock(conference_id, timeout=60) as acquired:
            if not acquired:
                raise TimeoutError(f"Could not acquire lock for conference {conference_id} within 60 seconds")
            
            def _populate_groups(committee_name):
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                group = openreview.tools.get_group(openreview_client, f'{conference_id}/{committee_name}')
                if len(group.members) == 0:
                    Helpers.post_profiles(client_v1, data)
                    members = data['groups'][f'{fake_data_source_id}/{committee_name}']['members']
                    openreview_client.add_members_to_group(f'{conference_id}/{committee_name}', members)
            
            def _post_publications(group_members):
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_publications(client_v1, openreview_client, data, group_members)

            def _post_submissions():
                with open('tests/data/fakeData.json') as json_file:
                    data = json.load(json_file)
                Helpers.post_submissions(data, f'{conference_id}/-/Submission', datasource_invitation=f'{fake_data_source_id}/-/Submission')

            def _post_expertise_selection():
                for user, label in post_expertise_selection.items():
                    Helpers.post_expertise_publication(
                        client_v1,
                        user,
                        conference_id,
                        label,
                        api_version=1
                    )

            client_v1 = openreview.Client(
                baseurl = 'http://localhost:3000', username='openreview.net', password=Helpers.strong_password
            )

            root_invitation = openreview.tools.get_invitation(
                openreview_client,
                'openreview.net/-/Edit'
            )
            if root_invitation is None:
                openreview_client.post_invitation_edit(invitations=None,
                    readers=['openreview.net'],
                    writers=['openreview.net'],
                    signatures=['~Super_User1'],
                    invitation=openreview.api.Invitation(id='openreview.net/-/Edit',
                        invitees=['openreview.net'],
                        readers=['openreview.net'],
                        signatures=['~Super_User1'],
                        edit=True
                    )
                )

            # If conference exists, select it
            request_form_note = None
            if openreview.tools.get_group(openreview_client, conference_id) is not None:
                request_forms = client_v1.get_all_notes(invitation='openreview.net/Support/-/Request_Form')
                for note in request_forms:
                    if note.content['venue_id'] == conference_id:
                        conference = openreview.conference.helpers.get_conference(client, note.id, support_user='openreview.net/Support')
                        request_form_note = note

            venue = Venue(openreview_client, conference_id, support_user='openreview.net/Support')
            venue.use_area_chairs = True
            venue.automatic_reviewer_assignment = True
            
            now = datetime.datetime.utcnow()

            venue.submission_stage = openreview.stages.SubmissionStage(
                double_blind=True,
                readers=[
                    openreview.builder.SubmissionStage.Readers.REVIEWERS_ASSIGNED
                ],
                due_date=now + datetime.timedelta(minutes=10),
                withdrawn_submission_reveal_authors=True,
                desk_rejected_submission_reveal_authors=True,
            )
            if request_form_note is None:
                venue.setup()
            venue.expertise_selection_stage = openreview.stages.ExpertiseSelectionStage()
            venue.create_submission_stage()

            pc_client = openreview.api.OpenReviewClient(
                baseurl = 'http://localhost:3001', username='openreview.net', password=Helpers.strong_password
            )
            pc_client.impersonate(conference_id)

            if post_reviewers:
                _populate_groups('Reviewers')
                reviewers = pc_client.get_group(f'{conference_id}/Reviewers')
                if post_publications:
                    _post_publications(reviewers.members)
            if post_area_chairs:
                _populate_groups('Area_Chairs')
                area_chairs = pc_client.get_group(f'{conference_id}/Area_Chairs')
                if post_publications:
                    _post_publications(area_chairs.members)
            if post_senior_area_chairs:
                _populate_groups('Senior_Area_Chairs')
                senior_area_chairs = pc_client.get_group(f'{conference_id}/Senior_Area_Chairs')
                if post_publications:
                    _post_publications(senior_area_chairs.members)

            if post_submissions:
                _post_submissions()
                Helpers.await_queue()

            if post_expertise_selection:
                _post_expertise_selection()

            return venue
    return _start_conference_v2