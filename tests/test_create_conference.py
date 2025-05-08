from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from flask import request
import openreview
import pytest
import time
import json

os.environ["OPENREVIEW_USERNAME"] = "OpenReview.net"
os.environ["OPENREVIEW_PASSWORD"] = 'Or$3cur3P@ssw0rd'

class TestConference():

    def test_create_conferences(self, client, openreview_client, helpers):

        venue = openreview.venue.Venue(openreview_client, 'API2', support_user='openreview.net/Support')

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

        venue.use_area_chairs = True
        
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
        venue.expertise_selection_stage = openreview.stages.ExpertiseSelectionStage()
        venue.setup()
        venue.create_submission_stage()

        now = datetime.datetime.utcnow()
        due_date = now + datetime.timedelta(days=3)
        first_date = now + datetime.timedelta(days=1)

        # Post the request form note
        # ABC is to be used for running the expertise model and has an inclusion invitation
        helpers.create_user('test@google.com', 'SomeTest', 'User')
        pc_client=helpers.create_user('pc@abc.cc', 'Program', 'ABCChair')

        request_form_note = pc_client.post_note(openreview.Note(
            invitation='openreview.net/Support/-/Request_Form',
            signatures=['~Program_ABCChair1'],
            readers=[
                'openreview.net/Support',
                '~Program_ABCChair1'
            ],
            writers=[],
            content={
                'title': 'Conference on Neural Information Processing Systems',
                'Official Venue Name': 'Conference on Neural Information Processing Systems',
                'Abbreviated Venue Name': 'ABC 2021',
                'Official Website URL': 'https://neurips.cc',
                'program_chair_emails': ['pc@abc.cc'],
                'contact_email': 'pc@abc.cc',
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
                'include_expertise_selection': 'Yes',
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

        helpers.await_queue()

        # Post a deploy note
        client.post_note(openreview.Note(
            content={'venue_id': 'ABC.cc'},
            forum=request_form_note.forum,
            invitation='openreview.net/Support/-/Request{}/Deploy'.format(request_form_note.number),
            readers=['openreview.net/Support'],
            referent=request_form_note.forum,
            replyto=request_form_note.forum,
            signatures=['openreview.net/Support'],
            writers=['openreview.net/Support']
        ))

        helpers.await_queue()

        assert client.get_group('ABC.cc')
        assert client.get_group('ABC.cc/Senior_Area_Chairs')
        acs=client.get_group('ABC.cc/Area_Chairs')
        assert acs
        assert 'ABC.cc/Senior_Area_Chairs' in acs.readers
        reviewers=client.get_group('ABC.cc/Reviewers')
        assert reviewers
        assert 'ABC.cc/Senior_Area_Chairs' in reviewers.readers
        assert 'ABC.cc/Area_Chairs' in reviewers.readers

        assert client.get_group('ABC.cc/Authors')


        # Post the request form note
        # DEF is to be used for unit testing create_dataset and has an exclusion invitation
        pc_client=helpers.create_user('pc@def.cc', 'Program', 'DEFChair')

        request_form_note = pc_client.post_note(openreview.Note(
            invitation='openreview.net/Support/-/Request_Form',
            signatures=['~Program_DEFChair1'],
            readers=[
                'openreview.net/Support',
                '~Program_DEFChair1'
            ],
            writers=[],
            content={
                'title': 'Conference on Neural Information Processing Systems',
                'Official Venue Name': 'Conference on Neural Information Processing Systems',
                'Abbreviated Venue Name': 'DEF 2021',
                'Official Website URL': 'https://neurips.cc',
                'program_chair_emails': ['pc@def.cc'],
                'contact_email': 'pc@def.cc',
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

        helpers.await_queue()

        # Post a deploy note
        client.post_note(openreview.Note(
            content={'venue_id': 'DEF.cc'},
            forum=request_form_note.forum,
            invitation='openreview.net/Support/-/Request{}/Deploy'.format(request_form_note.number),
            readers=['openreview.net/Support'],
            referent=request_form_note.forum,
            replyto=request_form_note.forum,
            signatures=['openreview.net/Support'],
            writers=['openreview.net/Support']
        ))

        helpers.await_queue()

        assert client.get_group('DEF.cc')
        assert client.get_group('DEF.cc/Senior_Area_Chairs')
        acs=client.get_group('DEF.cc/Area_Chairs')
        assert acs
        assert 'DEF.cc/Senior_Area_Chairs' in acs.readers
        reviewers=client.get_group('DEF.cc/Reviewers')
        assert reviewers
        assert 'DEF.cc/Senior_Area_Chairs' in reviewers.readers
        assert 'DEF.cc/Area_Chairs' in reviewers.readers

        assert client.get_group('DEF.cc/Authors')

        # Post the request form note
        # HIJ is to be used for testing for the inclusion invitation with the expertise API
        pc_client=helpers.create_user('pc@hij.cc', 'Program', 'HIJChair')

        request_form_note = pc_client.post_note(openreview.Note(
            invitation='openreview.net/Support/-/Request_Form',
            signatures=['~Program_HIJChair1'],
            readers=[
                'openreview.net/Support',
                '~Program_HIJChair1'
            ],
            writers=[],
            content={
                'title': 'Conference on Neural Information Processing Systems',
                'Official Venue Name': 'Conference on Neural Information Processing Systems',
                'Abbreviated Venue Name': 'HIJ 2021',
                'Official Website URL': 'https://neurips.cc',
                'program_chair_emails': ['pc@hij.cc'],
                'contact_email': 'pc@hij.cc',
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
                'include_expertise_selection': 'Yes',
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

        helpers.await_queue()

        # Post a deploy note
        client.post_note(openreview.Note(
            content={'venue_id': 'HIJ.cc'},
            forum=request_form_note.forum,
            invitation='openreview.net/Support/-/Request{}/Deploy'.format(request_form_note.number),
            readers=['openreview.net/Support'],
            referent=request_form_note.forum,
            replyto=request_form_note.forum,
            signatures=['openreview.net/Support'],
            writers=['openreview.net/Support']
        ))

        helpers.await_queue()

        assert client.get_group('HIJ.cc')
        assert client.get_group('HIJ.cc/Senior_Area_Chairs')
        acs=client.get_group('HIJ.cc/Area_Chairs')
        assert acs
        assert 'HIJ.cc/Senior_Area_Chairs' in acs.readers
        reviewers=client.get_group('HIJ.cc/Reviewers')
        assert reviewers
        assert 'HIJ.cc/Senior_Area_Chairs' in reviewers.readers
        assert 'HIJ.cc/Area_Chairs' in reviewers.readers

        assert client.get_group('HIJ.cc/Authors')

    def test_create_groups(self, client, openreview_client, helpers):
        # Post test data groups to the API

        def post_profiles(data):
            for profile_json in data['profiles']:
                if not client.search_profiles(ids=[profile_json['id']]):
                    # If the profile hasn't already been posted get the data and create the user
                    new_id = profile_json['id']
                    email = profile_json.get('content').get('preferredEmail') or profile_json.get('content').get('emails')[0]
                    first_name = profile_json['id'][1:-1].split('_')[0]
                    last_name = profile_json['id'][1:-1].split('_')[-1]
                    helpers.create_user(email, first_name, last_name)

                    assert client.get_profile(new_id)
                    assert client.get_profile(new_id).id == new_id
        
        # Post a small number of reviewers to the ABC.cc group used for testing the expertise model
        # to reduce test time
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_profiles(data)
        members = data['groups']['ABC.cc/Reviewers']['members']
        client.add_members_to_group('ABC.cc/Reviewers', members)

        # Post a large number of reviewers to the DEF.cc group used for testing the create_dataset functions
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_profiles(data)
        members = data['groups']['DEF.cc/Reviewers']['members']
        client.add_members_to_group('DEF.cc/Reviewers', members)

        # Post a small number of reviewers to the HIJ.cc group used only for testing the error message for no submissions
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_profiles(data)
        members = data['groups']['ABC.cc/Reviewers']['members']
        client.add_members_to_group('HIJ.cc/Reviewers', members)
        openreview_client.add_members_to_group('API2/Reviewers', members)

    def test_create_invitations(self, client, openreview_client):
        # Post invitations for submissions and publications

        reply = {
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

        # Post an invitation for publications, submissions to be tested with the expertise model and
        # submissions to be tested with create dataset

        # Archival/publication invitation
        invitation = openreview.Invitation(
            id = 'openreview.net/-/paper',
            writers = ['openreview.net'],
            signatures = ['openreview.net'],
            readers = ['everyone'],
            invitees = ['everyone'],
            reply=reply
        )
        client.post_invitation(invitation)
        assert client.get_invitation('openreview.net/-/paper')

    def test_post_submissions(self, client, openreview_client, helpers):

        def post_notes(data, invitation):
            test_user_client = openreview.Client(username='test@google.com', password=helpers.strong_password)
            for note_json in data['notes'][invitation]:
                content = note_json['content']
                content['authors'] = ['SomeTest User']
                content['authorids'] = ['~SomeTest_User1']
                cdate = note_json.get('cdate')

                note = openreview.Note(
                    invitation = invitation,
                    readers = [invitation.split('/')[0], '~SomeTest_User1'],
                    writers = [invitation.split('/')[0], '~SomeTest_User1'],
                    signatures = ['~SomeTest_User1'],
                    content = content,
                    cdate = cdate
                )
                note = test_user_client.post_note(note)

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'ABC.cc/-/Submission')

        pc_client=openreview.Client(username='pc@abc.cc', password=helpers.strong_password)
        request_form=client.get_notes(invitation='openreview.net/Support/-/Request_Form', sort='tcdate')[2]
        print(request_form)
        post_submission_note=pc_client.post_note(openreview.Note(
            content= {
                'force': 'Yes',
                'hide_fields': ['keywords'],
                'submission_readers': 'All program committee (all reviewers, all area chairs, all senior area chairs if applicable)'
            },
            forum= request_form.id,
            invitation= f'openreview.net/Support/-/Request1/Post_Submission',
            readers= ['ABC.cc/Program_Chairs', 'openreview.net/Support'],
            referent= request_form.id,
            replyto= request_form.id,
            signatures= ['~Program_ABCChair1'],
            writers= [],
        ))
        helpers.await_queue()

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'DEF.cc/-/Submission')

        pc_client=openreview.Client(username='pc@def.cc', password=helpers.strong_password)
        request_form=client.get_notes(invitation='openreview.net/Support/-/Request_Form', sort='tcdate')[1]
        print(request_form)
        post_submission_note=pc_client.post_note(openreview.Note(
            content= {
                'force': 'Yes',
                'hide_fields': ['keywords'],
                'submission_readers': 'All program committee (all reviewers, all area chairs, all senior area chairs if applicable)'
            },
            forum= request_form.id,
            invitation= f'openreview.net/Support/-/Request2/Post_Submission',
            readers= ['DEF.cc/Program_Chairs', 'openreview.net/Support'],
            referent= request_form.id,
            replyto= request_form.id,
            signatures= ['~Program_DEFChair1'],
            writers= [],
        ))
        helpers.await_queue()

    def test_post_publications(self, client, openreview_client):
        tmlr_editors = ['~Raia_Hadsell1', '~Kyunghyun_Cho1']

        def post_notes(data, api_invitation):
            for profile_json in data['profiles']:
                authorid = profile_json['id']
                if authorid not in tmlr_editors:
                    for idx, pub_json in enumerate(profile_json['publications']):
                        content = pub_json['content']
                        content['authorids'] = [authorid]
                        cdate = pub_json.get('cdate')

                        existing_pubs = list(openreview.tools.iterget_notes(client, content={'authorids': authorid}))
                        existing_titles = [pub.content.get('title') for pub in existing_pubs]

                        if content.get('title') not in existing_titles:
                            if idx % 2 == 0:
                                note = openreview.Note(
                                    invitation = api_invitation,
                                    readers = ['everyone'],
                                    writers = ['~SomeTest_User1'],
                                    signatures = ['~SomeTest_User1'],
                                    content = content,
                                    cdate = cdate
                                )
                                note = client.post_note(note)
                            else:
                                edit = openreview_client.post_note_edit(
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

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'openreview.net/-/paper')

        



