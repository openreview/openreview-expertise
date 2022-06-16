from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
import openreview
import pytest
import time
import json

os.environ["OPENREVIEW_USERNAME"] = "OpenReview.net"
os.environ["OPENREVIEW_PASSWORD"] = "1234"

class TestConference():

    def test_create_conferences(self, client, helpers):

        now = datetime.datetime.utcnow()
        due_date = now + datetime.timedelta(days=3)
        first_date = now + datetime.timedelta(days=1)

        # Post the request form note
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
                'Venue Start Date': '2021/12/01',
                'Submission Deadline': due_date.strftime('%Y/%m/%d'),
                'abstract_registration_deadline': first_date.strftime('%Y/%m/%d'),
                'Location': 'Virtual',
                'Paper Matching': [
                    'Reviewer Bid Scores',
                    'OpenReview Affinity'],
                'Author and Reviewer Anonymity': 'Double-blind',
                'reviewer_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                'area_chair_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                'senior_area_chair_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                'Open Reviewing Policy': 'Submissions and reviews should both be private.',
                'submission_readers': 'Program chairs and paper authors only',
                'How did you hear about us?': 'ML conferences',
                'Expected Submissions': '100'
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
                'Venue Start Date': '2021/12/01',
                'Submission Deadline': due_date.strftime('%Y/%m/%d'),
                'abstract_registration_deadline': first_date.strftime('%Y/%m/%d'),
                'Location': 'Virtual',
                'Paper Matching': [
                    'Reviewer Bid Scores',
                    'OpenReview Affinity'],
                'Author and Reviewer Anonymity': 'Double-blind',
                'reviewer_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                'area_chair_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                'senior_area_chair_identity': ['Program Chairs', 'Assigned Senior Area Chair', 'Assigned Area Chair', 'Assigned Reviewers'],
                'Open Reviewing Policy': 'Submissions and reviews should both be private.',
                'submission_readers': 'Program chairs and paper authors only',
                'How did you hear about us?': 'ML conferences',
                'Expected Submissions': '100'
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

    def test_create_groups(self, client, helpers):
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
                    
                    # A pre-generated user may not have 1 at the end
                    # In this case, add the same name to their profile, create a group for this new ID and rename the profile
                    curr_id = client.get_profile(email).id                   
                    if new_id != curr_id:
                        client.post_profile(openreview.Profile(
                            referent = curr_id, 
                            invitation = '~/-/invitation',
                            signatures = ['openreview.net'],
                            content = {},
                            metaContent = {
                                'names': { 
                                    'values': [{ 
                                        'first': first_name,
                                        'middle': '',
                                        'last': last_name,
                                        'username': new_id }],
                                    'weights': [1] }
                            }))
                        client.post_group(
                            openreview.Group(
                                id = new_id,
                                readers = [new_id],
                                writers = ['openreview.net'],
                                signatories = [new_id],
                                signatures = ['openreview.net']
                            )
                        )

                        client.rename_profile(curr_id, new_id)

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

    def test_create_invitations(self, client, openreview_client):
        # Post invitations for submissions and publications

        reply = {
            "forum": None,
            "replyto": None,
                "writers": {
                    "values": [
                        "openreview.net"
                    ]
                },
                "signatures": {
                    "description": "How your identity will be displayed with the above content.",
                    "values": [
                        "openreview.net"
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
                        "value-regex": "[^,\\n]*(,[^,\\n]+)*"
                    },
                    "authorids": {
                        "required": False,
                        "order": 4,
                        "description": "Comma separated list of author email addresses, in the same order as above.",
                        "value-regex": "[^,\\n]*(,[^,\\n]+)*"
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

        # create_dataset submission invitation to test deduplication
        invitation = openreview.Invitation(
                id = 'DEF.cc/-/Duplicated_Submission',
                writers = ['openreview.net'],
                signatures = ['openreview.net'],
                readers = ['everyone'],
                invitees = ['everyone'],
                reply=reply
            )
        client.post_invitation(invitation)
        assert client.get_invitation('DEF.cc/-/Duplicated_Submission')

        # create_dataset edge invitation to test exclusion
        '''
        invitation = openreview.Invitation(
                id = 'DEF.cc/-/Expertise_Selection',
                writers = ['openreview.net'],
                signatures = ['openreview.net'],
                readers = ['everyone'],
                invitees = ['everyone'],
                reply={
                    "forum": None,
                    "replyto": None,
                        "writers": {
                            "values": [
                                "openreview.net"
                            ]
                        },
                        "signatures": {
                            "description": "How your identity will be displayed with the above content.",
                            "values": [
                                "openreview.net"
                            ]
                        },
                        "readers": {
                            "description": "The users who will be allowed to read the above content.",
                            "values": [
                                "everyone"
                            ]
                        },
                        "content": {
                            "head": {
                                "type": "Note"
                            },
                            "tail": {
                                "type": "Profile"
                            },
                            "label": {
                                "value-radio": [
                                    "Exclude"
                                ],
                                "required": False
                            }
                        }
                }
            )
        client.post_invitation(invitation)
        assert client.get_invitation('DEF.cc/-/Expertise_Selection')
        '''

    def test_post_submissions(self, client, openreview_client):
        
        def post_notes(data, invitation):
            for note_json in data['notes'][invitation]:
                content = note_json['content']
                content['authors'] = ['Super User']
                content['authorids'] = ['~Super_User1']
                cdate = note_json.get('cdate')

                note = openreview.Note(
                    invitation = invitation,
                    readers = [invitation.split('/')[0], '~Super_User1'],
                    writers = ['~Super_User1'],
                    signatures = ['~Super_User1'],
                    content = content,
                    cdate = cdate
                )
                note = client.post_note(note)

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'ABC.cc/-/Submission')

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'DEF.cc/-/Submission')

    def test_post_publications(self, client, openreview_client):

        def post_notes(data, api_invitation):
            for profile_json in data['profiles']:
                authorid = profile_json['id']
                for pub_json in profile_json['publications']:
                    content = pub_json['content']
                    content['authorids'] = [authorid]
                    cdate = pub_json.get('cdate')

                    existing_pubs = list(openreview.tools.iterget_notes(client, content={'authorids': authorid}))
                    existing_titles = [pub.content.get('title') for pub in existing_pubs]

                    if content.get('title') not in existing_titles:
                        note = openreview.Note(
                            invitation = api_invitation,
                            readers = ['everyone'],
                            writers = ['openreview.net'],
                            signatures = ['openreview.net'],
                            content = content,
                            cdate = cdate
                        )
                        note = client.post_note(note)

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'openreview.net/-/paper')

        



