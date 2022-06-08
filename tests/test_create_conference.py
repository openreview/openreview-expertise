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
    @pytest.fixture(scope="class")
    def conference(self, client):
        pc_client=openreview.Client(username='pc@neurips.cc', password='1234')
        request_form=client.get_notes(invitation='openreview.net/Support/-/Request_Form', sort='tmdate')[0]

        conference=openreview.helpers.get_conference(pc_client, request_form.id)
        return conference

    def test_create_client(self, client, openreview_client):
        # Test connectivity to the API

        client = openreview.Client()
        assert client
        assert client.token
        assert client.profile
        assert '~Super_User1' == client.profile.id

        openreview_client = openreview.Client()
        assert openreview_client
        assert openreview_client.token
        assert openreview_client.profile
        assert '~Super_User1' == openreview_client.profile.id

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
        with open('tests/data/expertiseServiceData.json') as json_file:
            data = json.load(json_file)
        post_profiles(data)
        members = data['groups']['ABC.cc']['members']

        group = openreview.Group(
            id = 'ABC.cc',
            readers = ['everyone'],
            writers = ['openreview.net'],
            signatories = ['openreview.net'],
            signatures = ['openreview.net'],
            members = members
        )
        client.post_group(group)

        # Post a large number of reviewers to the DEF.cc group used for testing the create_dataset functions
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_profiles(data)
        members = data['groups']['ABC.cc']['members']

        group = openreview.Group(
            id = 'DEF.cc',
            readers = ['everyone'],
            writers = ['openreview.net'],
            signatories = ['openreview.net'],
            signatures = ['openreview.net'],
            members = members
        )
        client.post_group(group)

        # Post a small number of reviewers to the HIJ.cc group used for testing correctness
        with open('tests/data/api2Data.json') as json_file:
            data = json.load(json_file)
        post_profiles(data)
        members = data['groups']['HIJ.cc']['members']

        group = openreview.Group(
            id = 'HIJ.cc',
            readers = ['everyone'],
            writers = ['openreview.net'],
            signatories = ['openreview.net'],
            signatures = ['openreview.net'],
            members = members
        )
        client.post_group(group)

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

        # Expertise model submission invitation
        invitation = openreview.Invitation(
            id = 'ABC.cc/-/Submission',
            writers = ['openreview.net'],
            signatures = ['openreview.net'],
            readers = ['everyone'],
            invitees = ['everyone'],
            reply=reply
        )
        client.post_invitation(invitation)
        assert client.get_invitation('ABC.cc/-/Submission')

        # create_dataset submission invitation
        invitation = openreview.Invitation(
            id = 'DEF.cc/-/Submission',
            writers = ['openreview.net'],
            signatures = ['openreview.net'],
            readers = ['everyone'],
            invitees = ['everyone'],
            reply=reply
        )
        client.post_invitation(invitation)
        assert client.get_invitation('DEF.cc/-/Submission')

        # create_dataset submission invitation to test deduplication
        invitation = openreview.Invitation(
                id = 'DEF.cc/-/Blind_Submission',
                writers = ['openreview.net'],
                signatures = ['openreview.net'],
                readers = ['everyone'],
                invitees = ['everyone'],
                reply=reply
            )
        client.post_invitation(invitation)
        assert client.get_invitation('DEF.cc/-/Blind_Submission')

        # create_dataset edge invitation to test exclusion
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

    def test_post_submissions(self, client, openreview_client):
        
        def post_notes(data, data_invitation, api_invitation):
            for note_json in data['notes'][data_invitation]:
                content = note_json['content']
                cdate = note_json.get('cdate')

                note = openreview.Note(
                    invitation = api_invitation,
                    readers = ['everyone'],
                    writers = ['openreview.net'],
                    signatures = ['openreview.net'],
                    content = content,
                    cdate = cdate
                )
                note = client.post_note(note)

        with open('tests/data/expertiseServiceData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'ABC.cc/-/Submission', 'ABC.cc/-/Submission')

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'ABC.cc/-/Submission', 'DEF.cc/-/Submission')

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
        
        with open('tests/data/expertiseServiceData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'openreview.net/-/paper')

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'openreview.net/-/paper')

        



