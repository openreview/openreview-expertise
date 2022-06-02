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

        def post_profiles(data):
            for profile_json in data['profiles']:
                if not client.search_profiles(ids=[profile_json['id']]):
                    new_id = profile_json['id']
                    email = profile_json.get('content').get('preferredEmail') or profile_json.get('content').get('emails')[0]
                    first_name = profile_json['id'][1:-1].split('_')[0]
                    last_name = profile_json['id'][1:-1].split('_')[-1]
                    helpers.create_user(email, first_name, last_name)
                    
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

    def test_create_invitations(self, client, openreview_client):
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

        



