from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
import openreview
import pytest
import time
import json
from openreview.api import OpenReviewClient
from openreview.api import Note
from openreview.journal import Journal

os.environ["OPENREVIEW_USERNAME"] = "OpenReview.net"
os.environ["OPENREVIEW_PASSWORD"] = "1234"

class TestJournal():

    @pytest.fixture(scope="class")
    def journal(self):
        venue_id = 'TMLR'
        fabian_client=OpenReviewClient(username='fabian@mail.com', password='1234')
        fabian_client.impersonate('TMLR/Editors_In_Chief')
        journal=Journal(fabian_client, venue_id, '1234', contact_info='tmlr@jmlr.org', full_name='Transactions on Machine Learning Research', short_name='TMLR', submission_name='Submission')
        return journal

    def test_setup(self, openreview_client, helpers):
        venue_id = 'TMLR'

        ## Support Role
        helpers.create_user('fabian@mail.com', 'Fabian', 'Pedregosa')

        ## Editors in Chief
        # See api2Data.json

        journal=Journal(openreview_client, venue_id, '1234', contact_info='tmlr@jmlr.org', full_name='Transactions on Machine Learning Research', short_name='TMLR', submission_name='Submission')
        journal.setup(support_role='fabian@mail.com', editors=['~Raia_Hadsell1', '~Kyunghyun_Cho1'])
    
    def test_post_submissions(self, client, openreview_client, helpers):
        # Post submission with a test author id

        def post_notes(data, data_invitation):
            for note_json in data['notes'][data_invitation]:
                content = note_json['content']
                cdate = note_json.get('cdate')

                # Post note edit to journal submission invitation
                # TODO: Add in cdate to API2 notes
                submission_note = openreview_client.post_note_edit(
                    invitation = 'TMLR/-/Submission',
                    signatures = ['~Super_User1'],
                    note = Note(
                        #cdate = cdate,
                        content = {
                            'title': { 'value': content.get('title').get('value') },
                            'abstract': { 'value': content.get('abstract').get('value') },
                            'authors': { 'value': ['Test User']},
                            'authorids': { 'value': ['~SomeFirstName_User1']},
                            'pdf': {'value': '/pdf/' + 'p' * 40 +'.pdf' },
                            'supplementary_material': { 'value': '/attachment/' + 's' * 40 +'.zip'},
                            'competing_interests': { 'value': 'None beyond the authors normal conflict of interests'},
                            'human_subjects_reporting': { 'value': 'Not applicable'},
                            'submission_length': { 'value': 'Regular submission (no more than 12 pages of main content)'}
                        }
                    ))

        with open('tests/data/api2Data.json') as json_file:
            data = json.load(json_file)
        post_notes(data, 'ABC.cc/-/Blind_Submission')
        post_notes(data, 'HIJ.cc/-/Blind_Submission')

    def test_post_publications_to_journal(self, openreview_client):
        # Use the journal submission invitation to post publications in API2        

        def post_notes(data):
            for profile_json in data['profiles']:
                authorid = profile_json['id']
                name = ' '.join(authorid[1:-1].split('_'))
                for pub_json in profile_json['publications']:
                    content = pub_json['content']
                    cdate = pub_json.get('cdate')

                    existing_pubs = list(openreview.tools.iterget_notes(openreview_client, content={'authorids': authorid}))
                    existing_titles = [pub.content.get('title') for pub in existing_pubs]

                    if content.get('title') not in existing_titles:
                        publication_note = openreview_client.post_note_edit(
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
                                    'human_subjects_reporting': { 'value': 'Not applicable'},
                                    'submission_length': { 'value': 'Regular submission (no more than 12 pages of main content)'}
                                }
                            ))
        
        with open('tests/data/api2Data.json') as json_file:
            data = json.load(json_file)
        post_notes(data)


