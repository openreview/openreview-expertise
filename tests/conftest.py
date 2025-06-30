import openreview
import datetime
import pytest
import requests
import time
import json
from tests.conference_locks import conference_lock
from tests.test_utils import TestHelpers, ConferenceBuilder, JournalBuilder

from openreview.api import OpenReviewClient
from openreview.api import Note
from openreview.api import Group
from openreview.api import Invitation
from openreview.api import Edge
from openreview.venue import Venue
from openreview.journal import Journal

# Legacy alias for backward compatibility
Helpers = TestHelpers


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
        fake_data_source_id=None,
        exclude_expertise=True,
        post_reviewers=False,
        post_area_chairs=False,
        post_senior_area_chairs=False,
        post_submissions=False,
        post_publications=False,
        post_expertise_selection=None
    ):
        """Start a conference using the ConferenceBuilder."""
        builder = ConferenceBuilder(client, openreview_client)
        return builder.create_conference(
            conference_id=conference_id,
            fake_data_source_id=fake_data_source_id,
            exclude_expertise=exclude_expertise,
            post_reviewers=post_reviewers,
            post_area_chairs=post_area_chairs,
            post_senior_area_chairs=post_senior_area_chairs,
            post_submissions=post_submissions,
            post_publications=post_publications,
            post_expertise_selection=post_expertise_selection
        )
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