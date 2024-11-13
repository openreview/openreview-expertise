import openreview
import pytest
import requests
import time

class Helpers:
    strong_password = 'Or$3cur3P@ssw0rd'

    @staticmethod
    def create_user(email, first, last, alternates=[], institution=None):
        client = openreview.Client(baseurl = 'http://localhost:3000')
        assert client is not None, "Client is none"
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

@pytest.fixture(scope="class")
def helpers():
    return Helpers

@pytest.fixture(scope="session")
def test_client():
    Helpers.create_user('test@mail.com', 'SomeFirstName', 'User')
    yield openreview.Client(baseurl = 'http://localhost:3000', username='test@mail.com', password=Helpers.strong_password)

@pytest.fixture(scope="session")
def client():
    yield openreview.Client(baseurl = 'http://localhost:3000', username='openreview.net', password=Helpers.strong_password)

@pytest.fixture(scope="session")
def openreview_client():
    yield openreview.api.OpenReviewClient(baseurl = 'http://localhost:3001', username='openreview.net', password=Helpers.strong_password)
