from expertise.create_dataset import OpenReviewExpertise
from unittest.mock import patch, MagicMock
from collections import defaultdict
import openreview
import json
import random
from pathlib import Path
import sys
import pytest
import os
import time
import numpy as np
import shutil
import expertise.service
from expertise.dataset import ArchivesDataset, SubmissionsDataset

class TestExpertiseV2():

    job_id = None

    @pytest.fixture(scope='session')
    def celery_config(self):
        return {
            "broker_url": "redis://localhost:6379/10",
            "result_backend": "redis://localhost:6379/10",
            "task_track_started": True,
            "task_serializer": "pickle",
            "result_serializer": "pickle",
            "accept_content": ["pickle", "application/x-python-serialize"],
            "task_create_missing_queues": True,
        }

    @pytest.fixture(scope='session')
    def celery_includes(self):
        return ["expertise.service.celery_tasks"]

    @pytest.fixture(scope='session')
    def celery_worker_parameters(self):
        return {
            "queues": ("userpaper", "expertise"),
            "perform_ping_check": False,
            "concurrency": 1,
        }

    @pytest.fixture(scope='session')
    def openreview_context(self):
        """
        A pytest fixture for setting up a clean expertise-api test instance:
        `scope` argument is set to 'function', so each function will get a clean test instance.
        """
        config = {
            "LOG_FILE": "pytest.log",
            "OPENREVIEW_USERNAME": "openreview.net",
            "OPENREVIEW_PASSWORD": "Or$3cur3P@ssw0rd",
            "OPENREVIEW_BASEURL": "http://localhost:3000",
            "SUPERUSER_FIRSTNAME": "Super",
            "SUPERUSER_LASTNAME": "User",
            "SUPERUSER_TILDE_ID": "~Super_User1",
            "SUPERUSER_EMAIL": "info@openreview.net",
            "SPECTER_DIR": '../expertise-utils/specter/',
            "MFR_VOCAB_DIR": '../expertise-utils/multifacet_recommender/feature_vocab_file',
            "MFR_CHECKPOINT_DIR": '../expertise-utils/multifacet_recommender/mfr_model_checkpoint/',
            "WORKING_DIR": './tests/jobs',
            "CHECK_EVERY": 3600,
            "DELETE_AFTER": 3600,
            "IN_TEST": True
        }
        app = expertise.service.create_app(
            config=config
        )

        with app.app_context():
            yield {
                "app": app,
                "test_client": app.test_client(),
                "config": config
            }

    def test_get_publications(self, client, openreview_client):
        config = {
            'dataset': {
                'top_recent_pubs': 3,
            }
        }
        or_expertise = OpenReviewExpertise(client, openreview_client, config)
        publications = or_expertise.get_publications('~Carlos_Mondragon1')
        assert publications == []

        or_expertise = OpenReviewExpertise(client, openreview_client, config)
        publications = or_expertise.get_publications('~Harold_Rice1')
        assert len(publications) == 3 ## 3 top recent publications
        for pub in publications:
            content = pub['content']
            assert isinstance(content['title'], str)
            assert isinstance(content['abstract'], str)

    def test_get_submissions_from_invitation_v2(self, client, openreview_client):
        # Returns the V2 submissions
        config = {
            'use_email_ids': False,
            'match_group': 'ABC.cc',
            'paper_invitation': 'TMLR/-/Submission',
        }
        or_expertise = OpenReviewExpertise(client, openreview_client, config)
        retrieved_submissions = or_expertise.get_submissions()
        print(retrieved_submissions)
        retrieved_titles = [pub.get('content').get('title') for pub in retrieved_submissions.values()]
        assert len(retrieved_submissions) == 5
        for submission in retrieved_submissions.values():
            assert isinstance(submission['content']['title'], str)

        assert "Right Metatarsal, Endoscopic Approach" in retrieved_titles
        assert "Iliac Art to B Com Ilia, Perc Endo Approach" in retrieved_titles

    def test_get_by_submissions_from_paper_id(self, client, openreview_client):
        journal_papers = openreview_client.get_notes(invitation='TMLR/-/Submission')
        for paper in journal_papers:
            if paper.content['authorids']['value'][0] == '~SomeFirstName_User1':
                target_paper = paper
                break

        config = {
            'paper_id': target_paper.id,
        }
        or_expertise = OpenReviewExpertise(client, openreview_client, config)
        submissions = or_expertise.get_submissions()
        print(submissions)
        assert not isinstance(submissions[target_paper.id]['content']['title'], dict)
        assert not isinstance(submissions[target_paper.id]['content']['abstract'], dict)

    def test_get_by_submissions_from_paper_venueid(self, client, openreview_client):
        journal_papers = openreview_client.get_notes(invitation='TMLR/-/Submission')
        for paper in journal_papers:
            if paper.content['authorids']['value'][0] == '~SomeFirstName_User1':
                target_paper = paper
                break

        config = {
            'paper_venueid': target_paper.content['venueid']['value'],
        }
        or_expertise = OpenReviewExpertise(client, openreview_client, config)
        submissions = or_expertise.get_submissions()
        print(submissions)
        assert not isinstance(submissions[target_paper.id]['content']['title'], dict)
        assert not isinstance(submissions[target_paper.id]['content']['abstract'], dict)
    
    def test_journal_request_v2(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        # Fetch a paper ID
        journal_papers = openreview_client.get_notes(invitation='TMLR/-/Submission')
        for paper in journal_papers:
            if 'liver tumor' in paper.content['title']['value']:
                target_id = paper.id

        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Action_Editors",
                    },
                    "entityB": { 
                        'type': "Note",
                        'id': target_id
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'TMLR/Action_Editors'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['id'] == target_id

        openreview_context['job_id'] = job_id

        # Make a request for TMLR/Reviewers
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Reviewers",
                    },
                    "entityB": { 
                        'type': "Note",
                        'id': target_id
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        print(f"second request ret: {response.json}")
        print(f"setting job_id to {response.json['jobId']}")
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Test for paper id query
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'id': target_id}).json['results']
        assert len(response) == 2

        # Test for paper id and member of query
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'id': target_id, 'memberOf': 'TMLR/Reviewers'}).json['results']
        assert len(response) == 1

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'entityB.id': target_id, 'memberOf': 'TMLR/Reviewers'}).json['results']
        assert len(response) == 1

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'entityB.id': target_id, 'entityB.memberOf': 'TMLR/Reviewers'}).json['results']
        assert len(response) == 0

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'entityA.id': target_id}).json['results']
        assert len(response) == 0

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'id': 'DoesNotExist'}).json['results']
        assert len(response) == 0

        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f'{job_id}', 'deleteOnGet': True})
        assert not os.path.isdir(f"./tests/jobs/{job_id}")

    def test_get_journal_results(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 1
        response = response.json['results']
        medical_score, translation_score = 0, 0
        for item in response:
            print(item)
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1
            if profile_id == '~Raia_Hadsell1':
                medical_score = score
            if profile_id == '~Kyunghyun_Cho1':
                translation_score = score

        # Check for correctness
        assert medical_score > 0
        assert translation_score > 0
        assert medical_score > 0.5
        assert translation_score < 0.5
        
        # Clean up journal request
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")
        # Clean up directory
        shutil.rmtree(f"./tests/jobs/")
        if os.path.isfile('pytest.log'):
            os.remove('pytest.log')
        if os.path.isfile('default.log'):
            os.remove('default.log')

    def test_venueid_v2(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Action_Editors",
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted"
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

    def test_submission_content_v2(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Action_Editors",
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'track': 'no_track' }
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            print(f"resp: {response}")
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            try_time = time.time() - start_time

        assert response['status'] == 'Error'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Dimension out of range (expected to be in range of [-1, 0], but got 1). Please check that you have at least 1 submission submitted and that you have run the Post Submission stage.'

        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Action_Editors",
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        results = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id}).json['results']       
        assert len(results) == 15 # 3 editors x 5 submissions/publications from Raia/Kyunghyun

    def test_paperpaper_submission_content_v2(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        # Make a request that is not supported
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'Error' in response.json['name']
        assert 'bad request' in response.json['message'].lower()
        assert response.json['message'] == "Bad request: model specter+mfr does not support paper-paper scoring"

        abc_client = openreview.api.OpenReviewClient(token=openreview_client.token)
        abc_client.impersonate('ABC.cc/Program_Chairs')
        # Get a no publications error
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission",
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=abc_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            print(f"resp: {response}")
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            try_time = time.time() - start_time

        assert response['status'] == 'Error'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Dimension out of range (expected to be in range of [-1, 0], but got 1). Please check that you have access to the papers that you are querying for.'

        # Get a no submissions error
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission",
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=abc_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            print(f"resp: {response}")
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            try_time = time.time() - start_time

        assert response['status'] == 'Error'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Dimension out of range (expected to be in range of [-1, 0], but got 1). Please check that you have access to the papers that you are querying for.'

        # Make a request that is supported by the model
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        results = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id}).json['results']       
        assert len(results) == 25 # 5 submissions x 5 submissions/publications from Raia/Kyunghyun

    def test_specter2_scincl(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Action_Editors",
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'track': 'no_track' }
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            try_time = time.time() - start_time

        assert response['status'] == 'Error'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Dimension out of range (expected to be in range of [-1, 0], but got 1). Please check that you have at least 1 submission submitted and that you have run the Post Submission stage.'

        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "TMLR/Action_Editors",
                    },
                    "entityB": { 
                        'type': "Note",
                        'withVenueid': "TMLR/Submitted",
                        'withContent': { 'human_subjects_reporting': 'Not applicable' }
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'})
        metadata = response.json['metadata']
        response = response.json['results']
        for item in response:
            print(item)
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1
