from unittest.mock import patch, MagicMock
import random
from pathlib import Path
import openreview
import sys
import json
import pytest
import os
import time
import numpy as np
import shutil
import expertise.service
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import elmo
from expertise.service.utils import JobConfig, RedisDatabase


class TestExpertiseService():

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
            "OPENREVIEW_PASSWORD": "1234",
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
            "IN_TEST": True,
            "REDIS_ADDR": 'localhost',
            "REDIS_PORT": 6379,
            "REDIS_CONFIG_DB": 10,
            "REDIS_EMBEDDINGS_DB": 11,
            "model_params": {
                "use_redis": True
            }
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

    def test_on_redis_not_disk(self):
        # Load an example config and store it in Redis with no files
        redis = RedisDatabase(
            host='localhost',
            port=6379,
            db=10
        )

        # Find job using all jobs
        test_config = JobConfig(job_dir='./tests/jobs/abcde', job_id='abcde', user_id='test_user1@mail.com')
        redis.save_job(test_config)
        returned_configs = redis.load_all_jobs('test_user1@mail.com')
        assert returned_configs == []

        # Find job using job id
        test_config = JobConfig(job_dir='./tests/jobs/abcde', job_id='abcde', user_id='test_user1@mail.com')
        redis.save_job(test_config)
        try:
            returned_configs = redis.load_job(test_config.job_id, 'test_user1@mail.com')
        except openreview.OpenReviewException as e:
            assert str(e) == 'Job not found'

    def test_on_redis_on_disk(self):
        # Load an example config and store it in Redis with no files
        redis = RedisDatabase(
            host='localhost',
            port=6379,
            db=10
        )

        # Find job using all jobs
        #with open('./tests/data/example_config.json') as f:
        test_config = JobConfig(job_dir='./tests/jobs/abcde', job_id='abcde', user_id='test_user1@mail.com')
        redis.save_job(test_config)
        os.makedirs('./tests/jobs/abcde')
        returned_configs = redis.load_all_jobs('test_user1@mail.com')
        assert len(returned_configs) == 1
        assert returned_configs[0].user_id == 'test_user1@mail.com'
        assert returned_configs[0].job_id == 'abcde'

        # Find job using job id
        returned_config = redis.load_job(test_config.job_id, 'test_user1@mail.com')
        assert returned_config.user_id == 'test_user1@mail.com'
        assert returned_config.job_id == 'abcde'

        shutil.rmtree(f"./tests/jobs/")

    def test_request_expertise_with_no_config(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Submitting an empty config with no required fields
        response = test_client.post(
            '/expertise',
            data = json.dumps({}),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'Error' in response.json['name']
        assert 'bad request' in response.json['message'].lower()
        assert response.json['message'] == 'Bad request: required field missing in request: name'

    def test_request_expertise_with_no_second_entity(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field (only group entity)
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc" ,
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
        assert response.json['message'] == 'Bad request: required field missing in request: entityB'

    def test_request_expertise_with_empty_entity(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working config with an extra field that is not allowed
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": { },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        assert response.json['message'] == 'Bad request: required field missing in entityA: type'

    def test_request_expertise_with_missing_required_field_in_entity(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        assert response.json['message'] == 'Bad request: no valid Group properties in entityA'

    def test_request_expertise_with_unexpected_entity(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field (only group entity)
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityC": {},
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc" ,
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        assert response.json['message'] == "Bad request: unexpected fields in request: ['entityC']"

    def test_request_expertise_with_unexpected_field_in_entity(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field (only group entity)
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc",
                        'unexpected_field': 'unexpected_field'
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg',
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'Error' in response.json['name']
        assert 'bad request' in response.json['message'].lower()
        assert response.json['message'] == "Bad request: unexpected fields in entityA: ['unexpected_field']"

    def test_request_expertise_with_unexpected_model_param(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field (only group entity)
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc",
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
                    },
                    "model": {
                            "name": "specter+mfr",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'avg',
                            'unexpected_field': 'unexpected_field'
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'Error' in response.json['name']
        assert 'bad request' in response.json['message'].lower()
        assert response.json['message'] == "Bad request: unexpected fields in model: ['unexpected_field']"

    def test_request_expertise_with_empty_inclusion(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field (only group entity)
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc",
                        'expertise': { 'invitation': None }
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        assert response.json['message'] == "Bad request: Expertise invitation indicated but ID not provided"

    def test_request_expertise_with_valid_parameters(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc/Reviewers",
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'
        # assert response[0]['description'] == 'Server received config and allocated space'

        # # Attempt getting results of an incomplete job
        # time.sleep(5)
        # response = test_client.get('/expertise/results', query_string={'job_id': f'{job_id}'})
        # assert response.status_code == 500

        # Check for queued status
        #time.sleep(5)

        # response = test_client.get('/expertise/status', query_string={'job_id': f'{job_id}'}).json['results']
        # assert len(response) == 1
        # assert response[0]['name'] == 'test_run'
        # assert response[0]['status'] == 'Queued'
        # assert response[0]['description'] == 'Server received config and allocated space'

        # Check that the search returns an empty list before the job is completed when searching for completed
        response = test_client.get('/expertise/status/all', query_string={'status': 'Completed'}).json['results']
        assert len(response) == 0

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

        # After completion, check for non-empty completed list
        response = test_client.get('/expertise/status/all', query_string={'status': 'Completed'}).json['results']
        assert len(response) == 1
        assert response[0]['status'] == 'Completed'
        assert response[0]['name'] == 'test_run'

        response = test_client.get('/expertise/status/all', query_string={'status': 'Running'}).json['results']
        assert len(response) == 0

        openreview_context['job_id'] = job_id

        response = test_client.post(
            '/expertise',
            data=json.dumps({
                "name": "test_run2",
                "entityA": {
                    'type': "Group",
                    'memberOf': "ABC.cc/Reviewers",
                },
                "entityB": {
                    'type': "Note",
                    'invitation': "ABC.cc/-/Submission"
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

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        openreview_context['job_id2'] = job_id

    def test_status_all_query_params(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Test for status query
        response = test_client.get('/expertise/status/all', query_string={'status': 'Completed'}).json['results']
        assert len(response) == 2
        assert response[0]['status'] == 'Completed'
        assert response[1]['status'] == 'Completed'

        response = test_client.get('/expertise/status/all', query_string={'status': 'Running'}).json['results']
        assert len(response) == 0

        # Test for member query
        response = test_client.get('/expertise/status/all', query_string={'memberOf': 'ABC'}).json['results']
        assert len(response) == 2
        assert response[0]['request']['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert response[1]['request']['entityA']['memberOf'] == 'ABC.cc/Reviewers'

        response = test_client.get('/expertise/status/all', query_string={'memberOf': 'CBA'}).json['results']
        assert len(response) == 0

        # Test for invitation query
        response = test_client.get('/expertise/status/all', query_string={'paperInvitation': 'ABC.cc'}).json['results']
        assert len(response) == 2
        assert response[0]['request']['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response[1]['request']['entityB']['invitation'] == 'ABC.cc/-/Submission'

        response = test_client.get('/expertise/status/all', query_string={'paperInvitation': 'CBA'}).json['results']
        assert len(response) == 0

        # Test for combination
        response = test_client.get('/expertise/status/all', query_string={'status': 'Completed', 'memberOf': 'ABC'}).json['results']
        assert len(response) == 2
        assert response[0]['status'] == 'Completed'
        assert response[1]['status'] == 'Completed'

        response = test_client.get('/expertise/status/all', query_string={'status': 'Running', 'memberOf': 'ABC'}).json['results']
        assert len(response) == 0

        response = test_client.get('/expertise/status/all', query_string={'status': 'Running', 'memberOf': 'CBA'}).json['results']
        assert len(response) == 0

    def test_get_results_by_job_id(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        response = response.json['results']
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1

    def test_compare_results_for_identical_jobs(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        results_a = response.json['results']

        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id2']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        results_b = response.json['results']

        assert len(results_a) == len(results_b)
        for i in range(len(results_a)):
            assert results_a[i] == results_b[i]

    def test_inclusion_invitation(self, client, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID, HIJ has a single inclusion edge posted
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']
        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'HIJ.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityA']['expertise']['invitation'] == 'HIJ.cc/-/Expertise_Selection'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

    def test_inclusion_invitation_default(self, client, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID, ABC has no edges posted to it
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']
        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'ABC.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityA']['expertise']['invitation'] == 'ABC.cc/-/Expertise_Selection'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

    def test_exclusion_invitation(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'DEF.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityA']['expertise']['invitation'] == 'DEF.cc/-/Expertise_Selection'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

        # Assert size of archives folder is less than previous
        no_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{openreview_context['job_id']}/archives") if d.is_file())
        with_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{job_id}/archives") if d.is_file())
        assert with_exclusion < no_exclusion

    def test_get_results_for_all_jobs(self, openreview_context, celery_session_app, celery_session_worker):
        # Assert that there are two completed jobs belonging to this user
        test_client = openreview_context['test_client']
        response = test_client.get('/expertise/status/all', query_string={}).json['results']
        assert len(response) == 5
        for job_dict in response:
            assert job_dict['status'] == 'Completed'

    def test_get_results_and_delete_data(self, openreview_context, celery_session_app, celery_session_worker):
        # Clean up directories by setting the "delete_on_get" flag
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

        ## Assert the next expertise results should return empty result

    def test_request_expertise_with_model_errors(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a config with an error in the model field and return the job_id
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
                    },
                    "model": {
                            "name": "specter+mfr",
                            'sparseValue': 'notAnInt',
                            'useTitle': None, 
                            'useAbstract': None, 
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

        openreview_context['job_id'] = job_id

    def test_get_results_and_get_error(self, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        # Query until job is err
        time.sleep(5)
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}"})
        assert response.status_code == 404

        response = test_client.get('/expertise/status', query_string={'jobId': f"{openreview_context['job_id']}"}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f"{openreview_context['job_id']}"}).json
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['name'] == 'test_run'
        assert response['status'].strip() == 'Error'
        assert response['description'] == "'<' not supported between instances of 'int' and 'str'"
        assert response['cdate'] <= response['mdate']
        ###assert os.path.isfile(f"{server_config['WORKING_DIR']}/{job_id}/err.log")

        # Clean up error job by calling the delete endpoint
        response = test_client.get('/expertise/delete', query_string={'jobId': f"{openreview_context['job_id']}"}).json
        assert response['name'] == 'test_run'
        assert response['cdate'] <= response['mdate']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

    def test_request_expertise_with_no_submission_error(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a config with an error in the model field and return the job_id
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "HIJ.cc/Reviewers",
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "HIJ.cc/-/Submission" 
                    },
                    "model": {
                            "name": "specter+mfr",
                            'sparseValue': 'notAnInt',
                            'useTitle': None, 
                            'useAbstract': None, 
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

        openreview_context['job_id'] = job_id

    def test_get_results_and_get_no_submission_error(self, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        # Query until job is err
        time.sleep(5)
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}"})
        assert response.status_code == 404

        response = test_client.get('/expertise/status', query_string={'jobId': f"{openreview_context['job_id']}"}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f"{openreview_context['job_id']}"}).json
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['name'] == 'test_run'
        assert response['status'].strip() == 'Error'
        assert response['description'] == "Dimension out of range (expected to be in range of [-1, 0], but got 1). Please check that you have at least 1 submission submitted and that you have run the Post Submission stage."
        assert response['cdate'] <= response['mdate']
        ###assert os.path.isfile(f"{server_config['WORKING_DIR']}/{job_id}/err.log")

        # Clean up error job by calling the delete endpoint
        response = test_client.get('/expertise/delete', query_string={'jobId': f"{openreview_context['job_id']}"}).json
        assert response['name'] == 'test_run'
        assert response['cdate'] <= response['mdate']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")
    
    def test_request_journal(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        # Fetch a paper ID
        journal_papers = openreview_client.get_notes(invitation='TMLR/-/Submission')
        for paper in journal_papers:
            if paper.content['authorids']['value'][0] == '~SomeFirstName_User1':
                target_id = paper.id

        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['id'] == target_id
        openreview_context['job_id'] = job_id
    
    def test_get_journal_results(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 1
        response = response.json['results']
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1
        
        # Clean up journal request
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

    def test_high_load(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        test_client = openreview_context['test_client']
        num_requests = 3
        id_list = []

        # Fetch a paper ID
        journal_papers = openreview_client.get_notes(invitation='TMLR/-/Submission')
        for paper in journal_papers:
            if paper.content['authorids']['value'][0] == '~SomeFirstName_User1':
                target_id = paper.id

        # Make n requests
        for _ in range(num_requests):
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        "name": "test_run",
                        "entityA": {
                            'type': "Group",
                            'memberOf': "ABC.cc/Reviewers",
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
            id_list.append(job_id)
            time.sleep(2)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            assert response['name'] == 'test_run'
            assert response['status'] != 'Error'

        assert id_list is not None
        openreview_context['job_id'] = id_list
    
    def test_fetch_high_load_results(self, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 1200 # Timeout after 20 minutes
        assert openreview_context['job_id'] is not None
        id_list = openreview_context['job_id']
        num_requests = len(id_list)
        test_client = openreview_context['test_client']
        last_job_id = id_list[num_requests - 1]

        # Assert that the last request completes
        response = test_client.get('/expertise/status', query_string={'jobId': f'{last_job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{last_job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Now fetch and empty out all previous jobs
        for id in id_list:
            # Assert that they are complete
            response = test_client.get('/expertise/status', query_string={'jobId': f'{id}'}).json
            assert response['status'] == 'Completed'
            assert response['name'] == 'test_run'
            assert response['description'] == 'Job is complete and the computed scores are ready'

            response = test_client.get('/expertise/results', query_string={'jobId': f"{id}", 'deleteOnGet': True})
            metadata = response.json['metadata']
            assert metadata['submission_count'] == 1
            response = response.json['results']
            for item in response:
                submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
                assert len(submission_id) >= 1
                assert len(profile_id) >= 1
                assert profile_id.startswith('~')
                assert score >= 0 and score <= 1
            assert not os.path.isdir(f"./tests/jobs/{id}")

    def test_request_group_group(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc/Reviewers",
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'
        assert sum(1 for _ in os.scandir(f"./tests/jobs/{job_id}/archives")) > 1

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Group'
        assert req['entityB']['memberOf'] == 'ABC.cc/Reviewers'
        openreview_context['job_id'] = job_id
    
    def test_request_group_exclusion_exclusion(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'DEF.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'DEF.cc/-/Expertise_Selection' }
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
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
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Group'
        assert req['entityB']['memberOf'] == 'ABC.cc/Reviewers'

        # Assert size of archives folder is less than previous
        no_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{openreview_context['job_id']}/archives") if d.is_file())
        with_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{job_id}/archives") if d.is_file())
        assert with_exclusion < no_exclusion
        assert sum(1 for _ in os.scandir(f"./tests/jobs/{job_id}/archives")) > 1

        # Assert size of submissions file is less than previous
        no_exclusion = os.path.getsize(f"./tests/jobs/{openreview_context['job_id']}/submissions.json")
        with_exclusion = os.path.getsize(f"./tests/jobs/{job_id}/submissions.json")
        assert with_exclusion < no_exclusion
        openreview_context['exclusion_id'] = job_id

    def test_request_group_inclusion_exclusion(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'DEF.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'HIJ.cc/-/Expertise_Selection' }
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
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
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Group'
        assert req['entityB']['memberOf'] == 'ABC.cc/Reviewers'

        # Assert size of archives folder is less than previous
        no_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{openreview_context['job_id']}/archives") if d.is_file())
        with_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{job_id}/archives") if d.is_file())
        assert with_exclusion < no_exclusion
        assert sum(1 for _ in os.scandir(f"./tests/jobs/{job_id}/archives")) > 1

        # Assert size of submissions file is less than previous
        no_inclusion = os.path.getsize(f"./tests/jobs/{openreview_context['job_id']}/submissions.json")
        with_inclusion = os.path.getsize(f"./tests/jobs/{job_id}/submissions.json")
        with_exclusion = os.path.getsize(f"./tests/jobs/{openreview_context['exclusion_id']}/submissions.json")
        assert with_inclusion < no_inclusion
        assert with_inclusion < with_exclusion

    def test_request_group_inclusion_inclusion(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'HIJ.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'HIJ.cc/-/Expertise_Selection' }
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
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'jobId': f'{job_id}'}).json
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
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Group'
        assert req['entityB']['memberOf'] == 'ABC.cc/Reviewers'

        # Assert size of archives folder is less than previous
        no_inclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{openreview_context['job_id']}/archives") if d.is_file())
        with_inclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{job_id}/archives") if d.is_file())
        with_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{openreview_context['exclusion_id']}/archives") if d.is_file())
        assert sum(1 for _ in os.scandir(f"./tests/jobs/{job_id}/archives")) == 1
        assert with_inclusion < no_inclusion
        assert with_inclusion < with_exclusion

        # Assert size of submissions file is less than previous
        no_inclusion = os.path.getsize(f"./tests/jobs/{openreview_context['job_id']}/submissions.json")
        with_inclusion = os.path.getsize(f"./tests/jobs/{job_id}/submissions.json")
        with_exclusion = os.path.getsize(f"./tests/jobs/{openreview_context['exclusion_id']}/submissions.json")
        assert with_inclusion < no_inclusion
        assert with_inclusion < with_exclusion

    def test_get_group_results(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 8
        response = response.json['results']
        for item in response:
            match_id, submitter_id, score = item['match_member'], item['submission_member'], float(item['score'])
            assert len(match_id) >= 1
            assert len(submitter_id) >= 1
            assert match_id.startswith('~')
            assert submitter_id.startswith('~')
            assert score >= 0 and score <= 1
        
        # Clean up journal request
        response = test_client.get('/expertise/results', query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

        # Clean up directory
        shutil.rmtree(f"./tests/jobs/")
        if os.path.isfile('pytest.log'):
            os.remove('pytest.log')
        if os.path.isfile('default.log'):
            os.remove('default.log')