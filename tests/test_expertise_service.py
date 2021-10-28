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

    def test_request_expertise_with_no_config(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Submitting an empty config with no required fields
        response = test_client.post(
            '/expertise',
            data = json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'bad request' in response.json['error'].lower()
        assert response.json['error'] == 'Bad request: missing required field: name match_group paper_invitation'

    def test_request_expertise_with_missing_required_fields(self, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a partially filled out config without a required field
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    'name': 'test_run',
                    'match_group': ["ABC.cc"],
                    "model": "elmo",
                    "model_params": {
                        "use_title": False,
                        "use_abstract": True,
                        "average_score": True,
                        "max_score": False
                    }
                }
            ),
            content_type='application/json'
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'bad request' in response.json['error'].lower()
        assert response.json['error'] == 'Bad request: missing required field: paper_invitation'

    def test_request_expertise_with_invalid_field(self, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working config with an extra field that is not allowed
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    'name': 'test_run',
                    'match_group': ["ABC.cc"],
                    'paper_invitation': 'ABC.cc/-/Submission',
                    'unexpected_field': 'ABC.cc/-/Submission',
                    "model": "elmo",
                    "model_params": {
                        "use_title": False,
                        "use_abstract": True,
                        "average_score": True,
                        "max_score": False
                    }
                }
            ),
            content_type='application/json'
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'bad request' in response.json['error'].lower()
        assert response.json['error'] == 'Bad request: unexpected field: unexpected_field'

    def test_request_expertise_with_invalid_and_missing_required_field(self, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working config with an extra field that is not allowed
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    'match_group': ["ABC.cc"],
                    'paper_invitation': 'ABC.cc/-/Submission',
                    'unexpected_field': 'ABC.cc/-/Submission',
                    "model": "elmo",
                    "model_params": {
                        "use_title": False,
                        "use_abstract": True,
                        "average_score": True,
                        "max_score": False
                    }
                }
            ),
            content_type='application/json'
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'bad request' in response.json['error'].lower()
        assert response.json['error'] == 'Bad request: missing required field: name\nunexpected field: unexpected_field'

    def test_request_expertise_with_invalid_model_param(self, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working config with an extra model param field
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    'name': 'test_run',
                    'match_group': ["ABC.cc"],
                    'paper_invitation': 'ABC.cc/-/Submission',
                    "model": "elmo",
                    "model_params": {
                        'dummy_param': '64',
                        "use_title": False,
                        "use_abstract": True,
                        "average_score": True,
                        "max_score": False
                    }
                }
            ),
            content_type='application/json'
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'bad request' in response.json['error'].lower()
        assert response.json['error'] == 'Bad request: unexpected model param: dummy_param'

    def test_request_expertise_with_valid_parameters(self, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']
        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    'name': 'test_run',
                    'match_group': ["ABC.cc"],
                    'paper_invitation': 'ABC.cc/-/Submission',
                    "model": "elmo",
                    "model_params": {
                        "use_title": False,
                        "use_abstract": True,
                        "average_score": True,
                        "max_score": False
                    }
                }
            ),
            content_type='application/json'
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['job_id']
        time.sleep(2)
        response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json['results']
        assert len(response) == 1
        assert response[0]['name'] == 'test_run'
        assert response[0]['status'] != 'Error'
        # assert response[0]['description'] == 'Server received config and allocated space'

        # # Attempt getting results of an incomplete job
        # time.sleep(5)
        # response = test_client.get('/expertise/results', query_string={'id': f'{job_id}'})
        # assert response.status_code == 500

        # Check for queued status
        #time.sleep(5)

        # response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json['results']
        # assert len(response) == 1
        # assert response[0]['name'] == 'test_run'
        # assert response[0]['status'] == 'Queued'
        # assert response[0]['description'] == 'Server received config and allocated space'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json['results']
        assert len(response) == 1
        start_time = time.time()
        try_time = time.time() - start_time
        while response[0]['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json['results']
            if response[0]['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response[0]['status'] == 'Completed'
        assert response[0]['name'] == 'test_run'
        assert response[0]['description'] == 'Job is complete and the computed scores are ready'
        
        # Check config fields
        returned_config = response[0]['config']
        assert returned_config['name'] == 'test_run'
        assert returned_config['paper_invitation'] == 'ABC.cc/-/Submission'
        assert returned_config['model'] == 'elmo'
        assert 'token' not in returned_config
        assert 'baseurl' not in returned_config
        assert 'user_id' not in returned_config
        assert job_id is not None
        openreview_context['job_id'] = job_id

    def test_get_results_by_job_id(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'id': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        response = response.json['results']
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1

    def test_get_results_for_all_jobs(self, openreview_context, celery_session_app, celery_session_worker):
        # Assert that there are two completed jobs belonging to this user
        test_client = openreview_context['test_client']
        response = test_client.get('/expertise/status/all', query_string={}).json['results']
        assert len(response) == 1
        for job_dict in response:
            assert job_dict['status'] == 'Completed'

    def test_get_results_and_delete_data(self, openreview_context, celery_session_app, celery_session_worker):
        # Clean up directories by setting the "delete_on_get" flag
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        response = test_client.get('/expertise/results', query_string={'id': f"{openreview_context['job_id']}", 'delete_on_get': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

        ## Assert the next expertise results should return empty result

    def test_request_expertise_with_model_errors(self, openreview_context, celery_session_app, celery_session_worker):
        # Submit a config with an error in the model field and return the job_id
        test_client = openreview_context['test_client']
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    'name': 'test_run',
                    'paper_invitation': 'ABC.cc/-/Submission',
                    'match_group': ["ABC.cc"],
                    "model": "elmo",
                    "model_params": {
                        "use_title": False,
                        "use_abstract": False,
                        "average_score": None,
                        "max_score": None
                    }
                }
            ),
            content_type='application/json'
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['job_id']

        openreview_context['job_id'] = job_id

    def test_get_results_and_get_error(self, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        # Query until job is err
        time.sleep(5)
        response = test_client.get('/expertise/results', query_string={'id': f"{openreview_context['job_id']}"})
        assert response.status_code == 404

        response = test_client.get('/expertise/status', query_string={'id': f"{openreview_context['job_id']}"}).json['results']
        assert len(response) == 1
        start_time = time.time()
        try_time = time.time() - start_time
        while response[0]['status'] == 'Processing' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'id': f"{openreview_context['job_id']}"}).json['results']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response[0]['name'] == 'test_run'
        assert response[0]['status'].strip() == 'Error'
        assert response[0]['description'] == 'use_title and use_abstract cannot both be False'
        ###assert os.path.isfile(f"{server_config['WORKING_DIR']}/{job_id}/err.log")

        # Clean up error job
        shutil.rmtree(f"./tests/jobs/{openreview_context['job_id']}")
    
    def test_high_load(self, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        test_client = openreview_context['test_client']
        num_requests = 5
        id_list = []
        # Make n requests
        for _ in range(num_requests):
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        'name': 'test_run',
                        'match_group': ["ABC.cc"],
                        'paper_invitation': 'ABC.cc/-/Submission',
                        "model": "elmo",
                        "model_params": {
                            "use_title": False,
                            "use_abstract": True,
                            "average_score": True,
                            "max_score": False
                        }
                    }
                ),
                content_type='application/json'
            )
            assert response.status_code == 200, f'{response.json}'
            job_id = response.json['job_id']
            id_list.append(job_id)
            time.sleep(2)
            response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json['results']
            assert len(response) == 1
            assert response[0]['name'] == 'test_run'
            assert response[0]['status'] != 'Error'

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
        response = test_client.get('/expertise/status', query_string={'id': f'{last_job_id}'}).json['results']
        assert len(response) == 1
        start_time = time.time()
        try_time = time.time() - start_time
        while response[0]['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'id': f'{last_job_id}'}).json['results']
            if response[0]['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response[0]['status'] == 'Completed'
        assert response[0]['name'] == 'test_run'
        assert response[0]['description'] == 'Job is complete and the computed scores are ready'

        # Now fetch and empty out all previous jobs
        for id in id_list:
            # Assert that they are complete
            response = test_client.get('/expertise/status', query_string={'id': f'{id}'}).json['results']
            assert response[0]['status'] == 'Completed'
            assert response[0]['name'] == 'test_run'
            assert response[0]['description'] == 'Job is complete and the computed scores are ready'

            response = test_client.get('/expertise/results', query_string={'id': f"{id}", 'delete_on_get': True})
            metadata = response.json['metadata']
            assert metadata['submission_count'] == 2
            response = response.json['results']
            for item in response:
                submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
                assert len(submission_id) >= 1
                assert len(profile_id) >= 1
                assert profile_id.startswith('~')
                assert score >= 0 and score <= 1
            assert not os.path.isdir(f"./tests/jobs/{id}")

        # Clean up directory
        shutil.rmtree(f"./tests/jobs/")
        os.remove('pytest.log')
        os.remove('default.log')

# def test_elmo_queue(openreview_context, celery_session_app, celery_session_worker):
#     test_client = openreview_context['test_client']
#     server_config = openreview_context['config']
#     test_profile = '~Test_User1'


#     working_dir = openreview_context['config']['WORKING_DIR']
#     if os.path.isdir(working_dir):
#         shutil.rmtree(working_dir)

#     # Gather config
#     config = {
#         'name': 'test_run',
#         'match_group': ["ABC.cc"],
#         "model": "elmo",
#         "model_params": {
#             "use_title": False,
#             "use_abstract": True,
#             "average_score": True,
#             "max_score": False
#         }
#     }
#     # Test empty config
#     run_empty_config(test_client)

#     # Test missing required field
#     run_missing_required(test_client)

#     # Test unexpected field
#     run_unexpected_field(test_client)

#     # Test unexpected model param
#     run_unexpected_model_param(test_client)

#     # Submit correct config
#     job_id = run_correct_job_submit(test_client)

#     # Submit a second job
#     job_id_two = run_correct_job_submit(test_client)

#     # Check for queued status
#     run_check_queued_status(test_client, job_id_two)

#     # Query until job is complete
#     run_query_until_complete(test_client, job_id)

#     # Check for results
#     run_check_results(test_client, server_config, job_id)

#     # Query until second job is complete
#     run_query_until_complete(test_client, job_id_two)
#     run_check_two_completed_jobs(test_client)

#     # Clean up directories
#     run_clean_up_jobs(test_client, server_config, job_id)
#     run_clean_up_jobs(test_client, server_config, job_id_two)

#     # Gather second config with an error in the model field
#     job_id = run_submit_err_job(test_client)

#     # Query until job is complete
#     run_query_until_err(test_client, server_config, job_id)

#     # Clean up test
#     shutil.rmtree(f"{server_config['WORKING_DIR']}/")
#     os.remove('pytest.log')
#     os.remove('default.log')