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

    def __init__(self):
        self.job_id = None

    @pytest.fixture
    def create_elmo():
        def simple_elmo(config):
            archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
            submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

            elmoModel = elmo.Model(
                use_title=config['model_params'].get('use_title'),
                use_abstract=config['model_params'].get('use_abstract'),
                use_cuda=config['model_params'].get('use_cuda'),
                batch_size=config['model_params'].get('batch_size'),
                knn=config['model_params'].get('knn'),
                sparse_value=config['model_params'].get('sparse_value')
            )
            elmoModel.set_archives_dataset(archives_dataset)
            elmoModel.set_submissions_dataset(submissions_dataset)
            return elmoModel
        return simple_elmo

    @pytest.fixture()
    def openreview_context():
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

    @pytest.fixture(scope="session")
    def celery_config():
        return {
            "broker_url": "redis://localhost:6379/10",
            "result_backend": "redis://localhost:6379/10",
            "task_track_started": True,
            "task_serializer": "pickle",
            "result_serializer": "pickle",
            "accept_content": ["pickle", "application/x-python-serialize"],
            "task_create_missing_queues": True,
        }

    @pytest.fixture(scope="session")
    def celery_includes():
        return ["expertise.service.celery_tasks"]

    @pytest.fixture(scope="session")
    def celery_worker_parameters():
        return {
            "queues": ("userpaper", "expertise"),
            "perform_ping_check": False,
            "concurrency": 4,
        }

    def test_request_expertise_with_no_config(self, openreview_context, celery_app, celery_worker):
        test_client = openreview_context['test_client']
        # Submitting an empty config with no required fields
        response = test_client.post(
            '/expertise',
            data = json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'bad request' in response.json['error'].lower()

    def test_request_expertise_with_missing_required_fields(self, openreview_context, celery_app, celery_worker):
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

    def test_request_expertise_with_invalid_field(self, openreview_context, celery_app, celery_worker):
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

    def test_request_expertise_with_invalid_model_param(self, openreview_context, celery_app, celery_worker):
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

    def test_request_expertise_with_valid_parameters(self, openreview_context, celery_app, celery_worker):
        # Submit a working job and return the job ID
        test_client = openreview_context['test_client']
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

        # Attempt getting results of an incomplete job
        time.sleep(5)
        response = test_client.get('/expertise/results', query_string={'id': job_id})
        assert response.status_code == 500

        # Check for queued status
        time.sleep(5)

        response = test_client.get('/expertise/status', query_string={'id': job_id}).json['results']
        assert len(response) == 1
        assert response[0]['status'] == 'Queued'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'id': job_id}).json['results']
        assert len(response) == 1
        while response[0]['status'] == 'Processing':
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'id': job_id}).json['results']
        assert response[0]['status'] == 'Completed'
        assert response[0]['name'] == 'test_run'

        self.job_id = job_id

    def test_get_results_by_job_id(self, openreview_context, celery_app, celery_worker):
        test_client = openreview_context['test_client']
        # Searches for results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'id': self.job_id})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        response = response.json['results']
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1

def test_get_results_for_all_jobs(self, openreview_context, celery_app, celery_worker):
    # Assert that there are two completed jobs belonging to this user
    test_client = openreview_context['test_client']
    response = test_client.get('/expertise/status', query_string={}).json['results']
    assert len(response) == 2
    for job_dict in response:
        assert job_dict['status'] == 'Completed'

def test_get_results_and_delete_data(self, openreview_context, celery_app, celery_worker):
    # Clean up directories by setting the "delete_on_get" flag
    test_client = openreview_context['test_client']
    response = test_client.get('/expertise/results', query_string={'id': self.job_id, 'delete_on_get': True}).json['results']

    ## Assert the next expertise results should return empty result

def test_request_expertise_with_model_errors(self, openreview_context, celery_app, celery_worker):
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
                    "use_title": None,
                    "use_abstract": None,
                    "average_score": None,
                    "max_score": None
                }
            }
        ),
        content_type='application/json'
    )
    assert response.status_code == 200, f'{response.json}'
    job_id = response.json['job_id']

    self.job_id = job_id

def test_get_results_and_get_error(self, openreview_context, celery_app, celery_worker):
    test_client = openreview_context['test_client']
    # Query until job is err
    time.sleep(5)
    response = test_client.get('/expertise/results', query_string={'id': self.job_id})
    assert response.status_code == 500

    response = test_client.get('/expertise/status', query_string={}).json['results']
    assert len(response) == 1
    while response[0]['status'] == 'Processing':
        time.sleep(5)
        response = test_client.get('/expertise/status', query_string={}).json['results']

    assert response[0]['name'] == 'test_run'
    assert response[0]['status'].strip() == 'Error'
    assert 'error' in response[0].keys()
    ## TODO: assert the error string
    ###assert os.path.isfile(f"{server_config['WORKING_DIR']}/{job_id}/err.log")


# def test_elmo_queue(openreview_context, celery_app, celery_worker):
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