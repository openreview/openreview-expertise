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
        "WORKING_DIR": 'tmp',
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

def test_elmo_queue(openreview_context, celery_app, celery_worker):
    test_client = openreview_context['test_client']
    server_config = openreview_context['config']
    test_profile = '~Test_User1'
    
    if os.path.isdir(f'tmp/{test_profile}'):
        shutil.rmtree(f'tmp/{test_profile}')

    # Gather config
    config = {
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
    # Test missing required field
    response = test_client.post(
        '/expertise',
        data = json.dumps({**config}),
        content_type='application/json'
    )
    assert response.status_code == 500, f'{response.json}'

    # Test unexpected field
    config.update({'paper_invitation': 'ABC.cc/-/Submission'}) # Fill in required field
    config.update({'unexpected_field': 'ABC.cc/-/Submission'})
    response = test_client.post(
        '/expertise',
        data = json.dumps({**config}),
        content_type='application/json'
    )
    assert response.status_code == 500, f'{response.json}'

    # Test unexpected model param
    del config['unexpected_field']
    config.update({'model_params': {'dummy_param': '64'}})
    response = test_client.post(
        '/expertise',
        data = json.dumps({**config}),
        content_type='application/json'
    )
    assert response.status_code == 500, f'{response.json}'

    # Submit correct config
    del config['model_params']['dummy_param']
    response = test_client.post(
        '/expertise',
        data = json.dumps({**config}),
        content_type='application/json'
    )
    assert response.status_code == 200, f'{response.json}'
    job_id = response.json['job_id']

    # Attempt getting results of an incomplete job
    time.sleep(5)
    response = test_client.get('/results', query_string={'job_id': job_id})
    assert response.status_code == 500

    # Query until job is complete
    response = test_client.get('/jobs', query_string={}).json['results']
    assert len(response) == 1
    while response[0]['status'] == 'Processing':
        time.sleep(5)
        response = test_client.get('/jobs', query_string={}).json['results']
    assert response[0]['status'] == 'Completed'
    assert response[0]['name'] == 'test_run'

    # Check for results
    assert os.path.isdir(f"{server_config['WORKING_DIR']}/{test_profile}/{job_id}")
    assert os.path.isfile(f"{server_config['WORKING_DIR']}/{test_profile}/{job_id}/test_run.csv")
    response = test_client.get('/results', query_string={'job_id': job_id})
    metadata = response.json['metadata']
    assert metadata['submission_count'] == 2
    response = response.json['results']
    for item in response:
        submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
        assert len(submission_id) >= 1
        assert len(profile_id) >= 1
        assert profile_id.startswith('~')
        assert score >= 0 and score <= 1
    
    # Submit a second job
    response = test_client.post(
        '/expertise',
        data = json.dumps({**config}),
        content_type='application/json'
    )
    assert response.status_code == 200, f'{response.json}'
    job_id_two = response.json['job_id']

    # Query until second job is complete
    response = test_client.get('/jobs', query_string={}).json['results']
    assert len(response) == 2
    response = test_client.get('/jobs', query_string={'id': job_id_two}).json['results']
    assert len(response) == 1
    while response[0]['status'] == 'Processing':
        time.sleep(5)
        response = test_client.get('/jobs', query_string={'id': job_id_two}).json['results']
    assert response[0]['status'] == 'Completed'
    assert response[0]['name'] == 'test_run'

    # Clean up directories
    response = test_client.get('/results', query_string={'job_id': job_id, 'delete_on_get': True}).json['results']
    assert not os.path.isdir(f"{server_config['WORKING_DIR']}/{test_profile}/{job_id}")
    assert not os.path.isfile(f"{server_config['WORKING_DIR']}/~{test_profile}/{job_id}/test_run.csv")

    response = test_client.get('/results', query_string={'job_id': job_id_two, 'delete_on_get': True}).json['results']
    assert not os.path.isdir(f"{server_config['WORKING_DIR']}/{test_profile}/{job_id_two}")
    assert not os.path.isfile(f"{server_config['WORKING_DIR']}/~{test_profile}/{job_id_two}/test_run.csv")

    # Gather second config with an error in the model field
    config = {
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
    response = test_client.post(
        '/expertise',
        data = json.dumps({**config}),
        content_type='application/json'
    )
    assert response.status_code == 200, f'{response.json}'
    job_id = response.json['job_id']

    # Query until job is complete
    time.sleep(5)
    response = test_client.get('/results', query_string={'job_id': job_id})
    assert response.status_code == 500

    response = test_client.get('/jobs', query_string={}).json['results']
    assert len(response) == 1
    while response[0]['status'] == 'Processing':
        time.sleep(5)
        response = test_client.get('/jobs', query_string={}).json['results']
    
    assert 'Error' in response[0]['status']
    assert response[0]['name'] == 'test_run'
    assert len(response[0]['status'].strip()) > len('Error')
    assert os.path.isfile(f"{server_config['WORKING_DIR']}/{test_profile}/err.log")

    # Clean up test
    shutil.rmtree(f"{server_config['WORKING_DIR']}/")
    os.remove('pytest.log')
    os.remove('default.log')