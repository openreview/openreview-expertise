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
from expertise.service.utils import preprocess_config

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
        "CHECK_EVERY": 30,
        "DELETE_AFTER": 15,
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

def test_queue_evict(openreview_context, celery_app, celery_worker):
    test_profile = 'test_user1@mail.com'
    server_config = openreview_context['config']
    test_client = openreview_context['test_client']

    if os.path.isdir(f"{server_config['WORKING_DIR']}"):
        shutil.rmtree(f"{server_config['WORKING_DIR']}")

    # Create fake job directory and config file - no score file
    if not os.path.isdir(f"{server_config['WORKING_DIR']}"):
        os.makedirs(f"{server_config['WORKING_DIR']}")
    del_job_id = 'AT3FX'
    os.makedirs(f"{server_config['WORKING_DIR']}/{del_job_id}")

    config = {
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
    new_config = preprocess_config(config, del_job_id, test_profile, server_config)
    new_config['cdate'] = new_config['cdate']
    with open(os.path.join(server_config['WORKING_DIR'], del_job_id, 'config.cfg'), 'w+') as f:
        json.dump(new_config, f, ensure_ascii=False, indent=4)

    # Create fake job directory and config file - with score file
    keep_job_id = 'YYYYY'
    os.makedirs(f"{server_config['WORKING_DIR']}/{keep_job_id}")

    config = {
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
    new_config = preprocess_config(config, keep_job_id, test_profile, server_config)
    new_config['cdate'] = new_config['cdate']  + 45 # Theoretically made in future
    with open(os.path.join(server_config['WORKING_DIR'], keep_job_id, 'config.cfg'), 'w+') as f:
        json.dump(new_config, f, ensure_ascii=False, indent=4)
    with open(os.path.join(server_config['WORKING_DIR'], keep_job_id, 'test_run.csv'), 'w+') as f:
        f.writelines(['1jxcf,~Test_User1,0.9358935'])

    # Send a request to kickstart the thread - first directory gets deleted on first request
    response = test_client.get('/expertise/status', query_string={}).json['results']
    assert len(response) == 2

    # Check for existence of directories
    time.sleep(30)
    assert os.path.isdir(os.path.join(server_config['WORKING_DIR'], keep_job_id))
    assert not os.path.isdir(os.path.join(server_config['WORKING_DIR'], del_job_id))

    # Sleep and check again
    time.sleep(60)
    assert not os.path.isdir(os.path.join(server_config['WORKING_DIR'], keep_job_id))

    # Clean up test
    shutil.rmtree(f"{server_config['WORKING_DIR']}/")
    if os.path.isfile('pytest.log'):
        os.remove('pytest.log')
    if os.path.isfile('default.log'):
        os.remove('default.log')