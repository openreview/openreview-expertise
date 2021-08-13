from unittest.mock import patch, MagicMock
import random
from pathlib import Path
import openreview
import json
import pytest
import numpy as np
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
        "TEST_NUM": random.randint(1, 100000)
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
        "broker_url": "redis://localhost:6379/0",
        "result_backend": "redis://localhost:6379/0",
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