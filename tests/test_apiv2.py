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
from expertise.models import elmo
from expertise.service.utils import mock_client

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

    def test_get_publications(self):
        openreview_client = mock_client(version=1)
        openreview_client_v2 = mock_client(version=2)
        config = {
            'dataset': {
                'top_recent_pubs': 3,
            }
        }
        or_expertise = OpenReviewExpertise(openreview_client, openreview_client_v2, config)
        publications = or_expertise.get_publications('~Carlos_Mondragon1')
        assert publications == []

        or_expertise = OpenReviewExpertise(openreview_client, openreview_client_v2, config)
        publications = or_expertise.get_publications('~Harold_Rice8')
        assert len(publications) == 3
        for pub in publications:
            content = pub['content']
            assert isinstance(content['title'], str)
            assert isinstance(content['abstract'], str)


    def test_get_submissions_from_invitation_v1(self):
        # Returns the V1 submissions
        openreview_client = mock_client(version=1)
        openreview_client_v2 = mock_client(version=2)

        config = {
            'use_email_ids': False,
            'match_group': 'ABC.cc',
            'paper_invitation': 'ABC.cc/-/Submission',
        }
        or_expertise = OpenReviewExpertise(openreview_client, openreview_client_v2, config)
        submissions = or_expertise.get_submissions()
        print(submissions)
        assert not isinstance(submissions['KHnr1r7H']['content']['title'], dict)
        assert isinstance(submissions['KHnr1r7H']['content']['title'], str)
        assert not isinstance(submissions['KHnr1r7H']['content']['abstract'], dict)
        assert isinstance(submissions['KHnr1r7H']['content']['abstract'], str)
        assert json.dumps(submissions) == json.dumps({
            'KHnr1r7H': {
                "id": "KHnr1r7H",
                "content": {
                    "title": "Repair Right Metatarsal, Percutaneous Endoscopic Approach",
                    "abstract": "Nam ultrices, libero non mattis pulvinar, nulla pede ullamcorper augue, a suscipit nulla elit ac nulla. Sed vel enim sit amet nunc viverra dapibus. Nulla suscipit ligula in lacus.\n\nCurabitur at ipsum ac tellus semper interdum. Mauris ullamcorper purus sit amet nulla. Quisque arcu libero, rutrum ac, lobortis vel, dapibus at, diam."
                }
            },
            'YQtWeE8P': {
                "id": "YQtWeE8P",
                "content": {
                    "title": "Bypass L Com Iliac Art to B Com Ilia, Perc Endo Approach",
                    "abstract": "Nullam sit amet turpis elementum ligula vehicula consequat. Morbi a ipsum. Integer a nibh.\n\nIn quis justo. Maecenas rhoncus aliquam lacus. Morbi quis tortor id nulla ultrices aliquet.\n\nMaecenas leo odio, condimentum id, luctus nec, molestie sed, justo. Pellentesque viverra pede ac diam. Cras pellentesque volutpat dui."
                }
            }
        })

    def test_get_submissions_from_invitation_v2(self):
        # Returns the V1 submissions
        openreview_client = mock_client(version=1)
        openreview_client_v2 = mock_client(version=2)

        config = {
            'use_email_ids': False,
            'match_group': 'ABC.cc',
            'paper_invitation': 'ABC.cc/-/Blind_Submission',
        }
        or_expertise = OpenReviewExpertise(openreview_client, openreview_client_v2, config)
        submissions = or_expertise.get_submissions()
        print(submissions)
        assert not isinstance(submissions['KHnr1r7h']['content']['title'], dict)
        assert isinstance(submissions['KHnr1r7h']['content']['title'], str)
        assert not isinstance(submissions['KHnr1r7h']['content']['abstract'], dict)
        assert isinstance(submissions['KHnr1r7h']['content']['abstract'], str)
        assert json.dumps(submissions) == json.dumps({
            'KHnr1r7h': {
                "id": "KHnr1r7h",
                "content": {
                    "title": "Right Metatarsal, Endoscopic Approach",
                    "abstract": "Nam ultrices, libero non mattis pulvinar, nulla pede ullamcorper augue, a suscipit nulla elit ac nulla. Sed vel enim sit amet nunc viverra dapibus. Nulla suscipit ligula in lacus.\n\nCurabitur at ipsum ac tellus semper interdum. Mauris ullamcorper purus sit amet nulla. Quisque arcu libero, rutrum ac, lobortis vel."
                }
            },
            'YQtWeE8p': {
                "id": "YQtWeE8p",
                "content": {
                    "title": "Iliac Art to B Com Ilia, Perc Endo Approach",
                    "abstract": "Nullam sit amet turpis elementum ligula vehicula consequat. Morbi a ipsum. Integer a nibh.\n\nIn quis justo. Maecenas rhoncus aliquam lacus. Morbi quis tortor id nulla ultrices aliquet.\n\nMaecenas leo odio, condimentum id, luctus nec, molestie sed, justo. Pellentesque viverra pede ac diam."
                }
            }
        })

    def test_get_by_submissions_from_paper_id(self):
        openreview_client = mock_client(version=1)
        openreview_client_v2 = mock_client(version=2)

        config = {
            'paper_id': 'KHnr1r7h',
        }
        or_expertise = OpenReviewExpertise(openreview_client, openreview_client_v2, config)
        submissions = or_expertise.get_submissions()
        print(submissions)
        assert not isinstance(submissions['KHnr1r7h']['content']['title'], dict)
        assert not isinstance(submissions['KHnr1r7h']['content']['abstract'], dict)
        assert json.dumps(submissions) == json.dumps({
            'KHnr1r7h': {
                "id": "KHnr1r7h",
                "content": {
                    "title": "Right Metatarsal, Endoscopic Approach",
                    "abstract": "Nam ultrices, libero non mattis pulvinar, nulla pede ullamcorper augue, a suscipit nulla elit ac nulla. Sed vel enim sit amet nunc viverra dapibus. Nulla suscipit ligula in lacus.\n\nCurabitur at ipsum ac tellus semper interdum. Mauris ullamcorper purus sit amet nulla. Quisque arcu libero, rutrum ac, lobortis vel."
                }
            }
        })

    def test_journal_request_v2(self, openreview_context, celery_session_app, celery_session_worker):
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
                        'memberOf': "ABC.cc",
                    },
                    "entityB": { 
                        'type': "Note",
                        'id': 'KHnr1r7h'
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
            content_type='application/json'
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['job_id']
        time.sleep(2)
        response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', query_string={'id': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response[0]['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'
        
        # Check config fields
        returned_config = response['config']
        assert returned_config['name'] == 'test_run'
        assert returned_config['paper_id'] == 'KHnr1r7h'
        assert returned_config['model'] == 'specter+mfr'
        assert 'token' not in returned_config
        assert 'baseurl' not in returned_config
        assert 'user_id' not in returned_config
        assert job_id is not None
        openreview_context['job_id'] = job_id

    def test_get_journal_results(self, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', query_string={'id': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        response = response.json['results']
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            assert submission_id != 'dummy'
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1
        
        # Clean up journal request
        response = test_client.get('/expertise/results', query_string={'id': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")
        # Clean up directory
        shutil.rmtree(f"./tests/jobs/")
        if os.path.isfile('pytest.log'):
            os.remove('pytest.log')
        if os.path.isfile('default.log'):
            os.remove('default.log')