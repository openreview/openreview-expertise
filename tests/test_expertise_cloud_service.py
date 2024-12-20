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
from expertise.service.utils import JobConfig, RedisDatabase
from google.cloud.aiplatform_v1.types import PipelineState

class LocalMockBlob:
    def __init__(self, file_path, base_dir):
        self._file_path = file_path
        self._base_dir = base_dir
        # Blob name relative to the bucket base dir
        self.name = os.path.relpath(file_path, self._base_dir)

    def download_as_string(self):
        with open(self._file_path, 'rb') as f:
            return f.read()

    def upload_from_string(self, data='', content_type=None):
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        if not os.path.isdir(self._file_path): ## Only write to file if it's not a directory
            with open(self._file_path, 'w', encoding='utf-8') as f:
                f.write(data)

class LocalMockBucket:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def list_blobs(self, prefix=None, max_results=None):
        search_path = os.path.join(self.base_dir, prefix) if prefix else self.base_dir
        if not os.path.exists(search_path):
            return []
        blobs = []
        for root, dirs, files in os.walk(search_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                blobs.append(LocalMockBlob(file_path, self.base_dir))
        if max_results:
            blobs = blobs[:max_results]
        return blobs

    def blob(self, blob_name):
        file_path = os.path.join(self.base_dir, blob_name)
        return LocalMockBlob(file_path, self.base_dir)

class LocalMockClient:
    def __init__(self, project=None, root_dir=None):
        self.project = project
        self.root_dir = root_dir
        # Simulate a bucket directory inside root_dir
        self.bucket_dir = os.path.join(self.root_dir, 'test-bucket')
        os.makedirs(self.bucket_dir, exist_ok=True)

    def bucket(self, bucket_name):
        # We ignore bucket_name since we have only one test-bucket
        return LocalMockBucket(self.bucket_dir)

class TestExpertiseCloudService():

    job_id = None

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
            "IN_TEST": True,
            "REDIS_ADDR": 'localhost',
            "REDIS_PORT": 6379,
            "REDIS_CONFIG_DB": 10,
            "REDIS_EMBEDDINGS_DB": 11,
            "USE_GCP": True,
            "GCP_PROJECT_ID":'test_project',
            "GCP_PROJECT_NUMBER":'123456',
            "GCP_REGION":'us-central1',
            "GCP_PIPELINE_ROOT":'pipeline-root',
            "GCP_PIPELINE_NAME":'test-pipeline',
            "GCP_PIPELINE_REPO":'test-repo',
            "GCP_PIPELINE_TAG":'dev',
            "GCP_BUCKET_NAME":'test-bucket',
            "GCP_JOBS_FOLDER":'jobs',
            "GCP_SERVICE_LABEL":{'dev': 'expertise'},
            "POLL_INTERVAL": 1,
            "POLL_MAX_ATTEMPTS": 5,
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

    @patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob to avoid calling AI Platform
    def test_create_job_filesystem(self, mock_pipeline_job, openreview_client, openreview_context):
        redis = RedisDatabase(
            host=openreview_context['config']['REDIS_ADDR'],
            port=openreview_context['config']['REDIS_PORT'],
            db=openreview_context['config']['REDIS_CONFIG_DB'],
            sync_on_disk=False
        )

        # Clear Redis
        configs = redis.load_all_jobs(openreview_context['config']['OPENREVIEW_USERNAME'])
        for config in configs:
            redis.remove_job(openreview_context['config']['OPENREVIEW_USERNAME'], config.job_id)

        # Setup mock PipelineJob
        mock_pipeline_instance = MagicMock()
        mock_pipeline_job.return_value = mock_pipeline_instance

        # Mock PipelineJob.get()
        mock_pipeline_running = MagicMock()
        mock_pipeline_running.state = PipelineState.PIPELINE_STATE_RUNNING
        mock_pipeline_running.update_time.timestamp.return_value = time.time()

        mock_pipeline_succeeded = MagicMock()
        mock_pipeline_succeeded.state = PipelineState.PIPELINE_STATE_SUCCEEDED
        mock_pipeline_succeeded.update_time.timestamp.return_value = time.time()

        mock_pipeline_job.get.side_effect = [mock_pipeline_running] * 6 + [mock_pipeline_succeeded] * 7

        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']

        tmp_dir = Path('tests/gcp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Make a request
        # Patch storage.Client to return our local mock client that uses the filesystem
        with patch("google.cloud.storage.Client", new=lambda project=None: LocalMockClient(project=project, root_dir=tmp_dir)):
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
                        },
                        "dataset": {
                            'minimumPubDate': 0
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
            response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
            assert len(response) == 0

            time.sleep(
                openreview_context['config']['POLL_INTERVAL'] * openreview_context['config']['POLL_MAX_ATTEMPTS'] + 1
            )

            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            assert response['status'] == 'Completed'
            response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
            assert len(response) == 1

            assert response[0]['status'] == 'Completed'
            assert response[0]['name'] == 'test_run'
            assert response[0]['description'] == 'Job is complete and the computed scores are ready'

            # Build mock scores

            # Fetch the job config
            ## Convert current mocking to using file system
            config = redis.load_job(job_id, openreview_context['config']['OPENREVIEW_USERNAME'])
            
            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/metadata.json"), 'w') as f:
                f.write(json.dumps({"meta": "data"}))

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores.jsonl"), 'w') as f:
                f.write('{"submission": "abcd","user": "user_user1","score": 0.987}\n{"submission": "abcd","user": "user_user2","score": 0.987}')

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores_sparse.jsonl"), 'w') as f:
                f.write('{"submission": "abcde","user": "user_user1","score": 0.987}\n{"submission": "abcde","user": "user_user2","score": 0.987}')

            # Searches for journal results from the given job_id assuming the job has completed
            response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
            assert response.json["metadata"] == {"meta": "data"}
            assert response.json["results"] == [
                {"submission": "abcde","user": "user_user1","score": 0.987},
                {"submission": "abcde","user": "user_user2","score": 0.987}
            ]

            # Teardown tmp_dir
            shutil.rmtree(tmp_dir, ignore_errors=True)