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

    @patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob
    @patch("expertise.service.utils.storage.Client")  # Mock GCS Client
    def test_create_job(self, mock_storage_client, mock_pipeline_job, openreview_client, openreview_context):
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

        # Setup mock storage client
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Mock `upload_from_string` to simulate folder and file creation
        mock_blob.upload_from_string.return_value = None

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

        mock_pipeline_job.get.side_effect = [
            mock_pipeline_running,
            mock_pipeline_running,
            mock_pipeline_succeeded,
            mock_pipeline_succeeded
        ]

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
        time.sleep(2)

        # Setup Mock Bucket calls
        config = redis.load_job(job_id, openreview_context['config']['OPENREVIEW_USERNAME'])
        job_time = int(time.time() * 1000)
        mock_blob.name = f'{config.cloud_id}/request.json'
        mock_blob.download_as_string.return_value = json.dumps({
            "name": config.cloud_id,
            "user_id": "openreview.net",
            "cdate": int(time.time() * 1000)
        })
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = [mock_blob]

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
        assert len(response) == 0
        
        # Set complete
        mock_pipeline_ret = MagicMock()
        mock_pipeline_ret.state = PipelineState.PIPELINE_STATE_SUCCEEDED
        mock_pipeline_ret.update_time.timestamp.return_value = time.time()
        mock_pipeline_job.return_value.get.return_value = mock_pipeline_ret

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        if response['status'] == 'Error':
            assert False, response['description']
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
        assert len(response) == 1

        assert response[0]['status'] == 'Completed'
        assert response[0]['name'] == 'test_run'
        assert response[0]['description'] == 'Job is complete and the computed scores are ready'

        # Build mock scores
        mock_metadata_blob = MagicMock()
        mock_metadata_blob.name = f"jobs/{config.cloud_id}/metadata.json"
        mock_metadata_blob.download_as_string.return_value = json.dumps({"meta": "data"})

        mock_score_blob = MagicMock()
        mock_score_blob.name = f"jobs/{config.cloud_id}/scores.jsonl"
        mock_score_blob.download_as_string.return_value = '{"submission": "abcd","user": "user_user1","score": 0.987}\n{"submission": "abcd","user": "user_user2","score": 0.987}'

        mock_sparse_score_blob = MagicMock()
        mock_sparse_score_blob.name = f"jobs/{config.cloud_id}/scores_sparse.jsonl"
        mock_sparse_score_blob.download_as_string.return_value = '{"submission": "abcde","user": "user_user1","score": 0.987}\n{"submission": "abcde","user": "user_user2","score": 0.987}'

        mock_request_blob = MagicMock()
        mock_request_blob.name = f"jobs/{config.cloud_id}/request.json"
        mock_request_blob.download_as_string.return_value = json.dumps({
            "user_id": "test_user",
            "entityA": {"type": "Group"},
            "entityB": {"type": "Note"}
        })

        mock_storage_client.return_value.bucket.return_value.list_blobs.return_value = [
            mock_metadata_blob,
            mock_sparse_score_blob,
            mock_score_blob,
            mock_request_blob
        ]

        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
        assert response.json["metadata"] == {"meta": "data"}
        assert response.json["results"] == [
            {"submission": "abcde","user": "user_user1","score": 0.987},
            {"submission": "abcde","user": "user_user2","score": 0.987}
        ]