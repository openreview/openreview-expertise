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
from expertise.service.utils import JobConfig, RedisDatabase, get_user_id
from google.cloud.aiplatform_v1.types import PipelineState

# Default parameters for the module's common setup
DEFAULT_JOURNAL_ID = 'TMLR'
DEFAULT_CONF_ID = 'CLD.cc'
DEFAULT_POST_REVIEWERS = True
DEFAULT_POST_AREA_CHAIRS = False
DEFAULT_POST_SENIOR_AREA_CHAIRS = False
DEFAULT_POST_SUBMISSIONS = True
DEFAULT_POST_PUBLICATIONS = True

@pytest.fixture(scope="module", autouse=True)
def _setup_tmlr(clean_start_journal, client, openreview_client):
    clean_start_journal(
        openreview_client,
        DEFAULT_JOURNAL_ID,
        editors=['~Raia_Hadsell1', '~Kyunghyun_Cho1'],
        additional_editors=['~Margherita_Hilpert1'],
        post_submissions=True,
        post_publications=True,
        post_editor_data=True
    )

@pytest.fixture(scope="module", autouse=True)
def _setup_abc_cc(clean_start_conference, client, openreview_client):
    clean_start_conference(
        client,
        DEFAULT_CONF_ID,
        fake_data_source_id='ABC.cc',
        post_reviewers=DEFAULT_POST_REVIEWERS,
        post_area_chairs=DEFAULT_POST_AREA_CHAIRS,
        post_senior_area_chairs=DEFAULT_POST_SENIOR_AREA_CHAIRS,
        post_submissions=DEFAULT_POST_SUBMISSIONS,
        post_publications=DEFAULT_POST_PUBLICATIONS
    )

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
        prefix = prefix or ''
        blobs = []
        
        for root, dirs, files in os.walk(self.base_dir):
            for filename in files:
                rel_dir = os.path.relpath(root, self.base_dir)
                if rel_dir == '.':
                    rel_dir = ''
                object_path = os.path.join(rel_dir, filename)
                object_path = object_path.replace('\\', '/')
                
                if object_path.startswith(prefix):
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

@pytest.fixture(autouse=True)
def reset_run_once_state():
    import expertise.service.routes as rts
    rts.get_expertise_service.has_run = False
    rts.get_expertise_service.to_return = None
    yield
    rts.get_expertise_service.has_run = False
    rts.get_expertise_service.to_return = None


class TestExpertiseCloudService():

    job_id = None

    @pytest.fixture(scope='class')
    def openreview_context_cloud(self):
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
            "REDIS_CONFIG_DB": 9,
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
    def test_create_job_filesystem(self, mock_pipeline_job, openreview_client, openreview_context_cloud):
        def setup_job_mocks():
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

            mock_pipeline_job.get.side_effect = [mock_pipeline_running] * 4 + [mock_pipeline_succeeded] * 10

            return mock_pipeline_instance

        MAX_TIMEOUT = 300
        redis = RedisDatabase(
            host=openreview_context_cloud['config']['REDIS_ADDR'],
            port=openreview_context_cloud['config']['REDIS_PORT'],
            db=openreview_context_cloud['config']['REDIS_CONFIG_DB'],
            sync_on_disk=False
        )

        # Submit first job as CLD.cc/Program_Chairs
        abc_client = openreview.api.OpenReviewClient(
            token=openreview_client.token
        )
        abc_client.impersonate('CLD.cc/Program_Chairs')

        # Submit as TMLR/Editors_In_Chiefs
        tmlr_client = openreview.api.OpenReviewClient(
            token=openreview_client.token
        )
        tmlr_client.impersonate('TMLR/Editors_In_Chief')

        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context_cloud['test_client']

        tmp_dir = Path('tests/gcp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Make a request
        # Patch storage.Client to return our local mock client that uses the filesystem
        with patch("google.cloud.storage.Client", new=lambda project=None: LocalMockClient(project=project, root_dir=tmp_dir)):
            setup_job_mocks()
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        "name": "test_run",
                        "entityA": {
                            'type': "Group",
                            'memberOf': "CLD.cc/Area_Chairs",
                        },
                        "entityB": { 
                            'type': "Note",
                            'invitation': "CLD.cc/-/Submission" 
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
                headers=abc_client.headers
            )
            assert response.status_code == 200, f'{response.json}'
            job_id = response.json['jobId']
            time.sleep(0.5)

            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['name'] == 'test_run'
            assert response['status'] != 'Error'

            # Let request process
            time.sleep(openreview_context_cloud['config']['POLL_INTERVAL'] * openreview_context_cloud['config']['POLL_MAX_ATTEMPTS'])
            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            # Check proper user ID
            config = redis.load_job(job_id, openreview_context_cloud['config']['OPENREVIEW_USERNAME'])
            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/request.json"), 'r') as f:
                request = json.load(f)
                assert request['user_id'] == 'CLD.cc/Program_Chairs'
            
            setup_job_mocks()
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        "name": "test_run",
                        "entityA": {
                            'type': "Group",
                            'memberOf': "CLD.cc/Reviewers",
                        },
                        "entityB": { 
                            'type': "Note",
                            'invitation': "CLD.cc/-/Submission" 
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
                headers=tmlr_client.headers
            )
            assert response.status_code == 200, f'{response.json}'
            job_id = response.json['jobId']
            time.sleep(0.5)

            response = test_client.get('/expertise/status', headers=tmlr_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['name'] == 'test_run'
            assert response['status'] != 'Error'
            responses = test_client.get('/expertise/status/all', headers=tmlr_client.headers, query_string={'status': 'Completed'}).json['results']
            assert not any([r['jobId'] == job_id for r in responses])

            # Perform single query after waiting max time
            time.sleep(openreview_context_cloud['config']['POLL_INTERVAL'] * openreview_context_cloud['config']['POLL_MAX_ATTEMPTS'])
            response = test_client.get('/expertise/status', headers=tmlr_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            ## Expect 2*4 calls from the worker thread, 2*2 call from /expertise/status and 0 calls from /expertise/status/all
            print(mock_pipeline_job.get.call_args_list)
            assert len(mock_pipeline_job.get.call_args_list) == 12

            response = test_client.get('/expertise/status', headers=tmlr_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            responses = test_client.get('/expertise/status/all', headers=tmlr_client.headers, query_string={'status': 'Completed'}).json['results']
            assert any([r['jobId'] == job_id for r in responses])
            responses = test_client.get('/expertise/status/all', headers=tmlr_client.headers, query_string={
                "entityA.memberOf": "CLD.cc/Reviewers",
                "entityB.invitation": "CLD.cc/-/Submission"
            }).json['results']
            assert any([r['jobId'] == job_id for r in responses])
            responses = test_client.get('/expertise/status/all', headers=tmlr_client.headers, query_string={
                "entityA.memberOf": "CLD.cc/Reviewers",
                "entityB.invitation": "CLD.cc/-/Submission",
                'status': 'Completed'
            }).json['results']
            assert any([r['jobId'] == job_id for r in responses])

            job = [r for r in responses if r['jobId'] == job_id][0]
            assert job['status'] == 'Completed'
            assert job['name'] == 'test_run'
            assert job['description'] == 'Job is complete and the computed scores are ready'

            # Build mock scores

            # Fetch the job config
            ## Convert current mocking to using file system
            config = redis.load_job(job_id, openreview_context_cloud['config']['OPENREVIEW_USERNAME'])
            
            # Check proper user ID
            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/request.json"), 'r') as f:
                request = json.load(f)
                assert request['user_id'] == 'TMLR/Editors_In_Chief'

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/metadata.json"), 'w') as f:
                f.write(json.dumps({"meta": "data"}))

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores.jsonl"), 'w') as f:
                f.write('{"submission": "abcd","user": "user_user1","score": 0.987}\n{"submission": "abcd","user": "user_user2","score": 0.987}')

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores_sparse.jsonl"), 'w') as f:
                f.write('{"submission": "abcde","user": "user_user1","score": 0.987}\n{"submission": "abcde","user": "user_user2","score": 0.987}')

            # Searches for journal results from the given job_id assuming the job has completed
            response = test_client.get('/expertise/results', headers=tmlr_client.headers, query_string={'jobId': job_id})
            assert response.json["metadata"] == {"meta": "data"}
            assert response.json["results"] == [
                {"submission": "abcde","user": "user_user1","score": 0.987},
                {"submission": "abcde","user": "user_user2","score": 0.987}
            ]

            # Teardown tmp_dir
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob to avoid calling AI Platform
    def test_group_group_scores(self, mock_pipeline_job, openreview_client, openreview_context_cloud):
        def setup_job_mocks():
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

            mock_pipeline_job.get.side_effect = [mock_pipeline_running] * 4 + [mock_pipeline_succeeded] * 10

            return mock_pipeline_instance

        MAX_TIMEOUT = 300
        redis = RedisDatabase(
            host=openreview_context_cloud['config']['REDIS_ADDR'],
            port=openreview_context_cloud['config']['REDIS_PORT'],
            db=openreview_context_cloud['config']['REDIS_CONFIG_DB'],
            sync_on_disk=False
        )

        # Submit first job as CLD.cc/Program_Chairs
        abc_client = openreview.api.OpenReviewClient(
            token=openreview_client.token
        )
        abc_client.impersonate('CLD.cc/Program_Chairs')

        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context_cloud['test_client']

        tmp_dir = Path('tests/gcp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Make a request
        # Patch storage.Client to return our local mock client that uses the filesystem
        with patch("google.cloud.storage.Client", new=lambda project=None: LocalMockClient(project=project, root_dir=tmp_dir)):
            setup_job_mocks()
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        "name": "test_run",
                        "entityA": {
                            'type': "Group",
                            'memberOf': "CLD.cc/Area_Chairs",
                        },
                        "entityB": { 
                            'type': "Group",
                            'memberOf': "CLD.cc/Area_Chairs",
                        },
                        "model": {
                                "name": "specter2+scincl",
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
                headers=abc_client.headers
            )
            assert response.status_code == 200, f'{response.json}'
            job_id = response.json['jobId']

            # Let request process
            time.sleep(openreview_context_cloud['config']['POLL_INTERVAL'] * openreview_context_cloud['config']['POLL_MAX_ATTEMPTS'])
            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            ## Expect 1*5 calls from the worker thread, 1*1 call from /expertise/status and 0 calls from /expertise/status/all
            assert len(mock_pipeline_job.get.call_args_list) == 6

            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            responses = test_client.get('/expertise/status/all', headers=abc_client.headers, query_string={'status': 'Completed'}).json['results']
            assert any([r['jobId'] == job_id for r in responses])
            responses = test_client.get('/expertise/status/all', headers=abc_client.headers, query_string={
                "entityA.memberOf": "CLD.cc/Area_Chairs",
                "entityB.memberOf": "CLD.cc/Area_Chairs"
            }).json['results']
            assert any([r['jobId'] == job_id for r in responses])
            responses = test_client.get('/expertise/status/all', headers=abc_client.headers, query_string={
                "entityA.memberOf": "CLD.cc/Area_Chairs",
                "entityB.memberOf": "CLD.cc/Area_Chairs",
                'status': 'Completed'
            }).json['results']
            assert any([r['jobId'] == job_id for r in responses])

            job = [r for r in responses if r['jobId'] == job_id][0]
            assert job['status'] == 'Completed'
            assert job['name'] == 'test_run'
            assert job['description'] == 'Job is complete and the computed scores are ready'

            # Build mock scores

            # Fetch the job config
            ## Convert current mocking to using file system
            config = redis.load_job(job_id, openreview_context_cloud['config']['OPENREVIEW_USERNAME'])
            
            # Check proper user ID
            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/request.json"), 'r') as f:
                request = json.load(f)
                assert request['user_id'] == 'CLD.cc/Program_Chairs'

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/metadata.json"), 'w') as f:
                f.write(json.dumps({"meta": "data"}))

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores.jsonl"), 'w') as f:
                f.write('{"match_member": "user_user2","submission_member": "user_user1","score": 0.987}\n{"match_member": "user_user3","submission_member": "user_user2","score": 0.987}')

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores_sparse.jsonl"), 'w') as f:
                f.write('{"match_member": "user_user2","submission_member": "user_user1","score": 0.987}\n{"match_member": "user_user3","submission_member": "user_user2","score": 0.987}')

            # Searches for journal results from the given job_id assuming the job has completed
            response = test_client.get('/expertise/results', headers=abc_client.headers, query_string={'jobId': job_id})
            assert response.json["metadata"] == {"meta": "data"}
            assert response.json["results"] == [
                {"match_member": "user_user2","submission_member": "user_user1","score": 0.987},
                {"match_member": "user_user3","submission_member": "user_user2","score": 0.987}
            ]

            # Teardown tmp_dir
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob to avoid calling AI Platform
    def test_paper_paper_scores(self, mock_pipeline_job, openreview_client, openreview_context_cloud):
        def setup_job_mocks():
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

            mock_pipeline_job.get.side_effect = [mock_pipeline_running] * 4 + [mock_pipeline_succeeded] * 10

            return mock_pipeline_instance

        MAX_TIMEOUT = 300
        redis = RedisDatabase(
            host=openreview_context_cloud['config']['REDIS_ADDR'],
            port=openreview_context_cloud['config']['REDIS_PORT'],
            db=openreview_context_cloud['config']['REDIS_CONFIG_DB'],
            sync_on_disk=False
        )

        # Submit first job as CLD.cc/Program_Chairs
        abc_client = openreview.api.OpenReviewClient(
            token=openreview_client.token
        )
        abc_client.impersonate('CLD.cc/Program_Chairs')

        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context_cloud['test_client']

        tmp_dir = Path('tests/gcp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Make a request
        # Patch storage.Client to return our local mock client that uses the filesystem
        with patch("google.cloud.storage.Client", new=lambda project=None: LocalMockClient(project=project, root_dir=tmp_dir)):
            setup_job_mocks()
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        "name": "test_run",
                        "entityA": {
                            'type': "Note",
                            'invitation': "CLD.cc/-/Submission" 
                        },
                        "entityB": { 
                            'type': "Note",
                            'invitation': "CLD.cc/-/Submission" 
                        },
                        "model": {
                                "name": "specter2+scincl",
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
                headers=abc_client.headers
            )
            assert response.status_code == 200, f'{response.json}'
            job_id = response.json['jobId']

            # Let request process
            time.sleep(openreview_context_cloud['config']['POLL_INTERVAL'] * openreview_context_cloud['config']['POLL_MAX_ATTEMPTS'])
            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            ## Expect 1*5 calls from the worker thread, 1*1 call from /expertise/status and 0 calls from /expertise/status/all
            assert len(mock_pipeline_job.get.call_args_list) == 6

            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['status'] == 'Completed', f"Job status: {response['status']}"

            responses = test_client.get('/expertise/status/all', headers=abc_client.headers, query_string={'status': 'Completed'}).json['results']
            assert any([r['jobId'] == job_id for r in responses])
            responses = test_client.get('/expertise/status/all', headers=abc_client.headers, query_string={
                "entityA.invitation": "CLD.cc/-/Submission",
                "entityB.invitation": "CLD.cc/-/Submission"
            }).json['results']
            assert any([r['jobId'] == job_id for r in responses])
            responses = test_client.get('/expertise/status/all', headers=abc_client.headers, query_string={
                "entityA.invitation": "CLD.cc/-/Submission",
                "entityB.invitation": "CLD.cc/-/Submission",
                'status': 'Completed'
            }).json['results']
            assert any([r['jobId'] == job_id for r in responses])

            job = [r for r in responses if r['jobId'] == job_id][0]
            assert job['status'] == 'Completed'
            assert job['name'] == 'test_run'
            assert job['description'] == 'Job is complete and the computed scores are ready'

            # Build mock scores

            # Fetch the job config
            ## Convert current mocking to using file system
            config = redis.load_job(job_id, openreview_context_cloud['config']['OPENREVIEW_USERNAME'])
            
            # Check proper user ID
            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/request.json"), 'r') as f:
                request = json.load(f)
                assert request['user_id'] == 'CLD.cc/Program_Chairs'

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/metadata.json"), 'w') as f:
                f.write(json.dumps({"meta": "data"}))

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores.jsonl"), 'w') as f:
                f.write('{"match_submission": "abcd","submission": "edfg","score": 0.987}\n{"match_submission": "hijk","submission": "lmno","score": 0.987}')

            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/scores_sparse.jsonl"), 'w') as f:
                f.write('{"match_submission": "abcd","submission": "edfg","score": 0.987}\n{"match_submission": "hijk","submission": "lmno","score": 0.987}')

            # Searches for journal results from the given job_id assuming the job has completed
            response = test_client.get('/expertise/results', headers=abc_client.headers, query_string={'jobId': job_id})
            assert response.json["metadata"] == {"meta": "data"}
            assert response.json["results"] == [
                {"match_submission": "abcd","submission": "edfg","score": 0.987},
                {"match_submission": "hijk","submission": "lmno","score": 0.987}
            ]

            # Teardown tmp_dir
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob to avoid calling AI Platform
    def test_client_isolation(self, mock_pipeline_job, openreview_client, openreview_context_cloud):
        """
        This test ensures that the user_id polling the job and the user_id written to GCP storage are the same
        and aren't affected by parallel processing and sharing of the ExpertiseCloudService instance.
        1. Submit job as User A
        2. Wait briefly but don't let it fully process
        3. Submit job as User B
        4. Verify the user_id in GCP storage for User A's job
        """
        import time
        import os
        import json
        from pathlib import Path
        
        # Setup mock for pipeline jobs
        mock_pipeline_instance = MagicMock()
        mock_pipeline_job.return_value = mock_pipeline_instance
        
        # Mock the pipeline to stay in RUNNING state so we can 
        # submit another job before it completes
        mock_pipeline_running = MagicMock()
        mock_pipeline_running.state = PipelineState.PIPELINE_STATE_RUNNING
        mock_pipeline_running.update_time.timestamp.return_value = time.time()
        
        mock_pipeline_job.get.return_value = mock_pipeline_running
        
        tmp_dir = Path('tests/gcp_bug_test')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        
        try:
            # Create clients for two different users
            abc_client = openreview.api.OpenReviewClient(token=openreview_client.token)
            abc_client.impersonate('CLD.cc/Program_Chairs')
            
            tmlr_client = openreview.api.OpenReviewClient(token=openreview_client.token)
            tmlr_client.impersonate('TMLR/Editors_In_Chief')
            
            test_client = openreview_context_cloud['test_client']
            
            # Patch storage client to use local filesystem
            with patch("google.cloud.storage.Client", new=lambda project=None: LocalMockClient(project=project, root_dir=tmp_dir)):
                # 1. Submit job as User A
                response = test_client.post(
                    '/expertise',
                    data=json.dumps({
                        "name": "User_A_Job",
                        "entityA": {'type': "Group", 'memberOf': "CLD.cc/Area_Chairs"},
                        "entityB": {'type': "Note", 'invitation': "CLD.cc/-/Submission"},
                        "model": {"name": "specter+mfr"}
                    }),
                    content_type='application/json',
                    headers=abc_client.headers
                )
                
                assert response.status_code == 200, f"Failed to submit job: {response.json}"
                job_id_a = response.json['jobId']
                
                # Wait briefly for job to be queued but not finished
                time.sleep(1)
                
                # 2. Submit another job as User B to overwrite the client
                response = test_client.post(
                    '/expertise',
                    data=json.dumps({
                        "name": "User_B_Job",
                        "entityA": {'type': "Group", 'memberOf': "TMLR/Reviewers"},
                        "entityB": {'type': "Note", 'invitation': "TMLR/-/Submission"},
                        "model": {"name": "specter+mfr"}
                    }),
                    content_type='application/json',
                    headers=tmlr_client.headers
                )
                
                assert response.status_code == 200, f"Failed to submit job: {response.json}"
                job_id_b = response.json['jobId']
                
                # Get the service from routes to check its current state
                from expertise.service.routes import get_expertise_service
                with openreview_context_cloud['app'].app_context():
                    service = get_expertise_service(openreview_context_cloud['config'], openreview_context_cloud['app'].logger)
                    
                    # Check current client's user
                    current_user = get_user_id(service.cloud.client)
                    print(f"Current client user: {current_user}")
                    
                    # This should match User B (the last user to submit)
                    assert current_user == "TMLR/Editors_In_Chief", f"Expected client for TMLR/Editors_In_Chief but got {current_user}"
                
                # Wait for both jobs to be processed
                time.sleep(openreview_context_cloud['config']['POLL_INTERVAL'] * openreview_context_cloud['config']['POLL_MAX_ATTEMPTS'] * 2)
                
                # Get User A's job from Redis
                redis = RedisDatabase(
                    host=openreview_context_cloud['config']['REDIS_ADDR'],
                    port=openreview_context_cloud['config']['REDIS_PORT'],
                    db=openreview_context_cloud['config']['REDIS_CONFIG_DB'],
                    sync_on_disk=False
                )
                
                job_a = redis.load_job(job_id_a, "CLD.cc/Program_Chairs")
                assert job_a.cloud_id is not None, "Job A cloud_id is None"
                
                # Check what was stored in GCP for job A
                request_json_path = os.path.join(tmp_dir, "test-bucket", "jobs", job_a.cloud_id, "request.json")
                assert os.path.exists(request_json_path), f"File {request_json_path} doesn't exist"
                
                with open(request_json_path, 'r') as f:
                    stored_request = json.load(f)
                
                # This check will fail if the bug exists!
                # Due to the shared service instance, it will store User B's ID for User A's job
                assert stored_request.get('user_id') == "CLD.cc/Program_Chairs", \
                    f"Bug detected! Expected 'CLD.cc/Program_Chairs' but got '{stored_request.get('user_id')}'"
                
        finally:
            # Clean up
            pass
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    @patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob to avoid calling AI Platform
    def test_read_error_json(self, mock_pipeline_job, openreview_client, openreview_context_cloud):
        def setup_job_mocks():
            # Setup mock PipelineJob
            mock_pipeline_instance = MagicMock()
            mock_pipeline_job.return_value = mock_pipeline_instance

            # Mock PipelineJob.get()
            mock_pipeline_running = MagicMock()
            mock_pipeline_running.state = PipelineState.PIPELINE_STATE_RUNNING
            mock_pipeline_running.update_time.timestamp.return_value = time.time()

            mock_pipeline_failed = MagicMock()
            mock_pipeline_failed.state = PipelineState.PIPELINE_STATE_FAILED
            mock_pipeline_failed.update_time.timestamp.return_value = time.time()

            mock_pipeline_job.get.side_effect = [mock_pipeline_running] * 4 + [mock_pipeline_failed] * 10

            return mock_pipeline_instance

        MAX_TIMEOUT = 300
        redis = RedisDatabase(
            host=openreview_context_cloud['config']['REDIS_ADDR'],
            port=openreview_context_cloud['config']['REDIS_PORT'],
            db=openreview_context_cloud['config']['REDIS_CONFIG_DB'],
            sync_on_disk=False
        )
        # Use TMLR client to test permissions
        tmlr_client = openreview.api.OpenReviewClient(
            token=openreview_client.token
        )
        tmlr_client.impersonate('TMLR/Editors_In_Chief')

        abc_client = openreview.api.OpenReviewClient(token=openreview_client.token)
        abc_client.impersonate('CLD.cc/Program_Chairs')

        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context_cloud['test_client']

        tmp_dir = Path('tests/gcp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Make a request with no submissions
        # Patch storage.Client to return our local mock client that uses the filesystem
        with patch("google.cloud.storage.Client", new=lambda project=None: LocalMockClient(project=project, root_dir=tmp_dir)):
            setup_job_mocks()
            response = test_client.post(
                '/expertise',
                data = json.dumps({
                        "name": "test_run",
                        "entityA": {
                            'type': "Group",
                            'memberOf': "CLD.cc/Area_Chairs",
                        },
                        "entityB": { 
                            'type': "Note",
                            'invitation': "CLD_ERR.cc/-/Submission" 
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
                headers=abc_client.headers
            )
            assert response.status_code == 200, f'{response.json}'
            job_id = response.json['jobId']

            # Fetch the job config
            ## Wait for config to have a cloud_id
            while redis.load_job(job_id, openreview_context_cloud['config']['OPENREVIEW_USERNAME']).cloud_id is None:
                time.sleep(0.25)
            config = redis.load_job(job_id, openreview_context_cloud['config']['OPENREVIEW_USERNAME'])

            ## Prewrite the error.json file
            with open(os.path.join(tmp_dir, f"test-bucket/jobs/{config.cloud_id}/error.json"), 'w') as f:
                f.write('{"error": "Not Found Error: No papers found for: invitation_ids: [\'CLD_ERR.cc/-/Submission\']"}')

            ## Wait for the job to finish
            time.sleep(openreview_context_cloud['config']['POLL_INTERVAL'] * openreview_context_cloud['config']['POLL_MAX_ATTEMPTS'])

            response = test_client.get('/expertise/status', headers=abc_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['name'] == 'test_run'
            assert response['status'] == 'Error'
            assert response['description'] == "Not Found Error: No papers found for: invitation_ids: ['CLD_ERR.cc/-/Submission']"

            # Teardown tmp_dir
            shutil.rmtree(tmp_dir, ignore_errors=True)