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

# Default parameters for the module's common setup
DEFAULT_JOURNAL_ID = 'TMLR'
DEFAULT_CONF_ID = 'ABC.cc'
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
        post_reviewers=DEFAULT_POST_REVIEWERS,
        post_area_chairs=DEFAULT_POST_AREA_CHAIRS,
        post_senior_area_chairs=DEFAULT_POST_SENIOR_AREA_CHAIRS,
        post_submissions=DEFAULT_POST_SUBMISSIONS,
        post_publications=DEFAULT_POST_PUBLICATIONS
    )

@pytest.fixture(scope="module", autouse=True)
def _setup_hij_cc(clean_start_conference, client, openreview_client):
    clean_start_conference(
        client,
        'HIJ.cc',
        fake_data_source_id=DEFAULT_CONF_ID,
        exclude_expertise=False,
        post_reviewers=True,
        post_area_chairs=False,
        post_senior_area_chairs=False,
        post_submissions=False,
        post_publications=True
    )

@pytest.fixture(scope="module", autouse=True)
def _setup_upweight(clean_start_journal, client, openreview_client):
    clean_start_journal(
        openreview_client,
        'UPWEIGHT.cc',
        editors=['~Raia_Hadsell1', '~Kyunghyun_Cho1'],
        additional_editors=['~Margherita_Hilpert1'],
        post_submissions=False,
        post_publications=False,
        post_editor_data=False
    )

@pytest.fixture(scope="module", autouse=True)
def _setup_provided_submissions(clean_start_journal, client, openreview_client):
    clean_start_journal(
        openreview_client,
        'PROVIDEDSUBMISSIONS.cc',
        editors=['~Raia_Hadsell1', '~Kyunghyun_Cho1'],
        additional_editors=['~Margherita_Hilpert1'],
        post_submissions=False,
        post_publications=False,
        post_editor_data=False
    )


EXCLUSION_CONF_ID = 'EXCLUSION.cc'
EXPERTISE_SELECTION_POSTING = False
@pytest.fixture(scope="module", autouse=True)
def _setup_exclusion_cc(clean_start_conference, client, openreview_client):
    clean_start_conference(
        client,
        EXCLUSION_CONF_ID,
        fake_data_source_id=DEFAULT_CONF_ID,
        post_reviewers=DEFAULT_POST_REVIEWERS,
        post_area_chairs=DEFAULT_POST_AREA_CHAIRS,
        post_senior_area_chairs=DEFAULT_POST_SENIOR_AREA_CHAIRS,
        post_submissions=EXPERTISE_SELECTION_POSTING,
        post_publications=EXPERTISE_SELECTION_POSTING,
        post_expertise_selection={
            '~Harold_Rice1': 'Exclude'
        }
    )

INCLUSION_CONF_ID = 'INCLUSION.cc'
@pytest.fixture(scope="module", autouse=True)
def _setup_include_cc(clean_start_conference, client, openreview_client):
    clean_start_conference(
        client,
        INCLUSION_CONF_ID,
        exclude_expertise=False,
        fake_data_source_id=DEFAULT_CONF_ID,
        post_reviewers=DEFAULT_POST_REVIEWERS,
        post_area_chairs=DEFAULT_POST_AREA_CHAIRS,
        post_senior_area_chairs=DEFAULT_POST_SENIOR_AREA_CHAIRS,
        post_submissions=EXPERTISE_SELECTION_POSTING,
        post_publications=EXPERTISE_SELECTION_POSTING,
        post_expertise_selection={
            '~Harold_Rice1': 'Include'
        }
    )

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

    def test_request_expertise_with_invalid_model_venue_weights(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submitting a request with weightSpecification on an invalid model
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
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max'
                    },
                    "dataset": {
                        'minimumPubDate': 0,
                        "weightSpecification": [
                            {
                                "prefix": "UPWEIGHT",
                                "weight": 10
                            }
                        ]
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 400, f'{response.json}'
        assert 'Error' in response.json['name']
        assert 'bad request' in response.json['message'].lower()
        assert response.json['message'] == "Bad request: model specter+mfr does not support weighting by venue"

    def test_request_expertise_with_valid_parameters(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']
        
        # Post a submission and manually make it public and accepted
        submission = openreview.api.Note(
            content = {
                "title": { 'value': "test_weight" },
                "abstract": { 'value': "abstract weight" },
                "authors": { 'value': ['Royal Toy'] },
                "authorids": { 'value': ['~Royal_Toy1'] },
                'pdf': {'value': '/pdf/' + 'p' * 40 +'.pdf' },
                'competing_interests': {'value': 'aaa'},
                'human_subjects_reporting': {'value': 'bbb'}
            }
        )
        submission_edit = openreview_client.post_note_edit(
            invitation="UPWEIGHT.cc/-/Submission",
            signatures=['~Royal_Toy1'],
            note=submission
        )
        upweighted_note_id = submission_edit['note']['id']
        openreview_client.post_note_edit(
            invitation="UPWEIGHT.cc/-/Edit",
            readers=["UPWEIGHT.cc"],
            writers=["UPWEIGHT.cc"],
            signatures=["UPWEIGHT.cc"],
            note=openreview.api.Note(
                id=upweighted_note_id,
                content={
                    'venueid': {
                        'value': 'UPWEIGHT.cc/Withdrawn_Submission'
                    },
                    'venue': {
                        'value': 'UPWEIGHT Withdrawn Submission'
                    }
                },
                readers=['everyone'],
                pdate = 1554819115,
                license = 'CC BY-SA 4.0'
            )
        )

        # Make a request with weight specification
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
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max'
                    },
                    "dataset": {
                        'minimumPubDate': 0,
                        "weightSpecification": [
                            {
                                "articleSubmittedToOpenReview": True,
                                "weight": 0
                            }
                        ]
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'
        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time
        # Weight shifts scores onto a single submission
        # Build scores to reference later
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
        response = response.json['results']
        zeroed_royal_scores = {}
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            print(item)
            if profile_id == '~Royal_Toy1':
                zeroed_royal_scores[submission_id] = score

        # Check weights applied in both embedding files:
        specter_file = f"./tests/jobs/{job_id}/pub2vec_specter.jsonl"
        scincl_file = f"./tests/jobs/{job_id}/pub2vec_scincl.jsonl"

        all_publication_ids = set()
        with open(specter_file, 'r') as f, open(scincl_file, 'r') as g:
            for specter_line, scincl_line in zip(f, g):
                # Parse both lines
                specter_pub = json.loads(specter_line.strip())
                scincl_pub = json.loads(scincl_line.strip())

                # Validate both publications have weight field
                assert 'weight' in specter_pub, f"Missing weight in specter publication {specter_pub.get('paper_id')}"
                assert 'weight' in scincl_pub, f"Missing weight in scincl publication {scincl_pub.get('paper_id')}"

                # Check weights for both publications
                for pub, model_name in [(specter_pub, 'specter'), (scincl_pub, 'scincl')]:
                    all_publication_ids.add(pub['paper_id'])
                    expected_weight = 10 if pub['paper_id'] == upweighted_note_id else 1
                    assert pub['weight'] == expected_weight, f"{model_name} publication {pub['paper_id']} has weight {pub['weight']}, expected {expected_weight}"

        assert upweighted_note_id not in all_publication_ids

        # Make a request with weight specification, use articleSubmittedToOpenReview
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
                        'invitation': "ABC.cc/-/Submission",
                        'withContent': None
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max'
                    },
                    "dataset": {
                        'minimumPubDate': 0,
                        "weightSpecification": [
                            {
                                "articleSubmittedToOpenReview": True,
                                "weight": 0
                            },
                            {
                                "prefix": "UPWEIGHT",
                                "weight": 10
                            }
                        ]
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'
        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time
        # Weight shifts scores onto a single submission
        # Build scores to reference later
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
        response = response.json['results']
        openreview_royal_scores = {}
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            print(item)
            if profile_id == '~Royal_Toy1':
                openreview_royal_scores[submission_id] = score

        # Check weights applied in both embedding files (now it should be weight 5):
        specter_file = f"./tests/jobs/{job_id}/pub2vec_specter.jsonl"
        scincl_file = f"./tests/jobs/{job_id}/pub2vec_scincl.jsonl"

        with open(specter_file, 'r') as f, open(scincl_file, 'r') as g:
            for specter_line, scincl_line in zip(f, g):
                # Parse both lines
                specter_pub = json.loads(specter_line.strip())
                scincl_pub = json.loads(scincl_line.strip())

                # Validate both publications have weight field
                assert 'weight' in specter_pub, f"Missing weight in specter publication {specter_pub.get('paper_id')}"
                assert 'weight' in scincl_pub, f"Missing weight in scincl publication {scincl_pub.get('paper_id')}"

                # Check weights for both publications
                for pub, model_name in [(specter_pub, 'specter'), (scincl_pub, 'scincl')]:
                    expected_weight = 10 if pub['paper_id'] == upweighted_note_id else 1
                    assert pub['weight'] == expected_weight, f"{model_name} publication {pub['paper_id']} has weight {pub['weight']}, expected {expected_weight}"

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
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max'
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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

        # Check that the search returns both job beforejobs the new job is completed when searching for completed
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
        assert len(response) == 2

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'
        assert os.path.getsize(f"./tests/jobs/{job_id}/test_run.csv") == os.path.getsize(f"./tests/jobs/{job_id}/test_run_sparse.csv")

        # Check for API request
        req = response['request']
        assert req['name'] == 'test_run'
        assert req['entityA']['type'] == 'Group'
        assert req['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert 'model' in req.keys()
        assert 'dataset' in req.keys()
        assert req['dataset']['minimumPubDate'] == 0
        assert response['cdate'] <= response['mdate']

        # After completion, check for non-empty completed list
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
        assert len(response) == 3
        assert response[0]['status'] == 'Completed'
        assert response[0]['name'] == 'test_run'

        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Running'}).json['results']
        assert len(response) == 0

        openreview_context['job_id'] = job_id

        # Check that ~Royal_Toy1 has a lower score than in the venue-weighted job - look at full 
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
        response = response.json['results']

        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            print(item)
            if profile_id == '~Royal_Toy1':
                assert round(zeroed_royal_scores[submission_id], 5) == round(score, 5)
                assert openreview_royal_scores[submission_id] > score
                assert openreview_royal_scores[submission_id] > zeroed_royal_scores[submission_id]

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
                    "name": "specter2+scincl",
                    'useTitle': False,
                    'useAbstract': True,
                    'skipSpecter': False,
                    'scoreComputation': 'max'
                }
            }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        openreview_context['job_id2'] = job_id

    def test_status_all_query_params(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Test for status query
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
        assert len(response) == 4
        assert response[0]['status'] == 'Completed'
        assert response[1]['status'] == 'Completed'
        assert response[2]['status'] == 'Completed'
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Running'}).json['results']
        assert len(response) == 0

        # Test for member query
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'memberOf': 'ABC'}).json['results']
        assert len(response) == 4
        assert response[0]['request']['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert response[1]['request']['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        assert response[2]['request']['entityA']['memberOf'] == 'ABC.cc/Reviewers'
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'memberOf': 'CBA'}).json['results']
        assert len(response) == 0

        # Test for invitation query
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'invitation': 'ABC.cc'}).json['results']
        assert len(response) == 4
        assert response[0]['request']['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response[1]['request']['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response[2]['request']['entityB']['invitation'] == 'ABC.cc/-/Submission'

        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'invitation': 'CBA'}).json['results']
        assert len(response) == 0

        # Test for combination
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Completed', 'memberOf': 'ABC'}).json['results']
        assert len(response) == 4
        assert response[0]['status'] == 'Completed'
        assert response[1]['status'] == 'Completed'
        assert response[2]['status'] == 'Completed'

        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Running', 'memberOf': 'ABC'}).json['results']
        assert len(response) == 0

        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={'status': 'Running', 'memberOf': 'CBA'}).json['results']
        assert len(response) == 0

    def test_get_results_by_job_id(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        response = response.json['results']

        all_users = set()
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            all_users.add(profile_id)
            assert len(submission_id) >= 1
            assert len(profile_id) >= 1
            assert profile_id.startswith('~')
            assert score >= 0 and score <= 1

        # Check members
        assert "~Harold_Rice1" in all_users
        assert "~Zonia_Willms1" in all_users
        assert "~Royal_Toy1" in all_users
        assert "~C.V._Lastname1" in all_users


    def test_compare_results_for_identical_jobs(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 2
        results_a = response.json['results']

        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id2']}"})
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
                        'expertise': { 'invitation': 'INCLUSION.cc/-/Expertise_Selection'  }
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        assert req['entityA']['expertise']['invitation'] == 'INCLUSION.cc/-/Expertise_Selection'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

    def test_inclusion_invitation_default(self, client, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID, ABC is an inclusion invitation has no edges posted to it
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        # Submit a working job and return the job ID, DEF has a single exclude edge
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
                        'expertise': {  'invitation': 'EXCLUSION.cc/-/Expertise_Selection'  }
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        assert req['entityA']['expertise']['invitation'] == 'EXCLUSION.cc/-/Expertise_Selection'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

        # Assert size of archives folder is less than previous
        no_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{openreview_context['job_id']}/archives") if d.is_file())
        with_exclusion = sum(d.stat().st_size for d in os.scandir(f"./tests/jobs/{job_id}/archives") if d.is_file())
        assert with_exclusion < no_exclusion

    def test_get_results_for_all_jobs(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Assert that there are two completed jobs belonging to this user
        test_client = openreview_context['test_client']
        response = test_client.get('/expertise/status/all', headers=openreview_client.headers, query_string={}).json['results']
        assert len(response) == 7
        for job_dict in response:
            assert job_dict['status'] == 'Completed'

    def test_get_results_and_delete_data(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Clean up directories by setting the "delete_on_get" flag
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

        ## Assert the next expertise results should return empty result

    def test_paper_paper_request(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID, DEF has a single exclude edge
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']
        # Make a request
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
                    },
                    "entityB": { 
                        'type': "Note",
                        'invitation': "ABC.cc/-/Submission" 
                    },
                    "model": {
                        "name": "specter2+scincl",
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        assert req['entityA']['type'] == 'Note'
        assert req['entityA']['invitation'] == 'ABC.cc/-/Submission'
        assert req['entityB']['type'] == 'Note'
        assert req['entityB']['invitation'] == 'ABC.cc/-/Submission'
        assert response['cdate'] <= response['mdate']

        # Check archives folder
        assert os.path.isdir(f"./tests/jobs/{job_id}/archives")
        assert os.path.isfile(f"./tests/jobs/{job_id}/archives/match_submissions.jsonl")
        with open(f"./tests/jobs/{job_id}/archives/match_submissions.jsonl", 'r') as f:
            assert len(f.readlines()) == 2

        # Check scores matrix
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{job_id}", 'deleteOnGet': True})
        metadata = response.json['metadata']
        #assert metadata['submission_count'] == 1
        response = response.json['results']
        assert len(response) == 4 ## 2 papers x 2 papers = 4 entries in the score matrix
        print(response)
        for item in response:
            match_submission_id, submission_id, score = item['match_submission'], item['submission'], float(item['score'])
            assert len(submission_id) >= 1
            assert len(match_submission_id) >= 1
            assert score >= 0
            if match_submission_id == submission_id:
                assert score >= 0.99

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

    def test_get_results_and_get_error(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        # Query until job is err
        time.sleep(5)
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
        assert response.status_code == 404

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"}).json
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['name'] == 'test_run'
        assert response['status'].strip() == 'Error'
        assert response['description'] == "'<' not supported between instances of 'int' and 'str'"
        assert response['cdate'] <= response['mdate']
        ###assert os.path.isfile(f"{server_config['WORKING_DIR']}/{job_id}/err.log")

        # Clean up error job by calling the delete endpoint
        response = test_client.get('/expertise/delete', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"}).json
        assert response['name'] == 'test_run'
        assert response['cdate'] <= response['mdate']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

    def test_request_expertise_with_no_submission_error(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a config with no submissions
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
                            'sparseValue': 300,
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

    def test_get_results_and_get_no_submission_error(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        assert openreview_context['job_id'] is not None
        test_client = openreview_context['test_client']
        # Query until job is err
        time.sleep(5)
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
        assert response.status_code == 404

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Error' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"}).json
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['name'] == 'test_run'
        assert response['status'].strip() == 'Error'
        assert response['description'] == "Not Found Error: No papers found for: invitation_ids: ['HIJ.cc/-/Submission']"
        assert response['cdate'] <= response['mdate']
        ###assert os.path.isfile(f"{server_config['WORKING_DIR']}/{job_id}/err.log")

        # Clean up error job by calling the delete endpoint
        response = test_client.get('/expertise/delete', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"}).json
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
    
    def test_get_journal_results(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
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
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

    def test_high_load(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Submit a working job and return the job ID
        test_client = openreview_context['test_client']
        num_requests = 1
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
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            assert response['name'] == 'test_run'
            assert response['status'] != 'Error'

        assert id_list is not None
        openreview_context['job_id'] = id_list
    
    def test_fetch_high_load_results(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        MAX_TIMEOUT = 1200 # Timeout after 20 minutes
        assert openreview_context['job_id'] is not None
        id_list = openreview_context['job_id']
        num_requests = len(id_list)
        test_client = openreview_context['test_client']
        last_job_id = id_list[num_requests - 1]

        # Assert that the last request completes
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{last_job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{last_job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['description'] == 'Job is complete and the computed scores are ready'

        # Now fetch and empty out all previous jobs
        for id in id_list:
            # Assert that they are complete
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{id}'}).json
            assert response['status'] == 'Completed'
            assert response['name'] == 'test_run'
            assert response['description'] == 'Job is complete and the computed scores are ready'

            response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{id}", 'deleteOnGet': True})
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
        # Test group-group without any expertise selection
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        openreview_context['job_id'] = job_id ## Store no expertise selection job ID

        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{job_id}"})
        response = response.json['results']

        submission_users, match_users = set(), set()
        for item in response:
            match_id, submission_id, score = item['match_member'], item['submission_member'], float(item['score'])
            submission_users.add(submission_id)
            match_users.add(match_id)
            assert len(submission_id) >= 1
            assert len(match_id) >= 1
            assert match_id.startswith('~') and submission_id.startswith('~')
            assert score >= 0 and score <= 1

        # Check members
        assert "~Harold_Rice1" in submission_users
        assert "~Harold_Rice1" in match_users

        assert "~Zonia_Willms1" in submission_users
        assert "~Zonia_Willms1" in match_users

        assert "~Royal_Toy1" in submission_users
        assert "~Royal_Toy1" in match_users

        assert "~C.V._Lastname1" in submission_users
        assert "~C.V._Lastname1" in match_users
    
    def test_request_group_exclusion_exclusion(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Test expertise exclusion - both the archives and submissions should be smaller than the previous test's
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
                        'expertise': {  'invitation': 'EXCLUSION.cc/-/Expertise_Selection'  }
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'EXCLUSION.cc/-/Expertise_Selection' }
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        openreview_context['exclusion_id'] = job_id ## Store the job ID of a job with exclusion-exclusion selections

    def test_request_group_inclusion_exclusion(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        # Switch alternate match group to inclusion, do same checks on archives but submissions should be smaller than both previous cases (1 submission)
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
                        'expertise': { 'invitation': 'EXCLUSION.cc/-/Expertise_Selection'  }
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'INCLUSION.cc/-/Expertise_Selection' }
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        # With 2 inclusions, the match group should have fully populated archives for 2 users (default), and one of them should only have 1 submission
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
                        'expertise': { 'invitation': 'INCLUSION.cc/-/Expertise_Selection' }
                    },
                    "entityB": { 
                        'type': "Group",
                        'memberOf': "ABC.cc/Reviewers",
                        'expertise': { 'invitation': 'INCLUSION.cc/-/Expertise_Selection' }
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
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
        print(list(os.scandir(f"./tests/jobs/{job_id}/archives")))
        assert sum(1 for _ in os.scandir(f"./tests/jobs/{job_id}/archives")) == 4 # One member as no publications, the other only has TMLR submissions
        assert os.path.getsize(f"./tests/jobs/{job_id}/archives/~Harold_Rice1.jsonl") < os.path.getsize(f"./tests/jobs/{openreview_context['job_id']}/archives/~Harold_Rice1.jsonl")
        assert with_inclusion < no_inclusion
        assert with_inclusion < with_exclusion

        # Assert size of submissions file is less than previous
        no_inclusion = os.path.getsize(f"./tests/jobs/{openreview_context['job_id']}/submissions.json")
        with_inclusion = os.path.getsize(f"./tests/jobs/{job_id}/submissions.json")
        with_exclusion = os.path.getsize(f"./tests/jobs/{openreview_context['exclusion_id']}/submissions.json")
        assert with_inclusion < no_inclusion
        assert with_inclusion < with_exclusion

    def test_get_group_results(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
        test_client = openreview_context['test_client']
        # Searches for journal results from the given job_id assuming the job has completed
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}"})
        metadata = response.json['metadata']
        assert metadata['submission_count'] == 10 ## Additional from new conferences, 10 from new publication
        response = response.json['results']
        for item in response:
            match_id, submitter_id, score = item['match_member'], item['submission_member'], float(item['score'])
            assert len(match_id) >= 1
            assert len(submitter_id) >= 1
            assert match_id.startswith('~')
            assert submitter_id.startswith('~')
            assert score >= 0 and score <= 1

        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'status': 'Completed'}).json['results']
        # Check that all results are sorted in desc cdate
        for i in range(len(response) - 1):
            assert response[i]['cdate'] >= response[i + 1]['cdate']
        
        # Clean up journal request
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': f"{openreview_context['job_id']}", 'deleteOnGet': True}).json['results']
        assert not os.path.isdir(f"./tests/jobs/{openreview_context['job_id']}")

        # Clean up directory
        shutil.rmtree(f"./tests/jobs/")
        if os.path.isfile('pytest.log'):
            os.remove('pytest.log')
        if os.path.isfile('default.log'):
            os.remove('default.log')

    def test_request_expertise_with_submissions(self, openreview_client, openreview_context):
        # Submit a working job and return the job ID
        MAX_TIMEOUT = 600 # Timeout after 10 minutes
        test_client = openreview_context['test_client']
        
        # Post a submission and manually make it public and accepted
        submission = openreview.api.Note(
            content = {
                "title": { 'value': "test_weight" },
                "abstract": { 'value': "abstract weight" },
                "authors": { 'value': ['Royal Toy'] },
                "authorids": { 'value': ['~Royal_Toy1'] },
                'pdf': {'value': '/pdf/' + 'p' * 40 +'.pdf' },
                'competing_interests': {'value': 'aaa'},
                'human_subjects_reporting': {'value': 'bbb'}
            }
        )
        submission_edit = openreview_client.post_note_edit(
            invitation="PROVIDEDSUBMISSIONS.cc/-/Submission",
            signatures=['~Royal_Toy1'],
            note=submission
        )
        provided_note_id = submission_edit['note']['id']
        openreview_client.post_note_edit(
            invitation="PROVIDEDSUBMISSIONS.cc/-/Edit",
            readers=["PROVIDEDSUBMISSIONS.cc"],
            writers=["PROVIDEDSUBMISSIONS.cc"],
            signatures=["PROVIDEDSUBMISSIONS.cc"],
            note=openreview.api.Note(
                id=provided_note_id,
                content={
                    'venueid': {
                        'value': 'PROVIDEDSUBMISSIONS.cc/Withdrawn_Submission'
                    },
                    'venue': {
                        'value': 'PROVIDEDSUBMISSIONS Withdrawn Submission'
                    }
                },
                readers=['everyone'],
                pdate = 1554819115,
                license = 'CC BY-SA 4.0'
            )
        )

        # Make a request with weight specification
        response = test_client.post(
            '/expertise',
            data = json.dumps({
                    "name": "test_run",
                    "entityA": {
                        'type': "Group",
                        'reviewerIds': [
                            "~Harold_Rice1",
                            "~Zonia_Willms1",
                            "~Royal_Toy1",
                            "~C.V._Lastname1",
                        ]
                    },
                    "entityB": { 
                        'type': "Note",
                        'submissions': [
                            {
                                "id": "ASDFASDF",
                                "title": "Test Submission",
                                "abstract": "Test Abstract",
                            },
                            {
                                "id": "SFGSDFGSDFG",
                                "title": "Test Submission 2",
                                "abstract": "Test Abstract 2",
                            }
                        ]
                    },
                    "model": {
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max'
                    },
                    "dataset": {
                        'minimumPubDate': 0,
                    }
                }
            ),
            content_type='application/json',
            headers=openreview_client.headers
        )
        assert response.status_code == 200, f'{response.json}'
        job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'
        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time
        # Weight shifts scores onto a single submission
        # Build scores to reference later
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
        response = response.json['results']
        zeroed_royal_scores = {}
        for item in response:
            submission_id, profile_id, score = item['submission'], item['user'], float(item['score'])
            print(item)
            if profile_id == '~Royal_Toy1':
                zeroed_royal_scores[submission_id] = score

        # Check weights applied in both embedding files:
        specter_file = f"./tests/jobs/{job_id}/pub2vec_specter.jsonl"
        scincl_file = f"./tests/jobs/{job_id}/pub2vec_scincl.jsonl"

        with open(specter_file, 'r') as f, open(scincl_file, 'r') as g:
            for specter_line, scincl_line in zip(f, g):
                # Parse both lines
                specter_pub = json.loads(specter_line.strip())
                scincl_pub = json.loads(scincl_line.strip())

                # Check that publications have embeddings
                for pub, model_name in [(specter_pub, 'specter'), (scincl_pub, 'scincl')]:
                    assert 'embedding' in pub, f"{model_name} publication {pub.get('paper_id')} missing embedding"

    def test_request_expertise_with_normalization_field(self, openreview_client, openreview_context, celery_session_app, celery_session_worker):
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
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max'
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
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'
        assert os.path.getsize(f"./tests/jobs/{job_id}/test_run.csv") == os.path.getsize(f"./tests/jobs/{job_id}/test_run_sparse.csv")

        # Get normalized mean
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': job_id})
        response = response.json['results']

        all_scores = []
        for item in response:
            _, _, score = item['submission'], item['user'], float(item['score'])
            all_scores.append(score)
        norm_mean_score = sum(all_scores) / len(all_scores)

        # Make a request for unnormalized scores
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
                            "name": "specter2+scincl",
                            'useTitle': False, 
                            'useAbstract': True, 
                            'skipSpecter': False,
                            'scoreComputation': 'max',
                            'normalizeScores': False
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
        unnorm_job_id = response.json['jobId']
        time.sleep(2)
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{unnorm_job_id}'}).json
        assert response['name'] == 'test_run'
        assert response['status'] != 'Error'

        # Query until job is complete
        response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{unnorm_job_id}'}).json
        start_time = time.time()
        try_time = time.time() - start_time
        while response['status'] != 'Completed' and try_time <= MAX_TIMEOUT:
            time.sleep(5)
            response = test_client.get('/expertise/status', headers=openreview_client.headers, query_string={'jobId': f'{unnorm_job_id}'}).json
            if response['status'] == 'Error':
                assert False, response['description']
            try_time = time.time() - start_time

        assert try_time <= MAX_TIMEOUT, 'Job has not completed in time'
        assert response['status'] == 'Completed'
        assert response['name'] == 'test_run'
        assert response['description'] == 'Job is complete and the computed scores are ready'
        assert os.path.getsize(f"./tests/jobs/{unnorm_job_id}/test_run.csv") == os.path.getsize(f"./tests/jobs/{unnorm_job_id}/test_run_sparse.csv")

        # Check that normalized scores are different
        response = test_client.get('/expertise/results', headers=openreview_client.headers, query_string={'jobId': unnorm_job_id})
        response = response.json['results']

        all_scores = []
        for item in response:
            _, _, score = item['submission'], item['user'], float(item['score'])
            all_scores.append(score)
        unnorm_mean_score = sum(all_scores) / len(all_scores)

        assert unnorm_mean_score > norm_mean_score, 'Unnormalized mean score should be greater than normalized mean score'
