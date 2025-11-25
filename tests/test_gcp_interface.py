from unittest.mock import patch, MagicMock
import shortuuid
import pytest
import json
import datetime
import time
import openreview
from copy import deepcopy
from expertise.service.utils import GCPInterface, JobDescription, JobStatus, JobConfig, APIRequest
from google.cloud.aiplatform_v1.types import PipelineState
from expertise.utils.utils import generate_job_id

# Default parameters for the module's common setup
DEFAULT_JOURNAL_ID = 'TMLR'
DEFAULT_CONF_ID = 'GCP.cc'
DEFAULT_POST_REVIEWERS = True
DEFAULT_POST_AREA_CHAIRS = False
DEFAULT_POST_SENIOR_AREA_CHAIRS = False
DEFAULT_POST_SUBMISSIONS = True
DEFAULT_POST_PUBLICATIONS = True

def collect_generator_results(generator):
    """
    Collects results from a generator that yields chunks of data with 'metadata' and 'results'.
    
    Args:
        generator: A generator that yields dictionaries with 'metadata' and/or 'results' keys
        
    Returns:
        A dictionary with combined 'metadata' and 'results' from all chunks
    """
    all_results = []
    metadata = None
    
    # Iterate through all chunks
    for chunk in generator:
        # Save metadata when encountered
        if chunk.get('metadata') is not None:
            metadata = chunk['metadata']
        
        # Collect results
        if chunk.get('results'):
            all_results.extend(chunk['results'])
    
    # Return a dictionary with all results and metadata
    return {
        'metadata': metadata or {},
        'results': all_results
    }

@pytest.fixture(scope="module", autouse=True)
def _setup_tmlr(clean_start_journal, client, openreview_client):
    clean_start_journal(
        openreview_client,
        DEFAULT_JOURNAL_ID,
        editors=['~Raia_Hadsell1', '~Kyunghyun_Cho1'],
        additional_editors=['~Margherita_Hilpert1'],
        post_submissions=True,
        post_publications=True,
        post_editor_data=True,
        
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

# Test case for the `create_job` method
@patch("expertise.service.utils.time.time")  # Mock time.time()
@patch("expertise.service.utils.aip.PipelineJob")  # Mock PipelineJob
@patch("expertise.service.utils.storage.Client")  # Mock GCS Client
def test_create_job(mock_storage_client, mock_pipeline_job, mock_time, openreview_client):
    # Mock time.time() to return a fixed value
    mock_time.return_value = 1234567890.123  # Fixed timestamp for testing
    
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

    # Initialize the GCPInterface with test parameters
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Prepare input request
    json_request = {
        "name": "test_run2",
        "entityA": {
            'type': "Group",
            'memberOf': "GCP.cc/Reviewers",
        },
        "entityB": {
            'type': "Note",
            'invitation': "GCP.cc/-/Submission"
        },
        "model": {
            "name": "specter+mfr",
            'useTitle': False,
            'useAbstract': True,
            'skipSpecter': False,
            'scoreComputation': 'avg'
        }
    }

    # Generate a test job_id
    test_job_id = generate_job_id()
    
    # Call the `create_job` method
    # deepcopy because APIRequest() destroys the original
    result = gcp_interface.create_job(deepcopy(json_request), job_id=test_job_id, machine_type='small')
    
    expected_timestamp_ms = int(1234567890.123 * 1000)  # 1234567890123
    expected_valid_vertex_id = f"{test_job_id}-{expected_timestamp_ms}"
    assert result == expected_valid_vertex_id
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.startswith(test_job_id)

    # Assertions
    # 1. Verify folder creation in GCS
    mock_storage_client.return_value.bucket.assert_any_call("test-bucket")
    assert 3 == mock_storage_client.return_value.bucket.call_count
    mock_bucket.blob.assert_any_call(f"jobs/{result}/")
    mock_blob.upload_from_string.assert_any_call("")

    # 2. Verify JSON file upload
    expected_folder_path = f"jobs/{result}"
    mock_bucket.blob.assert_any_call(f"{expected_folder_path}/request.json")
    json_arg = mock_blob.upload_from_string.call_args_list[1]
    submitted_json = json.loads(json_arg.kwargs["data"])
    assert submitted_json['name'] == result
    assert submitted_json['entityA'] == json_request['entityA']
    assert submitted_json['entityB'] == json_request['entityB']
    assert submitted_json['token'] == openreview_client.token
    assert submitted_json['baseurl_v1'] == 'http://localhost:3000'
    assert submitted_json['baseurl_v2'] == 'http://localhost:3001'
    assert submitted_json['gcs_folder'] == f"gs://test-bucket/{expected_folder_path}"
    assert submitted_json['user_id'] == 'openreview.net'
    assert submitted_json['cdate'] == expected_timestamp_ms  # Verify timestamp is stored

    # 3. Verify PipelineJob submission
    mock_pipeline_job.assert_called_once_with(
        display_name=result,
        template_path=(
            "https://us-central1-kfp.pkg.dev/test_project/"
            "test-repo/test-pipeline/latest"
        ),
        job_id=result,
        pipeline_root="gs://test-bucket/pipeline-root",
        parameter_values={"gcs_request_path": f"gs://test-bucket/{expected_folder_path}/request.json", "machine_type": "small"},
        labels={"test": "label"}
    )
    mock_pipeline_instance.submit.assert_called_once()

# Test case for the `get_job_status_by_job_id` method
@patch("expertise.service.utils.aip.PipelineJob.get")  # Mock PipelineJob.get
@patch("expertise.service.utils.storage.Client")  # Mock GCS Client
def test_get_job_status_by_job_id(mock_storage_client, mock_pipeline_job_get, openreview_client):
    # Mock storage client
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    job_time = int(time.time() * 1000)
    mock_blob.name = 'test_job/request.json'
    mock_blob.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "cdate": int(time.time() * 1000)
    })
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob]

    # Mock PipelineJob.get()
    mock_pipeline_job = MagicMock()
    mock_pipeline_job.state = PipelineState.PIPELINE_STATE_RUNNING
    mock_pipeline_job.update_time.timestamp.return_value = time.time()
    mock_pipeline_job_get.return_value = mock_pipeline_job

    # Initialize GCPInterface with test parameters
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Call the `get_job_status_by_job_id` method
    user_id = "openreview.net"
    job_id = "test_job"
    config = JobConfig(
        cloud_id = job_id
    )
    config.api_request = APIRequest(
        {
            "name": "test_run",
            "entityA": {
                'type': "Group",
                'memberOf': "ABC.cc/Area_Chairs",
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
    )
    result = gcp_interface.get_job_status_by_job_id(user_id, config)

    # Assertions
    assert result["name"] == job_id
    assert result["tauthor"] == user_id
    assert result["status"] == JobStatus.RUN_EXPERTISE
    assert result["description"] == JobDescription.VALS.value[JobStatus.RUN_EXPERTISE]
    assert result["cdate"] > 0
    assert result["mdate"] > 0

    # Verify GCS interactions
    mock_bucket.list_blobs.assert_called_once_with(prefix=f"jobs/{job_id}")
    mock_blob.download_as_string.assert_called_once()

    # Verify Vertex AI Pipeline interaction
    mock_pipeline_job_get.assert_called_once_with(
        f"projects/123456/locations/us-central1/pipelineJobs/{job_id}"
    )

# Test case for job not found
@patch("expertise.service.utils.storage.Client")
def test_get_job_status_by_job_id_job_not_found(mock_storage_client, openreview_client):
    # Mock storage client with no blobs
    mock_bucket = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = []

    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Verify that an exception is raised when no job is found
    config = JobConfig(
        cloud_id = "test_job"
    )
    config.api_request = APIRequest(
        {
            "name": "test_run",
            "entityA": {
                'type': "Group",
                'memberOf': "ABC.cc/Area_Chairs",
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
    )
    with pytest.raises(openreview.OpenReviewException, match="Job not found"):
        gcp_interface.get_job_status_by_job_id("test_user", config)

# Test case for insufficient permissions
@patch("expertise.service.utils.storage.Client")
def test_get_job_status_by_job_id_insufficient_permissions(mock_storage_client, openreview_client):
    # Mock storage client with a blob not matching the user ID
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.name = 'test_job/request.json'
    mock_blob.download_as_string.return_value = json.dumps({
        "user_id": "other_user",
        "cdate": int(time.time() * 1000)
    })
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob]

    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Verify that an exception is raised for insufficient permissions
    config = JobConfig(
        cloud_id = 'test_job'
    )
    config.api_request = APIRequest(
        {
            "name": "test_run",
            "entityA": {
                'type': "Group",
                'memberOf': "ABC.cc/Area_Chairs",
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
    )
    with pytest.raises(openreview.OpenReviewException, match="Forbidden: Insufficient permissions to access job"):
        gcp_interface.get_job_status_by_job_id("test_user", config)

# Test case for multiple requests found
@patch("expertise.service.utils.aip.PipelineJob.get")  # Mock PipelineJob.get
@patch("expertise.service.utils.storage.Client")
def test_get_job_status_by_job_id_multiple_requests(mock_storage_client, mock_pipeline_job_get, openreview_client):
    # Mock storage client with multiple blobs matching the user ID
    mock_bucket = MagicMock()
    mock_blob_1 = MagicMock()
    mock_blob_2 = MagicMock()
    mock_blob_1.name = 'test_job/request.json'
    mock_blob_2.name = 'test_job/request.json'
    mock_blob_1.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "cdate": int(time.time() * 1000)
    })
    mock_blob_2.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "cdate": int(time.time() * 1000)
    })
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob_1, mock_blob_2]

    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Verify that an exception is raised for multiple requests
    config = JobConfig(
        cloud_id = 'test_job'
    )
    config.api_request = APIRequest(
        {
            "name": "test_run",
            "entityA": {
                'type': "Group",
                'memberOf': "ABC.cc/Area_Chairs",
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
    )
    with pytest.raises(openreview.OpenReviewException, match="Internal Error: Multiple requests found for job"):
        gcp_interface.get_job_status_by_job_id("openreview.net", config)

# Test case for the `get_job_status` method
@patch("expertise.service.utils.aip.PipelineJob.get")  # Mock PipelineJob.get
@patch("expertise.service.utils.storage.Client")  # Mock GCS Client
def test_get_job_status(mock_storage_client, mock_pipeline_job_get, openreview_client):
    # Mock storage client
    mock_bucket = MagicMock()
    mock_blob_inv = MagicMock()
    mock_blob_id = MagicMock()
    mock_blob_grp = MagicMock()

    mock_blob_inv.name = 'test_inv/request.json'
    mock_blob_id.name = 'test_id/request.json'
    mock_blob_grp.name = 'test_grp/request.json'

    mock_blob_inv.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "name": "job_1",
        "cdate": int(time.time() * 1000),
        "entityA": {"memberOf": "TestGroup.cc/Reviewers"},
        "entityB": {"invitation": "TestGroup.cc/-/Submission"}
    })
    mock_blob_id.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "name": "job_2",
        "cdate": int(time.time() * 1000),
        "entityA": {"memberOf": "TestGroup.cc/Action_Editors"},
        "entityB": {"id": "thisIsATestId"}
    })
    mock_blob_grp.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "name": "job_3",
        "cdate": int(time.time() * 1000),
        "entityA": {"memberOf": "TestGroup.cc/Senior_Area_Chairs"},
        "entityB": {"memberOf": "TestGroup.cc/Area_Chairs"}
    })
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob_inv, mock_blob_id, mock_blob_grp]

    # Mock PipelineJob.get()
    mock_pipeline_job = MagicMock()
    mock_pipeline_job.state = PipelineState.PIPELINE_STATE_SUCCEEDED
    mock_pipeline_job.update_time.timestamp.return_value = time.time()
    mock_pipeline_job_get.return_value = mock_pipeline_job

    # Initialize GCPInterface with test parameters
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Call the method with query_params
    user_id = "openreview.net"
    query_params = {"entityA.memberOf": "TestGroup.cc/Reviewers"}
    result = gcp_interface.get_job_status(user_id, query_params)

    # Assertions
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "job_1"
    assert result["results"][0]["status"] == JobStatus.COMPLETED
    assert result["results"][0]["request"]["entityA"]["memberOf"] == "TestGroup.cc/Reviewers"

    query_params = {"entityB.id": "thisIsATestId"}
    result = gcp_interface.get_job_status(user_id, query_params)

    # Assertions
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "job_2"
    assert result["results"][0]["status"] == JobStatus.COMPLETED
    assert result["results"][0]["request"]["entityA"]["memberOf"] == "TestGroup.cc/Action_Editors"
    assert result["results"][0]["request"]["entityB"]["id"] == "thisIsATestId"

    query_params = {"entityB.memberOf": "TestGroup.cc/Area_Chairs"}
    result = gcp_interface.get_job_status(user_id, query_params)

    # Assertions
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "job_3"
    assert result["results"][0]["status"] == JobStatus.COMPLETED
    assert result["results"][0]["request"]["entityA"]["memberOf"] == "TestGroup.cc/Senior_Area_Chairs"
    assert result["results"][0]["request"]["entityB"]["memberOf"] == "TestGroup.cc/Area_Chairs"

    # Verify GCS interactions
    mock_bucket.list_blobs.assert_called()
    mock_blob_inv.download_as_string.assert_called()
    mock_blob_id.download_as_string.assert_called()
    mock_blob_grp.download_as_string.assert_called()

    # Verify Vertex AI Pipeline interaction
    assert len(
        [call for call in mock_pipeline_job_get.call_args_list if call.args[0] == "projects/123456/locations/us-central1/pipelineJobs/job_1"]
    ) == 1
    assert len(
        [call for call in mock_pipeline_job_get.call_args_list if call.args[0] == "projects/123456/locations/us-central1/pipelineJobs/job_2"]
    ) == 1
    assert len(
        [call for call in mock_pipeline_job_get.call_args_list if call.args[0] == "projects/123456/locations/us-central1/pipelineJobs/job_3"]
    ) == 1


# Test case for multiple filters
@patch("expertise.service.utils.aip.PipelineJob.get")
@patch("expertise.service.utils.storage.Client")
def test_get_job_status_multiple_filters(mock_storage_client, mock_pipeline_job_get, openreview_client):
    # Mock storage client
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.name = 'test_job/request.json'

    mock_blob.download_as_string.return_value = json.dumps({
        "user_id": "openreview.net",
        "name": "job_3",
        "cdate": int(time.time() * 1000),
        "entityA": {"memberOf": "TestGroup.cc/Senior_Area_Chairs"},
        "entityB": {"invitation": "TestGroup.cc/-/Submission"}
    })
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob]

    # Mock PipelineJob.get()
    mock_pipeline_job = MagicMock()
    mock_pipeline_job.state = PipelineState.PIPELINE_STATE_FAILED
    mock_pipeline_job.update_time.timestamp.return_value = time.time()
    mock_pipeline_job_get.return_value = mock_pipeline_job

    # Initialize GCPInterface
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Call the method with combined filters
    user_id = "openreview.net"
    query_params = {
        "entityA.memberOf": "TestGroup.cc/Senior_Area_Chairs",
        "entityB.invitation": "TestGroup.cc/-/Submission"
    }
    result = gcp_interface.get_job_status(user_id, query_params)

    # Assertions
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "job_3"
    assert result["results"][0]["status"] == JobStatus.ERROR
    assert result["results"][0]["request"]["entityB"]["invitation"] == "TestGroup.cc/-/Submission"


# Test case for permissions
@patch("expertise.service.utils.storage.Client")
def test_get_job_status_insufficient_permissions(mock_storage_client, openreview_client):
    # Mock storage client
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_blob.download_as_string.return_value = json.dumps({
        "user_id": "other_user",
        "name": "job_4",
        "cdate": int(time.time() * 1000)
    })
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob]

    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Call the method
    user_id = "test_user"
    result = gcp_interface.get_job_status(user_id, {})

    # Assertions
    assert len(result["results"]) == 0

# Test case for the `get_job_results` method
@patch("expertise.service.utils.storage.Client")  # Mock GCS Client
def test_get_job_results(mock_storage_client, openreview_client):
    # Mock GCS blobs
    mock_metadata_blob = MagicMock()
    mock_metadata_blob.name = "jobs/job_1/metadata.json"
    mock_metadata_blob.download_as_string.return_value = json.dumps({"meta": "data"})

    mock_score_blob = MagicMock()
    mock_score_blob.name = "jobs/job_1/scores.jsonl"
    mock_score_blob.download_as_string.return_value = '{"submission": "abcd","user": "user_user1","score": 0.987}\n{"submission": "abcd","user": "user_user2","score": 0.987}'

    # Create a mock file-like object for sparse score blob
    mock_file = MagicMock()
    mock_file.readline.side_effect = [
        '{"submission": "abcde","user": "user_user1","score": 0.987}',
        '{"submission": "abcde","user": "user_user2","score": 0.987}',
        ''  # Empty string to terminate the loop
    ]
    mock_file.close.return_value = None

    mock_sparse_score_blob = MagicMock()
    mock_sparse_score_blob.name = "jobs/job_1/scores_sparse.jsonl"
    mock_sparse_score_blob.download_as_string.return_value = '{"submission": "abcde","user": "user_user1","score": 0.987}\n{"submission": "abcde","user": "user_user2","score": 0.987}'
    mock_sparse_score_blob.open.return_value = mock_file

    mock_request_blob = MagicMock()
    mock_request_blob.name = "jobs/job_1/request.json"
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

    # Initialize GCPInterface with test parameters
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Call the method
    user_id = "test_user"
    job_id = "job_1"
    result_generator = gcp_interface.get_job_results(user_id, job_id)
    result = collect_generator_results(result_generator)

    # Assertions
    assert result["metadata"] == {"meta": "data"}
    assert result["results"] == [
        {"submission": "abcde", "user": "user_user1", "score": 0.987},
        {"submission": "abcde", "user": "user_user2", "score": 0.987}
    ]

    # Verify GCS interactions
    mock_storage_client.return_value.bucket.return_value.list_blobs.assert_called_once_with(prefix="jobs/job_1/")
    mock_metadata_blob.download_as_string.assert_called_once()
    mock_sparse_score_blob.open.assert_called_once_with('r')

# Test case for JSON streaming
@patch("expertise.service.utils.storage.Client")  # Mock GCS Client
def test_get_job_results_json_streaming(mock_storage_client, openreview_client):
    mock_metadata_blob = MagicMock()
    mock_metadata_blob.name = "jobs/job_1/metadata.json"
    mock_metadata_blob.download_as_string.return_value = json.dumps({"meta": "data"})

    mock_sparse_jsonl_blob = MagicMock()
    mock_sparse_jsonl_blob.name = "jobs/job_1/scores_sparse.jsonl"
    mock_jsonl_file = MagicMock()
    mock_jsonl_file.readline.side_effect = [
        '{"submission": "abcde","user": "user_user1","score": 0.987}\n',
        '{"submission": "abcde","user": "user_user2","score": 0.987}',
        ''
    ]
    mock_jsonl_file.close.return_value = None
    mock_sparse_jsonl_blob.open.return_value = mock_jsonl_file

    mock_score_blob = MagicMock()
    mock_score_blob.name = "jobs/job_1/scores.jsonl"
    mock_score_blob.download_as_string.return_value = '{"submission": "abcde","user": "user_user1","score": 0.987}'

    mock_request_blob = MagicMock()
    mock_request_blob.name = "jobs/job_1/request.json"
    mock_request_blob.download_as_string.return_value = json.dumps({
        "user_id": "test_user",
        "entityA": {"type": "Group"},
        "entityB": {"type": "Note"}
    })

    mock_storage_client.return_value.bucket.return_value.list_blobs.return_value = [
        mock_metadata_blob,
        mock_sparse_jsonl_blob,
        mock_score_blob,
        mock_request_blob
    ]

    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    user_id = "test_user"
    job_id = "job_1"
    result_generator = gcp_interface.get_job_results(user_id, job_id)

    # First chunk should contain metadata
    first_chunk = next(result_generator)
    assert first_chunk["metadata"] == {"meta": "data"}
    assert first_chunk["results"] == []
    
    # Subsequent chunks should contain JSON objects
    result_chunks = list(result_generator)
    assert len(result_chunks) > 0
    # Flatten all results from chunks
    all_results = []
    for chunk in result_chunks:
        all_results.extend(chunk.get('results', []))
    
    assert len(all_results) == 2
    assert all_results[0]['submission'] == 'abcde'
    assert all_results[0]['user'] == 'user_user1'
    assert all_results[0]['score'] == 0.987
    assert all_results[1]['submission'] == 'abcde'
    assert all_results[1]['user'] == 'user_user2'
    assert all_results[1]['score'] == 0.987

    mock_sparse_jsonl_blob.open.assert_called_once_with('r')
    mock_jsonl_file.close.assert_called_once()

# Test case for missing metadata file
@patch("expertise.service.utils.storage.Client")
def test_get_job_results_missing_metadata(mock_storage_client, openreview_client):
    # Mock GCS blobs
    mock_score_blob = MagicMock()
    mock_score_blob.name = "jobs/job_1/scores.jsonl"
    mock_score_blob.download_as_string.return_value = '{"submission": abcd,"user": "user_user","score": 0.987}\n{"submission": abcd,"user": "user_user","score": 0.987}'

    mock_request_blob = MagicMock()
    mock_request_blob.name = "jobs/job_1/request.json"
    mock_request_blob.download_as_string.return_value = json.dumps({"user_id": "test_user"})

    mock_storage_client.return_value.bucket.return_value.list_blobs.return_value = [
        mock_score_blob,
        mock_request_blob
    ]

    # Initialize GCPInterface
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Verify exception is raised
    user_id = "test_user"
    job_id = "job_1"

    with pytest.raises(openreview.OpenReviewException, match="incorrect metadata files found"):
        result_generator = gcp_interface.get_job_results(user_id, job_id)
        collect_generator_results(result_generator)

# Test case for insufficient permissions
@patch("expertise.service.utils.storage.Client")
def test_get_job_results_insufficient_permissions(mock_storage_client, openreview_client):
    # Mock GCS blobs
    mock_request_blob = MagicMock()
    mock_request_blob.name = "jobs/job_1/request.json"
    mock_request_blob.download_as_string.return_value = json.dumps({"user_id": "other_user"})

    mock_storage_client.return_value.bucket.return_value.list_blobs.return_value = [
        mock_request_blob
    ]

    # Initialize GCPInterface
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={'test': 'label'}
    )

    # Verify exception is raised
    user_id = "test_user"
    job_id = "job_1"
    
    with pytest.raises(openreview.OpenReviewException, match="Forbidden: Insufficient permissions to access job"):
        result_generator = gcp_interface.get_job_results(user_id, job_id)
        collect_generator_results(result_generator)

# Test case for group scoring
@patch("expertise.service.utils.storage.Client")
def test_get_job_results_group_scoring(mock_storage_client):
    # Mock GCS blobs
    mock_metadata_blob = MagicMock()
    mock_metadata_blob.name = "jobs/job_1/metadata.json"
    mock_metadata_blob.download_as_string.return_value = json.dumps({"meta": "data"})

    # Create a mock file-like object for group score blob
    mock_file = MagicMock()
    mock_file.readline.side_effect = [
        '{"match_member": "m_user1","submission_member": "s_user1","score": 0.987}',
        '{"match_member": "m_user2","submission_member": "s_user2","score": 0.987}',
        ''  # Empty string to terminate the loop
    ]
    mock_file.close.return_value = None

    mock_group_score_blob = MagicMock()
    mock_group_score_blob.name = "jobs/job_1/group_scores.jsonl"
    mock_group_score_blob.download_as_string.return_value = '{"match_member": "m_user1","submission_member": "s_user1","score": 0.987}\n{"match_member": "m_user2","submission_member": "s_user2","score": 0.987}'
    mock_group_score_blob.open.return_value = mock_file

    mock_request_blob = MagicMock()
    mock_request_blob.name = "jobs/job_1/request.json"
    mock_request_blob.download_as_string.return_value = json.dumps({
        "user_id": "test_user",
        "entityA": {"type": "Group"},
        "entityB": {"type": "Group"}
    })

    mock_storage_client.return_value.bucket.return_value.list_blobs.return_value = [
        mock_metadata_blob,
        mock_group_score_blob,
        mock_request_blob
    ]

    # Initialize GCPInterface
    gcp_interface = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=MagicMock()
    )

    # Call the method
    user_id = "test_user"
    job_id = "job_1"
    result_generator = gcp_interface.get_job_results(user_id, job_id)
    result = collect_generator_results(result_generator)

    # Assertions
    assert result["metadata"] == {"meta": "data"}
    assert result["results"] == [
        {"match_member": "m_user1","submission_member": "s_user1","score": 0.987},
        {"match_member": "m_user2","submission_member": "s_user2","score": 0.987}
    ]

    # Verify GCS interactions
    mock_storage_client.return_value.bucket.return_value.list_blobs.assert_called_once_with(prefix="jobs/job_1/")
    mock_metadata_blob.download_as_string.assert_called_once()
    mock_group_score_blob.open.assert_called_once_with('r')

@patch("expertise.service.utils.aip.PipelineJob.get")
@patch("expertise.service.utils.storage.Client")
def test_get_job_status_by_job_id_returns_redis_when_no_cloud_id(mock_storage_client, mock_pipeline_job_get, openreview_client):
    from expertise.service.utils import APIRequest, JobConfig, GCPInterface, JobStatus, JobDescription
    # Minimal request and config with no cloud_id
    api_req = APIRequest({
        "name": "test_job",
        "entityA": {"type": "Group", "memberOf": "Some.Venue/Reviewers"},
        "entityB": {"type": "Note", "invitation": "Some.Venue/-/Submission"}
    })
    cfg = JobConfig(
        name="test",
        user_id="openreview.net",
        job_id="job_no_cloud",
        cloud_id=None,
        cdate=1234567890000,
        mdate=1234567890000,
        status=JobStatus.QUEUED,
        description=JobDescription.VALS.value[JobStatus.QUEUED],
    )
    cfg.api_request = api_req

    gcp = GCPInterface(
        project_id="test_project",
        project_number="123456",
        region="us-central1",
        pipeline_root="pipeline-root",
        pipeline_name="test-pipeline",
        pipeline_repo="test-repo",
        bucket_name="test-bucket",
        jobs_folder="jobs",
        openreview_client=openreview_client,
        service_label={"test": "label"},
    )

    # Early return
    result = gcp.get_job_status_by_job_id("openreview.net", cfg)

    # Assert
    assert result["name"] == "test"
    assert result["tauthor"] == "openreview.net"
    assert result["jobId"] == "job_no_cloud"
    assert result["status"] == JobStatus.QUEUED
    assert result["description"] == JobDescription.VALS.value[JobStatus.QUEUED]
    assert result["cdate"] == 1234567890000
    assert result["mdate"] == 1234567890000
    assert result["request"] == api_req.to_json()

    # No cloud lookups
    mock_storage_client.return_value.bucket.return_value.list_blobs.assert_not_called()
    mock_pipeline_job_get.assert_not_called()