import pytest
from unittest.mock import patch, MagicMock
import json
import os
import shutil
import openreview

# Default parameters for the module's common setup
DEFAULT_JOURNAL_ID = 'TMLR'
DEFAULT_CONF_ID = 'PIPELINE.cc'
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
def _setup_pipeline_cc(clean_start_conference, client, openreview_client):
    clean_start_conference(
        client,
        DEFAULT_CONF_ID,
        post_reviewers=DEFAULT_POST_REVIEWERS,
        post_area_chairs=DEFAULT_POST_AREA_CHAIRS,
        post_senior_area_chairs=DEFAULT_POST_SENIOR_AREA_CHAIRS,
        post_submissions=DEFAULT_POST_SUBMISSIONS,
        post_publications=DEFAULT_POST_PUBLICATIONS
    )

# Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.storage.Client")  # Mock GCS Client
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline(mock_load_model_artifacts, mock_gcs_client, mock_execute_expertise, openreview_client):
    # Mock GCS client and bucket
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_gcs_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Mock blob.upload_from_string to do nothing
    mock_blob.upload_from_string.return_value = None

    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = None

    # Prepare input API request string
    api_request_str = json.dumps({
        "name": "test_run2",
        "entityA": {
            'type': "Group",
            'memberOf': "PIPELINE.cc/Reviewers",
        },
        "entityB": {
            'type': "Note",
            'invitation': "PIPELINE.cc/-/Submission"
        },
        "model": {
            "name": "specter+mfr",
            'useTitle': False,
            'useAbstract': True,
            'skipSpecter': False,
            'scoreComputation': 'avg'
        },
        "user_id": "openreview.net",
        "token": openreview_client.token,
        "baseurl_v1": "http://localhost:3000",
        "baseurl_v2": "http://localhost:3001",
        "gcs_folder": "gs://test_bucket/test_prefix",
        "dump_embs": True,
        "dump_archives": True,
    })

    # Prepare environment variables
    os.environ["SPECTER_DIR"] = "/path/to/specter"
    os.environ["MFR_VOCAB_DIR"] = "/path/to/mfr_vocab"
    os.environ["MFR_CHECKPOINT_DIR"] = "/path/to/mfr_checkpoint"

    # Build files
    working_dir = './test_pipeline'
    os.makedirs(working_dir, exist_ok=True)

    ## Build scores file
    scores_file = os.path.join(working_dir, 'scores.csv')
    with open(scores_file, 'w') as f:
        f.write("test_user,note1,0.5\ntest_user,note2,0.5")
    sparse_file = os.path.join(working_dir, 'scores_sparse.csv')
    with open(scores_file, 'w') as f:
        f.write("test_user,note1,0.5\ntest_user,note2,0.5")

    ## Build embeddings
    embeddings_dir = os.path.join(working_dir, 'pub2vec.jsonl')
    with open(embeddings_dir, 'w') as f:
        f.write(json.dumps({"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str, working_dir)

    # Assertions
    # Check that blobs were created and data was uploaded to GCS
    mock_gcs_client.assert_called_once()
    mock_bucket.blob.assert_any_call("test_prefix/job_config.json")
    mock_blob.upload_from_string.assert_called()  # Ensure upload_from_string was called

    # Ensure execute_create_dataset and execute_expertise were called
    mock_execute_expertise.assert_called_once()

    mock_blob.upload_from_string.assert_any_call(
        '{"submission": "test_user", "user": "note1", "score": 0.5}\n{"submission": "test_user", "user": "note2", "score": 0.5}'
    )
    mock_blob.upload_from_string.assert_any_call(
        '{"submission_count": 2, "no_publications_count": 0, "no_publications": [], "no_profile": []}'
    )
    mock_blob.upload_from_string.assert_any_call(
        json.dumps({"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]})
    )
    publication_calls = [call for call in mock_blob.upload_from_string.call_args_list if '"content":' in call.args[0]]
    assert len(publication_calls) == 4

    shutil.rmtree(working_dir)  # Clean up

# Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.storage.Client")  # Mock GCS Client
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline_group(mock_load_model_artifacts, mock_gcs_client, mock_execute_expertise, openreview_client):
    # Mock GCS client and bucket
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_gcs_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Mock blob.upload_from_string to do nothing
    mock_blob.upload_from_string.return_value = None

    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = None

    # Prepare input API request string
    api_request_str = json.dumps({
        "name": "test_run2",
        "entityA": {
            'type': "Group",
            'memberOf': "PIPELINE.cc/Reviewers",
        },
        "entityB": {
            'type': "Group",
            'memberOf': "PIPELINE.cc/Reviewers"
        },
        "model": {
            "name": "specter+mfr",
            'useTitle': False,
            'useAbstract': True,
            'skipSpecter': False,
            'scoreComputation': 'avg'
        },
        "user_id": "openreview.net",
        "token": openreview_client.token,
        "baseurl_v1": "http://localhost:3000",
        "baseurl_v2": "http://localhost:3001",
        "gcs_folder": "gs://test_bucket/test_prefix",
        "dump_embs": True,
        "dump_archives": True,
    })

    # Prepare environment variables
    os.environ["SPECTER_DIR"] = "/path/to/specter"
    os.environ["MFR_VOCAB_DIR"] = "/path/to/mfr_vocab"
    os.environ["MFR_CHECKPOINT_DIR"] = "/path/to/mfr_checkpoint"

    # Build files
    working_dir = './test_pipeline'
    os.makedirs(working_dir, exist_ok=True)

    ## Build scores file
    scores_file = os.path.join(working_dir, 'scores.csv')
    with open(scores_file, 'w') as f:
        f.write("test_user,sub_user,0.5\ntest_user,sub_user,0.5")
    sparse_file = os.path.join(working_dir, 'scores_sparse.csv')
    with open(scores_file, 'w') as f:
        f.write("test_user,sub_user,0.5\ntest_user,sub_user,0.5")

    ## Build embeddings
    embeddings_dir = os.path.join(working_dir, 'pub2vec.jsonl')
    with open(embeddings_dir, 'w') as f:
        f.write(json.dumps({"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str, working_dir)

    # Assertions
    # Check that blobs were created and data was uploaded to GCS
    mock_gcs_client.assert_called_once()
    mock_bucket.blob.assert_any_call("test_prefix/job_config.json")
    mock_blob.upload_from_string.assert_called()  # Ensure upload_from_string was called

    # Ensure execute_create_dataset and execute_expertise were called
    mock_execute_expertise.assert_called_once()
    print(mock_blob.upload_from_string.call_args_list)

    mock_blob.upload_from_string.assert_any_call(
        '{"match_member": "test_user", "submission_member": "sub_user", "score": 0.5}\n{"match_member": "test_user", "submission_member": "sub_user", "score": 0.5}'
    )
    mock_blob.upload_from_string.assert_any_call(
        '{"submission_count": 7, "no_publications_count": 0, "no_publications": [], "no_profile": [], "no_profile_submission": []}'
    )
    mock_blob.upload_from_string.assert_any_call(
        json.dumps({"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]})
    )
    publication_calls = [call for call in mock_blob.upload_from_string.call_args_list if '"content":' in call.args[0]]
    assert len(publication_calls) == 4

    shutil.rmtree(working_dir)  # Clean up

# Test case for the `run_pipeline` function for paper-paper matching
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.storage.Client")  # Mock GCS Client
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline_paper_paper(mock_load_model_artifacts, mock_gcs_client, mock_execute_expertise, openreview_client):
    # Mock GCS client and bucket
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_gcs_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Mock blob.upload_from_string to do nothing
    mock_blob.upload_from_string.return_value = None

    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = None

    # Prepare input API request string
    api_request_str = json.dumps({
        "name": "test_run2",
        "entityA": {
            'type': "Note",
            'invitation': "PIPELINE.cc/-/Submission",
        },
        "entityB": {
            'type': "Note",
            'invitation': "PIPELINE.cc/-/Submission"
        },
        "model": {
            "name": "specter2+scincl",
            'useTitle': False,
            'useAbstract': True,
            'skipSpecter': False,
            'scoreComputation': 'avg'
        },
        "user_id": "openreview.net",
        "token": openreview_client.token,
        "baseurl_v1": "http://localhost:3000",
        "baseurl_v2": "http://localhost:3001",
        "gcs_folder": "gs://test_bucket/test_prefix",
        "dump_embs": False,
        "dump_archives": False,
    })

    # Build files
    working_dir = './test_pipeline'
    os.makedirs(working_dir, exist_ok=True)

    ## Build scores file
    scores_file = os.path.join(working_dir, 'scores.csv')
    with open(scores_file, 'w') as f:
        f.write("sub_one,sub_two,0.5\nsub_one,sub_two,0.5")
    sparse_file = os.path.join(working_dir, 'scores_sparse.csv')
    with open(scores_file, 'w') as f:
        f.write("sub_one,sub_two,0.5\nsub_one,sub_two,0.5")

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str, working_dir)

    # Assertions
    # Check that blobs were created and data was uploaded to GCS
    mock_gcs_client.assert_called_once()
    mock_bucket.blob.assert_any_call("test_prefix/job_config.json")
    mock_blob.upload_from_string.assert_called()  # Ensure upload_from_string was called

    # Ensure execute_create_dataset and execute_expertise were called
    mock_execute_expertise.assert_called_once()
    print(mock_blob.upload_from_string.call_args_list)

    mock_blob.upload_from_string.assert_any_call(
        '{"match_submission": "sub_one", "submission": "sub_two", "score": 0.5}\n{"match_submission": "sub_one", "submission": "sub_two", "score": 0.5}'
    )
    mock_blob.upload_from_string.assert_any_call(
        '{"submission_count": 2, "no_publications_count": 0, "no_publications": []}'
    )

    shutil.rmtree(working_dir)  # Clean up

    # Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.storage.Client")  # Mock GCS Client
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_runtime_errors(mock_load_model_artifacts, mock_gcs_client, mock_execute_expertise, openreview_client):
    # Mock GCS client and bucket
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_gcs_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # Mock blob.upload_from_string to do nothing
    mock_blob.upload_from_string.return_value = None

    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = None

    # Use TMLR client to test permissions
    tmlr_client = openreview.api.OpenReviewClient(
        token=openreview_client.token
    )
    tmlr_client.impersonate('TMLR/Editors_In_Chief')

    # Prepare input API request string
    api_request_str = json.dumps({
        "name": "test_run2",
        "entityA": {
            'type': "Group",
            'memberOf': "PIPELINE.cc/Reviewers",
        },
        "entityB": { 
            'type': "Note",
            'invitation': "PIPELINE_ERR.cc/-/Submission" 
        },
        "model": {
            "name": "specter+mfr",
            'useTitle': False,
            'useAbstract': True,
            'skipSpecter': False,
            'scoreComputation': 'avg'
        },
        "user_id": "openreview.net",
        "token": openreview_client.token,
        "baseurl_v1": "http://localhost:3000",
        "baseurl_v2": "http://localhost:3001",
        "gcs_folder": "gs://test_bucket/test_prefix",
        "dump_embs": True,
        "dump_archives": True,
    })

    # Prepare environment variables
    os.environ["SPECTER_DIR"] = "/path/to/specter"
    os.environ["MFR_VOCAB_DIR"] = "/path/to/mfr_vocab"
    os.environ["MFR_CHECKPOINT_DIR"] = "/path/to/mfr_checkpoint"

    # Build files
    working_dir = './test_pipeline'
    os.makedirs(working_dir, exist_ok=True)

    ## Skip file building - never happens in error

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    try:
        run_pipeline(api_request_str, working_dir)
    except Exception as e:
        assert str(e) == 'Not Found Error: No papers found for: invitation_ids: [\'PIPELINE_ERR.cc/-/Submission\']'

    # Assertions
    # Check that blobs were created and data was uploaded to GCS
    mock_gcs_client.assert_called_once()
    mock_blob.upload_from_string.assert_called()  # Ensure upload_from_string was called

    mock_blob.upload_from_string.assert_any_call(
        '{"error": "Not Found Error: No papers found for: invitation_ids: [\'PIPELINE_ERR.cc/-/Submission\']"}'
    )

    shutil.rmtree(working_dir)  # Clean up
