import pytest
from unittest.mock import patch, MagicMock
import json
import os
import shutil
import openreview
from conftest import GCSTestHelper

# Default parameters for the module's common setup
DEFAULT_JOURNAL_ID = 'TMLR'
DEFAULT_CONF_ID = 'PIPELINE.cc'
DEFAULT_POST_REVIEWERS = True
DEFAULT_POST_AREA_CHAIRS = False
DEFAULT_POST_SENIOR_AREA_CHAIRS = False
DEFAULT_POST_SUBMISSIONS = True
DEFAULT_POST_PUBLICATIONS = True

GCS_TEST_BUCKET = GCSTestHelper.GCS_TEST_BUCKET

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
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
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
        "gcs_folder": f"gs://{GCS_TEST_BUCKET}/{gcs_jobs_prefix}/test_prefix",
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
    with open(sparse_file, 'w') as f:
        f.write("test_user,note1,0.5\ntest_user,note2,0.5")

    ## Build embeddings
    embeddings_dir = os.path.join(working_dir, 'pub2vec.jsonl')
    with open(embeddings_dir, 'w') as f:
        f.write(json.dumps({"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str, working_dir)

    # Assertions

    # Ensure execute_create_dataset and execute_expertise were called
    # Use the gcs_test_bucket fixture to get actual 
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix/"

    # Check for scores.jsonl file
    scores_blob = bucket.blob(f"{prefix}scores.jsonl")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    assert '{"submission": "test_user", "user": "note1", "score": 0.5}' in scores_content
    assert '{"submission": "test_user", "user": "note2", "score": 0.5}' in scores_content

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 2
    assert metadata_content["no_publications_count"] == 0

    # Check for pub2vec.jsonl file
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert pub2vec_blob.exists()
    pub2vec_content = pub2vec_blob.download_as_text()
    assert '{"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]}' in pub2vec_content

    # Check archives subdirectory for 4 files
    archives_blobs = list(bucket.list_blobs(prefix=f"{prefix}archives/"))
    assert len(archives_blobs) == 4

# Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline_group(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
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
        "gcs_folder": f"gs://{GCS_TEST_BUCKET}/{gcs_jobs_prefix}/test_prefix_grp",
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
    with open(sparse_file, 'w') as f:
        f.write("test_user,sub_user,0.5\ntest_user,sub_user,0.5")

    ## Build embeddings
    embeddings_dir = os.path.join(working_dir, 'pub2vec.jsonl')
    with open(embeddings_dir, 'w') as f:
        f.write(json.dumps({"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str, working_dir)

    # Assertions
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_grp/"

    # Ensure execute_create_dataset and execute_expertise were called
    mock_execute_expertise.assert_called_once()

    # Check for scores.jsonl file
    scores_blob = bucket.blob(f"{prefix}scores.jsonl")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    assert '{"submission": "test_user", "user": "sub_user", "score": 0.5}' in scores_content

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 7
    assert metadata_content["no_publications_count"] == 0

    # Check for pub2vec.jsonl file
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert pub2vec_blob.exists()
    pub2vec_content = pub2vec_blob.download_as_text()
    assert '{"paper_id": "paperId", "embedding": [0.1, 0.2, 0.3]}' in pub2vec_content

    # Check archives subdirectory for 4 files
    archives_blobs = list(bucket.list_blobs(prefix=f"{prefix}archives/"))
    assert len(archives_blobs) == 4

    shutil.rmtree(working_dir)  # Clean up

# Test case for the `run_pipeline` function for paper-paper matching
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline_paper_paper(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
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
        "gcs_folder": f"gs://{GCS_TEST_BUCKET}/{gcs_jobs_prefix}/test_prefix_pap",
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
    with open(sparse_file, 'w') as f:
        f.write("sub_one,sub_two,0.5\nsub_one,sub_two,0.5")

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str, working_dir)

    # Assertions
    # Check that blobs were created and data was uploaded to GCS
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_pap/"

    # Ensure execute_create_dataset and execute_expertise were called
    mock_execute_expertise.assert_called_once()

    # Check for scores.jsonl file
    scores_blob = bucket.blob(f"{prefix}scores.jsonl")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    assert '{"match_submission": "sub_one", "submission": "sub_two", "score": 0.5}' in scores_content

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 2
    assert metadata_content["no_publications_count"] == 0

    shutil.rmtree(working_dir)  # Clean up

    # Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_runtime_errors(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
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
        "gcs_folder": f"gs://{GCS_TEST_BUCKET}/{gcs_jobs_prefix}/test_prefix_err",
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
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_err/"

    # Check for error.json file
    error_blob = bucket.blob(f"{prefix}error.json")
    assert error_blob.exists()
    error_content = json.loads(error_blob.download_as_text())
    assert error_content["error"] == 'Not Found Error: No papers found for: invitation_ids: [\'PIPELINE_ERR.cc/-/Submission\']'

    shutil.rmtree(working_dir)  # Clean up
