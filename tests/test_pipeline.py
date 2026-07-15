import pytest
from unittest.mock import patch, MagicMock
import csv
import io
import json
import os
import shutil
import openreview
from conftest import GCSTestHelper
from expertise.service.utils import ExpectedDataError

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
    mock_execute_expertise.return_value = {'pub2vec.jsonl': {}}

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
        f.write("note1,test_user,0.5\nnote2,test_user,0.5")
    sparse_file = os.path.join(working_dir, 'scores_sparse.csv')
    with open(sparse_file, 'w') as f:
        f.write("note1,test_user,0.5\nnote2,test_user,0.5")

    ## Build metadata
    metadata_file = os.path.join(working_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        f.write(json.dumps({"submission_count": 2, "archives_count": 4, "no_publications_count": 0}))

    ## Build archives
    archives_dir = os.path.join(working_dir, 'archives')
    os.makedirs(archives_dir, exist_ok=True)
    for i in range(4):
        with open(os.path.join(archives_dir, f'archive_{i}.jsonl'), 'w') as f:
            f.write(json.dumps({"id": f"user_{i}", "content": {"title": f"Paper {i}"}}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str=api_request_str, working_dir=working_dir)

    # Assertions

    # Pipeline worker pulled only the artifacts the requested model needs — read
    # from raw_request['model']['name']. For 'specter+mfr': specter HF + MFR,
    # and nothing from the specter2/scincl family.
    mock_load_model_artifacts.assert_called_once()
    downloaded_subdirs = mock_load_model_artifacts.call_args.kwargs['subdirs']
    assert set(downloaded_subdirs) == {'hf_models/specter', 'multifacet_recommender'}
    assert 'hf_models/specter2_base' not in downloaded_subdirs
    assert 'hf_models/specter2_adapter' not in downloaded_subdirs
    assert 'hf_models/scincl' not in downloaded_subdirs

    # Ensure execute_expertise was called with in-memory cached embeddings.
    mock_execute_expertise.assert_called_once()
    assert mock_execute_expertise.call_args.kwargs['cached_publication_embeddings'] == {'specter': {}}
    assert mock_execute_expertise.call_args.kwargs['cached_submission_embeddings'] == {'specter': {}}

    # Use the gcs_test_bucket fixture to get actual
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix/"

    # Pipeline uploads scores.csv directly (no upload-time JSONL conversion).
    # The entityA/entityB column-swap for mixed Group/Note matching happens at
    # read time in the cloud reader, not here. So the GCS CSV rows match the
    # source CSV byte-for-byte: cols [test_id, reviewer_id, score].
    scores_blob = bucket.blob(f"{prefix}scores.csv")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    scores_rows = list(csv.reader(io.StringIO(scores_content)))
    assert ['note1', 'test_user', '0.5'] in scores_rows
    assert ['note2', 'test_user', '0.5'] in scores_rows

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 2
    assert metadata_content["archives_count"] == 4
    assert metadata_content["no_publications_count"] == 0

    # Embedding JSONL files are no longer uploaded to the job folder.
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert not pub2vec_blob.exists()

    # Check archives subdirectory for 4 files
    archives_blobs = list(bucket.list_blobs(prefix=f"{prefix}archives/"))
    assert len(archives_blobs) == 4

# Test case for the `run_pipeline` function using gcs_dir
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline_gcsdir(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = {'pub2vec.jsonl': {}}

    # Prepare input API request string
    api_request = {
        "name": "test_run_gcs",
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
        "baseurl_v2": "http://localhost:3001",
        "gcs_folder": f"gs://{GCS_TEST_BUCKET}/{gcs_jobs_prefix}/test_prefix_gcs_dir",
        "dump_embs": True,
        "dump_archives": True,
    }

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
        f.write("note1,test_user,0.5\nnote2,test_user,0.5")
    sparse_file = os.path.join(working_dir, 'scores_sparse.csv')
    with open(sparse_file, 'w') as f:
        f.write("note1,test_user,0.5\nnote2,test_user,0.5")

    ## Build metadata
    metadata_file = os.path.join(working_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        f.write(json.dumps({"submission_count": 2, "archives_count": 4, "no_publications_count": 0}))

    ## Write a request file to GCS
    request_blob_name = f"{gcs_jobs_prefix}/test_prefix_gcs_dir/request.json"
    blob = gcs_test_bucket.blob(request_blob_name)
    blob.upload_from_string(
        data=json.dumps(api_request),
        content_type="application/json"
    )

    # Call the function
    request_blob_path = f"gs://{GCS_TEST_BUCKET}/{request_blob_name}"
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(gcs_dir=request_blob_path, working_dir=working_dir)

    # Assertions

    # Ensure execute_expertise was called with in-memory cached embeddings.
    mock_execute_expertise.assert_called_once()
    assert mock_execute_expertise.call_args.kwargs['cached_publication_embeddings'] == {'specter': {}}
    assert mock_execute_expertise.call_args.kwargs['cached_submission_embeddings'] == {'specter': {}}

    # Use the gcs_test_bucket fixture to get actual
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_gcs_dir/"

    # Pipeline uploads scores.csv directly (no upload-time JSONL conversion).
    # The entityA/entityB column-swap for mixed Group/Note matching happens at
    # read time in the cloud reader, not here. So the GCS CSV rows match the
    # source CSV byte-for-byte: cols [test_id, reviewer_id, score].
    scores_blob = bucket.blob(f"{prefix}scores.csv")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    scores_rows = list(csv.reader(io.StringIO(scores_content)))
    assert ['note1', 'test_user', '0.5'] in scores_rows
    assert ['note2', 'test_user', '0.5'] in scores_rows

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 2
    assert metadata_content["archives_count"] == 4
    assert metadata_content["no_publications_count"] == 0

    # Embedding JSONL files are no longer uploaded to the job folder.
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert not pub2vec_blob.exists()

    # Check archives subdirectory for 4 files
    archives_blobs = list(bucket.list_blobs(prefix=f"{prefix}archives/"))
    assert len(archives_blobs) == 4

# Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_run_pipeline_group(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = {'pub2vec.jsonl': {}}

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

    ## Build metadata
    metadata_file = os.path.join(working_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        f.write(json.dumps({"submission_count": 7, "archives_count": 4, "no_publications_count": 0}))

    ## Build archives
    archives_dir = os.path.join(working_dir, 'archives')
    os.makedirs(archives_dir, exist_ok=True)
    for i in range(4):
        with open(os.path.join(archives_dir, f'archive_{i}.jsonl'), 'w') as f:
            f.write(json.dumps({"id": f"user_{i}", "content": {"title": f"Paper {i}"}}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str=api_request_str, working_dir=working_dir)

    # Assertions
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_grp/"

    # Ensure execute_expertise was called with in-memory cached embeddings.
    mock_execute_expertise.assert_called_once()
    assert mock_execute_expertise.call_args.kwargs['cached_publication_embeddings'] == {'specter': {}}
    assert mock_execute_expertise.call_args.kwargs['cached_submission_embeddings'] == {'specter': {}}

    # Pipeline uploads scores.csv directly (group-group matching: cols are
    # already [entityA, entityB, score] in canonical order).
    scores_blob = bucket.blob(f"{prefix}scores.csv")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    scores_rows = list(csv.reader(io.StringIO(scores_content)))
    assert ['test_user', 'sub_user', '0.5'] in scores_rows

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 7
    assert metadata_content["archives_count"] == 4
    assert metadata_content["no_publications_count"] == 0

    # Embedding JSONL files are no longer uploaded to the job folder.
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert not pub2vec_blob.exists()

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
    mock_execute_expertise.return_value = {'pub2vec.jsonl': {}}

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

    ## Build metadata
    metadata_file = os.path.join(working_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        f.write(json.dumps({"submission_count": 2, "archives_count": 0, "no_publications_count": 0}))

    # Call the function
    from expertise.execute_pipeline import run_pipeline  # Replace with the actual module path
    run_pipeline(api_request_str=api_request_str, working_dir=working_dir)

    # Assertions
    # Pipeline worker pulled only the artifacts needed for 'specter2+scincl':
    # specter2 base + adapter + scincl, and nothing MFR/legacy-specter related.
    mock_load_model_artifacts.assert_called_once()
    downloaded_subdirs = mock_load_model_artifacts.call_args.kwargs['subdirs']
    assert set(downloaded_subdirs) == {
        'hf_models/specter2_base',
        'hf_models/specter2_adapter',
        'hf_models/scincl',
    }
    assert 'hf_models/specter' not in downloaded_subdirs
    assert 'multifacet_recommender' not in downloaded_subdirs

    # Check that blobs were created and data was uploaded to GCS
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_pap/"

    # Ensure execute_create_dataset and execute_expertise were called
    mock_execute_expertise.assert_called_once()

    # Pipeline uploads scores.csv directly (paper-paper matching: cols are
    # already [entityA, entityB, score] in canonical order).
    scores_blob = bucket.blob(f"{prefix}scores.csv")
    assert scores_blob.exists()
    scores_content = scores_blob.download_as_text()
    scores_rows = list(csv.reader(io.StringIO(scores_content)))
    assert ['sub_one', 'sub_two', '0.5'] in scores_rows

    # Check for metadata.json file
    metadata_blob = bucket.blob(f"{prefix}metadata.json")
    assert metadata_blob.exists()
    metadata_content = json.loads(metadata_blob.download_as_text())
    assert metadata_content["submission_count"] == 2
    assert metadata_content["archives_count"] == 0
    assert metadata_content["no_publications_count"] == 0

    shutil.rmtree(working_dir)  # Clean up

    # Test case for the `run_pipeline` function
@patch("expertise.execute_pipeline.execute_expertise")  # Mock execute_expertise
@patch("expertise.execute_pipeline.load_model_artifacts")  # Mock load_model_artifacts
def test_runtime_errors(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
    # Mock other external dependencies
    mock_load_model_artifacts.return_value = None
    error_message = 'No papers found for: invitation_ids: [\'PIPELINE_ERR.cc/-/Submission\']'
    mock_execute_expertise.side_effect = ExpectedDataError(error_message)

    pipeline_client = openreview.api.OpenReviewClient(
        token=openreview_client.token
    )
    pipeline_client.impersonate('PIPELINE.cc')

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
        "token": pipeline_client.token,
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

    # Call the function
    from expertise.execute_pipeline import run_pipeline
    try:
        run_pipeline(api_request_str=api_request_str, working_dir=working_dir)
    except Exception as e:
        assert str(e) == error_message

    # Assertions
    # Check that blobs were created and data was uploaded to GCS
    bucket = gcs_test_bucket
    prefix = f"{gcs_jobs_prefix}/test_prefix_err/"

    # Check for error.json file
    error_blob = bucket.blob(f"{prefix}error.json")
    assert error_blob.exists()
    error_content = json.loads(error_blob.download_as_text())
    assert error_content["error"] == 'No papers found for: invitation_ids: [\'PIPELINE_ERR.cc/-/Submission\']'
    assert error_content["expected"] == True  # This is an expected data error

    shutil.rmtree(working_dir)  # Clean up
