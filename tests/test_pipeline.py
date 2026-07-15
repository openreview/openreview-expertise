import pytest
from unittest.mock import patch, MagicMock
import csv
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
import openreview
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
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

    ## Build embeddings JSONL files so the upload path is exercised
    with open(os.path.join(working_dir, 'pub2vec.jsonl'), 'w') as f:
        f.write(json.dumps({"paper_id": "note1", "embedding": [0.1, 0.2]}) + '\n')
    with open(os.path.join(working_dir, 'sub2vec.jsonl'), 'w') as f:
        f.write(json.dumps({"paper_id": "note1", "embedding": [0.1, 0.2]}) + '\n')

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

    # Embedding JSONL files are uploaded to the job folder for backwards compat.
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert pub2vec_blob.exists()
    sub2vec_blob = bucket.blob(f"{prefix}sub2vec.jsonl")
    assert sub2vec_blob.exists()

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

    ## Build embeddings JSONL files so the upload path is exercised
    with open(os.path.join(working_dir, 'pub2vec.jsonl'), 'w') as f:
        f.write(json.dumps({"paper_id": "note1", "embedding": [0.1, 0.2]}) + '\n')
    with open(os.path.join(working_dir, 'sub2vec.jsonl'), 'w') as f:
        f.write(json.dumps({"paper_id": "note1", "embedding": [0.1, 0.2]}) + '\n')

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

    # Embedding JSONL files are uploaded to the job folder for backwards compat.
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert pub2vec_blob.exists()
    sub2vec_blob = bucket.blob(f"{prefix}sub2vec.jsonl")
    assert sub2vec_blob.exists()

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

    ## Build embeddings JSONL files so the upload path is exercised
    with open(os.path.join(working_dir, 'pub2vec.jsonl'), 'w') as f:
        f.write(json.dumps({"paper_id": "note1", "embedding": [0.1, 0.2]}) + '\n')
    with open(os.path.join(working_dir, 'sub2vec.jsonl'), 'w') as f:
        f.write(json.dumps({"paper_id": "note1", "embedding": [0.1, 0.2]}) + '\n')

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

    # Embedding JSONL files are uploaded to the job folder for backwards compat.
    pub2vec_blob = bucket.blob(f"{prefix}pub2vec.jsonl")
    assert pub2vec_blob.exists()
    sub2vec_blob = bucket.blob(f"{prefix}sub2vec.jsonl")
    assert sub2vec_blob.exists()

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


@patch("expertise.execute_pipeline.execute_expertise")
@patch("expertise.execute_pipeline.load_model_artifacts")
def test_run_pipeline_stale_cache_triggers_recompute(mock_load_model_artifacts, mock_execute_expertise, openreview_client, gcs_test_bucket, gcs_jobs_prefix):
    """If a paper's mdate is newer than the cached embedding_date, the cache entry
    is treated as stale and execute_expertise receives empty cached embeddings,
    forcing recomputation."""
    mock_load_model_artifacts.return_value = None
    mock_execute_expertise.return_value = {'pub2vec.jsonl': {'paper1': [0.1, 0.2]}}

    # Build a local parquet dataset with a stale embedding (older than paper mdate)
    with tempfile.TemporaryDirectory() as tmpdir:
        table = pa.table({
            "paper_id": pa.array(["paper1"], pa.string()),
            "embedding": pa.array([[0.1, 0.2]], pa.list_(pa.float32())),
            "model": pa.array(["specter"], pa.string()),
            "year_month": pa.array(["2024-01"], pa.string()),
            "embedding_date": pa.array(["2024-01-15T00:00:00Z"], pa.string()),
            "job_id": pa.array(["old-job"], pa.string()),
        })
        part_dir = Path(tmpdir) / "model=specter" / "year_month=2024-01"
        part_dir.mkdir(parents=True)
        pq.write_table(table, part_dir / "part-00000.parquet")

        # Patch _get_dataset so GlobalEmbeddingsCache reads the local dataset
        def _patched_get_dataset(self):
            if self._dataset is not None:
                return self._dataset
            self._dataset = ds.dataset(str(tmpdir), partitioning="hive")
            return self._dataset

        with patch("expertise.execute_pipeline.GlobalEmbeddingsCache._get_dataset", _patched_get_dataset):
            api_request_str = json.dumps({
                "name": "test_stale_cache",
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
                "gcs_folder": f"gs://{GCS_TEST_BUCKET}/{gcs_jobs_prefix}/test_stale_cache",
                "dump_embs": True,
                "dump_archives": False,
            })

            os.environ["SPECTER_DIR"] = "/path/to/specter"
            os.environ["MFR_VOCAB_DIR"] = "/path/to/mfr_vocab"
            os.environ["MFR_CHECKPOINT_DIR"] = "/path/to/mfr_checkpoint"

            working_dir = './test_pipeline'
            os.makedirs(working_dir, exist_ok=True)

            with open(os.path.join(working_dir, 'scores.csv'), 'w') as f:
                f.write("n1,u1,0.5")
            with open(os.path.join(working_dir, 'scores_sparse.csv'), 'w') as f:
                f.write("n1,u1,0.5")
            with open(os.path.join(working_dir, 'metadata.json'), 'w') as f:
                f.write(json.dumps({"submission_count": 1, "archives_count": 1, "no_publications_count": 0}))

            archives_dir = os.path.join(working_dir, 'archives')
            os.makedirs(archives_dir, exist_ok=True)
            with open(os.path.join(archives_dir, 'author.jsonl'), 'w') as f:
                f.write(json.dumps({"id": "paper1", "mdate": "2024-06-01T00:00:00Z", "content": {"title": "T"}}))

            with open(os.path.join(working_dir, 'pub2vec.jsonl'), 'w') as f:
                f.write(json.dumps({"paper_id": "paper1", "embedding": [0.1, 0.2]}) + '\n')
            with open(os.path.join(working_dir, 'sub2vec.jsonl'), 'w') as f:
                f.write(json.dumps({"paper_id": "n1", "embedding": [0.3, 0.4]}) + '\n')

            from expertise.execute_pipeline import run_pipeline
            run_pipeline(api_request_str=api_request_str, working_dir=working_dir)

            # The cache has paper1 with embedding_date 2024-01-15, but the paper's mdate is 2024-06-01.
            # GlobalEmbeddingsCache should filter it out, so execute_expertise gets empty embeddings.
            mock_execute_expertise.assert_called_once()
            assert mock_execute_expertise.call_args.kwargs['cached_publication_embeddings'] == {'specter': {}}
            assert mock_execute_expertise.call_args.kwargs['cached_submission_embeddings'] == {'specter': {}}

            shutil.rmtree(working_dir)

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
