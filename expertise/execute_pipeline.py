import argparse
import datetime
import os
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
import pyarrow.compute as pc
from expertise.execute_expertise import execute_expertise
from expertise.service import load_model_artifacts, artifacts_for_model
from expertise.service.utils import APIRequest, JobConfig, ExpectedDataError
from expertise.utils.utils import generate_job_id
from google.cloud import storage

DEFAULT_CONFIG = {
    "dataset": {},
    "model": "specter+mfr",
    "model_params": {
        "use_title": True,
        "sparse_value": 400,
        "specter_batch_size": 16,
        "mfr_batch_size": 50,
        "use_abstract": True,
        "average_score": False,
        "max_score": True,
        "skip_specter": False,
        "use_cuda": True,
        "use_redis": False
    }
}
DELETED_FIELDS = ['user_id', 'cdate', 'machine_type']

def load_gcs(gcs_path):
    """Return client and bucket for a GCS path."""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    # Parse GCS path: gs://bucket_name/path/to/file
    bucket_name = gcs_path.split('/')[2]
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)

    return gcs_client, bucket

def download_from_gcs(gcs_path):
    """Download JSON content from a GCS path."""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    # Parse GCS path: gs://bucket_name/path/to/file
    _, bucket = load_gcs(gcs_path)

    blob_name = '/'.join(gcs_path.split('/')[3:])

    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gcs_path}")

    # Download as JSON
    content = blob.download_as_text(encoding='utf-8')
    return json.loads(content)

def download_dataset_from_gcs(gcs_path, local_dir):
    """Download a dataset tarball from GCS and extract it into local_dir."""
    _, bucket = load_gcs(gcs_path)
    blob_name = '/'.join(gcs_path.split('/')[3:])

    os.makedirs(local_dir, exist_ok=True)

    tarball_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            tarball_path = tmp.name
        bucket.blob(blob_name).download_to_filename(tarball_path)
        print(f"Downloaded {blob_name} to {tarball_path}")

        # Safely extract tarball, guarding against path traversal
        with tarfile.open(tarball_path, 'r:gz') as tar:
            for member in tar.getmembers():
                member_path = os.path.join(local_dir, member.name)
                real_member_path = os.path.realpath(member_path)
                real_local_dir = os.path.realpath(local_dir)
                if not real_member_path.startswith(real_local_dir + os.sep) and real_member_path != real_local_dir:
                    raise ValueError(f"Tarball member {member.name} attempts path traversal")
                tar.extract(member, local_dir)
        print(f"Extracted dataset into {local_dir}")
    finally:
        if tarball_path and os.path.exists(tarball_path):
            os.remove(tarball_path)


def run_pipeline(
        api_request_str=None,
        gcs_dir=None,
        working_dir=None,
        dataset_gcs_path=None
    ):

    if gcs_dir is None and api_request_str is None:
        raise ValueError("Either gcs_dir or api_request_str must be provided")

    # Initialize variables for exception handling and cleanup
    bucket = None
    blob_prefix = None
    config = None
    working_dir_created = False
    dump_archives = False
    validated_request = None

    try:
        if api_request_str is not None:
            try:
                raw_request: dict = json.loads(api_request_str)
                print("Parsed request as JSON string")
            except:
                if not os.path.exists(api_request_str):
                    raise FileNotFoundError(f"File {api_request_str} not found")
                with open(api_request_str, 'r') as f:
                    raw_request = json.load(f)
                print(f"Loaded request from local file: {api_request_str}")
        elif gcs_dir is not None:
            raw_request = download_from_gcs(gcs_dir)
            print("Parsed request from GCS folder")

        # Pop pipeline-only metadata. The pipeline doesn't authenticate against
        # OpenReview, so token/baseurl_v2 are not part of the upload anymore.
        print('Popping variables')
        for field in DELETED_FIELDS:
            raw_request.pop(field, None)
        destination_prefix = raw_request.pop('gcs_folder')
        # dump_embs removed - embeddings always uploaded
        dump_archives = False if 'dump_archives' not in raw_request else raw_request.pop('dump_archives')
        server_config = {
            'OPENREVIEW_BASEURL_V2': os.getenv('OPENREVIEW_BASEURL_V2', ''),
            'SPECTER_DIR': os.getenv('SPECTER_DIR'),
            'MFR_VOCAB_DIR': os.getenv('MFR_VOCAB_DIR'),
            'MFR_CHECKPOINT_DIR': os.getenv('MFR_CHECKPOINT_DIR'),
        }
        _, bucket = load_gcs(destination_prefix)
        blob_prefix = '/'.join(destination_prefix.split('/')[3:])

        # Download only the artifacts required for this model — a pipeline worker
        # handles a single job and pulling unused models wastes startup time.
        requested_model = raw_request.get('model', {}).get('name', DEFAULT_CONFIG['model'])
        required_artifacts = artifacts_for_model(requested_model)
        print(f'Loading model artifacts for model={requested_model}: {required_artifacts}')
        load_model_artifacts(subdirs=required_artifacts)

        print('Creating job ID')
        job_id = generate_job_id()
        if working_dir is None:
            working_dir = f"/app/{job_id}"
            working_dir_created = True
        os.makedirs(working_dir, exist_ok=True)

        # APIRequest.validate(client) is NOT called here — the pipeline performs
        # no remote lookups and consumes nothing that validate() resolves
        # (user_id, edge invitation labels). Permissions were enforced on the
        # worker side. from_request runs as pure transformation.
        validated_request = APIRequest({
            'name': raw_request['name'],
            'entityA': raw_request['entityA'],
            'entityB': raw_request['entityB'],
            **{k: raw_request[k] for k in ['model', 'dataset', 'machineType'] if raw_request.get(k) is not None}
        })

        print('Creating job config')
        config = JobConfig.from_request(
            api_request = validated_request,
            starting_config = DEFAULT_CONFIG,
            server_config = server_config,
            working_dir = working_dir,
        )

        if working_dir is not None:
            path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
            config.job_dir = working_dir
            config.dataset['directory'] = working_dir
            for field in path_fields:
                config.model_params[field] = working_dir

        if dataset_gcs_path:
            print(f'Downloading pre-created dataset from {dataset_gcs_path}')
            download_dataset_from_gcs(dataset_gcs_path, working_dir)

        # Query the global parquet embedding cache for note embeddings.
        # Missing embeddings are computed from the paper data below and appended
        # back to the cache after scoring.
        try:
            from expertise.embeddings_cache import GlobalEmbeddingsCache
            archives_path = Path(config.job_dir) / 'archives'
            submissions_path = Path(config.job_dir) / 'submissions'
            submissions_json = Path(config.job_dir) / 'submissions.json'

            paper_mdates = {}
            publication_ids = set()
            if archives_path.exists():
                for author_file in archives_path.iterdir():
                    if not author_file.name.endswith('.jsonl'):
                        continue
                    with open(author_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                pub = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            pid = pub.get('id')
                            if pid:
                                publication_ids.add(pid)
                                if pub.get('mdate'):
                                    paper_mdates[pid] = int(pub['mdate'])

            submission_ids = set()
            if submissions_path.exists():
                for submission_file in submissions_path.iterdir():
                    if not submission_file.name.endswith('.jsonl'):
                        continue
                    note_id = submission_file.stem
                    if note_id:
                        submission_ids.add(note_id)
                        with open(submission_file) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    sub = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if sub.get('mdate'):
                                    paper_mdates[note_id] = int(sub['mdate'])
            if submissions_json.exists():
                with open(submissions_json) as f:
                    for note_id, note in json.load(f).items():
                        if note_id:
                            submission_ids.add(note_id)
                            if isinstance(note, dict) and note.get('mdate'):
                                paper_mdates[note_id] = int(note['mdate'])

            note_ids = publication_ids | submission_ids
            model_to_cache_key = {
                'specter2+scincl': ['specter', 'scincl'],
                'specter2': ['specter'],
                'scincl': ['scincl'],
                'specter': ['specter'],
                'specter+mfr': ['specter'],
            }
            targets = model_to_cache_key.get(config.model, [])

            cached_embeddings = {}
            if note_ids and targets:
                cache_prefix = 'embeddings-cache-dev' if 'jobs-dev' in destination_prefix else 'embeddings-cache'
                cache = GlobalEmbeddingsCache(
                    bucket_name=os.getenv('EMBEDDING_CACHE_BUCKET', 'openreview-expertise'),
                    cache_prefix=cache_prefix
                )
                cached_embeddings = {cache_key: {} for cache_key in targets}

                embeddings_by_model = cache.get_embeddings_for_models(list(note_ids), targets, paper_mdates=paper_mdates)

                for cache_key in targets:
                    model_embeddings = embeddings_by_model.get(cache_key, {})
                    cached_embeddings[cache_key] = model_embeddings
                    count = len(model_embeddings)
                    if count > 0:
                        print(f"Pre-populated {count} embeddings from global cache ({cache_key})", flush=True)
            else:
                print("No note IDs found; skipping global cache lookup", flush=True)
        except Exception as e:
            print(f"Global cache lookup failed: {e}", flush=True)

        print('Executing expertise')
        new_embeddings = execute_expertise(
            config.to_json(),
            cached_embeddings=cached_embeddings,
        )

        # Fetch and write to storage
        print('Fetching and writing to storage')

        # Upload the score CSV(s) to GCS as-is (no transformation). The CSV
        # rows are in the model's natural order: [test_id, reviewer_id, score]
        # for mixed Group/Note matching; [test_id, train_id, score] for
        # symmetric matching. The matching-type-aware entityA/entityB swap
        # happens at read time in both the local and cloud reader paths,
        # consistent with how ExpertiseService.get_expertise_results has
        # always handled it.
        print("Uploading score CSV(s)...", flush=True)
        for csv_file in [d for d in os.listdir(config.job_dir) if d.endswith('.csv')]:
            dest_name = 'scores_sparse.csv' if '_sparse' in csv_file else 'scores.csv'
            destination_blob = f"{blob_prefix}/{dest_name}"
            blob = bucket.blob(destination_blob)
            src = os.path.join(config.job_dir, csv_file)
            blob.upload_from_filename(src)
        print("Finished uploading score CSV(s)", flush=True)

        # Upload the full scores matrix (.pt) directly.
        print("Dumping scores matrix(es)", flush=True)
        for pt_file in [d for d in os.listdir(config.job_dir) if d.endswith('.pt')]:
            destination_blob = f"{blob_prefix}/{pt_file}"
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(os.path.join(config.job_dir, pt_file))

        # Dump config
        print("Dumping job config", flush=True)
        destination_blob = f"{blob_prefix}/job_config.json"
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(json.dumps(config.to_json()))

        # Dump metadata file
        print("Dumping metadata files", flush=True)
        for json_file in [d for d in os.listdir(config.job_dir) if 'metadata' in d]:
            destination_blob = f"{blob_prefix}/{json_file}"
            with open(os.path.join(config.job_dir, json_file), 'r') as f:
                blob = bucket.blob(destination_blob)
                contents = json.dumps(json.load(f))
                blob.upload_from_string(contents)

        # Dump archives
        archives_dir = os.path.join(config.job_dir, 'archives')
        if dump_archives and os.path.isdir(archives_dir):
            print("Dumping archives", flush=True)
            for jsonl_file in os.listdir(archives_dir):
                result = []
                destination_blob = f"{blob_prefix}/archives/{jsonl_file}"
                with open(os.path.join(archives_dir, jsonl_file), 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        result.append({
                            'id': data['id'],
                            'content': data['content']
                        })
                blob = bucket.blob(destination_blob)
                contents = '\n'.join([json.dumps(r) for r in result])
                blob.upload_from_string(contents)

        # Dump embeddings JSONL files to the job folder for backwards
        # compatibility. These can be used to revert to master if needed.
        print("Dumping embeddings", flush=True)
        for emb_file in [d for d in os.listdir(config.job_dir) if '.jsonl' in d]:
            destination_blob = f"{blob_prefix}/{emb_file}"
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(os.path.join(config.job_dir, emb_file))

        # Append newly computed embeddings to the global parquet cache so future
        # jobs can reuse them without recomputation.
        try:
            _append_embeddings_to_global_cache(new_embeddings, blob_prefix, bucket)
        except Exception as e:
            print(f"Global cache append failed (non-critical): {e}", flush=True)

    except Exception as e:
        # Write error to single JSONL line in GCS if bucket is available
        if bucket is not None and blob_prefix is not None:
            error_message = {
                'error': str(e),
                'expected': isinstance(e, ExpectedDataError)
            }
            destination_blob = f"{blob_prefix}/error.json"
            blob = bucket.blob(destination_blob)
            blob.upload_from_string(json.dumps(error_message))

        exception = e.with_traceback(e.__traceback__)
        raise exception
    finally:
        # Clean up working directory to prevent disk exhaustion
        # For Vertex AI, the container is ephemeral, but explicit cleanup
        # ensures no disk leak when running locally or in tests
        if working_dir_created and working_dir and os.path.isdir(working_dir):
            print(f'Cleaning up working directory: {working_dir}')
            shutil.rmtree(working_dir)

def _append_embeddings_to_global_cache(new_embeddings, blob_prefix, bucket):
    """Append newly computed in-memory embeddings to the Hive-partitioned GCS Parquet cache.

    new_embeddings is a dict mapping filename (e.g. 'pub2vec_specter.jsonl') to a
    dict of paper_id -> embedding (list of floats). Uses current UTC time for the
    year_month partition and embedding_date.

    Returns None on success. Returns {} and logs a warning if pyarrow/pandas are unavailable.
    Failures during write are caught and logged as non-critical.
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.dataset as ds
    except ImportError:
        print("pyarrow/pandas not available; skipping global cache append", flush=True)
        return {}

    cache_prefix = 'embeddings-cache-dev' if 'jobs-dev' in blob_prefix else 'embeddings-cache'

    dt = datetime.datetime.now(tz=datetime.timezone.utc)
    year_month = dt.strftime("%Y-%m")

    def _model_for_emb_file(emb_file):
        if 'scincl' in emb_file:
            return 'scincl'
        return 'specter'

    embedding_date = dt.replace(tzinfo=None)

    job_id = blob_prefix.split('/')[-1]
    records = []
    for emb_file, embeddings in (new_embeddings or {}).items():
        model = _model_for_emb_file(emb_file)
        if not embeddings:
            continue
        for pid, emb in embeddings.items():
            records.append({
                'paper_id': pid,
                'embedding': emb,
                'model': model,
                'year_month': year_month,
                'embedding_date': embedding_date,
                'job_id': job_id,
            })

    if not records:
        print("No new embeddings to append", flush=True)
        return

    df = pd.DataFrame(records)
    gcs_path = f"gs://{bucket.name}/{cache_prefix}"
    try:
        existing_dataset = ds.dataset(gcs_path, partitioning="hive")
        for model in df["model"].unique().tolist():
            existing_table = existing_dataset.to_table(
                columns=["paper_id", "embedding_date"],
                filter=(pc.field("model") == model)
            )
            if existing_table.num_rows == 0:
                continue
            latest_table = existing_table.group_by("paper_id").aggregate([("embedding_date", "max")])
            latest_rows = latest_table.to_pydict()
            latest_embedding_date = {
                pid: edate for pid, edate in zip(latest_rows["paper_id"], latest_rows["embedding_date_max"])
                if edate is not None
            }
            mask = (df["model"] == model) & df["paper_id"].isin(latest_embedding_date)
            existing_dates = pd.Series(latest_embedding_date)
            mapped = df.loc[mask, "paper_id"].map(existing_dates)
            stale_idx = df.loc[mask].index[df.loc[mask, "embedding_date"] <= mapped]
            df = df.drop(stale_idx)
    except Exception as e:
        print(f"No existing cache to filter against: {e}", flush=True)

    if df.empty:
        print("No new embeddings to append; all paper_ids already cached", flush=True)
        return

    try:
        filtered = df.to_dict('records')
        table = pa.table({
            'paper_id': [r['paper_id'] for r in filtered],
            'embedding': [r['embedding'] for r in filtered],
            'model': [r['model'] for r in filtered],
            'year_month': [r['year_month'] for r in filtered],
            'embedding_date': pa.array([r['embedding_date'] for r in filtered], pa.timestamp('ms')),
            'job_id': [r['job_id'] for r in filtered],
        })
        ds.write_dataset(
            table,
            gcs_path,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([
                    ("model", pa.string()),
                    ("year_month", pa.string()),
                    ("job_id", pa.string()),
                ]),
                flavor="hive",
            ),
            existing_data_behavior="overwrite_or_ignore",
        )
    except Exception as e:
        print(f"Global cache write_dataset failed: {e}", flush=True)
        return
    print(f"Appended {len(df)} new embeddings to {gcs_path}", flush=True)


if __name__ == '__main__':
    print('Starting pipeline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_request_str', help='a JSON string or file containing all other arguments')
    parser.add_argument('--gcs_dir', help='GCS directory containing request.json')
    parser.add_argument('--dataset_gcs_path', help='GCS path to pre-created dataset; skips dataset creation when provided')
    args = parser.parse_args()

    if args.gcs_dir:
        run_pipeline(gcs_dir=args.gcs_dir, dataset_gcs_path=args.dataset_gcs_path)
    else:
        run_pipeline(api_request_str=args.api_request_str, dataset_gcs_path=args.dataset_gcs_path)
