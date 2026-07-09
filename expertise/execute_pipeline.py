import argparse
import os
import json
import csv
import tarfile
import tempfile
import shutil
import time
from pathlib import Path
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
    """Download a dataset tarball from GCS and extract it into local_dir.

    Cached publication embeddings (cached_pub2vec_*.jsonl) are stored alongside
    the dataset/ folder at the job's GCS root rather than inside the tarball
    (see upload_dataset for rationale). Pull them directly into local_dir so
    the predictor finds them next to pub2vec_*.jsonl as before.
    """
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

    job_root_prefix = blob_name.rsplit('/', 2)[0] + '/'
    for blob in bucket.list_blobs(prefix=job_root_prefix, delimiter='/'):
        name = blob.name.rsplit('/', 1)[-1]
        if name.startswith('cached_pub2vec_') and name.endswith('.jsonl'):
            dest = os.path.join(local_dir, name)
            blob.download_to_filename(dest)
            print(f"Downloaded cached embeddings {name} to {dest}")


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

        # Query the global parquet embedding cache for publication embeddings.
        # Missing embeddings are computed from the paper data below and appended
        # back to the cache after scoring.
        cache_start = time.time()
        try:
            from expertise.embeddings_cache import GlobalEmbeddingsCache
            archives_path = Path(config.job_dir) / 'archives'
            paper_ids = set()
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
                                paper_ids.add(pid)
            if paper_ids:
                cache_prefix = 'embeddings-cache-dev' if 'jobs-dev' in destination_prefix else 'embeddings-cache'
                cache = GlobalEmbeddingsCache(
                    bucket_name=os.getenv('EMBEDDING_CACHE_BUCKET', 'openreview-expertise'),
                    cache_prefix=cache_prefix
                )
                model_to_cache_name = {
                    'specter2+scincl': [
                        ('specter', 'cached_pub2vec_specter.jsonl'),
                        ('scincl', 'cached_pub2vec_scincl.jsonl'),
                    ],
                    'specter2': [('specter', 'cached_pub2vec.jsonl')],
                    'scincl': [('scincl', 'cached_pub2vec.jsonl')],
                    'specter': [('specter', 'cached_pub2vec.jsonl')],
                }
                targets = model_to_cache_name.get(config.model, [])
                total_cached = 0
                for cache_key, dest_name in targets:
                    dest_path = Path(config.job_dir) / dest_name
                    count = cache.write_cache_file(list(paper_ids), cache_key, str(dest_path))
                    total_cached += count
                    if count > 0:
                        print(f"Pre-populated {count} embeddings from global cache ({cache_key})", flush=True)
                print(f"Global cache lookup completed in {time.time() - cache_start:.2f}s ({total_cached}/{len(paper_ids)} embeddings found)", flush=True)
            else:
                print("No paper IDs found; skipping global cache lookup", flush=True)
        except Exception as e:
            print(f"Global cache lookup failed: {e}", flush=True)
            raise

        print('Executing expertise')
        execute_expertise(config.to_json())

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

        # Upload the full scores matrix (.pt) directly. Same direct-streaming
        # pattern as the embedding files — source bytes go disk → resumable-
        # upload buffer → GCS without ever being parsed in Python.
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

        # Always dump embeddings to bucket. Source jsonl already has exactly
        # {'paper_id', 'embedding'} (see Predictor._build_embedding_jsonl), so
        # there's nothing to transform — upload the file directly via a
        # resumable upload to avoid loading 500k embeddings (~12GB of Python
        # objects) into memory just to re-serialize them.
        print("Dumping embeddings", flush=True)
        for emb_file in [d for d in os.listdir(config.job_dir) if '.jsonl' in d]:
            destination_blob = f"{blob_prefix}/{emb_file}"
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(os.path.join(config.job_dir, emb_file))

        # Append newly computed embeddings to the global parquet cache so future
        # jobs can reuse them without recomputation.
        try:
            _append_embeddings_to_global_cache(config.job_dir, blob_prefix, bucket)
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

def _append_embeddings_to_global_cache(job_dir, blob_prefix, bucket):
    """Append pub2vec JSONL embeddings to the Hive-partitioned GCS Parquet cache.

    Uses the job timestamp embedded in blob_prefix for the year_month partition.
    Metadata columns (venueid, title, invitation, domain, cdate, mdate) are left
    empty/null for now and will be backfilled once the pipeline carries that
    information through.
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.dataset as ds
    except ImportError:
        print("pyarrow/pandas not available; skipping global cache append", flush=True)
        return

    cache_prefix = 'embeddings-cache-dev' if 'jobs-dev' in blob_prefix else 'embeddings-cache'

    ts_ms = None
    for part in blob_prefix.split('/'):
        if '-' in part:
            try:
                ts_ms = int(part.split('-')[-1])
                break
            except ValueError:
                pass
    if ts_ms is None:
        import time
        ts_ms = int(time.time() * 1000)
    import datetime
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
    year_month = dt.strftime("%Y-%m")

    model_map = {
        'pub2vec_specter.jsonl': 'specter',
        'pub2vec_scincl.jsonl': 'scincl',
        'pub2vec.jsonl': 'specter',
    }

    records = []
    for emb_file, model in model_map.items():
        local_path = os.path.join(job_dir, emb_file)
        if not os.path.exists(local_path):
            continue
        with open(local_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = entry.get('paper_id')
                emb = entry.get('embedding')
                if pid is None or emb is None:
                    continue
                records.append({
                    'paper_id': pid,
                    'job_id': blob_prefix.split('/')[-1],
                    'embedding': emb,
                    'uri': f"gs://{bucket.name}/{blob_prefix}/{emb_file}",
                    'model': model,
                    'year_month': year_month,
                    'venueid': '',
                    'title': '',
                    'invitation': '',
                    'domain': None,
                    'cdate': None,
                    'mdate': None,
                })

    if not records:
        return

    df = pd.DataFrame(records)
    gcs_path = f"gs://{bucket.name}/{cache_prefix}"
    existing_ids = set()
    try:
        existing_dataset = ds.dataset(gcs_path, partitioning="hive")
        existing_table = existing_dataset.to_table(
            columns=["paper_id"],
            filter=(pc.field("model").isin(df["model"].unique().tolist()))
        )
        if existing_table.num_rows > 0:
            existing_ids = set(existing_table.column("paper_id").to_pylist())
    except Exception as e:
        print(f"No existing cache to filter against: {e}", flush=True)

    if existing_ids:
        df = df[~df["paper_id"].isin(existing_ids)]

    if df.empty:
        print("No new embeddings to append; all paper_ids already cached", flush=True)
        return

    for model, model_df in df.groupby("model"):
        partition_path = f"{gcs_path}/model={model}/year_month={year_month}"
        blobs = list(bucket.list_blobs(prefix=partition_path + "/"))
        part_files = [b.name for b in blobs if b.name.endswith(".parquet")]
        next_idx = len(part_files)
        dest_blob_name = f"{cache_prefix}/model={model}/year_month={year_month}/part-{next_idx:08d}.parquet"
        table = pa.Table.from_pandas(model_df)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            local_path = tmp.name
        try:
            pq.write_table(table, local_path, compression="zstd")
            bucket.blob(dest_blob_name).upload_from_filename(local_path)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
        print(f"Appended {len(model_df)} new embeddings to gs://{bucket.name}/{dest_blob_name}", flush=True)


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
