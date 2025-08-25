import argparse
import os
import openreview
import shortuuid
import json
import csv
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service import load_model_artifacts
from expertise.service.utils import APIRequest, JobConfig
from google.cloud import storage

DEFAULT_CONFIG = {
    "dataset": {},
    "model": "specter+mfr",
    "model_params": {
        "use_title": True,
        "sparse_value": 600,
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

def run_pipeline(
        api_request_str=None, 
        gcs_dir=None,
        working_dir=None
    ):

    if gcs_dir is None and api_request_str is None:
        raise ValueError("Either gcs_dir or api_request_str must be provided")
    
    # Initialize variables for exception handling
    bucket = None
    blob_prefix = None
    
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
        
        # Pop token, base URLs and other expected variable
        print('Popping variables')
        for field in DELETED_FIELDS:
            raw_request.pop(field, None)
        token = raw_request.pop('token')
        baseurl_v1 = raw_request.pop('baseurl_v1')
        baseurl_v2 = raw_request.pop('baseurl_v2')
        destination_prefix = raw_request.pop('gcs_folder')
        dump_embs = False if 'dump_embs' not in raw_request else raw_request.pop('dump_embs')
        dump_archives = False if 'dump_archives' not in raw_request else raw_request.pop('dump_archives')
        specter_dir = os.getenv('SPECTER_DIR')
        mfr_vocab_dir = os.getenv('MFR_VOCAB_DIR')
        mfr_checkpoint_dir = os.getenv('MFR_CHECKPOINT_DIR')
        server_config ={
            'OPENREVIEW_BASEURL': baseurl_v1,
            'OPENREVIEW_BASEURL_V2': baseurl_v2,
            'SPECTER_DIR': specter_dir,
            'MFR_VOCAB_DIR': mfr_vocab_dir,
            'MFR_CHECKPOINT_DIR': mfr_checkpoint_dir,
        }
        _, bucket = load_gcs(destination_prefix)
        blob_prefix = '/'.join(destination_prefix.split('/')[3:])

        print('Loading model artifacts')
        load_model_artifacts()

        print('Logging into OpenReview')
        client_v1 = openreview.Client(baseurl=baseurl_v1, token=token)
        client_v2 = openreview.api.OpenReviewClient(baseurl_v2, token=token)

        print('Creating job ID')
        job_id = shortuuid.ShortUUID().random(length=5)
        if working_dir is None:
            working_dir = f"/app/{job_id}"
        os.makedirs(working_dir, exist_ok=True)

        print('Creating job config')
        validated_request = APIRequest(raw_request)
        config = JobConfig.from_request(
            api_request = validated_request,
            starting_config = DEFAULT_CONFIG,
            openreview_client= client_v1,
            openreview_client_v2= client_v2,
            server_config = server_config,
            working_dir = working_dir
        )

        if working_dir is not None:
            path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
            config.job_dir = working_dir
            config.dataset['directory'] = working_dir
            for field in path_fields:
                config.model_params[field] = working_dir

        # Create Dataset and Execute Expertise
        print('Creating dataset and executing expertise')
        execute_create_dataset(client_v1, client_v2, config.to_json())
        execute_expertise(config.to_json())
    except Exception as e:
        # Write error to single JSONL line in GCS if bucket is available
        if bucket is not None and blob_prefix is not None:
            error_message = {
                'error': str(e)
            }
            destination_blob = f"{blob_prefix}/error.json"
            blob = bucket.blob(destination_blob)
            blob.upload_from_string(json.dumps(error_message))

        exception = e.with_traceback(e.__traceback__)
        raise exception

    # Fetch and write to storage
    print('Fetching and writing to storage')
    group_group_matching = validated_request.entityA.get('type', '') == 'Group' and \
        validated_request.entityB.get('type', '') == 'Group'
    paper_paper_matching = validated_request.entityA.get('type', '') == 'Note' and \
        validated_request.entityB.get('type', '') == 'Note'

    for csv_file in [d for d in os.listdir(config.job_dir) if '.csv' in d]:
        result = []
        destination_blob = f"{blob_prefix}/{csv_file.replace('.csv', '.jsonl')}"
        with open(os.path.join(config.job_dir, csv_file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if group_group_matching:
                    result.append({
                        'match_member': row[0],
                        'submission_member': row[1],
                        'score': float(row[2])
                    })
                elif paper_paper_matching:
                    result.append({
                        'match_submission': row[0],
                        'submission': row[1],
                        'score': float(row[2])
                    })
                else:
                    result.append({
                        'submission': row[0],
                        'user': row[1],
                        'score': float(row[2])
                    })
                    
        blob = bucket.blob(destination_blob)
        contents = '\n'.join([json.dumps(r) for r in result])
        blob.upload_from_string(contents)

    # Dump config
    destination_blob = f"{blob_prefix}/job_config.json"
    blob = bucket.blob(destination_blob)
    blob.upload_from_string(json.dumps(config.to_json()))

    # Dump metadata file
    for json_file in [d for d in os.listdir(config.job_dir) if 'metadata' in d]:
        destination_blob = f"{blob_prefix}/{json_file}"
        with open(os.path.join(config.job_dir, json_file), 'r') as f:
            blob = bucket.blob(destination_blob)
            contents = json.dumps(json.load(f))
            blob.upload_from_string(contents)

    # Dump archives
    if dump_archives:
        for jsonl_file in os.listdir(os.path.join(config.job_dir, 'archives')):
            result = []
            destination_blob = f"{blob_prefix}/archives/{jsonl_file}"
            with open(os.path.join(config.job_dir, 'archives' ,jsonl_file), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    result.append({
                        'id': data['id'],
                        'content': data['content']
                    })
            blob = bucket.blob(destination_blob)
            contents = '\n'.join([json.dumps(r) for r in result])
            blob.upload_from_string(contents)

    # Dump embeddings
    if dump_embs:
        for emb_file in [d for d in os.listdir(config.job_dir) if '.jsonl' in d]:
            result = []
            destination_blob = f"{blob_prefix}/{emb_file}"
            with open(os.path.join(config.job_dir, emb_file), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    result.append({
                        'paper_id': data['paper_id'],
                        'embedding': data['embedding']
                    })
            blob = bucket.blob(destination_blob)
            contents = '\n'.join([json.dumps(r) for r in result])
            blob.upload_from_string(contents)

if __name__ == '__main__':
    print('Starting pipeline')
    parser = argparse.ArgumentParser()
    parser.add_argument('api_request_str', help='a JSON file containing all other arguments')
    args = parser.parse_args()
    run_pipeline(args.api_request_str)
