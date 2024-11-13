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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('api_request_str', help='a JSON file containing all other arguments')
    args = parser.parse_args()
    raw_request: dict = json.loads(args.api_request_str)

    # Pop token, base URLs and other expected variables
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
    load_model_artifacts()

    client_v1 = openreview.Client(baseurl=baseurl_v1, token=token)
    client_v2 = openreview.api.OpenReviewClient(baseurl_v2, token=token)

    job_id = shortuuid.ShortUUID().random(length=5)
    working_dir = f"/app/{job_id}"
    os.makedirs(working_dir, exist_ok=True)  

    validated_request = APIRequest(raw_request)
    config = JobConfig.from_request(
        api_request = validated_request,
        starting_config = DEFAULT_CONFIG,
        openreview_client= client_v1,
        openreview_client_v2= client_v2,
        server_config = server_config,
        working_dir = working_dir
    )

    # Create Dataset and Execute Expertise
    execute_create_dataset(client_v1, client_v2, config.to_json())
    execute_expertise(config.to_json())

    # Fetch and write to storage
    bucket_name = destination_prefix.split('/')[2]
    blob_prefix = '/'.join(destination_prefix.split('/')[3:])
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    for csv_file in [d for d in os.listdir(config.job_dir) if '.csv' in d]:
        result = []
        destination_blob = f"{blob_prefix}/{csv_file.replace('.csv', '.jsonl')}"
        with open(os.path.join(config.job_dir, csv_file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
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