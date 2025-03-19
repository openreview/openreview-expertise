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
DELETED_FIELDS = ['user_id', 'cdate']

def run_pipeline(api_request_str, working_dir=None):
    raw_request: dict = json.loads(api_request_str)

    # Pop token, base URLs and other expected variable
    print('Popping variables')
    for field in DELETED_FIELDS:
        raw_request.pop(field, None)
    token = raw_request.pop('token')
    baseurl_v1 = raw_request.pop('baseurl_v1')
    baseurl_v2 = raw_request.pop('baseurl_v2')
    destination_prefix = raw_request.pop('gcs_folder')
    skip_artifacts = raw_request.pop('skip_artifacts', False)
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

    print('Loading model artifacts')
    if not skip_artifacts:
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

    # Fetch and write to storage
    print('Fetching and writing to storage')
    group_group_matching = validated_request.entityA.get('type', '') == 'Group' and \
        validated_request.entityB.get('type', '') == 'Group'
    paper_paper_matching = validated_request.entityA.get('type', '') == 'Note' and \
        validated_request.entityB.get('type', '') == 'Note'

    # GCS Debug: Print environment variables
    print(f"GCS Debug: STORAGE_EMULATOR_HOST: {os.environ.get('STORAGE_EMULATOR_HOST')}")
    print(f"GCS Debug: GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    
    # GCS Debug: Verify destination path
    bucket_name = destination_prefix.split('/')[2]
    blob_prefix = '/'.join(destination_prefix.split('/')[3:])
    print(f"GCS Debug: Parsed bucket name: {bucket_name}")
    print(f"GCS Debug: Parsed blob prefix: {blob_prefix}")
    
    # Create a test file in the output directory
    try:
        with open(os.path.join(config.job_dir, "request.json"), "w") as f:
            json.dump({"test": "data"}, f)
        print(f"GCS Debug: Created test file in output dir: {os.path.join(config.job_dir, 'request.json')}")
    except Exception as e:
        print(f"GCS Debug: Failed to create test file: {str(e)}")
    
    # GCS Debug: List directory contents before upload
    print(f"GCS Debug: Working directory: {config.job_dir}")
    try:
        print(f"GCS Debug: Files in working directory: {os.listdir(config.job_dir)}")
    except Exception as e:
        print(f"GCS Debug: Failed to list directory: {str(e)}")
    
    # Initialize GCS client with debug info
    try:
        print("GCS Debug: Initializing storage client")
        gcs_client = storage.Client()
        print("GCS Debug: Storage client initialized successfully")
        
        # Try a simple operation to verify connection
        print("GCS Debug: Listing buckets to test connection")
        buckets = list(gcs_client.list_buckets())
        print(f"GCS Debug: Found {len(buckets)} buckets")
        
        print(f"GCS Debug: Getting bucket: {bucket_name}")
        bucket = gcs_client.bucket(bucket_name)
        print(f"GCS Debug: Got bucket: {bucket}")
        
        # Try to upload a simple test blob
        test_blob = bucket.blob(f"{blob_prefix}/test_connection.txt")
        print(f"GCS Debug: Uploading test blob: {test_blob.name}")
        test_blob.upload_from_string("Test connection successful")
        print("GCS Debug: Test blob uploaded successfully")
        
        # CSV File upload
        csv_files = [d for d in os.listdir(config.job_dir) if '.csv' in d]
        print(f"GCS Debug: Found {len(csv_files)} CSV files to upload")
        
        for csv_file in csv_files:
            print(f"GCS Debug: Processing CSV file: {csv_file}")
            result = []
            destination_blob = f"{blob_prefix}/{csv_file.replace('.csv', '.jsonl')}"
            print(f"GCS Debug: Destination blob: {destination_blob}")
            
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
            
            print(f"GCS Debug: Processed {len(result)} rows from CSV")
            blob = bucket.blob(destination_blob)
            contents = '\n'.join([json.dumps(r) for r in result])
            print(f"GCS Debug: Uploading {len(contents)} bytes to blob: {blob.name}")
            
            try:
                blob.upload_from_string(contents)
                print(f"GCS Debug: Successfully uploaded blob: {blob.name}")
            except Exception as e:
                print(f"GCS Debug: Failed to upload blob {blob.name}: {str(e)}")
        
        # Dump config
        print("GCS Debug: Uploading job config")
        destination_blob = f"{blob_prefix}/job_config.json"
        blob = bucket.blob(destination_blob)
        try:
            blob.upload_from_string(json.dumps(config.to_json()))
            print(f"GCS Debug: Successfully uploaded job config")
        except Exception as e:
            print(f"GCS Debug: Failed to upload job config: {str(e)}")

        # Dump metadata files
        metadata_files = [d for d in os.listdir(config.job_dir) if 'metadata' in d]
        print(f"GCS Debug: Found {len(metadata_files)} metadata files to upload")
        
        for json_file in metadata_files:
            destination_blob = f"{blob_prefix}/{json_file}"
            print(f"GCS Debug: Processing metadata file: {json_file} -> {destination_blob}")
            try:
                with open(os.path.join(config.job_dir, json_file), 'r') as f:
                    blob = bucket.blob(destination_blob)
                    contents = json.dumps(json.load(f))
                    blob.upload_from_string(contents)
                    print(f"GCS Debug: Successfully uploaded metadata file: {json_file}")
            except Exception as e:
                print(f"GCS Debug: Failed to upload metadata file {json_file}: {str(e)}")
        
        # List uploaded files to verify
        print("GCS Debug: Listing blobs in bucket to verify uploads")
        blobs = list(bucket.list_blobs(prefix=blob_prefix))
        print(f"GCS Debug: Found {len(blobs)} blobs with prefix {blob_prefix}:")
        for blob in blobs:
            print(f"  - {blob.name}")
            
    except Exception as e:
        print(f"GCS Debug: Error during GCS operations: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print('Starting pipeline')
    parser = argparse.ArgumentParser()
    parser.add_argument('api_request_str', help='a JSON file containing all other arguments')
    args = parser.parse_args()
    
    run_pipeline(args.api_request_str)