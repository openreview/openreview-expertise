import openreview
import os
import sys
import json
import time
from tests.conftest import GCSTestHelper

# Helper script to generate a token for the container tests
# and populate an example JSON file with the token

# Writes out the test JSON to test_input.json

def generate_token(path_to_json):
    try:
        from google.cloud import storage
        import datetime
        
        client = openreview.api.OpenReviewClient(
            "http://localhost:3001", 
            username="openreview.net", 
            password="Or$3cur3P@ssw0rd"
        )

        # Get project ID from environment
        bucket_name = GCSTestHelper.GCS_TEST_BUCKET

        # Load JSON template
        with open(path_to_json, "r") as f:
            data = json.load(f)

        gcs_folder = data['gcs_folder']
        # Insert timestamp into the folder path
        case = gcs_folder.split('/')[-1]
        gcs_folder = '/'.join(gcs_folder.split('/')[:-1])
        folder_path = f"{gcs_folder}/{int(time.time())}/{case}"
        relative_folder_path = '/'.join(folder_path.split('/')[3:])
        
        # Write to GCS (mimicking create_job behavior)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Create folder if it doesn't exist
        folder_blob = bucket.blob(f"{relative_folder_path}/")
        folder_blob.upload_from_string('')

        data['gcs_folder'] = folder_path
        data['token'] = client.token
        
        # Write request.json
        request_blob = bucket.blob(f"{relative_folder_path}/request.json")
        request_blob.upload_from_string(
            data=json.dumps(data),
            content_type="application/json"
        )
        
        # Output the GCS path for CircleCI to use
        gcs_request_path = f"{folder_path}/request.json"
        
        # Write GCS path to file for CircleCI to read
        with open("gcs_request_path.txt", "w") as f:
            f.write(gcs_request_path)
        
        print(f"Request written to GCS: {gcs_request_path}")
        print("GCS path written to: gcs_request_path.txt")
        return 0
        
    except Exception as e:
        print(f"Error generating token and writing to GCS: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(generate_token(sys.argv[1]))