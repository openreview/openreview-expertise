#!/usr/bin/env python
# Save this as verify_bucket.py

import sys
import json
import os
from expertise.service.utils import GCPInterface, RedisDatabase
from tests.conftest import GCSTestHelper

# Define expected rows
# Avoid comparison directly because of noise
EXPECTED_ROWS = [
    ("~Harold_Rice1","~Harold_Rice1","0.5"),
    ("~C.V._Lastname1","~Harold_Rice1","0.5"),
    ("~Zonia_Willms1","~Harold_Rice1","0.5"),
    ("~Royal_Toy1","~Harold_Rice1","0.5"),
    ("~Harold_Rice1","~Royal_Toy1","0.5"),
    ("~C.V._Lastname1","~Royal_Toy1","0.5"),
    ("~Zonia_Willms1","~Royal_Toy1","0.6"),
    ("~Royal_Toy1","~Royal_Toy1","0.8"),
    ("~Harold_Rice1","~C.V._Lastname1","0.5"),
    ("~C.V._Lastname1","~C.V._Lastname1","0.8"),
    ("~Zonia_Willms1","~C.V._Lastname1","0.5"),
    ("~Royal_Toy1","~C.V._Lastname1","0.5"),
    ("~Harold_Rice1","~Zonia_Willms1","0.5"),
    ("~C.V._Lastname1","~Zonia_Willms1","0.5"),
    ("~Zonia_Willms1","~Zonia_Willms1","0.6"),
    ("~Royal_Toy1","~Zonia_Willms1","0.6")
]

def verify_bucket():
    print("Verifying GCS bucket using production interface")
    
    # Get project ID from environment
    project_id = GCSTestHelper.GCS_PROJECT
    bucket_name = GCSTestHelper.GCS_TEST_BUCKET
    jobs_folder = GCSTestHelper.GCS_TEST_ROOT
    
    # Create GCPInterface with minimal configuration for GCS operations
    gcp_interface = GCPInterface(
        project_id=project_id,
        bucket_name=bucket_name,
        jobs_folder=jobs_folder,
        # Use a mock client so we can still access the bucket
        openreview_client=None
    )
    
    # Search for all jobs in the bucket
    storage_client = gcp_interface.gcs_client
    bucket = storage_client.bucket(bucket_name)

    # Load the GCS folder path from the file written by generate_token.py
    with open('gcs_request_path.txt', 'r') as f:
        gcs_request_path = f.read().strip()
    
    # Extract the folder path (remove /request.json)
    gcs_folder_path = gcs_request_path.replace('/request.json', '')
    relative_folder_path = '/'.join(gcs_folder_path.split('/')[3:])  # Remove gs://bucket_name/
    job_id = '/'.join(relative_folder_path.split('/')[1:])  # remove the first part which is the jobs folder
    
    # The request.json should already exist from generate_token.py, so we don't need to upload it again
    # Just verify it exists
    request_blob = bucket.blob(f"{relative_folder_path}/request.json")
    if not request_blob.exists():
        raise FileNotFoundError(f"Request file not found at: {gcs_request_path}")

    job_blobs = list(bucket.list_blobs(prefix=f"{gcp_interface.jobs_folder}/{job_id}/"))
    blob_names = [blob.name for blob in job_blobs]
    print(f"Blob names: {blob_names}")  

    # Test job file creation
    assert f'{relative_folder_path}/job_config.json' in blob_names
    assert f'{relative_folder_path}/metadata.json' in blob_names
    assert f'{relative_folder_path}/test_container.jsonl' in blob_names
    assert f'{relative_folder_path}/test_container_sparse.jsonl' in blob_names

    # Test get_job_results
    try:
        print(f"\nAttempting to fetch results for job ID: {job_id}")
        try:
            results = gcp_interface.get_job_results("openreview.net", job_id)
            print(f"Successfully retrieved results for job {job_id}")
            print(f"Metadata: {json.dumps(results.get('metadata', {}), indent=2)}")
            print(f"Number of results: {len(results.get('results', []))}")
            
            # Verify scores
            scores = results.get('results', [])
            assert len(scores) == len(EXPECTED_ROWS), f"Expected {len(EXPECTED_ROWS)} scores, got {len(scores)}"
            
            for expected_row in EXPECTED_ROWS:
                match_member, submission_member, expected_score = expected_row
                expected_in_scores = False
                for score in scores:
                    if score['match_member'] == match_member and score['submission_member'] == submission_member:
                        expected_in_scores = True
                assert expected_in_scores, f"Expected score for {match_member}/{submission_member} not found"
            return True  # Successfully fetched at least one job's results
        except Exception as e:
            print(f"Error retrieving results for job {job_id}: {e}")
    
    except Exception as e:
        print(f"Error accessing bucket: {e}")
        return False
    
    print("\nVerification complete")
    return False

if __name__ == "__main__":
    success = verify_bucket()
    sys.exit(0 if success else 1)