#!/usr/bin/env python
# Save this as verify_bucket.py

import sys
import json
import os
from expertise.service.utils import GCPInterface, RedisDatabase

# Define expected rows
EXPECTED_ROWS = [
    ("SlmLlyHt31x","SlmLlyHt31x","0.9999999105930328")
    ("SelIx1SK31x","SlmLlyHt31x","0.87")
    ("SlmLlyHt31x","SelIx1SK31x","0.87")
    ("SelIx1SK31x","SelIx1SK31x","1.0")
]

def verify_bucket():
    print("Verifying GCS bucket using production interface")
    
    # Print environment variables for debugging
    print(f"STORAGE_EMULATOR_HOST: {os.environ.get('STORAGE_EMULATOR_HOST')}")
    print(f"GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    
    # Get project ID from environment
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'test-project')
    
    # Create GCPInterface with minimal configuration for GCS operations
    gcp_interface = GCPInterface(
        project_id=project_id,
        bucket_name="test-bucket",
        jobs_folder="jobs",
        # Use a mock client so we can still access the bucket
        openreview_client=None
    )
    job_id = "paper_paper_scores"
    
    # Search for all jobs in the bucket
    storage_client = gcp_interface.gcs_client
    bucket = storage_client.bucket("test-bucket")

    # Simulate writing request.json to the bucket
    with open('tests/container_jsons/container_paper_paper.json', 'r') as f:
        request = json.load(f)
    blob = bucket.blob(f"{gcp_interface.jobs_folder}/{job_id}/request.json")
    blob.upload_from_string(
        data=json.dumps(request),
        content_type="application/json"
    )

    job_blobs = list(bucket.list_blobs(prefix=f"{gcp_interface.jobs_folder}/{job_id}/"))
    blob_names = [blob.name for blob in job_blobs]
    print(f"Blob names: {blob_names}")  

    # Test job file creation
    assert 'jobs/paper_paper_scores/job_config.json' in blob_names
    assert 'jobs/paper_paper_scores/metadata.json' in blob_names
    assert 'jobs/paper_paper_scores/test_container.jsonl' in blob_names
    assert 'jobs/paper_paper_scores/test_container_sparse.jsonl' in blob_names

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
                match_submission_id, submission_id, expected_score = expected_row
                for score in scores:
                    # Special case for self-self score
                    if score['match_submission_id'] == score['submission_id']:
                        assert score['score'] >= 0.99, \
                            f"Expected score for {match_submission_id}/{submission_id} to start with {expected_score}, got {score['score']}"
                    elif score['match_submission_id'] == match_submission_id and score['submission_id'] == submission_id:
                        assert str(score['score']).startswith(expected_score), \
                            f"Expected score for {match_submission_id}/{submission_id} to start with {expected_score}, got {score['score']}"
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