#!/usr/bin/env python
# Save this as verify_bucket.py

import sys
import json
import os
from expertise.service.utils import GCPInterface, RedisDatabase

# Define expected rows
EXPECTED_ROWS = [
    ('SxNx1Xf83yl', '~Harold_Rice1', '0.50'),
    ('HgbgymMI21e', '~Harold_Rice1', '0.52'),
    ('SxNx1Xf83yl', '~C.V._Lastname1', '0.49'),
    ('HgbgymMI21e', '~C.V._Lastname1', '0.46'),
    ('SxNx1Xf83yl', '~Zonia_Willms1', '0.68'),
    ('HgbgymMI21e', '~Zonia_Willms1', '0.59'),
    ('SxNx1Xf83yl', '~Royal_Toy1', '0.66'),
    ('HgbgymMI21e', '~Royal_Toy1', '0.56')
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
        jobs_folder="test-folder",
        # Use a mock client so we can still access the bucket
        openreview_client=None
    )
    
    # Try to connect to Redis to get job IDs
    print("Connecting to Redis to get job IDs...")
    try:
        redis_db = RedisDatabase(
            host="localhost",
            port=6379,
            db=10
        )
        # List all jobs
        job_configs = redis_db.load_all_jobs("openreview.net")
        
        if job_configs:
            print(f"Found {len(job_configs)} jobs in Redis")
            for config in job_configs:
                print(f"Job ID: {config.cloud_id}, Name: {config.name}")
                print(f"Job config: {json.dumps(config.to_json(), indent=2)}")
        else:
            print("No jobs found in Redis")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        print("Proceeding with direct bucket verification")
    
    try:
        # Try to fetch results for each job ID
        for config in job_configs:
            job_id = config.cloud_id
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
                    submission_id, user_id, expected_score = expected_row
                    for score in scores:
                        if score['submission'] == submission_id and score['user'] == user_id:
                            assert str(score['score']).startswith(expected_score), \
                                f"Expected score for {submission_id}/{user_id} to start with {expected_score}, got {score['score']}"
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