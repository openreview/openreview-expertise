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

[{'match_member': '~Harold_Rice1', 'score': 0.59, 'submission_member': '~Harold_Rice1'}, {'match_member': '~C.V._Lastname1', 'score': 0.56, 'submission_member': '~Harold_Rice1'}, {'match_member': '~Zonia_Willms1', 'score': 0.53, 'submission_member': '~Harold_Rice1'}, {'match_member': '~Royal_Toy1', 'score': 0.51, 'submission_member': '~Harold_Rice1'}, {'match_member': '~Harold_Rice1', 'score': 0.51, 'submission_member': '~Royal_Toy1'}, {'match_member': '~C.V._Lastname1', 'score': 0.5, 'submission_member': '~Royal_Toy1'}, {'match_member': '~Zonia_Willms1', 'score': 0.6, 'submission_member': '~Royal_Toy1'}, {'match_member': '~Royal_Toy1', 'score': 0.8, 'submission_member': '~Royal_Toy1'}, {'match_member': '~Harold_Rice1', 'score': 0.56, 'submission_member': '~C.V._Lastname1'}, {'match_member': '~C.V._Lastname1', 'score': 0.8, 'submission_member': '~C.V._Lastname1'}, {'match_member': '~Zonia_Willms1', 'score': 0.58, 'submission_member': '~C.V._Lastname1'}, {'match_member': '~Royal_Toy1', 'score': 0.5, 'submission_member': '~C.V._Lastname1'}, {'match_member': '~Harold_Rice1', 'score': 0.54, 'submission_member': '~Zonia_Willms1'}, {'match_member': '~C.V._Lastname1', 'score': 0.59, 'submission_member': '~Zonia_Willms1'}, {'match_member': '~Zonia_Willms1', 'score': 0.68, 'submission_member': '~Zonia_Willms1'}, {'match_member': '~Royal_Toy1', 'score': 0.6, 'submission_member': '~Zonia_Willms1'}]

[{'match_submission': 'SlmLlyHt31x', 'score': 0.9999999105930328, 'submission': 'SlmLlyHt31x'}, {'match_submission': 'SelIx1SK31x', 'score': 0.8732473254203796, 'submission': 'SlmLlyHt31x'}, {'match_submission': 'SlmLlyHt31x', 'score': 0.8732473254203796, 'submission': 'SelIx1SK31x'}, {'match_submission': 'SelIx1SK31x', 'score': 1.0, 'submission': 'SelIx1SK31x'}]

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
    job_id = "group_paper_scores"
    
    # Search for all jobs in the bucket
    storage_client = gcp_interface.gcs_client
    bucket = storage_client.bucket("test-bucket")

    # Simulate writing request.json to the bucket
    with open('tests/container_jsons/container_paper_group.json', 'r') as f:
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
    assert 'jobs/group_paper_scores/job_config.json' in blob_names
    assert 'jobs/group_paper_scores/metadata.json' in blob_names
    assert 'jobs/group_paper_scores/test_container.jsonl' in blob_names
    assert 'jobs/group_paper_scores/test_container_sparse.jsonl' in blob_names

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