import pytest
import re
import time
from collections import Counter
from expertise.utils.utils import generate_job_id, JOB_ID_ALPHABET


def test_generate_job_id_gcp_compliance():
    """
    Test that generated job IDs comply with GCP Vertex AI pipeline requirements:
    - Must be less than 128 characters
    - Valid characters are [a-z][0-9]-
    - First character must be a letter
    
    Using the regex pattern from GCP Vertex AI error message:
    "Expecting an ID following the regex pattern '[a-z][-a-z0-9]{0,127}'"
    """
    # GCP pipeline job ID pattern from Vertex AI error message
    gcp_pattern = re.compile(r'^[a-z][-a-z0-9]{0,127}$')
    
    # Generate a large number of IDs to test
    num_ids = 10000
    generated_ids = []
    
    for i in range(num_ids):
        job_id = generate_job_id()
        generated_ids.append(job_id)
        
        # Test that it matches the EXACT GCP pattern from Vertex AI
        assert gcp_pattern.match(job_id), (
            f"Job ID '{job_id}' does not match GCP Vertex AI requirements. "
            f"Must match regex pattern '[a-z][-a-z0-9]{{0,127}}' "
            f"(start with lowercase letter, followed by lowercase letters, numbers, or hyphens)"
        )
        
        # Verify first character is a letter
        assert job_id[0].isalpha() and job_id[0].islower(), (
            f"Job ID '{job_id}' must start with a lowercase letter"
        )
        
        # Verify no uppercase letters
        assert job_id.islower(), f"Job ID '{job_id}' contains uppercase letters"
        
        # Verify only valid characters (lowercase letters, numbers, hyphens)
        for char in job_id:
            assert char in 'abcdefghijklmnopqrstuvwxyz0123456789-', (
                f"Job ID '{job_id}' contains invalid character '{char}'"
            )
    
    # Test uniqueness
    unique_ids = set(generated_ids)
    duplicates = len(generated_ids) - len(unique_ids)
    duplicate_rate = duplicates / num_ids * 100
    
    # Allow a very small duplicate rate for random generation (should be near 0%)
    assert duplicate_rate < 0.1, (
        f"Too many duplicate IDs: {duplicates} out of {num_ids} ({duplicate_rate:.2f}%)"
    )
    
    # Verify the exact length is 10 as specified in generate_job_id
    for job_id in generated_ids[:100]:  # Check first 100
        assert len(job_id) == 10, f"Job ID '{job_id}' should be exactly 10 characters, got {len(job_id)}"