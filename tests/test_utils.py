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