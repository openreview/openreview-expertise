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
        
        # Test length requirement
        assert len(job_id) < 128, f"Job ID '{job_id}' exceeds 128 character limit (length: {len(job_id)})"
        
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


def test_generate_job_id_alphabet_compliance():
    """
    Test that generated job IDs use only the custom alphabet
    and exclude confusing characters (0, l)
    """
    # Characters that should be excluded
    excluded_chars = set('0l')
    
    # Generate many IDs to ensure alphabet compliance
    num_ids = 1000
    all_chars_used = set()
    
    for _ in range(num_ids):
        job_id = generate_job_id()
        all_chars_used.update(job_id)
        
        # Verify no excluded characters appear
        for char in excluded_chars:
            assert char not in job_id, (
                f"Job ID '{job_id}' contains excluded character '{char}'"
            )
    
    # Verify all used characters are from the custom alphabet
    for char in all_chars_used:
        assert char in JOB_ID_ALPHABET, (
            f"Character '{char}' not in custom alphabet"
        )
    
    # Print statistics for verification
    print(f"Characters used in {num_ids} IDs: {sorted(all_chars_used)}")
    print(f"Total unique characters: {len(all_chars_used)}")
    print(f"Alphabet size: {len(JOB_ID_ALPHABET)}")
    
    # Verify reasonable character distribution (should use most of the alphabet)
    assert len(all_chars_used) > len(JOB_ID_ALPHABET) * 0.8, (
        f"Only {len(all_chars_used)} out of {len(JOB_ID_ALPHABET)} "
        f"alphabet characters were used, suggesting poor randomization"
    )


def test_generate_job_id_never_starts_with_number():
    """
    Test that generated job IDs NEVER start with a number.
    This test is based on an actual error encountered:
    'ValueError: Generated job ID: 6dZTMTK29i is illegal as a Vertex pipelines job ID. 
    Expecting an ID following the regex pattern "[a-z][-a-z0-9]{0,127}"'
    """
    # The exact regex pattern from the GCP error message
    gcp_exact_pattern = re.compile(r'^[a-z][-a-z0-9]{0,127}$')
    
    # Generate many IDs to ensure none start with a number
    num_ids = 10000
    ids_starting_with_number = []
    ids_failing_gcp_pattern = []
    
    for _ in range(num_ids):
        job_id = generate_job_id()
        
        # Check if starts with a number
        if job_id[0].isdigit():
            ids_starting_with_number.append(job_id)
        
        # Check if it fails the exact GCP pattern
        if not gcp_exact_pattern.match(job_id):
            ids_failing_gcp_pattern.append(job_id)
    
    # Assert no IDs start with numbers
    assert len(ids_starting_with_number) == 0, (
        f"Found {len(ids_starting_with_number)} IDs starting with numbers! "
        f"Examples: {ids_starting_with_number[:5]}. "
        f"This would cause GCP Vertex AI to reject the job ID."
    )
    
    # Assert all IDs pass the exact GCP pattern
    assert len(ids_failing_gcp_pattern) == 0, (
        f"Found {len(ids_failing_gcp_pattern)} IDs failing GCP pattern! "
        f"Examples: {ids_failing_gcp_pattern[:5]}. "
        f"Pattern required: '[a-z][-a-z0-9]{{0,127}}'"
    )
    
    print(f"✅ All {num_ids} IDs correctly start with a letter and match GCP pattern")


def test_generate_job_id_performance():
    """
    Stress test for job ID generation performance and uniqueness at scale
    """
    num_ids = 100000
    
    # Measure generation time
    start_time = time.time()
    generated_ids = [generate_job_id() for _ in range(num_ids)]
    end_time = time.time()
    
    generation_time = end_time - start_time
    ids_per_second = num_ids / generation_time
    
    print(f"Generated {num_ids} IDs in {generation_time:.2f} seconds")
    print(f"Rate: {ids_per_second:.0f} IDs per second")
    
    # Performance assertion - should generate at least 10,000 IDs per second
    assert ids_per_second > 10000, (
        f"Performance too slow: {ids_per_second:.0f} IDs/sec (expected > 10,000)"
    )
    
    # Check uniqueness at scale
    unique_ids = set(generated_ids)
    duplicates = num_ids - len(unique_ids)
    duplicate_rate = duplicates / num_ids * 100
    
    print(f"Duplicates: {duplicates} out of {num_ids} ({duplicate_rate:.4f}%)")
    
    # With 10-character IDs from a 33-character alphabet, collision probability is very low
    # Allow up to 0.01% duplicate rate (10 duplicates in 100,000)
    assert duplicate_rate < 0.01, (
        f"Too many duplicates at scale: {duplicates} out of {num_ids} ({duplicate_rate:.4f}%)"
    )
    
    # Verify all IDs still comply with EXACT GCP requirements from Vertex AI
    gcp_exact_pattern = re.compile(r'^[a-z][-a-z0-9]{0,127}$')
    sample_size = min(1000, len(unique_ids))
    sample_ids = list(unique_ids)[:sample_size]
    
    for job_id in sample_ids:
        assert gcp_exact_pattern.match(job_id), (
            f"ID '{job_id}' fails GCP Vertex AI compliance. "
            f"Must match pattern '[a-z][-a-z0-9]{{0,127}}'"
        )


def test_generate_job_id_character_distribution():
    """
    Test that the character distribution in generated IDs is reasonably uniform
    """
    num_ids = 10000
    all_chars = []
    
    for _ in range(num_ids):
        job_id = generate_job_id()
        all_chars.extend(list(job_id))
    
    # Count character frequencies
    char_counts = Counter(all_chars)
    total_chars = len(all_chars)
    
    # Calculate expected frequency for uniform distribution
    unique_chars = len(char_counts)
    expected_freq = total_chars / unique_chars
    
    # Check that no character is over or underrepresented by more than 50%
    for char, count in char_counts.items():
        ratio = count / expected_freq
        assert 0.5 < ratio < 1.5, (
            f"Character '{char}' appears {count} times (expected ~{expected_freq:.0f}), "
            f"ratio: {ratio:.2f}"
        )
    
    print(f"Character distribution test passed with {unique_chars} unique characters")
    print(f"Most common: {char_counts.most_common(5)}")
    print(f"Least common: {char_counts.most_common()[-5:]}")
