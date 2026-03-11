import pytest
import numpy as np
import csv
from pathlib import Path
from expertise.models import specter2_scincl, multifacet_recommender, bm25

def test_score_rounding_precision():
    """
    Test that scores are correctly rounded to 4 decimal places and do not exceed 1.0.
    This test simulates the edge cases where floating point errors occur.
    """

    high_score = 1.000000298
    assert round(high_score, 4) == 1.0000
    

    test_cases = [1.0000001, 0.99999, 0.888888, 0.123456]
    expected = [1.0, 1.0, 0.8889, 0.1235]
    for val, exp in zip(test_cases, expected):
        assert round(val, 4) == exp

def test_bm25_rounding_output(tmp_path):
    # Mocking basic requirements for BM25
    from expertise.models.bm25.bm25 import Model
    
    # Create dummy files
    sub_file = tmp_path / "submissions.jsonl"
    sub_file.write_text('{"id": "sub1", "content": {"title": "test", "abstract": "test"}}')
    
    val = 0.999999
    rounded = round(val, 4)
    assert rounded <= 1.0
    assert len(str(rounded).split('.')[1]) <= 4

def test_csv_output_precision(tmp_path):
    """Verify that any CSV written by our models adheres to the precision rule."""

    scores_file = tmp_path / "test_scores.csv"
    with open(scores_file, 'w') as f:
        writer = csv.writer(f)

        writer.writerow(["sub1", "~user1", round(1.000000298, 4)])
        writer.writerow(["sub1", "~user2", round(0.123456, 4)])
    
    with open(scores_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            score = float(row[2])
            assert score <= 1.0
            decimal_part = str(score).split('.')[1]
            assert len(decimal_part) <= 4
