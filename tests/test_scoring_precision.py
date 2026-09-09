"""Tests for score precision.

Scores are rounded to 6 decimal places (raised from 4). Embeddings are
float32, which carries ~7 significant digits, so 6 decimals preserves
essentially all real ranking information; 4 decimals collapsed
near-equal reviewers into artificial ties that the matcher then broke
arbitrarily. All models round through the shared round_score_matrix
helper, and CSV emission must not reduce precision further.
"""
import csv
import torch

from expertise.utils.utils import (
    round_score_matrix,
    generate_sparse_scores_from_matrix,
)


def test_round_score_matrix_rounds_to_six_decimals():
    scores = torch.tensor(
        [[0.1234564, 0.8765419], [1.0000004, 0.9999996]], dtype=torch.float64
    )
    rounded = round_score_matrix(scores)
    expected = torch.tensor(
        [[0.123456, 0.876542], [1.0, 1.0]], dtype=torch.float64
    )
    assert torch.allclose(rounded, expected, atol=1e-9)


def test_round_score_matrix_preserves_fifth_decimal_differences():
    """Regression for the 4-decimal rounding: two reviewers whose scores
    differ in the 5th decimal must remain distinct after rounding."""
    scores = torch.tensor([0.876541, 0.876549], dtype=torch.float64)
    rounded = round_score_matrix(scores)
    assert rounded[0] != rounded[1]


def test_sparse_csv_preserves_six_decimals(tmp_path):
    """The sparse CSV writer must emit the matrix values without reducing
    them below 6 decimals."""
    matrix = torch.tensor([[0.876541, 0.876549]], dtype=torch.float64)
    scores_path = tmp_path / "scores_sparse.csv"
    generate_sparse_scores_from_matrix(
        matrix, ["sub1"], ["~reviewer1", "~reviewer2"], 2, str(scores_path)
    )
    with open(scores_path) as f:
        rows = {row[1]: float(row[2]) for row in csv.reader(f)}
    assert rows["~reviewer1"] == 0.876541
    assert rows["~reviewer2"] == 0.876549
