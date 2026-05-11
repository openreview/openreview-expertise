"""Unit tests for generate_sparse_scores_from_matrix.

These tests pin down three properties separately from any model-side code,
using small synthetic score matrices:

1. Top-K selection picks the (test, reviewer) pairs with the highest scores
   along each axis, unioned across both axes.
2. Output rows are sorted (test_id desc, score desc) — same convention as the
   legacy tuple-based generate_sparse_scores. Consumers may depend on this
   ordering.
3. The new matrix-based implementation agrees with the legacy tuple-based
   generate_sparse_scores on a random matrix (with unique scores so no ties).
"""
import csv
import torch
import pytest
from expertise.utils.utils import (
    generate_sparse_scores,
    generate_sparse_scores_from_matrix,
)


def _read_csv_rows(path):
    """Read sparse CSV back as list[(test_id, reviewer_id, score)]."""
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                rows.append((row[0], row[1], float(row[2])))
    return rows


def _legacy_sparse_pairs(matrix, test_ids, reviewer_ids, sparse_value, scores_path):
    """Run legacy tuple-based generate_sparse_scores on the equivalent
    flattened (test, reviewer, score) list. Returns the set of (test_id,
    reviewer_id) pairs.

    The legacy function mutates the input list and uses `round` on the way out
    to the CSV; we mirror that by re-reading from the CSV.
    """
    scores = matrix.tolist()
    flat = []
    for i, t in enumerate(test_ids):
        for j, r in enumerate(reviewer_ids):
            flat.append((t, r, round(scores[i][j], 4)))
    generate_sparse_scores(flat, sparse_value, scores_path)
    return {(row[0], row[1]) for row in _read_csv_rows(scores_path)}


def test_top_k_selection(tmp_path):
    """Top-1 per row + top-1 per column on a 3x3 matrix. Verify the
    union of (max-per-row, max-per-column) pairs is emitted. Crafted so
    row-max and column-max pick different (test, reviewer) pairs."""
    matrix = torch.tensor([
        [0.9, 0.1, 0.2],   # row max: r0
        [0.8, 0.4, 0.3],   # row max: r0
        [0.7, 0.6, 0.5],   # row max: r0
    ])
    test_ids = ['t0', 't1', 't2']
    reviewer_ids = ['r0', 'r1', 'r2']

    out_path = tmp_path / 'sparse.csv'
    generate_sparse_scores_from_matrix(matrix, test_ids, reviewer_ids, sparse_value=1, scores_path=out_path)
    rows = _read_csv_rows(out_path)
    pairs = {(t, r) for t, r, _ in rows}

    # Top-1 per row picks r0 for every test paper:
    #   (t0,r0), (t1,r0), (t2,r0)
    # Top-1 per col picks the row-max for each column:
    #   r0->t0 (0.9), r1->t2 (0.6), r2->t2 (0.5)
    #   -> (t0,r0), (t2,r1), (t2,r2)
    expected = {('t0', 'r0'), ('t1', 'r0'), ('t2', 'r0'), ('t2', 'r1'), ('t2', 'r2')}
    assert pairs == expected


def test_output_ordering_matches_legacy(tmp_path):
    """Output is sorted (test_id desc, score desc) to match legacy output."""
    matrix = torch.tensor([
        [0.10, 0.20, 0.30],
        [0.40, 0.50, 0.60],
        [0.70, 0.80, 0.90],
    ])
    test_ids = ['aaa', 'mmm', 'zzz']           # alphabetical
    reviewer_ids = ['r0', 'r1', 'r2']

    out_path = tmp_path / 'sparse.csv'
    # sparse_value=3 -> all 9 pairs included for this 3x3 matrix.
    generate_sparse_scores_from_matrix(matrix, test_ids, reviewer_ids, sparse_value=3, scores_path=out_path)
    rows = _read_csv_rows(out_path)

    assert len(rows) == 9

    # Legacy convention: sort by (test_id desc, score desc).
    # → 'zzz' rows first (then 'mmm', then 'aaa'); within each test_id,
    # scores descending.
    test_id_order = [r[0] for r in rows]
    assert test_id_order == ['zzz', 'zzz', 'zzz', 'mmm', 'mmm', 'mmm', 'aaa', 'aaa', 'aaa']

    # Within each test_id, scores must be in descending order.
    for tid in ['zzz', 'mmm', 'aaa']:
        block = [s for t, _, s in rows if t == tid]
        assert block == sorted(block, reverse=True), \
            f"Scores within test_id={tid!r} not descending: {block}"


def test_score_values_correct(tmp_path):
    """Each emitted row carries the score from the matrix at that (test, reviewer)
    cell (rounded to 4 decimals)."""
    matrix = torch.tensor([
        [0.1234, 0.5678, 0.9876],
        [0.4321, 0.8765, 0.2109],
    ])
    test_ids = ['s0', 's1']
    reviewer_ids = ['~Reviewer_A1', '~Reviewer_B1', '~Reviewer_C1']

    out_path = tmp_path / 'sparse.csv'
    generate_sparse_scores_from_matrix(matrix, test_ids, reviewer_ids, sparse_value=2, scores_path=out_path)
    rows = _read_csv_rows(out_path)

    # Build a lookup back to the source matrix for verification.
    expected = {}
    for i, t in enumerate(test_ids):
        for j, r in enumerate(reviewer_ids):
            expected[(t, r)] = round(matrix[i, j].item(), 4)

    for test_id, reviewer_id, score in rows:
        assert (test_id, reviewer_id) in expected
        assert abs(score - expected[(test_id, reviewer_id)]) < 1e-9


def test_sparse_value_larger_than_axis(tmp_path):
    """When sparse_value > min(num_test, num_reviewers), the output is the
    full matrix (clamped internally)."""
    matrix = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4],
    ])
    test_ids = ['t0', 't1']
    reviewer_ids = ['r0', 'r1']

    out_path = tmp_path / 'sparse.csv'
    generate_sparse_scores_from_matrix(matrix, test_ids, reviewer_ids, sparse_value=100, scores_path=out_path)
    rows = _read_csv_rows(out_path)
    pairs = {(t, r) for t, r, _ in rows}

    assert pairs == {('t0', 'r0'), ('t0', 'r1'), ('t1', 'r0'), ('t1', 'r1')}


def test_equivalent_to_legacy_on_random_matrix(tmp_path):
    """On a random matrix with unique scores, the matrix-based and tuple-based
    implementations must agree on the set of pairs they emit. (Ordering is
    also asserted to match by sort key in test_output_ordering_matches_legacy.)
    """
    torch.manual_seed(0)
    # Use a wide enough range and large enough size that ties are unlikely.
    matrix = torch.rand((20, 30)) * 1000
    # Ensure uniqueness — add tiny per-cell offset to break any accidental ties.
    matrix = matrix + torch.arange(matrix.numel()).reshape(matrix.shape).float() * 1e-6
    test_ids = [f't{i:02d}' for i in range(matrix.shape[0])]
    reviewer_ids = [f'~Reviewer{j:03d}1' for j in range(matrix.shape[1])]

    new_path = tmp_path / 'new.csv'
    generate_sparse_scores_from_matrix(matrix, test_ids, reviewer_ids, sparse_value=5, scores_path=new_path)
    new_pairs = {(t, r) for t, r, _ in _read_csv_rows(new_path)}

    legacy_path = tmp_path / 'legacy.csv'
    legacy_pairs = _legacy_sparse_pairs(matrix, test_ids, reviewer_ids, sparse_value=5, scores_path=legacy_path)

    assert new_pairs == legacy_pairs, (
        f"New impl picked pairs that differ from legacy.\n"
        f"  only-in-new: {new_pairs - legacy_pairs}\n"
        f"  only-in-legacy: {legacy_pairs - new_pairs}"
    )


def test_zero_sparse_value_produces_empty_file(tmp_path):
    """sparse_value=0 emits an empty file rather than crashing."""
    matrix = torch.tensor([[0.5, 0.7]])
    out_path = tmp_path / 'sparse.csv'
    generate_sparse_scores_from_matrix(matrix, ['t0'], ['r0', 'r1'], sparse_value=0, scores_path=out_path)
    assert out_path.exists()
    assert out_path.read_text() == ''


def test_empty_reviewer_axis_produces_empty_file(tmp_path):
    """When the model produces a [num_test, 0] matrix because no reviewers
    were eligible (every train_note_id_list empty or all publications were
    in the bad_id_set), the sparse generator must emit an empty file rather
    than crash on torch.topk(k=0, dim=1)."""
    matrix = torch.empty((3, 0))
    out_path = tmp_path / 'sparse.csv'
    generate_sparse_scores_from_matrix(matrix, ['t0', 't1', 't2'], [], sparse_value=10, scores_path=out_path)
    assert out_path.exists()
    assert out_path.read_text() == ''


def test_empty_test_axis_produces_empty_file(tmp_path):
    """Symmetric: 0 test rows, some reviewers. Empty file, no crash."""
    matrix = torch.empty((0, 3))
    out_path = tmp_path / 'sparse.csv'
    generate_sparse_scores_from_matrix(matrix, [], ['r0', 'r1', 'r2'], sparse_value=10, scores_path=out_path)
    assert out_path.exists()
    assert out_path.read_text() == ''
