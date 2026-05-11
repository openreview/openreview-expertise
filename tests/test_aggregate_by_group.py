"""Unit tests for aggregate_by_group covering the matrix-based path.

aggregate_by_group is the group-vs-group flow: scores are produced per
(entityB_paper, entityA_member) by the model, then re-averaged into
per-(entityB_member, entityA_member) by this function. After the matrix
refactor, the model writes {name}.pt instead of {name}.csv, so
aggregate_by_group must load the .pt file when present and aggregate in
matrix space rather than reading the (now non-existent) CSV.

This file pins down both code paths:
- matrix path (.pt exists): aggregation happens via tensor ops.
- legacy path (.csv exists, .pt does not): aggregation via the original
  nested-dict iteration.

Both paths must produce the same output for the same input.
"""
import csv as csv_mod
import json
import os
from pathlib import Path

import torch

from expertise.utils.utils import aggregate_by_group


def _setup_job_dir(tmp_path, *, archive_members, publications_by_profile_id):
    """Build the directory layout aggregate_by_group expects:
        <dataset_dir>/archives/<member>.jsonl  (entityA members)
        <dataset_dir>/publications_by_profile_id.json  (entityB members -> papers)
    Returns the dataset_dir, scores_dir, and config dict to pass in.
    """
    dataset_dir = tmp_path / 'dataset'
    archives_dir = dataset_dir / 'archives'
    archives_dir.mkdir(parents=True)
    for member in archive_members:
        (archives_dir / f'{member}.jsonl').write_text('')

    with open(dataset_dir / 'publications_by_profile_id.json', 'w') as f:
        json.dump(publications_by_profile_id, f)

    scores_dir = tmp_path / 'scores'
    scores_dir.mkdir()

    config = {
        'name': 'test_run',
        'model_params': {'scores_path': str(scores_dir)},
        'dataset': {'directory': str(dataset_dir)},
    }
    return dataset_dir, scores_dir, config


def test_aggregate_by_group_matrix_path(tmp_path):
    """When {name}.pt exists, aggregate_by_group loads it and averages each
    entityB member's paper rows -> per-(entityB_member, entityA_member) score.
    """
    # entityB members and their papers
    publications_by_profile_id = {
        '~AuthorA1': [{'id': 'paperA1'}, {'id': 'paperA2'}],
        '~AuthorB1': [{'id': 'paperB1'}],
    }
    # entityA members (reviewers, on the archives side)
    archive_members = ['~Rev1', '~Rev2']

    _, scores_dir, config = _setup_job_dir(
        tmp_path,
        archive_members=archive_members,
        publications_by_profile_id=publications_by_profile_id,
    )

    # Matrix: rows = entityB papers (test_ids), cols = entityA members (reviewer_ids)
    #          ~Rev1  ~Rev2
    # paperA1   0.8    0.2
    # paperA2   0.4    0.6
    # paperB1   0.5    0.7
    scores_matrix = torch.tensor([
        [0.8, 0.2],
        [0.4, 0.6],
        [0.5, 0.7],
    ])
    test_ids = ['paperA1', 'paperA2', 'paperB1']
    reviewer_ids = ['~Rev1', '~Rev2']
    torch.save(
        {'scores': scores_matrix, 'test_ids': test_ids, 'reviewer_ids': reviewer_ids},
        scores_dir / 'test_run.pt',
    )

    preliminary_scores = aggregate_by_group(config)

    # AuthorA papers: [paperA1, paperA2] -> rows 0, 1
    #   Rev1 avg = (0.8 + 0.4) / 2 = 0.60
    #   Rev2 avg = (0.2 + 0.6) / 2 = 0.40
    # AuthorB papers: [paperB1] -> row 2
    #   Rev1 = 0.50
    #   Rev2 = 0.70
    expected = {
        ('~Rev1', '~AuthorA1'): 0.60,
        ('~Rev2', '~AuthorA1'): 0.40,
        ('~Rev1', '~AuthorB1'): 0.50,
        ('~Rev2', '~AuthorB1'): 0.70,
    }

    actual = {(rev, sub): score for rev, sub, score in preliminary_scores}
    assert actual == expected

    # aggregate_by_group also writes the result as {name}.csv.
    csv_path = scores_dir / 'test_run.csv'
    assert csv_path.exists()
    with open(csv_path) as f:
        csv_rows = {(row[0], row[1]): float(row[2]) for row in csv_mod.reader(f) if row}
    assert csv_rows == expected


def test_aggregate_by_group_csv_path_unchanged(tmp_path):
    """Legacy path: {name}.csv exists, {name}.pt doesn't. Original nested-dict
    aggregation is used and produces equivalent output to the matrix path.
    """
    publications_by_profile_id = {
        '~AuthorA1': [{'id': 'paperA1'}, {'id': 'paperA2'}],
        '~AuthorB1': [{'id': 'paperB1'}],
    }
    archive_members = ['~Rev1', '~Rev2']

    _, scores_dir, config = _setup_job_dir(
        tmp_path,
        archive_members=archive_members,
        publications_by_profile_id=publications_by_profile_id,
    )

    # Write the legacy {name}.csv (paper_id, ac_id, score).
    csv_rows = [
        ('paperA1', '~Rev1', 0.8), ('paperA1', '~Rev2', 0.2),
        ('paperA2', '~Rev1', 0.4), ('paperA2', '~Rev2', 0.6),
        ('paperB1', '~Rev1', 0.5), ('paperB1', '~Rev2', 0.7),
    ]
    with open(scores_dir / 'test_run.csv', 'w') as f:
        w = csv_mod.writer(f)
        for r in csv_rows:
            w.writerow(r)

    preliminary_scores = aggregate_by_group(config)
    actual = {(rev, sub): score for rev, sub, score in preliminary_scores}
    expected = {
        ('~Rev1', '~AuthorA1'): 0.60,
        ('~Rev2', '~AuthorA1'): 0.40,
        ('~Rev1', '~AuthorB1'): 0.50,
        ('~Rev2', '~AuthorB1'): 0.70,
    }
    assert actual == expected


def test_aggregate_by_group_matrix_skips_authors_with_no_known_papers(tmp_path):
    """If an entityB member's paper_ids are all missing from the matrix's
    test_ids (e.g. they weren't part of the scoring run), their entry should
    be skipped — same as the legacy code which produced no per-archive entry
    when score_length was 0.
    """
    publications_by_profile_id = {
        '~AuthorA1': [{'id': 'paperA1'}],
        '~AuthorMissing1': [{'id': 'ghost_paper'}],  # not in test_ids
    }
    archive_members = ['~Rev1']

    _, scores_dir, config = _setup_job_dir(
        tmp_path,
        archive_members=archive_members,
        publications_by_profile_id=publications_by_profile_id,
    )

    scores_matrix = torch.tensor([[0.9]])
    test_ids = ['paperA1']
    reviewer_ids = ['~Rev1']
    torch.save(
        {'scores': scores_matrix, 'test_ids': test_ids, 'reviewer_ids': reviewer_ids},
        scores_dir / 'test_run.pt',
    )

    preliminary_scores = aggregate_by_group(config)
    actual = {(rev, sub): score for rev, sub, score in preliminary_scores}
    # AuthorMissing1 had no papers in test_ids -> skipped.
    assert actual == {('~Rev1', '~AuthorA1'): 0.90}


def test_aggregate_by_group_matrix_preserves_zero_scores(tmp_path):
    """The legacy code emits per-(archive, profile) rows for every reviewer
    that scored at least one of the profile's papers, regardless of the
    rounded value. The matrix path matches this: zero scores aren't dropped.
    """
    publications_by_profile_id = {
        '~AuthorA1': [{'id': 'paperA1'}],
    }
    archive_members = ['~Rev1', '~Rev2']
    _, scores_dir, config = _setup_job_dir(
        tmp_path,
        archive_members=archive_members,
        publications_by_profile_id=publications_by_profile_id,
    )

    # Rev2 score 0.001 rounds to 0.0 — kept; Rev1 score 0.9 -> 0.90.
    scores_matrix = torch.tensor([[0.9, 0.001]])
    torch.save(
        {'scores': scores_matrix, 'test_ids': ['paperA1'], 'reviewer_ids': ['~Rev1', '~Rev2']},
        scores_dir / 'test_run.pt',
    )

    preliminary_scores = aggregate_by_group(config)
    actual = {(rev, sub): score for rev, sub, score in preliminary_scores}
    assert actual == {
        ('~Rev1', '~AuthorA1'): 0.90,
        ('~Rev2', '~AuthorA1'): 0.00,
    }


def test_aggregate_by_group_matrix_and_csv_paths_produce_same_output(tmp_path):
    """Direct cross-check: run both code paths on equivalent input data and
    assert they produce identical preliminary_scores. Catches any drift
    between the legacy CSV-based aggregation and the new matrix-based one.
    """
    publications_by_profile_id = {
        '~AuthorA1': [{'id': 'paperA1'}, {'id': 'paperA2'}],
        '~AuthorB1': [{'id': 'paperB1'}],
    }
    archive_members = ['~Rev1', '~Rev2']

    # Same matrix values, written to both .pt and .csv forms under separate dirs.
    csv_rows = [
        ('paperA1', '~Rev1', 0.8), ('paperA1', '~Rev2', 0.2),
        ('paperA2', '~Rev1', 0.4), ('paperA2', '~Rev2', 0.6),
        ('paperB1', '~Rev1', 0.5), ('paperB1', '~Rev2', 0.7),
    ]
    matrix = torch.tensor([
        [0.8, 0.2],
        [0.4, 0.6],
        [0.5, 0.7],
    ])

    # Matrix path
    _, matrix_scores_dir, matrix_config = _setup_job_dir(
        tmp_path / 'matrix',
        archive_members=archive_members,
        publications_by_profile_id=publications_by_profile_id,
    )
    torch.save(
        {'scores': matrix, 'test_ids': ['paperA1', 'paperA2', 'paperB1'], 'reviewer_ids': ['~Rev1', '~Rev2']},
        matrix_scores_dir / 'test_run.pt',
    )
    matrix_result = aggregate_by_group(matrix_config)

    # CSV path
    _, csv_scores_dir, csv_config = _setup_job_dir(
        tmp_path / 'csv',
        archive_members=archive_members,
        publications_by_profile_id=publications_by_profile_id,
    )
    with open(csv_scores_dir / 'test_run.csv', 'w') as f:
        w = csv_mod.writer(f)
        for r in csv_rows:
            w.writerow(r)
    csv_result = aggregate_by_group(csv_config)

    matrix_dict = {(rev, sub): score for rev, sub, score in matrix_result}
    csv_dict = {(rev, sub): score for rev, sub, score in csv_result}
    assert matrix_dict == csv_dict
