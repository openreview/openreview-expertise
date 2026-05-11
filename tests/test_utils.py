import os
import pytest
import re
import json
import time
from collections import Counter
from expertise.utils.utils import generate_job_id, JOB_ID_ALPHABET, generate_sparse_scores
import expertise.service
from expertise.service import artifacts_for_model


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


@pytest.fixture
def temp_env_cfg(tmp_path, monkeypatch):
    """
    Writes default.cfg and a <env>.cfg into an isolated temp directory and
    monkeypatches Flask so create_app() uses it as the instance path. This
    avoids mutating the package's source tree during tests.

    Yields (env_name, sentinel_log_path).
    """
    env_name = 'pytest_env'
    sentinel_log = tmp_path / 'sentinel-log-file.log'

    # Copy the real default.cfg into the temp config dir so create_app()'s
    # from_pyfile('default.cfg') succeeds.
    real_config_dir = os.path.join(
        os.path.dirname(expertise.service.__file__), 'config'
    )
    with open(os.path.join(real_config_dir, 'default.cfg')) as f:
        default_cfg = f.read()
    (tmp_path / 'default.cfg').write_text(default_cfg)
    (tmp_path / f'{env_name}.cfg').write_text(f"LOG_FILE='{sentinel_log}'\n")

    # Wrap flask.Flask so create_app()'s instance_path points at tmp_path.
    original_flask = expertise.service.flask.Flask

    def _flask_with_tmp_instance(*args, **kwargs):
        kwargs['instance_path'] = str(tmp_path)
        return original_flask(*args, **kwargs)

    monkeypatch.setattr(expertise.service.flask, 'Flask', _flask_with_tmp_instance)

    yield env_name, str(sentinel_log)


def test_create_app_loads_env_cfg(monkeypatch, temp_env_cfg):
    """
    create_app() should load <EXPERTISE_ENV>.cfg on top of default.cfg and
    expose EXPERTISE_ENV via app.config.
    """
    env_name, sentinel = temp_env_cfg
    monkeypatch.setenv('EXPERTISE_ENV', env_name)

    app = expertise.service.create_app()

    assert app.config['EXPERTISE_ENV'] == env_name
    assert app.config['LOG_FILE'] == sentinel


def test_create_app_defaults_to_production(monkeypatch, tmp_path, temp_env_cfg):
    """
    When EXPERTISE_ENV is not set, it should default to 'production'.
    Uses temp_env_cfg to isolate instance_path and passes LOG_FILE so the
    logger doesn't write into the repo/CI workspace.
    """
    monkeypatch.delenv('EXPERTISE_ENV', raising=False)
    log_file = str(tmp_path / 'default.log')

    app = expertise.service.create_app(config={'LOG_FILE': log_file})

    assert app.config['EXPERTISE_ENV'] == 'production'


def test_create_app_config_dict_overrides_env(monkeypatch, tmp_path, temp_env_cfg):
    """
    Callers should be able to override EXPERTISE_ENV via the config dict
    passed to create_app(). Uses temp_env_cfg to isolate instance_path and
    passes LOG_FILE to keep the test hermetic.
    """
    monkeypatch.setenv('EXPERTISE_ENV', 'production')
    log_file = str(tmp_path / 'override.log')

    app = expertise.service.create_app(
        config={'EXPERTISE_ENV': 'override', 'LOG_FILE': log_file}
    )

    assert app.config['EXPERTISE_ENV'] == 'override'


@pytest.mark.parametrize("model, expected_subdirs", [
    ('bm25', []),
    ('specter', ['hf_models/specter']),
    ('mfr', ['multifacet_recommender']),
    ('specter+mfr', ['hf_models/specter', 'multifacet_recommender']),
    ('specter2', ['hf_models/specter2_base', 'hf_models/specter2_adapter']),
    ('scincl', ['hf_models/scincl']),
    ('specter2+scincl', [
        'hf_models/specter2_base',
        'hf_models/specter2_adapter',
        'hf_models/scincl',
    ]),
])
def test_artifacts_for_model_returns_only_needed_subdirs(model, expected_subdirs):
    """Pipeline workers download only the artifacts their model needs — this
    map is the contract wiring that selective-download into run_pipeline."""
    assert artifacts_for_model(model) == expected_subdirs


def test_artifacts_for_model_unknown_model_falls_back_to_all():
    """Unknown model names fall back to downloading everything (None) so a new
    model added to the API doesn't break the pipeline before the map is updated."""
    assert artifacts_for_model('some-unreleased-model') is None
    assert artifacts_for_model(None) is None


# ---------------------------------------------------------------------------
# extract_venue_key / parse_cloud_id_timestamp — venue-scoped embedding cache.
# These are pure-logic helpers used to discover prior-job artifacts on GCS.
# ---------------------------------------------------------------------------
from expertise.service.utils import extract_venue_key, parse_cloud_id_timestamp


@pytest.mark.parametrize("request_dict, expected", [
    # entityA Group / memberOf — strip role suffix
    ({'entityA': {'type': 'Group', 'memberOf': 'ICLR.cc/2026/Conference/Reviewers'},
      'entityB': {'type': 'Note', 'invitation': 'ICLR.cc/2026/Conference/-/Submission'}},
     'ICLR.cc/2026/Conference'),
    # entityA Note / invitation — split on /-/
    ({'entityA': {'type': 'Note', 'invitation': 'TMLR/-/Submission'},
      'entityB': {'type': 'Note', 'invitation': 'TMLR/-/Submission'}},
     'TMLR'),
    # entityA has neither memberOf nor invitation; entityB has memberOf
    ({'entityA': {'type': 'Note', 'id': 'abc'},
      'entityB': {'type': 'Group', 'memberOf': 'NeurIPS.cc/2025/Workshop/Reviewers'}},
     'NeurIPS.cc/2025/Workshop'),
    # memberOf without slashes — return as-is
    ({'entityA': {'type': 'Group', 'memberOf': 'JustAGroup'}, 'entityB': {}},
     'JustAGroup'),
    # nothing matchable
    ({'entityA': {}, 'entityB': {}}, None),
    # invitation without /-/ falls through to None
    ({'entityA': {'invitation': 'no-divider'}, 'entityB': {}}, None),
    # entityB Note / withVenueid — split jobs use this when invitation is absent.
    # Under-review submissions carry a '/Submission' suffix; strip it so the
    # key matches what memberOf-style requests produce.
    ({'entityA': {'type': 'Group', 'reviewerIds': ['~A1', '~B1']},
      'entityB': {'type': 'Note', 'withVenueid': 'ICLR.cc/2026/Conference/Submission'}},
     'ICLR.cc/2026/Conference'),
    # withVenueid alongside invitation — invitation wins (checked first)
    ({'entityA': {'type': 'Note', 'invitation': 'TMLR/-/Submission', 'withVenueid': 'OTHER'},
      'entityB': {}},
     'TMLR'),
    # non-dict input
    (None, None),
    ('not-a-dict', None),
])
def test_extract_venue_key(request_dict, expected):
    assert extract_venue_key(request_dict) == expected


@pytest.mark.parametrize("cloud_id, expected", [
    ('abc12345-1719500000000', 1719500000000),
    ('multi-dash-job-id-99', 99),
    ('no-suffix-letters', None),
    ('', None),
    (None, None),
])
def test_parse_cloud_id_timestamp(cloud_id, expected):
    assert parse_cloud_id_timestamp(cloud_id) == expected


# ---------------------------------------------------------------------------
# Predictor base-class helpers: cached publication embeddings
# ---------------------------------------------------------------------------
# Bypass the package __init__ which imports transformers (version conflict
# in this env prevents loading specter.py/scincl.py).
import importlib.util
_predictor_spec = importlib.util.spec_from_file_location(
    "predictor",
    os.path.join(os.path.dirname(expertise.service.__file__), "../models/specter2_scincl/predictor.py")
)
_predictor_mod = importlib.util.module_from_spec(_predictor_spec)
_predictor_spec.loader.exec_module(_predictor_mod)
Predictor = _predictor_mod.Predictor


def test_load_cached_publication_embeddings(tmp_path):
    """Predictor loads cached embeddings from adjacent cached_<basename>.jsonl."""
    base_path = tmp_path / "pub2vec_specter.jsonl"
    cache_path = tmp_path / "cached_pub2vec_specter.jsonl"

    predictor = Predictor()

    # No cache file -> empty dicts
    assert predictor._load_cached_publication_embeddings(str(base_path)) == ({}, {})

    # Write cache with two valid papers and one bad json line
    cache_path.write_text(
        '{"paper_id": "p1", "embedding": [0.1, 0.2]}\n'
        'bad json line\n'
        '{"paper_id": "p2", "embedding": [0.3, 0.4]}\n'
    )
    lookup, jsonl_lines = predictor._load_cached_publication_embeddings(str(base_path))
    assert lookup == {
        "p1": [0.1, 0.2],
        "p2": [0.3, 0.4],
    }
    # jsonl_lines contains the original lines for direct writing
    assert jsonl_lines == {
        "p1": '{"paper_id": "p1", "embedding": [0.1, 0.2]}\n',
        "p2": '{"paper_id": "p2", "embedding": [0.3, 0.4]}\n',
    }


# ---------------------------------------------------------------------------
# generate_sparse_scores — output contract
#
# Master computes the sparse CSV from a list of (sub, rev, score) tuples via
# two sorts and a per-id counter. feature/improve-sorting computes it via
# torch.topk on the score matrix. Both must produce the same set of rows:
#
#   For every submission_id: top-`sparse_value` reviewers by score.
#   For every reviewer_id:   top-`sparse_value` submissions by score.
#   Output = union of those selections (deduplicated).
#
# Tests below pin that contract against the current implementation; the same
# expected sets should hold for any replacement.
# ---------------------------------------------------------------------------
import csv
from itertools import groupby


def _reference_sparse_rows(matrix, test_ids, reviewer_ids, sparse_value):
    """Reference contract: top-k per row + top-k per column, union'd. Scores
    are returned at full precision; callers apply rounding to match
    production (which rounds to 4 decimals before sparsification)."""
    n_test, n_rev, k = len(test_ids), len(reviewer_ids), sparse_value
    selected = set()
    for i in range(n_test):
        row = sorted(((matrix[i][j], j) for j in range(n_rev)), key=lambda x: x[0], reverse=True)
        for score, j in row[:k]:
            selected.add((test_ids[i], reviewer_ids[j], score))
    for j in range(n_rev):
        col = sorted(((matrix[i][j], i) for i in range(n_test)), key=lambda x: x[0], reverse=True)
        for score, i in col[:k]:
            selected.add((test_ids[i], reviewer_ids[j], score))
    return selected


def _flatten_matrix(matrix, test_ids, reviewer_ids, decimals=4):
    """Build the (sub, rev, score) tuple list that generate_sparse_scores
    consumes. Scores rounded to `decimals` to match what all_scores feeds in."""
    return [
        (t, r, round(float(matrix[i][j]), decimals))
        for i, t in enumerate(test_ids)
        for j, r in enumerate(reviewer_ids)
    ]


def _expected_rounded(matrix, test_ids, reviewer_ids, sparse_value, decimals=4):
    return {
        (t, r, round(float(s), decimals))
        for (t, r, s) in _reference_sparse_rows(matrix, test_ids, reviewer_ids, sparse_value)
    }


def _parse_sparse_csv(path):
    rows = set()
    with open(path) as f:
        for line in csv.reader(f):
            if not line:
                continue
            rows.add((line[0], line[1], float(line[2])))
    return rows


def test_sparse_top_k_per_submission_and_reviewer(tmp_path):
    """Both axes show up in the sparse output. Scores chosen so that the
    top-2-per-submission and top-2-per-reviewer selections partially
    overlap — the union exercises both halves of the contract."""
    matrix = [
        [0.90, 0.10, 0.50, 0.60],
        [0.20, 0.80, 0.30, 0.40],
        [0.70, 0.50, 0.20, 0.15],
    ]
    test_ids = ["subA", "subB", "subC"]
    reviewer_ids = ["R1", "R2", "R3", "R4"]
    out = tmp_path / "sparse.csv"
    generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 2, out)
    assert _parse_sparse_csv(out) == _expected_rounded(matrix, test_ids, reviewer_ids, 2)


def test_sparse_value_one(tmp_path):
    """sparse_value=1: only the per-axis argmax survives on each side."""
    matrix = [
        [0.9, 0.1, 0.5],
        [0.2, 0.8, 0.3],
    ]
    test_ids = ["subA", "subB"]
    reviewer_ids = ["R1", "R2", "R3"]
    out = tmp_path / "sparse.csv"
    generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 1, out)
    assert _parse_sparse_csv(out) == _expected_rounded(matrix, test_ids, reviewer_ids, 1)


def test_sparse_value_larger_than_groups_keeps_everything(tmp_path):
    """When sparse_value >= num_submissions and >= num_reviewers, the result
    is the full score set — no-op sparsification."""
    matrix = [
        [0.9, 0.5],
        [0.2, 0.8],
        [0.4, 0.6],
    ]
    test_ids = ["subA", "subB", "subC"]
    reviewer_ids = ["R1", "R2"]
    out = tmp_path / "sparse.csv"
    generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 100, out)
    expected_full = {
        (t, r, round(float(matrix[i][j]), 4))
        for i, t in enumerate(test_ids)
        for j, r in enumerate(reviewer_ids)
    }
    assert _parse_sparse_csv(out) == expected_full


def test_sparse_single_pair(tmp_path):
    out = tmp_path / "sparse.csv"
    generate_sparse_scores([("subA", "R1", 0.42)], 1, out)
    assert _parse_sparse_csv(out) == {("subA", "R1", 0.42)}


def test_sparse_output_grouped_by_submission_with_scores_descending(tmp_path):
    """Downstream JSONL streaming relies on the CSV being grouped by
    submission_id with scores descending within each group (the final sort
    in generate_sparse_scores is on (sub_id, score) reverse=True)."""
    matrix = [
        [0.90, 0.50, 0.70],
        [0.20, 0.80, 0.30],
    ]
    test_ids = ["subA", "subB"]
    reviewer_ids = ["R1", "R2", "R3"]
    out = tmp_path / "sparse.csv"
    generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 3, out)
    with open(out) as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    seen_groups = [sid for sid, _ in groupby(rows, key=lambda r: r[0])]
    assert seen_groups == sorted(seen_groups, reverse=True), (
        f"Submission groups not contiguous/descending: {seen_groups}"
    )
    for sid, group in groupby(rows, key=lambda r: r[0]):
        scores = [float(r[2]) for r in group]
        assert scores == sorted(scores, reverse=True), (
            f"Scores within group {sid} not descending: {scores}"
        )


def test_sparse_return_value_matches_csv(tmp_path):
    """The function's return value must contain the same rows it writes to
    disk — callers use the return in-memory and the CSV for upload."""
    matrix = [
        [0.9, 0.4, 0.6],
        [0.1, 0.8, 0.3],
        [0.7, 0.2, 0.5],
    ]
    test_ids = ["subA", "subB", "subC"]
    reviewer_ids = ["R1", "R2", "R3"]
    out = tmp_path / "sparse.csv"
    returned = generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 2, out)
    assert {(t, r, float(s)) for (t, r, s) in returned} == _parse_sparse_csv(out)


def test_sparse_every_submission_and_reviewer_appears(tmp_path):
    """Sparsification trims rows, never entire ids — every submission_id and
    reviewer_id from the input must appear at least once. test_spectermfr
    already asserts this end-to-end; this covers arbitrary synthetic input."""
    matrix = [
        [0.9, 0.1, 0.5, 0.3],
        [0.2, 0.8, 0.4, 0.7],
        [0.6, 0.3, 0.2, 0.5],
    ]
    test_ids = ["subA", "subB", "subC"]
    reviewer_ids = ["R1", "R2", "R3", "R4"]
    out = tmp_path / "sparse.csv"
    generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 1, out)
    rows = _parse_sparse_csv(out)
    assert {t for (t, _, _) in rows} == set(test_ids)
    assert {r for (_, r, _) in rows} == set(reviewer_ids)


def test_sparse_matches_reference_at_several_k(tmp_path):
    """Larger dense matrix at several sparse_value settings — exercises the
    typical production shape (every submission scored against every
    reviewer)."""
    import random
    rng = random.Random(0xC0FFEE)
    test_ids = [f"sub{i}" for i in range(6)]
    reviewer_ids = [f"~Rev{j}" for j in range(5)]
    matrix = [[rng.random() for _ in reviewer_ids] for _ in test_ids]
    for k in (1, 2, 3, 5, 10):
        out = tmp_path / f"sparse_k{k}.csv"
        generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), k, out)
        assert _parse_sparse_csv(out) == _expected_rounded(
            matrix, test_ids, reviewer_ids, k
        ), f"Mismatch at sparse_value={k}"


def test_sparse_ties_preserve_required_rows(tmp_path):
    """When several scores tie at the cutoff, any tie break is valid — but
    the contract still requires `sparse_value` rows from a fully-tied
    submission, and deterministic top-N rows from a non-tied submission."""
    matrix = [
        [0.5, 0.5, 0.5, 0.5],  # subA: all reviewers tied
        [0.9, 0.1, 0.2, 0.3],  # subB: clear ordering
    ]
    test_ids = ["subA", "subB"]
    reviewer_ids = ["R1", "R2", "R3", "R4"]
    out = tmp_path / "sparse.csv"
    generate_sparse_scores(_flatten_matrix(matrix, test_ids, reviewer_ids), 2, out)
    rows = _parse_sparse_csv(out)
    sub_a_rows = [r for (t, r, _) in rows if t == "subA"]
    assert len(sub_a_rows) >= 2
    sub_b_set = {(r, s) for (t, r, s) in rows if t == "subB"}
    assert ("R1", 0.9) in sub_b_set
    # R4's top submission is subB (0.3 > 0.0 vs subA's tied 0.5 — but axis 1
    # selects R4's top-2 submissions by score; subA(0.5) and subB(0.3) survive).
    assert ("R4", 0.3) in sub_b_set