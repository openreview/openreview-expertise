import os
import pytest
import re
import json
import time
from collections import Counter
from expertise.utils.utils import generate_job_id, JOB_ID_ALPHABET
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