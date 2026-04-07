import os
import pytest
import re
import time
from collections import Counter
from expertise.utils.utils import generate_job_id, JOB_ID_ALPHABET
import expertise.service


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
def temp_env_cfg(tmp_path):
    """
    Writes a temporary <env>.cfg file into the service config directory and
    cleans it up after the test. Yields (env_name, sentinel_value).
    """
    config_dir = os.path.join(
        os.path.dirname(expertise.service.__file__), 'config'
    )
    env_name = 'pytest_env'
    sentinel = 'sentinel-log-file.log'
    cfg_path = os.path.join(config_dir, f'{env_name}.cfg')
    with open(cfg_path, 'w') as f:
        f.write(f"LOG_FILE='{sentinel}'\n")
    try:
        yield env_name, sentinel
    finally:
        if os.path.exists(cfg_path):
            os.remove(cfg_path)


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


def test_create_app_defaults_to_production(monkeypatch):
    """
    When EXPERTISE_ENV is not set, it should default to 'production'.
    """
    monkeypatch.delenv('EXPERTISE_ENV', raising=False)

    app = expertise.service.create_app()

    assert app.config['EXPERTISE_ENV'] == 'production'


def test_create_app_config_dict_overrides_env(monkeypatch):
    """
    Callers should be able to override EXPERTISE_ENV via the config dict
    passed to create_app().
    """
    monkeypatch.setenv('EXPERTISE_ENV', 'production')

    app = expertise.service.create_app(config={'EXPERTISE_ENV': 'override'})

    assert app.config['EXPERTISE_ENV'] == 'override'