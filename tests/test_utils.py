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