"""
Tests for build_pipeline.py - Pipeline compilation and configuration tests.
"""
import pytest
import yaml


def test_pipeline_compilation_includes_timeout(tmp_path):
    """
    End-to-end test that compiles the pipeline and verifies timeout is in the YAML.
    This replicates the pipeline construction from build_pipeline.py without uploading.
    """
    from kfp import compiler
    from kfp.dsl import pipeline, component, If, Elif, Else
    from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
    from expertise.build_pipeline import parse_config_file, CONFIG_FILE_PATH

    config = parse_config_file(CONFIG_FILE_PATH)

    # Create a minimal test component
    @component(base_image="python:3.9")
    def test_component_op(gcs_request_path: str) -> None:
        pass

    # Create custom training jobs with timeout (same as build_pipeline.py)
    small_job = create_custom_training_job_from_component(
        test_component_op,
        display_name=config['PIPELINE_NAME_SMALL'],
        machine_type=config['PIPELINE_MACHINE_SMALL'],
        accelerator_type=config['PIPELINE_GPU_SMALL'],
        accelerator_count=config['PIPELINE_GPU_COUNT_SMALL'],
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=config['PIPELINE_DISK_SIZE_SMALL'],
        timeout=config['PIPELINE_TIMEOUT_SMALL'],
    )

    medium_job = create_custom_training_job_from_component(
        test_component_op,
        display_name=config['PIPELINE_NAME_MEDIUM'],
        machine_type=config['PIPELINE_MACHINE_MEDIUM'],
        accelerator_type=config['PIPELINE_GPU_MEDIUM'],
        accelerator_count=config['PIPELINE_GPU_COUNT_MEDIUM'],
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=config['PIPELINE_DISK_SIZE_MEDIUM'],
        timeout=config['PIPELINE_TIMEOUT_MEDIUM'],
    )

    large_job = create_custom_training_job_from_component(
        test_component_op,
        display_name=config['PIPELINE_NAME_LARGE'],
        machine_type=config['PIPELINE_MACHINE_LARGE'],
        accelerator_type=config['PIPELINE_GPU_LARGE'],
        accelerator_count=config['PIPELINE_GPU_COUNT_LARGE'],
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=config['PIPELINE_DISK_SIZE_LARGE'],
        timeout=config['PIPELINE_TIMEOUT_LARGE'],
    )

    @pipeline(name='test-pipeline')
    def test_pipeline(gcs_request_path: str, machine_type: str = 'small'):
        with If(machine_type == config['SMALL_NAME']):
            small_job(project='test', location='us-central1', gcs_request_path=gcs_request_path)
        with Elif(machine_type == config['MEDIUM_NAME']):
            medium_job(project='test', location='us-central1', gcs_request_path=gcs_request_path)
        with Else():
            large_job(project='test', location='us-central1', gcs_request_path=gcs_request_path)

    # Compile the pipeline
    yaml_path = tmp_path / 'test_pipeline.yaml'
    compiler.Compiler().compile(
        pipeline_func=test_pipeline,
        package_path=str(yaml_path)
    )

    # Read and verify the YAML
    yaml_content = yaml_path.read_text()

    # Assert timeout values are present in compiled YAML
    assert config['PIPELINE_TIMEOUT_SMALL'] in yaml_content, \
        f"Timeout {config['PIPELINE_TIMEOUT_SMALL']} not found in compiled pipeline"
    assert config['PIPELINE_TIMEOUT_MEDIUM'] in yaml_content, \
        f"Timeout {config['PIPELINE_TIMEOUT_MEDIUM']} not found in compiled pipeline"
    assert config['PIPELINE_TIMEOUT_LARGE'] in yaml_content, \
        f"Timeout {config['PIPELINE_TIMEOUT_LARGE']} not found in compiled pipeline"

    # Parse YAML and verify structure
    with open(yaml_path, 'r') as f:
        pipeline_spec = yaml.safe_load(f)

    # The timeout should appear in the deploymentSpec of custom job components
    assert 'deploymentSpec' in str(pipeline_spec) or 'timeout' in yaml_content


def test_parse_config_file_timeout_values(tmp_path):
    """Test that timeout values are correctly parsed from config file."""
    from expertise.build_pipeline import parse_config_file

    # Create a temporary config file
    config_content = """
PIPELINE_TIMEOUT_SMALL = '86400s'
PIPELINE_TIMEOUT_MEDIUM = '172800s'
PIPELINE_TIMEOUT_LARGE = '259200s'
PIPELINE_NAME_SMALL = 'expertise-job-small'
PIPELINE_GPU_COUNT_SMALL = 1
"""
    config_file = tmp_path / "test_config.cfg"
    config_file.write_text(config_content)

    # Parse the config
    config = parse_config_file(str(config_file))

    # Verify timeout values are parsed as strings (not converted to numbers)
    assert config['PIPELINE_TIMEOUT_SMALL'] == '86400s'
    assert config['PIPELINE_TIMEOUT_MEDIUM'] == '172800s'
    assert config['PIPELINE_TIMEOUT_LARGE'] == '259200s'

    # Verify other values still parse correctly
    assert config['PIPELINE_NAME_SMALL'] == 'expertise-job-small'
    assert config['PIPELINE_GPU_COUNT_SMALL'] == 1
