# pip install kfp google-cloud-pipeline-components
from kfp import compiler
from kfp.dsl import (
    pipeline,
    component,
    container_component,
    InputPath,
    OutputPath,
    ContainerSpec,
)
from kfp.registry import RegistryClient
import argparse
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component
)
import os

# Make config path relative to this script's directory
_BUILD_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(_BUILD_DIR, 'service', 'config', 'default.cfg')

def parse_config_file(config_path):
    def _coerce_numeric(value):
        """Try to parse value as int, then float, fallback to string."""
        for parser in (int, float):
            try:
                return parser(value)
            except ValueError:
                pass
        return value
    """
    Parse a configuration file line-by-line.
    
    Reads lines containing '=' and splits them into key-value pairs.
    Values are parsed as strings (if quoted), integers, or floats.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary with parsed values
    """
    config = {}
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Parse value based on type
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            # Remove quotes and store as string
                            config[key] = value[1:-1]
                        else:
                            # Try to parse as numeric, fallback to string
                            config[key] = _coerce_numeric(value)
                                
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except Exception as e:
        print(f"Error parsing config file: {e}")
        
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Builds and Uploads a Kubeflow Pipeline for the Expertise Model")
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region for Docker Images in Artifact Registry"
    )
    parser.add_argument(
        "--kfp_region",
        type=str,
        required=True,
        help="Region Kubeflow Pipelines in Artifact Registry"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="GCP Project ID"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Name of the Artifact Registry Docker Repository"
    )
    parser.add_argument(
        "--kfp_repo",
        type=str,
        required=True,
        help="Name of the Artifact Registry Kubeflow Repository"
    )
    parser.add_argument(
        "--kfp_name",
        type=str,
        required=True,
        help="Name of the Kubeflow Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Name of the Docker Image"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=False,
        default='latest',
        help="Tag of the Docker Image"
    )
    parser.add_argument(
        "--kfp_description",
        type=str,
        required=False,
        default="Latest Kubeflow Pipeline for OpenReview Expertise",
        help="Description of the latest Kubeflow Pipeline"
    )
    args = parser.parse_args()
    config = parse_config_file(CONFIG_FILE_PATH)

    @component(
        base_image=f"{args.region}-docker.pkg.dev/{args.project}/{args.repo}/{args.image}:{args.tag}"
    )
    def execute_expertise_pipeline_op(
        gcs_request_path: str,
        dataset_gcs_path: str = ''
    ) -> None:
        from expertise.execute_pipeline import run_pipeline
        run_pipeline(
            gcs_dir=gcs_request_path,
            dataset_gcs_path=dataset_gcs_path if dataset_gcs_path else None
        )

    small_expertise_job_from_file_input = create_custom_training_job_from_component(
        execute_expertise_pipeline_op,
        display_name=config['PIPELINE_NAME_SMALL'],
        machine_type=config['PIPELINE_MACHINE_SMALL'],
        accelerator_type=config['PIPELINE_GPU_SMALL'],
        accelerator_count=config['PIPELINE_GPU_COUNT_SMALL'],
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=config['PIPELINE_DISK_SIZE_SMALL'],
    )

    medium_expertise_job_from_file_input = create_custom_training_job_from_component(
        execute_expertise_pipeline_op,
        display_name=config['PIPELINE_NAME_MEDIUM'],
        machine_type=config['PIPELINE_MACHINE_MEDIUM'],
        accelerator_type=config['PIPELINE_GPU_MEDIUM'],
        accelerator_count=config['PIPELINE_GPU_COUNT_MEDIUM'],
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=config['PIPELINE_DISK_SIZE_MEDIUM'],
    )

    large_expertise_job_from_file_input = create_custom_training_job_from_component(
        execute_expertise_pipeline_op,
        display_name=config['PIPELINE_NAME_LARGE'],
        machine_type=config['PIPELINE_MACHINE_LARGE'],
        accelerator_type=config['PIPELINE_GPU_LARGE'],
        accelerator_count=config['PIPELINE_GPU_COUNT_LARGE'],
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=config['PIPELINE_DISK_SIZE_LARGE'],
    )

    # Three separate pipelines — one per tier. Machine type is pre-computed by the
    # BullMQ worker, so no conditional branching is needed inside any pipeline.

    @pipeline(
        name=f"{args.kfp_name}-{config['SMALL_NAME']}",
        description='Expertise pipeline for small jobs'
    )
    def small_expertise_pipeline(
        gcs_request_path: str,
        dataset_gcs_path: str = ''
    ):
        small_expertise_job_from_file_input(
            project=args.project,
            location=args.kfp_region,
            gcs_request_path=gcs_request_path,
            dataset_gcs_path=dataset_gcs_path
        ).set_display_name("Running Small Expertise Pipeline")

    @pipeline(
        name=f"{args.kfp_name}-{config['MEDIUM_NAME']}",
        description='Expertise pipeline for medium jobs'
    )
    def medium_expertise_pipeline(
        gcs_request_path: str,
        dataset_gcs_path: str = ''
    ):
        medium_expertise_job_from_file_input(
            project=args.project,
            location=args.kfp_region,
            gcs_request_path=gcs_request_path,
            dataset_gcs_path=dataset_gcs_path
        ).set_display_name("Running Medium Expertise Pipeline")

    @pipeline(
        name=f"{args.kfp_name}-{config['LARGE_NAME']}",
        description='Expertise pipeline for large jobs'
    )
    def large_expertise_pipeline(
        gcs_request_path: str,
        dataset_gcs_path: str = ''
    ):
        large_expertise_job_from_file_input(
            project=args.project,
            location=args.kfp_region,
            gcs_request_path=gcs_request_path,
            dataset_gcs_path=dataset_gcs_path
        ).set_display_name("Running Large Expertise Pipeline")

    pipelines = [
        (small_expertise_pipeline,  f"{args.kfp_name}-{config['SMALL_NAME']}",  'expertise_pipeline_small.yaml'),
        (medium_expertise_pipeline, f"{args.kfp_name}-{config['MEDIUM_NAME']}", 'expertise_pipeline_medium.yaml'),
        (large_expertise_pipeline,  f"{args.kfp_name}-{config['LARGE_NAME']}",  'expertise_pipeline_large.yaml'),
    ]

    registry_client = RegistryClient(host=f"https://{args.kfp_region}-kfp.pkg.dev/{args.project}/{args.kfp_repo}")

    for pipeline_func, pipeline_name, package_path in pipelines:
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=package_path)

        version_exists = False
        try:
            existing_tag = registry_client.get_tag(package_name=pipeline_name, tag=args.tag)
            if existing_tag:
                version_exists = True
                print(f"Pipeline '{pipeline_name}' with tag '{args.tag}' already exists in registry")
        except Exception:
            print(f"Pipeline tag '{args.tag}' does not exist for '{pipeline_name}', uploading new version")

        if version_exists:
            print(f"Skipping upload for '{pipeline_name}'.")
            continue

        print(f"Uploading pipeline '{pipeline_name}' with tag: {args.tag}")
        templateName, versionName = registry_client.upload_pipeline(
            file_name=package_path,
            tags=[args.tag],
            extra_headers={"description": args.kfp_description}
        )
        print(f"Pipeline uploaded: templateName='{templateName}', versionName='{versionName}'")