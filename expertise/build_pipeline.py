# pip install kfp google-cloud-pipeline-components
from kfp import compiler
from kfp.dsl import (
    pipeline,
    component,
    container_component,
    InputPath,
    OutputPath,
    ContainerSpec
)
from kfp.registry import RegistryClient
import argparse
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component
)

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

    # New component to write a string (our JSON config) to a file
    @component(base_image='python:3.9-slim')
    def write_string_to_file_op(
        data: str,
        output_file: OutputPath(str)
    ):
        """Writes string data to a file."""
        with open(output_file, 'w') as f:
            f.write(data)

    @component(
        base_image=f"{args.region}-docker.pkg.dev/{args.project}/{args.repo}/{args.image}:{args.tag}"
    )
    def execute_expertise_pipeline_op(
        job_config_file: str
    ) -> None:
        from expertise.execute_pipeline import run_pipeline
        run_pipeline(job_config_file)

    custom_expertise_job_from_file_input = create_custom_training_job_from_component(
        execute_expertise_pipeline_op,
        display_name="expertise-job",
        machine_type="n1-highmem-64",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=4,
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=200,
    )

    @pipeline(
        name=args.kfp_name,
        description='Processes request for user-paper expertise scores using file-based config'
    )
    def expertise_pipeline(
        job_config: str
    ):
        # Write the raw JSON string to a file
        write_config_task = write_string_to_file_op(
            data=job_config
        ).set_display_name("Write Job Config to File")

        # Pass the path to this file to the custom job
        run_expertise_task = custom_expertise_job_from_file_input(
            project=args.project,
            location=args.kfp_region,
            job_config_file=write_config_task.outputs['output_file'] # Pass the output file path
        ).set_display_name("Running Expertise Pipeline")

    compiler.Compiler().compile(
        pipeline_func=expertise_pipeline,
        package_path='expertise_pipeline.yaml'
    )

    client = RegistryClient(host=f"https://{args.kfp_region}-kfp.pkg.dev/{args.project}/{args.kfp_repo}")

    try:
        client.delete_tag(
            package_name=args.kfp_name,
            tag='latest'
        )
        print(f"Successfully deleted tag 'latest' for pipeline '{args.kfp_name}'.")
    except Exception as e:
        print(f"Could not delete tag 'latest' for pipeline '{args.kfp_name}' (it might not exist): {e}")

    upload_tags = [args.tag]
    if args.tag != 'latest':
        upload_tags.append('latest')

    print(f"Uploading pipeline '{args.kfp_name}' with tags: {upload_tags}")

    templateName, versionName = client.upload_pipeline(
        file_name="expertise_pipeline.yaml",
        tags=upload_tags,
        extra_headers={"description": args.kfp_description}
    )
    print(f"Pipeline uploaded: templateName='{templateName}', versionName='{versionName}'")