# pip install kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import pipeline
from kfp.registry import RegistryClient
import argparse

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
    args = parser.parse_args()

    @dsl.container_component
    def execute_expertise_pipeline_op(job_config: str):
        return dsl.ContainerSpec(
            image=f'{args.region}-docker.pkg.dev/{args.project}/{args.repo}/{args.image}:{args.tag}',
            command=['python', '-m', 'expertise.execute_pipeline'],
            args=[job_config]
        )

    @pipeline(
        name=args.kfp_name,
        description='Processes request for user-paper expertise scores'
    )
    def expertise_pipeline(job_config: str):
        import os
        # Setting environment variables within the function
        os.environ["AIP_STORAGE_URI"] = "gs://openreview-expertise/expertise-utils/"
        os.environ["SPECTER_DIR"] = "/app/expertise-utils/specter/"
        os.environ["MFR_VOCAB_DIR"] = "/app/expertise-utils/multifacet_recommender/feature_vocab_file"
        os.environ["MFR_CHECKPOINT_DIR"] = "/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/"
        op = (execute_expertise_pipeline_op(job_config=job_config)
        .set_cpu_limit('4')
        .set_memory_limit('32G')
        .add_node_selector_constraint('NVIDIA_TESLA_T4')
        .set_accelerator_limit('1')
        )


    compiler.Compiler().compile(
        pipeline_func=expertise_pipeline,
        package_path='expertise_pipeline.yaml'
    )

    client = RegistryClient(host=f"https://{args.kfp_region}-kfp.pkg.dev/{args.project}/{args.kfp_repo}")
    client.delete_tag(
        args.kfp_name,
        'latest'
    )

    tags = [args.tag]
    if 'latest' not in tags:
        tags.append('latest')
    templateName, versionName = client.upload_pipeline(
        tags=tags,
        file_name="expertise_pipeline.yaml"
    )