import argparse
import os
import openreview
import shortuuid
import json
from execute_expertise import execute_create_dataset, execute_expertise
from expertise.config import ModelConfig
from expertise.service import load_model_artifacts
from expertise.service.utils import APIRequest, JobConfig

DEFAULT_CONFIG = {
    "dataset": {},
    "model": "specter+mfr",
    "model_params": {
        "use_title": True,
        "sparse_value": 600,
        "specter_batch_size": 16,
        "mfr_batch_size": 50,
        "use_abstract": True,
        "average_score": False,
        "max_score": True,
        "skip_specter": False,
        "use_cuda": True,
        "use_redis": False
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('api_request_str', help='a JSON file containing all other arguments')
    args = parser.parse_args()
    raw_request: dict = json.loads(args.api_request)

    # Pop token, base URLs and other expected variables
    token = raw_request.pop('token')
    baseurl_v1 = raw_request.pop('baseurl_v1')
    baseurl_v2 = raw_request.pop('baseurl_v2')
    specter_dir = os.getenv('SPECTER_DIR')
    mfr_vocab_dir = os.getenv('MFR_VOCAB_DIR')
    mfr_checkpoint_dir = os.getenv('MFR_CHECKPOINT_DIR')
    server_config ={
        'OPENREVIEW_BASEURL': baseurl_v1,
        'OPENREVIEW_BASEURL_V2': baseurl_v2,
        'SPECTER_DIR': specter_dir,
        'MFR_VOCAB_DIR': mfr_vocab_dir,
        'MFR_CHECKPOINT_DIR': mfr_checkpoint_dir,
    }
    load_model_artifacts()

    client_v1 = openreview.Client(baseurl=baseurl_v1, token=token)
    client_v2 = openreview.api.OpenReviewClient(baseurl_v2, token=token)

    job_id = shortuuid.ShortUUID().random(length=5)
    working_dir = f"/app/{job_id}"
    os.makedirs(working_dir, exist_ok=True)  

    validated_request = APIRequest(raw_request)
    config = JobConfig.from_request(
        api_request = validated_request,
        starting_config = DEFAULT_CONFIG,
        openreview_client= client_v1,
        openreview_client_v2= client_v2,
        server_config = server_config,
        working_dir = working_dir
    )

    # Create Dataset and Execute Expertise
    execute_create_dataset(client_v1, client_v2, config.to_json())
    execute_expertise(config.to_json())
