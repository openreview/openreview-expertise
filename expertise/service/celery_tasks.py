from functools import update_wrapper
import logging, json, os, shutil
from .utils import mock_client
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service.expertise import JobStatus, JobDescription
from expertise.service.server import celery_app as celery
import openreview

def update_status(job_dir, new_status, desc=None):
    """
    Updates the config of a given job to the new status
    Optionally allows manual setting of the description

    :param job_dir: The directory of a given job which contains a config file
    :type job_dir: str

    :param new_status: The new status for the job - a value from the JobStatus enumeration
    :type new_status: str
    """
    descriptions = JobDescription.VALS.value
    with open(os.path.join(job_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    config['status'] = new_status
    if desc is None:
        config['description'] = descriptions[new_status]
    else:
        config['description'] = desc
    with open(os.path.join(config['job_dir'], 'config.json'), 'w+') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

@celery.task(name='userpaper', track_started=True, bind=True, time_limit=3600 * 24)
def run_userpaper(self, config: dict, logger: logging.Logger):
    try:
        update_status(config['job_dir'], JobStatus.FETCHING_DATA)
        if config.get('token'):
            openreview_client = openreview.Client(
                token=config['token'],
                baseurl=config['baseurl']
            )
        else:
            openreview_client = mock_client()
        execute_create_dataset(openreview_client, config=config)
        update_status(config['job_dir'], JobStatus.EXPERTISE_QUEUED)
        run_expertise.apply_async(
                (config, logger),
                queue='expertise',
        )
    except Exception as exc:
        # Write error, clean up working directory and store log
        logger.error(f"Error in job: {config['job_id']}, {str(exc)}")
        update_status(config['job_dir'], JobStatus.ERROR, str(exc))

@celery.task(name='expertise', track_started=True, bind=True, time_limit=3600 * 24)
def run_expertise(self, config: dict, logger: logging.Logger):
    try:
        update_status(config['job_dir'], JobStatus.RUN_EXPERTISE)
        execute_expertise(config=config)
        update_status(config['job_dir'], JobStatus.COMPLETED)
    except Exception as exc:
        # Write error, clean up working directory and store log
        logger.error(f"Error in job: {config['job_id']}, {str(exc)}")
        update_status(config['job_dir'], JobStatus.ERROR, str(exc))

