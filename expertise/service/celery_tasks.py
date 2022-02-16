from functools import update_wrapper
import logging, json, os, shutil, time
from .utils import mock_client, JobStatus, JobDescription, JobConfig
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service.server import celery_app as celery
import openreview

def update_status(config, new_status, desc=None):
    """
    Updates the config of a given job to the new status
    Optionally allows manual setting of the description

    :param config: JobConfig of a given job
    :type config: JobConfig

    :param new_status: The new status for the job - a value from the JobStatus enumeration
    :type new_status: str
    """
    descriptions = JobDescription.VALS.value
    config.status = new_status
    if desc is None:
        config.description = descriptions[new_status]
    else:
        config.description = desc
    config.mdate = int(time.time() * 1000)
    config.save()

@celery.task(name='userpaper', track_started=True, bind=True, time_limit=3600 * 24)
def run_userpaper(self, config: JobConfig, token: str, logger: logging.Logger):
    try:
        update_status(config, JobStatus.FETCHING_DATA)
        if token:
            openreview_client = openreview.Client(
                token=token,
                baseurl=config.baseurl
            )
            openreview_client_v2 = openreview.api.OpenReviewClient(
                token=token,
                baseurl=config.baseurl_v2
            )
        else:
            openreview_client = mock_client(version=1)
            openreview_client_v2 = mock_client(version=2)
        execute_create_dataset(openreview_client, openreview_client_v2, config=config.to_json())
        update_status(config, JobStatus.EXPERTISE_QUEUED)
        run_expertise.apply_async(
                (config, logger),
                queue='expertise',
        )
    except Exception as exc:
        # Write error, clean up working directory and store log
        logger.error(f"Error in job: {config.job_id}, {str(exc)}")
        update_status(config, JobStatus.ERROR, str(exc))

@celery.task(name='expertise', track_started=True, bind=True, time_limit=3600 * 24)
def run_expertise(self, config: dict, logger: logging.Logger):
    try:
        update_status(config, JobStatus.RUN_EXPERTISE)
        execute_expertise(config=config.to_json())
        update_status(config, JobStatus.COMPLETED)
    except Exception as exc:
        # Write error, clean up working directory and store log
        logger.error(f"Error in job: {config.job_id}, {str(exc)}")
        update_status(config, JobStatus.ERROR, str(exc))

