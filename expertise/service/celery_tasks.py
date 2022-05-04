from functools import update_wrapper
import logging, json, os, shutil, time

import redis
from .utils import mock_client, JobStatus, JobDescription, JobConfig, RedisDatabase
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service.server import celery_app as celery_server
from expertise.service.server import redis_config_pool
import openreview, celery

def update_status(config, new_status, redis_args, desc=None):
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
    redis_db = RedisDatabase(connection_pool=redis_config_pool)
    redis_db.save_job(config)

def on_failure_userpaper(self, exc, task_id, args, kwargs, einfo):
    config, logger = args[0], args[2]
    logger.error(f"Error in job: {config.job_id}, {str(exc)}")
    update_status(config, JobStatus.ERROR, str(exc))

def on_failure_expertise(self, exc, task_id, args, kwargs, einfo):
    config, logger = args[0], args[1]
    logger.error(f"Error in job: {config.job_id}, {str(exc)}")
    update_status(config, JobStatus.ERROR, str(exc))

def after_userpaper_return(self, status, retval, task_id, args, kwargs, einfo):
    config, logger = args[0], args[2]
    if config.status != JobStatus.ERROR:
        logger.info(f"New status: {JobStatus.RUN_EXPERTISE}")
        update_status(args[0], JobStatus.RUN_EXPERTISE)

def after_expertise_return(self, status, retval, task_id, args, kwargs, einfo):
    config, logger = args[0], args[1]
    if config.status != JobStatus.ERROR:
        logger.info(f"New status: {JobStatus.COMPLETED}")
        update_status(args[0], JobStatus.COMPLETED)

@celery_server.task(
    name='userpaper',
    after_return=after_userpaper_return,
    on_failure=on_failure_userpaper,
    track_started=True,
    bind=True,
    time_limit=3600 * 24
)
def run_userpaper(self, config: JobConfig, token: str, logger: logging.Logger):
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
    logger.info('CREATING DATASET')
    execute_create_dataset(openreview_client, openreview_client_v2, config=config.to_json())
    run_expertise.apply_async(
            (config, logger),
            queue='expertise',
    )
    logger.info('FINISHED USERPAPER')

@celery_server.task(
    name='expertise',
    after_return=after_expertise_return,
    on_failure=on_failure_expertise,
    track_started=True,
    bind=True,
    time_limit=3600 * 24
)
def run_expertise(self, config: dict, logger: logging.Logger):
    execute_expertise(config=config.to_json())

