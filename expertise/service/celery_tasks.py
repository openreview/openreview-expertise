from functools import update_wrapper
import logging, json, os, shutil, time

import redis
from .utils import JobStatus, JobDescription, JobConfig, RedisDatabase
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service.server import celery_app as celery_server
from expertise.service.server import redis_config_pool
import openreview, celery

def get_config(config):
    """
    Get the latest config from Redis
    """
    return RedisDatabase(connection_pool=redis_config_pool).load_job(config.job_id, config.user_id)

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
        # Add user friendly translation
        if 'num_samples=0' in desc:
            desc += '. Please check that there is at least 1 member of the match group with at least 1 publication on OpenReview.'
        if 'Dimension out of range' in desc:
            desc += '. Please check that you have at least 1 submission submitted and that you have run the Post Submission stage.'
        config.description = desc
    config.mdate = int(time.time() * 1000)
    redis_db = RedisDatabase(connection_pool=redis_config_pool)
    redis_db.save_job(config)

def check_revoked(config: JobConfig):
    """
    Gets the status given a config - used to check for revoked status

    :param config: JobConfig of a given job
    :type config: JobConfig

    :return True if revoked:
    """
    redis_db = RedisDatabase(connection_pool=redis_config_pool)
    new_config = redis_db.load_job(config.job_id, config.user_id)
    return new_config.status == JobStatus.REVOKED

def clean_revoked(config: JobConfig):
    """
    Deletes the data for a revoked job

    :param config: JobConfig of a given job
    :type config: JobConfig
    """
    redis_db = RedisDatabase(connection_pool=redis_config_pool)
    if os.path.isdir(config.job_dir):
        shutil.rmtree(config.job_dir)
    redis_db.remove_job(config.user_id, config.job_id)

def on_failure_userpaper(self, exc, task_id, args, kwargs, einfo):
    config, logger = get_config(args[0]), args[2]
    logger.error(f"Error in job: {config.job_id}, {str(exc)}")
    update_status(config, JobStatus.ERROR, str(exc))

def on_failure_expertise(self, exc, task_id, args, kwargs, einfo):
    config, logger = get_config(args[0]), args[1]
    logger.error(f"Error in job: {config.job_id}, {str(exc)}")
    update_status(config, JobStatus.ERROR, str(exc))

def before_userpaper_start(self, task_id, args, kwargs):
    config, logger = get_config(args[0]), args[2]
    if config.status != JobStatus.ERROR and config.status != JobStatus.REVOKED:
        logger.info(f"New status: {JobStatus.FETCHING_DATA}")
        update_status(config, JobStatus.FETCHING_DATA)

def before_expertise_start(self, task_id, args, kwargs):
    config, logger = get_config(args[0]), args[1]
    if config.status != JobStatus.ERROR and config.status != JobStatus.REVOKED:
        logger.info(f"New status: {JobStatus.RUN_EXPERTISE}")
        update_status(config, JobStatus.RUN_EXPERTISE)

def after_userpaper_return(self, status, retval, task_id, args, kwargs, einfo):
    config, logger = get_config(args[0]), args[2]
    if config.status != JobStatus.ERROR and config.status != JobStatus.REVOKED:
        logger.info(f"New status: {JobStatus.EXPERTISE_QUEUED}")
        update_status(config, JobStatus.EXPERTISE_QUEUED)

def after_expertise_return(self, status, retval, task_id, args, kwargs, einfo):
    config, logger = get_config(args[0]), args[1]
    if config.status != JobStatus.ERROR and config.status != JobStatus.REVOKED:
        logger.info(f"New status: {JobStatus.COMPLETED}")
        update_status(config, JobStatus.COMPLETED)

@celery_server.task(
    name='userpaper',
    before_start=before_userpaper_start,
    after_return=after_userpaper_return,
    on_failure=on_failure_userpaper,
    track_started=True,
    bind=True,
    time_limit=3600 * 24
)
def run_userpaper(self, config: JobConfig, token: str, logger: logging.Logger, config_json: dict):
    openreview_client = openreview.Client(
        token=token,
        baseurl=config.baseurl
    )
    openreview_client_v2 = openreview.api.OpenReviewClient(
        token=token,
        baseurl=config.baseurl_v2
    )
    logger.info('CREATING DATASET')
    execute_create_dataset(openreview_client, openreview_client_v2, config=config.to_json())
    run_expertise.apply_async(
            (config, logger, config.to_json()),
            queue='expertise',
            task_id=config.job_id + '_expertise'
    )
    logger.info('FINISHED USERPAPER')

@celery_server.task(
    name='expertise',
    before_start=before_expertise_start,
    after_return=after_expertise_return,
    on_failure=on_failure_expertise,
    track_started=True,
    bind=True,
    time_limit=3600 * 24
)
def run_expertise(self, config: JobConfig, logger: logging.Logger, config_json: dict):
    # Run if not revoked
    if not check_revoked(config):
        execute_expertise(config=config.to_json())

    # If revoked while running, or revoked at all, clean up job
    if check_revoked(config):
        logger.info(f"Deleting {config.job_dir} for {config.user_id}")
        clean_revoked(config)
