import logging, json, os, shutil
from .utils import mock_client
from expertise.execute_expertise import *
from expertise.service.server import celery_app as celery

@celery.task(name='userpaper', track_started=True, bind=True, time_limit=3600 * 24)
def run_userpaper(self, config: dict, logger: logging.Logger):
    try:
        if 'token' in config.keys():
            openreview_client = openreview.Client(
                token=config['token'],
                baseurl=config['baseurl']
            )
        else:
            openreview_client = mock_client()
            logger.info('Creating dataset')
        execute_create_dataset(openreview_client, config=config)
        run_expertise.apply_async(
                (config, logger),
                queue='expertise',
        )
    except Exception as exc:
        working_dir = os.path.join(config['profile_dir'], str(config['job_id']))
        logger.error(f'Removing dir: {working_dir}')
        shutil.rmtree(working_dir)
        logger.error(f"Writing to: {os.path.join(config['profile_dir'], 'err.log')}")
        with open(os.path.join(config['profile_dir'], 'err.log'), 'a+') as f:
            f.write(f"{config['job_id']},{exc}\n")
        logger.error('Error: {}'.format(exc))

@celery.task(name='expertise', track_started=True, bind=True, time_limit=3600 * 24)
def run_expertise(self, config: dict, logger: logging.Logger):
    try:
        logger.info('Executing expertise')
        execute_expertise(config=config)
        logger.info('Returned from expertise')
    except Exception as exc:
        working_dir = os.path.join(config['profile_dir'], str(config['job_id']))
        logger.error(f'Removing dir: {working_dir}')
        shutil.rmtree(working_dir)
        logger.error(f"Writing to: {os.path.join(config['profile_dir'], 'err.log')}")
        with open(os.path.join(config['profile_dir'], 'err.log'), 'a+') as f:
            f.write(f"{config['job_id']},{exc}\n")
        logger.error('Error: {}'.format(exc))

