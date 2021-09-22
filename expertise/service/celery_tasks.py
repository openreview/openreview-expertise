import logging, json, os, shutil
from .utils import mock_client
from expertise.execute_expertise import *
from expertise.service.server import celery_app as celery

@celery.task(name='userpaper', track_started=True, bind=True, time_limit=3600 * 24)
def run_userpaper(self, config: dict, logger: logging.Logger):
    try:
        logger.info('Creating dataset')
        if config.get('token'):
            openreview_client = openreview.Client(
                token=config['token'],
                baseurl=config['baseurl']
            )
        else:
            openreview_client = mock_client()
        execute_create_dataset(openreview_client, config=config)
        run_expertise.apply_async(
                (config, logger),
                queue='expertise',
        )
    except Exception as exc:
        # Write error, clean up working directory and store log
        logger.error(f"Error in job: {config['job_id']}, {str(exc)}")
        working_dir = config['job_dir']
        with open(os.path.join(working_dir, 'err.log'), 'a+') as f:
            f.write(f"{config['job_id']},{config['name']},{exc}\n")

@celery.task(name='expertise', track_started=True, bind=True, time_limit=3600 * 24)
def run_expertise(self, config: dict, logger: logging.Logger):
    try:
        logger.info('Executing expertise')
        execute_expertise(config=config)
    except Exception as exc:
        # Write error, clean up working directory and store log
        logger.error(f"Error in job: {config['job_id']}, {str(exc)}")
        working_dir = config['job_dir']
        with open(os.path.join(working_dir, 'err.log'), 'a+') as f:
            f.write(f"{config['job_id']},{config['name']},{exc}\n")

