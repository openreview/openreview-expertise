import logging
from ..execute_expertise import *
from .server import celery_app as celery


@celery.task(name='userpaper', track_started=True, bind=True, time_limit=3600 * 24)
def run_userpaper(self, config: dict, logger: logging.Logger):
    try:
        openreview_client = openreview.Client(
            token=config['token'],
            baseurl=config['baseurl']
        )
        execute_create_dataset(openreview_client, config_file=config)
        execute_expertise(config_file=config)
    except Exception as exc:
        logger.error('Error: {}'.format(exc))