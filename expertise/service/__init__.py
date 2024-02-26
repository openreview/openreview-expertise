import os
import flask
import logging, logging.handlers
import redis
import time

from celery import Celery
from celery.app.control import Control
from expertise.service.utils import JobStatus, JobDescription, RedisDatabase

def configure_logger(app):
    '''
    Configures the app's logger object.
    '''
    app.logger.removeHandler(flask.logging.default_handler)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: [in %(pathname)s:%(lineno)d] %(threadName)s %(message)s')

    file_handler = logging.handlers.RotatingFileHandler(
        filename=app.config['LOG_FILE'],
        mode='a',
        maxBytes=1*1000*1000,
        backupCount=20)

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    app.logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    if app.config['ENV'] == 'development':
        app.logger.addHandler(stream_handler)

    app.logger.setLevel(logging.DEBUG)
    app.logger.debug('Starting app')

    return app.logger


def create_app(config=None):
    '''
    Implements the "app factory" pattern, recommended by Flask documentation.
    '''

    app = flask.Flask(
        __name__,
        instance_path=os.path.join(os.path.dirname(__file__), 'config'),
        instance_relative_config=True
    )

    # app.config['ENV'] is automatically set by the FLASK_ENV environment variable.
    # by default, app.config['ENV'] == 'production'
    app.config.from_pyfile('default.cfg')
    app.config.from_pyfile('{}.cfg'.format(app.config.get('ENV')), silent=True)

    if config and isinstance(config, dict):
        app.config.from_mapping(config)

    configure_logger(app)

    # The placement of this import statement is important!
    # It must come after the app is initialized, and imported in the same scope.
    from . import routes
    app.register_blueprint(routes.BLUEPRINT)

    # Any jobs that are running must be marked with cancel
    redis_config_pool, _ = create_redis(app)
    redis = RedisDatabase(
        connection_pool=redis_config_pool
    )
    app.logger.info('Running server startup code')
    for config in redis.load_all_jobs('~Super_User1'):
        status = config.status
        if status in [
            JobStatus.QUEUED,
            JobStatus.FETCHING_DATA,
            JobStatus.EXPERTISE_QUEUED,
            JobStatus.RUN_EXPERTISE
        ]:
            app.logger.info(f"{config.job_id} was running - canceling job")
            config.status = JobStatus.ERROR
            config.description = 'Server restarted while job was running'
            config.mdate = int(time.time() * 1000)
            redis.save_job(config)

    return app

def create_celery(app):
    """
    Initializes a celery application using Flask App
    """

    def set_config_error(job_config):
        status = job_config.status
        if status in [
            JobStatus.QUEUED,
            JobStatus.FETCHING_DATA,
            JobStatus.EXPERTISE_QUEUED,
            JobStatus.RUN_EXPERTISE
        ]:
            app.logger.info(f"{job_config.job_id} was running - canceling job")
            job_config.status = JobStatus.ERROR
            job_config.description = 'Server restarted while job was running'
            job_config.mdate = int(time.time() * 1000)
            redis.save_job(job_config)

    config_source = app.config["CELERY_CONFIG"]
    celery = Celery(
        app.import_name,
        include=["expertise.service.celery_tasks"],
        config_source=config_source
    )

    # Clear the celery queues
    redis_config_pool, _ = create_redis(app)
    redis = RedisDatabase(
        connection_pool=redis_config_pool
    )

    app.logger.info('Running celery startup code')
    control = Control(celery)
    inspect = control.inspect()
    active = inspect.active()
    scheduled = inspect.scheduled()
    reserved = inspect.reserved()

    app.logger.info('Active queue:')
    app.logger.info(active)
    if active:
        for job_list in active.values():
            for job in job_list:
                app.logger.info(f"Revoking task {job['id']}")
                control.revoke(job['id'])
                set_config_error(job['args'][0])

    app.logger.info('Scheduled queue:')
    app.logger.info(scheduled)
    if scheduled:
        for job_list in scheduled.values():
            for job in job_list:
                app.logger.info(f"Revoking task {job['id']}")
                control.revoke(job['id'])
                set_config_error(job['args'][0])

    app.logger.info('Reserved queue:')
    app.logger.info(reserved)
    if reserved:
        for job_list in reserved.values():
            for job in job_list:
                app.logger.info(f"Revoking task {job['id']}")
                control.revoke(job['id'])
                set_config_error(job['args'][0])

    return celery

def create_redis(app):
    """
    Initializes a redis connection pool
    """
    config_pool = redis.ConnectionPool(
        host=app.config['REDIS_ADDR'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_CONFIG_DB']
    )

    embedding_pool = redis.ConnectionPool(
        host=app.config['REDIS_ADDR'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_EMBEDDINGS_DB']
    )
    return config_pool, embedding_pool