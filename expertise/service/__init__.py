import os
import flask
import logging, logging.handlers
import redis

from celery import Celery

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

    return app

def create_celery(app):
    """
    Initializes a celery application using Flask App
    """
    config_source = app.config["CELERY_CONFIG"]
    celery = Celery(
        app.import_name,
        include=["expertise.service.celery_tasks"],
        config_source=config_source
    )

    return celery

def create_redis(app):
    """
    Initializes a redis connection pool
    """
    pool = redis.ConnectionPool(
        host=app.config['REDIS_ADDR'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_DB']
    )
    return pool