import os
import flask
import logging, logging.handlers
import redis
from threading import Event
from google.cloud import storage

from celery import Celery

artifact_loading_started = Event()
model_ready = Event()

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

def load_model_artifacts():
    """Copy all files from a GCS bucket directory to a local directory."""
    artifact_loading_started.set()

    # Extract the bucket name and path from the environment variable
    aip_storage_uri = os.getenv('AIP_STORAGE_URI')
    print(f"Loading from... {aip_storage_uri}")
    if not aip_storage_uri:
        raise ValueError("AIP_STORAGE_URI environment variable is not set")

    # Assuming AIP_STORAGE_URI is in the format gs://bucket_name/path_to_directory
    bucket_name = aip_storage_uri.split('/')[2]
    print(f"Bucket={bucket_name}")

    # The directory to copy the artifacts to, and the subdirectory name you want
    destination_dir = "/app/expertise-utils" ## TODO: Parameterize this
    source_blob_prefix = '/'.join(aip_storage_uri.split('/')[3:])

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    for blob in blobs:
        destination_path = os.path.join(destination_dir, os.path.relpath(blob.name, start=source_blob_prefix))
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)
        print(f"Copied {blob.name} to {destination_path}")
    
    print("Model artifacts loaded")
    model_ready.set()