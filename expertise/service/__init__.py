import os
import flask
import logging, logging.handlers
import redis
from threading import Event
from google.cloud import storage

artifact_loading_started = Event()
model_ready = Event()

# Maps an API `model` value to the GCS subdirectories (relative to AIP_STORAGE_URI)
# that must be present on disk before the model can run. Subdirectories live under
# `gs://<bucket>/<aip_prefix>/<subdir>/` and are mirrored to `/app/expertise-utils/<subdir>/`.
MODEL_ARTIFACTS = {
    'bm25': [],
    'specter': ['hf_models/specter'],
    'mfr': ['multifacet_recommender'],
    'specter+mfr': ['hf_models/specter', 'multifacet_recommender'],
    'specter2': ['hf_models/specter2_base', 'hf_models/specter2_adapter'],
    'scincl': ['hf_models/scincl'],
    'specter2+scincl': [
        'hf_models/specter2_base',
        'hf_models/specter2_adapter',
        'hf_models/scincl',
    ],
}


def artifacts_for_model(model):
    """Return the list of GCS subdirectories required by `model`.

    Unknown models fall back to downloading every artifact so the service keeps
    working if a new model is introduced without updating this map.
    """
    if model in MODEL_ARTIFACTS:
        return MODEL_ARTIFACTS[model]
    return None

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

    if app.config.get('EXPERTISE_ENV', 'production') == 'development':
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

    # Use our own EXPERTISE_ENV variable to pick the config file to load.
    # FLASK_ENV / app.config['ENV'] are deprecated in Flask 2.3, so we avoid
    # them entirely. Defaults to 'production'. Callers may override the env
    # via the config dict, so resolve the final value first and then load the
    # matching cfg file to keep app.config['EXPERTISE_ENV'] consistent with
    # the settings actually loaded.
    env = os.getenv('EXPERTISE_ENV', 'production')
    if config and isinstance(config, dict) and 'EXPERTISE_ENV' in config:
        env = config['EXPERTISE_ENV']

    app.config['EXPERTISE_ENV'] = env
    app.config.from_pyfile('default.cfg')
    app.config.from_pyfile('{}.cfg'.format(env), silent=True)

    if config and isinstance(config, dict):
        app.config.from_mapping(config)

    configure_logger(app)

    # The placement of this import statement is important!
    # It must come after the app is initialized, and imported in the same scope.
    from . import routes
    app.register_blueprint(routes.BLUEPRINT)

    return app


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

def load_model_artifacts(subdirs=None, destination_dir=None):
    """Copy model artifacts from the GCS bucket to the local directory.

    If `subdirs` is None, every file under AIP_STORAGE_URI is mirrored (used by
    the long-lived Flask service). If `subdirs` is a list of subdirectory names,
    only those subdirectories are downloaded (used by per-job pipeline workers
    so they don't pull models they won't run).

    An empty list downloads nothing — valid for the `bm25` model.

    `destination_dir` defaults to env `EXPERTISE_UTILS_DIR` or `/app/expertise-utils`
    (the container default). CI jobs running outside the container override it to
    mirror into the CI workspace.
    """
    artifact_loading_started.set()

    # Extract the bucket name and path from the environment variable
    aip_storage_uri = os.getenv('AIP_STORAGE_URI')
    print(f"Loading from... {aip_storage_uri}")
    if not aip_storage_uri:
        raise ValueError("AIP_STORAGE_URI environment variable is not set")

    # Assuming AIP_STORAGE_URI is in the format gs://bucket_name/path_to_directory
    bucket_name = aip_storage_uri.split('/')[2]
    print(f"Bucket={bucket_name}")

    if destination_dir is None:
        destination_dir = os.getenv('EXPERTISE_UTILS_DIR', '/app/expertise-utils')
    base_prefix = '/'.join(aip_storage_uri.split('/')[3:]).rstrip('/')

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    if subdirs is None:
        print(f"[artifacts] Downloading ALL artifacts under gs://{bucket_name}/{base_prefix}/", flush=True)
        prefixes = [base_prefix]
    elif len(subdirs) == 0:
        print("[artifacts] No artifacts required for this job, skipping GCS download.", flush=True)
        model_ready.set()
        return
    else:
        print(f"[artifacts] Selective download from gs://{bucket_name}/{base_prefix}/ — subdirs: {subdirs}", flush=True)
        prefixes = [f"{base_prefix}/{s.strip('/')}" for s in subdirs]

    total_files = 0
    total_bytes = 0
    for prefix in prefixes:
        print(f"[artifacts] Downloading gs://{bucket_name}/{prefix}/", flush=True)
        blobs = bucket.list_blobs(prefix=prefix + '/')
        found = False
        for blob in blobs:
            found = True
            destination_path = os.path.join(destination_dir, os.path.relpath(blob.name, start=base_prefix))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            total_files += 1
            total_bytes += blob.size or 0
            print(f"[artifacts] copied {blob.name} ({blob.size} bytes) -> {destination_path}", flush=True)
        if not found:
            raise FileNotFoundError(
                f"No artifacts found at gs://{bucket_name}/{prefix}/ — "
                f"check that the model files are uploaded to this location."
            )

    print(f"[artifacts] Done. Downloaded {total_files} files, {total_bytes / (1024*1024):.1f} MB from GCS bucket.", flush=True)
    model_ready.set()