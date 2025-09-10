import os
from expertise.service import (
    create_app, create_celery, create_redis
)
from dotenv import load_dotenv

base_dir = os.path.dirname(__file__)

# Load defaults first
default_path = os.path.join(base_dir, "config", "default.cfg")
load_dotenv(default_path, override=False)

# Load production, override only the matching ones
prod_path = os.path.join(base_dir, "config", "production.cfg")
load_dotenv(prod_path, override=True)

app = create_app()
redis_config_pool, redis_embeddings_pool = create_redis(app)
celery_app = create_celery(app)
