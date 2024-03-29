from expertise.service import (
    create_app, create_celery, create_redis
)
from dotenv import load_dotenv

load_dotenv()

app = create_app()
redis_config_pool, redis_embeddings_pool = create_redis(app)
celery_app = create_celery(app)
