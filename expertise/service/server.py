from expertise.service import (
    create_app, create_celery
)
from dotenv import load_dotenv

load_dotenv()

app = create_app()
celery_app = create_celery(app)