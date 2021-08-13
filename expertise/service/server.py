from expertise.service import (
    create_app, create_celery
)

app = create_app()
celery_app = create_celery(app, config_source='expertise.service.config.celery_config')