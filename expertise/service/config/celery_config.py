from kombu import Exchange, Queue
from kombu.serialization import registry

task_default_queue = 'default'
task_queues = (
    Queue('userpaper', routing_key='expertise.service.celery_tasks.run_userpaper'),
    Queue('expertise', routing_key='expertise.service.celery_tasks.run_expertise')
)
# CELERY_IMPORTS = ('tasks')
task_ignore_result = False
# broker_url = 'redis://localhost:6379/0'
broker_url = 'redis://localhost:6379/10'
result_backend = 'redis://localhost:6379/10'
# CELERY_DEFAULT_EXCHANGE_TYPE = 'direct'
task_serializer = 'pickle'
result_serializer = 'pickle'
accept_content = ['pickle', 'application/x-python-serialize']
task_create_missing_queues = True

worker_prefetch_multiplier = 1
# result_backend = 'redis://localhost:6379/0'

broker_transport_options = {
    'visibility_timeout': 43200,
    'max_retries': 0
}

registry.enable('pickle')
registry.enable('application/x-python-serialize')