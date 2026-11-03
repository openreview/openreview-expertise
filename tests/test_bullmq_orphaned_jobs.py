"""
Regression tests for jobs orphaned in Redis (bullmq >= 2.25.4 required).

An "orphaned" job is a job id that is still present in one of BullMQ's state
sets/lists (wait, completed, failed, ...) while its job hash has been removed
from Redis. With bullmq 2.11.0, Queue.getJobs raised
"IndexError: list index out of range" when it encountered one, which
get_expertise_all_status swallowed, so /expertise/status/all returned an empty
result list for every query.

These tests only need Redis on localhost:6379 (no OpenReview API stack):
    docker run -d --rm --name expertise-test-redis -p 6379:6379 redis:7-alpine
"""
import asyncio
import logging
import threading
import time

import openreview
import pytest
import redis
from bullmq import Queue

from expertise.service.expertise import BaseExpertiseService

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 9
QUEUE_PREFIX = 'bullmq:test-orphan'
QUEUE_NAME = 'Expertise'

# Same states that get_expertise_all_status queries
JOB_STATES = [
    'completed',
    'failed',
    'active',
    'delayed',
    'paused',
    'waiting',
    'waiting-children',
    'prioritized',
]


def _job_data(job_id, member_of, user_id='~Test_User1'):
    cdate = int(time.time() * 1000)
    venue = member_of.split('/')[0]
    return {
        'user_id': user_id,
        'status': 'Queued',
        'description': 'Your job has been queued',
        'config': {
            'name': 'test_run',
            'user_id': user_id,
            'job_id': job_id,
            'cdate': cdate,
            'mdate': cdate,
        },
        'api_request': {
            'name': 'test_run',
            'entityA': {'type': 'Group', 'memberOf': member_of},
            'entityB': {'type': 'Note', 'invitation': f'{venue}/-/Submission'},
        },
    }


def _clear_queue_keys(redis_client):
    keys = redis_client.keys(f'{QUEUE_PREFIX}:{QUEUE_NAME}:*')
    if keys:
        redis_client.delete(*keys)


@pytest.fixture
def queue_env():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    queue = Queue(
        QUEUE_NAME,
        {
            'prefix': QUEUE_PREFIX,
            'connection': {
                'host': REDIS_HOST,
                'port': REDIS_PORT,
                'db': REDIS_DB,
            },
        },
    )
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    _clear_queue_keys(redis_client)

    yield queue, loop, redis_client

    _clear_queue_keys(redis_client)
    asyncio.run_coroutine_threadsafe(queue.close(), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    redis_client.close()


def _run(coro, loop):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)


def _add_jobs_and_orphan_one(queue, loop, redis_client):
    """Add three jobs, then delete one job's hash so its id is orphaned in the wait list."""
    _run(queue.add('test_run', _job_data('job-1', 'ABC.cc/Reviewers'), {'jobId': 'job-1'}), loop)
    _run(queue.add('test_run', _job_data('job-2', 'TMLR/Action_Editors'), {'jobId': 'job-2'}), loop)
    _run(queue.add('test_run', _job_data('job-3', 'TMLR/Action_Editors'), {'jobId': 'job-3'}), loop)

    assert redis_client.delete(f'{QUEUE_PREFIX}:{QUEUE_NAME}:job-2') == 1


def _make_service(queue, loop):
    # BaseExpertiseService.__init__ starts a live worker thread that would
    # consume the fabricated jobs, so build a bare instance with only the
    # attributes get_expertise_all_status uses.
    service = object.__new__(BaseExpertiseService)
    service.queue = queue
    service.queue_loop = loop
    service.logger = logging.getLogger('test-bullmq-orphan')
    return service


def test_get_jobs_tolerates_orphaned_job_id(queue_env):
    queue, loop, redis_client = queue_env
    _add_jobs_and_orphan_one(queue, loop, redis_client)

    # bullmq 2.11.0 raised IndexError here when the state sets referenced
    # a job id whose hash no longer exists
    jobs = _run(queue.getJobs(JOB_STATES), loop)

    returned_ids = {job.id for job in jobs if job.data}
    assert returned_ids == {'job-1', 'job-3'}


def test_status_all_returns_results_despite_orphaned_job(queue_env):
    queue, loop, redis_client = queue_env
    _add_jobs_and_orphan_one(queue, loop, redis_client)

    service = _make_service(queue, loop)

    # Same query as client.get_expertise_status(group_id='TMLR/Action_Editors')
    result = service.get_expertise_all_status('~Test_User1', {'entityA.memberOf': 'TMLR/Action_Editors'})
    assert [record['jobId'] for record in result['results']] == ['job-3']

    # An unfiltered query returns every surviving job
    result = service.get_expertise_all_status('~Test_User1', {})
    assert sorted(record['jobId'] for record in result['results']) == ['job-1', 'job-3']


def test_status_all_raises_when_queue_read_fails(queue_env):
    queue, loop, redis_client = queue_env
    service = _make_service(queue, loop)

    class BrokenQueue:
        async def getJobs(self, states):
            raise RuntimeError('connection reset')

    # A failure reading the queue must surface as an error,
    # not be swallowed into an empty result list
    service.queue = BrokenQueue()
    with pytest.raises(openreview.OpenReviewException, match='Failed to list jobs from queue'):
        service.get_expertise_all_status('~Test_User1', {})
