import shortuuid
import shutil
import time
import os
import json
import torch
import gc
import datetime
from csv import reader
import openreview
from openreview import OpenReviewException
from enum import Enum
from threading import Lock
from pathlib import Path
import multiprocessing
from bullmq import Queue, Worker, Job
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service.utils import GCPInterface
from copy import deepcopy
import asyncio
import threading
import traceback
from google.api_core import exceptions as google_exceptions
from google.rpc import code_pb2

from .utils import JobConfig, APIRequest, JobDescription, JobStatus, SUPERUSER_IDS, get_user_id, ExpectedDataError

user_index_file_lock = Lock()

class BaseExpertiseService:
    def __init__(
        self,
        config,
        logger,
        containerized=False,
        sync_on_disk=True,
        worker_attempts=1,
        worker_backoff_delay=60000,
        worker_concurrency=None,
        worker_lock_duration=None,
        worker_autorun=False,
    ):
        """
        :param config:         Your server configuration dictionary
        :param logger:         Logger instance for logging
        :param containerized:  Whether your service is running in containerized mode
        :param sync_on_disk:   Whether to write job files to disk
        :param worker_attempts: (Optional) number of attempts for the BullMQ worker
        :param worker_backoff_delay: (Optional) backoff delay 2 ^ attempts * delay (ms) for the BullMQ worker
        :param worker_concurrency: (Optional) concurrency for the BullMQ worker
        :param worker_lock_duration: (Optional) lock duration (ms) for the BullMQ worker
        :param worker_autorun: (Optional) whether the worker should start automatically
        """
        self.logger = logger
        self.server_config = config
        self.containerized = containerized
        self.sync_on_disk = sync_on_disk  # Whether to actually save jobs on disk
        self.default_expertise_config = config.get('DEFAULT_CONFIG')
        self._submit_lock = threading.Lock()
        self._pending_request_keys = set()
        self.worker_attempts = worker_attempts
        self.worker_backoff_delay = worker_backoff_delay
        self.bullmq_remove_on_complete_age = config.get('BULLMQ_REMOVE_ON_COMPLETE_AGE', 1209600)
        self.bullmq_remove_on_fail_age = config.get('BULLMQ_REMOVE_ON_FAIL_AGE', 1209600)
        self.working_dir = config.get('WORKING_DIR')
        self.specter_dir = config.get('SPECTER_DIR')
        self.mfr_feature_vocab_file = config.get('MFR_VOCAB_DIR')
        self.mfr_checkpoint_dir = config.get('MFR_CHECKPOINT_DIR')

        # Create the BullMQ queue
        self.queue = Queue(
            'Expertise',
            {
                'prefix': 'bullmq:expertise',
                'connection': {
                    "host": config['REDIS_ADDR'],
                    "port": config['REDIS_PORT'],
                    "db": config['REDIS_CONFIG_DB'],
                }
            }
        )
        self.start_queue_in_thread()

        self.worker_settings = {
            'prefix': 'bullmq:expertise',
            'connection': {
                "host": config['REDIS_ADDR'],
                "port": config['REDIS_PORT'],
                "db": config['REDIS_CONFIG_DB'],
            },
            'autorun': False
        }
        if worker_concurrency is not None:
            self.worker_settings['concurrency'] = worker_concurrency
        if worker_lock_duration is not None:
            self.worker_settings['lockDuration'] = worker_lock_duration

        self.worker = None
        self.start_worker_in_thread()

        # Define required/optional fields if they are reused
        self.req_fields = ['name', 'match_group', 'user_id', 'job_id']
        self.optional_model_params = ['use_title', 'use_abstract', 'average_score', 'max_score', 'skip_specter']
        self.optional_fields = [
            'model', 'model_params', 'exclusion_inv', 'token', 'baseurl',
            'baseurl_v2', 'paper_invitation', 'paper_id'
        ]
        self.path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']

        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)

    @staticmethod
    def expertise_worker(config_json, queue):
        try:
            config = json.loads(config_json)
            execute_expertise(config=config)
        except Exception as e:
            queue.put(e)
        finally:
            # Cleanup resources
            torch.cuda.empty_cache()
            gc.collect()

    def start_queue_in_thread(self):
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.queue_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_event_loop, args=(self.queue_loop,), daemon=True)
        thread.start()

    def start_worker_in_thread(self):
        def run_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.worker = Worker(
                'Expertise',
                self.worker_process,
                self.worker_settings
            )
            loop.run_until_complete(self.worker.run())

        thread = threading.Thread(target=run_event_loop, daemon=True)
        thread.start()

    async def close(self):
        if self.worker is not None:
            await self.worker.close()
        await self.queue.close()

    def worker_process(self, job, token):
        """
        Override in child classes
        """
        raise NotImplementedError("worker_process must be implemented in a child class.")

    def _build_description(self, new_status, desc=None, config=None):
        descriptions = JobDescription.VALS.value
        if desc is None:
            return descriptions[new_status]

        if config is not None:
            paper_scoring = config.api_request.entityA.get('type') == 'Note' and config.api_request.entityB.get('type') == 'Note'
            if 'num_samples=0' in desc:
                if paper_scoring:
                    desc += '. Please check that you have access to the papers that you are querying for.'
                else:
                    desc += '. Please check that there is at least 1 member of the match group with some publication.'
            if 'Dimension out of range' in desc:
                if paper_scoring:
                    desc += '. Please check that you have access to the papers that you are querying for.'
                else:
                    desc += '. Please check that you have at least 1 submission submitted and that you have run the Post Submission stage.'
        return desc

    async def _update_job_status(self, job, new_status, desc=None, error=None):
        """
        Write status, description and optional error into the BullMQ job data.
        """
        description = self._build_description(new_status, desc)
        data = {**job.data, 'status': new_status, 'description': description}
        if error is not None:
            data['error'] = error
        job.data = data
        future = asyncio.run_coroutine_threadsafe(
            self.queue.scripts.updateData(job.id, data),
            self.queue_loop
        )
        await asyncio.wrap_future(future)

    def _get_job_from_queue(self, job_id):
        try:
            future = asyncio.run_coroutine_threadsafe(
                Job.fromId(self.queue, job_id),
                self.queue_loop
            )
            return future.result()
        except Exception as e:
            self.logger.warning(f"Failed to fetch job {job_id} from queue: {e}")
            return None

    def _get_job_status_from_queue(self, job_id):
        """
        Query BullMQ for the canonical status of a job.
        Returns (status, description) or (None, None) if the job
        is no longer in the queue (archived).
        """
        job = self._get_job_from_queue(job_id)
        if job is None:
            return None, None

        descriptions = JobDescription.VALS.value
        data = job.data
        status = data.get('status')
        description = data.get('description')
        if status is not None:
            return status, (description or descriptions.get(status, ''))

        return None, None

    def _config_from_job_data(self, data):
        config = JobConfig.from_json(deepcopy(data.get('config', {})))
        api_request_data = data.get('api_request')
        if api_request_data is not None:
            config.api_request = APIRequest(deepcopy(api_request_data))
        return config

    def _load_job_from_queue(self, user_id, job_id):
        job = self._get_job_from_queue(job_id)
        if job is None:
            raise openreview.OpenReviewException(f'Job {job_id} not found in queue')
        job_user_id = job.data.get('user_id')
        if job_user_id != user_id and user_id not in SUPERUSER_IDS:
            raise openreview.OpenReviewException('Forbidden: Insufficient permissions to access job')
        return job

    def _record_for_config(self, config, status, description):
        self._filter_config(config)
        return {
            'name': config.name,
            'tauthor': config.user_id,
            'jobId': config.job_id,
            'status': status,
            'description': description,
            'cdate': config.cdate,
            'mdate': config.mdate,
            'request': config.api_request.to_json() if config.api_request else {}
        }

    def _job_matches_query(self, config, status, query_obj):
        def check_status():
            search_status = query_obj.get('status', '')
            return not search_status or status.lower().startswith(search_status.lower())

        def check_member():
            search_member, memberOf = '', ''
            if 'memberOf' in query_obj.keys():
                memberOf = config.api_request.entityA.get('memberOf', '') or config.api_request.entityB.get('memberOf', '')
                search_member = query_obj['memberOf']
            elif 'memberOf' in query_obj.get('entityA', {}).keys():
                memberOf = config.api_request.entityA.get('memberOf', '')
                search_member = query_obj['entityA']['memberOf']
            elif 'memberOf' in query_obj.get('entityB', {}).keys():
                memberOf = config.api_request.entityB.get('memberOf', '')
                search_member = query_obj['entityB']['memberOf']
            return not search_member or memberOf.lower().startswith(search_member.lower())

        def check_invitation():
            search_invitation, inv = '', ''
            if 'invitation' in query_obj.keys():
                inv = config.api_request.entityA.get('invitation', '') or config.api_request.entityB.get('invitation', '')
                search_invitation = query_obj['invitation']
            elif 'invitation' in query_obj.get('entityA', {}).keys():
                inv = config.api_request.entityA.get('invitation', '')
                search_invitation = query_obj['entityA']['invitation']
            elif 'invitation' in query_obj.get('entityB', {}).keys():
                inv = config.api_request.entityB.get('invitation', '')
                search_invitation = query_obj['entityB']['invitation']
            return not search_invitation or inv.lower().startswith(search_invitation.lower())

        def check_paper_id():
            search_paper_id, paper_id = '', ''
            if 'id' in query_obj.keys():
                paper_id = config.api_request.entityA.get('id', '') or config.api_request.entityB.get('id', '')
                search_paper_id = query_obj['id']
            elif 'id' in query_obj.get('entityA', {}).keys():
                paper_id = config.api_request.entityA.get('id', '')
                search_paper_id = query_obj['entityA']['id']
            elif 'id' in query_obj.get('entityB', {}).keys():
                paper_id = config.api_request.entityB.get('id', '')
                search_paper_id = query_obj['entityB']['id']
            return not search_paper_id or paper_id.lower().startswith(search_paper_id.lower())

        return False not in [
            check_status(),
            check_member(),
            check_invitation(),
            check_paper_id()
        ]

    def get_expertise_all_status(self, user_id, query_params):
        """
        Searches the server for all jobs submitted by a user that satisfies
        the HTTP GET query parameters. Jobs are read from the BullMQ queue.
        """
        result = {'results': []}
        query_obj = {}

        for query, value in query_params.items():
            if query.find('.') < 0:
                query_obj[query] = value
            else:
                entity, query_by = query.split('.')
                if entity not in query_obj.keys():
                    query_obj[entity] = {}
                query_obj[entity][query_by] = value

        self.logger.info(f"Searching for jobs with query: {query_obj}")

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.queue.getJobs([
                    'completed',
                    'failed',
                    'active',
                    'delayed',
                    'paused',
                    'waiting',
                    'waiting-children',
                    'prioritized',
                ]),
                self.queue_loop,
            )
            jobs = future.result()
        except Exception as e:
            self.logger.warning(f"Failed to list jobs from queue: {e}")
            jobs = []

        for job in jobs:
            job_user_id = job.data.get('user_id')
            if job_user_id != user_id and user_id not in SUPERUSER_IDS:
                continue

            config = self._config_from_job_data(job.data)

            status = job.data.get('status')
            description = job.data.get('description')
            if status is None:
                continue

            if self._job_matches_query(config, status, query_obj):
                result['results'].append(self._record_for_config(config, status, description))

        # Sort results by cdate
        result['results'] = sorted(result['results'], key=lambda x: x['cdate'], reverse=True)

        return result

    def get_expertise_status(self, user_id, job_id):
        """
        Searches the server for all jobs submitted by a user
        Only fetch the status of the given job id

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param job_id: ID of the specific job to look up
        :type job_id: str

        :returns: A dictionary with the key 'results' containing a list of job statuses
        """
        job = self._load_job_from_queue(user_id, job_id)
        config = self._config_from_job_data(job.data)
        status = job.data.get('status')
        description = job.data.get('description')
        if status is None:
            raise openreview.OpenReviewException(f"Job {job_id} not found in queue")

        return self._record_for_config(config, status, description)

    def _filter_config(self, running_config):
        """
        Filters out certain server-side fields of a config file in order to
        form a presentable config to the user

        :param running_config: Contains the config JSON as read from the servver
        :type running_config: JobConfig

        :returns config: A modified version of config without the server fields
        """

        running_config.baseurl = None
        running_config.baseurl_v2 = None
        running_config.user_id = None

    def _validate_request(self, client, request):
        """
        Validate and build a JobConfig for a request.

        Wraps the request in an APIRequest, calls APIRequest.validate(client)
        to resolve OpenReview-dependent fields (user_id, expertise-edge
        invitation labels) and enforce caller permissions (machine_type),
        then transforms it into a JobConfig via JobConfig.from_request.

        :param client: Authenticated OpenReview client for the calling user
        :param request: Submitted job request body
        :returns: The resolved JobConfig
        """

        self.logger.info(f"Incoming request - {request}")
        validated_request = APIRequest(request)
        validated_request.validate(client)
        config = JobConfig.from_request(
            api_request = validated_request,
            starting_config = self.default_expertise_config,
            server_config = self.server_config,
            working_dir = self.working_dir
        )
        self.logger.info(f"Config validation passed - {config.to_json()}")
        return config

    def _save_config(self, config: JobConfig):
        """Create job directory and write config.json and request.json to disk."""
        if not os.path.isdir(config.dataset['directory']):
            os.makedirs(config.dataset['directory'])
        with open(os.path.join(config.job_dir, 'config.json'), 'w+') as f:
            json.dump(config.to_json(), f, ensure_ascii=False, indent=4)
        if config.api_request is not None:
            with open(os.path.join(config.job_dir, 'request.json'), 'w+') as f:
                json.dump(config.api_request.to_json(), f, ensure_ascii=False, indent=4)
        self.logger.info(f"Saving processed config to {os.path.join(config.job_dir, 'config.json')}")
        return config

    def _get_score_and_metadata_dir(self, search_dir):
        """
        Searches the given directory for a possible score file and the metadata file

        :param search_dir: The root directory to search in
        :type search_dir: str

        :returns file_dir: The directory of the score file, if it exists, starting from the given directory
        :returns metadata_dir: The directory of the metadata file, if it exists, starting from the given directory
        """
        # Get results always returns sparse scores. The dispatch in
        # execute_expertise unconditionally produces {name}_sparse.csv (
        # sparse_value is a required positive integer, validated at config
        # creation), so if the file isn't on disk something went wrong with
        # the job and we surface that as an error rather than silently
        # serving the wrong artifact.
        file_dir, metadata_dir = None, None
        with open(os.path.join(search_dir, 'config.json'), 'r') as f:
            config = JobConfig.from_json(json.load(f))

        sparse_csv_path = os.path.join(search_dir, f"{config.name}_sparse.csv")
        if not os.path.isfile(sparse_csv_path):
            raise OpenReviewException("Sparse score file not found for job {job_id}".format(job_id=config.job_id))
        file_dir = sparse_csv_path

        if os.path.isfile(os.path.join(search_dir, 'metadata.json')):
            metadata_dir = os.path.join(search_dir, 'metadata.json')
        else:
            raise OpenReviewException("Metadata file not found for job {job_id}".format(job_id=config.job_id))

        return file_dir, metadata_dir

    def del_expertise_job(self, user_id, job_id):
        """Remove job artifacts from disk and the BullMQ queue, returning a sanitized config."""
        job = self._load_job_from_queue(user_id, job_id)
        config = self._config_from_job_data(job.data)

        # Only allow deletion when job has completed, errored out, or been
        # archived from BullMQ (no longer trackable).
        allowed_states = {
            JobStatus.COMPLETED, JobStatus.DATA_ERROR, JobStatus.ERROR
        }
        status = job.data.get('status')
        if status is not None and status not in allowed_states:
            raise openreview.OpenReviewException(
                f"Bad request: cannot delete job in status {status}"
            )

        self.logger.info(f"Deleting {config.job_dir} for {user_id}")
        if os.path.isdir(config.job_dir):
            shutil.rmtree(config.job_dir)
        else:
            self.logger.info("No files found - only removing queue entry")

        self._filter_config(config)
        return config.to_json()

    def _get_job_name(self, request):
        job_name_parts = [request.get('name', 'No name provided')]
        entities = []
        if request.get('entityA', {}).get('type'):
            entities.append(request['entityA'])
        else:
            job_name_parts.append('No Entity A Type Found')
        if request.get('entityB', {}).get('type'):
            entities.append(request['entityB'])
        else:
            job_name_parts.append('No Entity B Type Found')

        for entity in entities:

            job_name_parts.append(
                APIRequest.extract_from_entity(
                    entity,
                    get_value=True
                )
            )

        return f'{job_name_parts[0]}: {job_name_parts[1]} - {job_name_parts[2]}'

    def _get_log_from_request(self, request):
        log = []
        if request.get('entityA'):
            log.append(f"Entity A: {json.dumps(request.get('entityA', {}), indent=4)}")
        if request.get('entityB'):
            log.append(f"Entity B: {json.dumps(request.get('entityB', {}), indent=4)}")

        return '\n'.join(log)

    def _get_log_from_config(self, config):
        log = []
        if config.name:
            log.append(f"Job name: {config.name}")
        if config.paper_id:
            log.append(f"Paper ID: {config.paper_id}")
        if config.paper_invitation:
            log.append(f"Paper invitation: {config.paper_invitation}")
        if config.paper_venueid:
            log.append(f"Paper venue ID: {config.paper_venueid}")
        if config.match_group:
            log.append(f"Match group: {config.match_group}")
        if config.alternate_match_group:
            log.append(f"Alternate match group: {config.alternate_match_group}")
        if config.model:
            log.append(f"Model: {config.model}")
        if config.model_params:
            log.append(f"Model params: {json.dumps(config.to_json().get('model_params', {}), indent=4)}")

        return '\n'.join(log)

    def get_key_from_request(self, request):
        key_parts = []
        entities = []
        if request.get('entityA', {}).get('type'):
            entities.append(request['entityA'])
        else:
            key_parts.append('NoEntityA')

        if request.get('entityB', {}).get('type'):
            entities.append(request['entityB'])
        else:
            key_parts.append('NoEntityB')

        for entity in entities:
            key_parts.extend(
                APIRequest.extract_from_entity(
                    entity,
                    get_value=True,
                    return_as_list=True
                )
            )

        if request.get('model', {}).get('name'):
            key_parts.append(request['model']['name'])

        return ':'.join(key_parts)

class ExpertiseService(BaseExpertiseService):

    def __init__(self, config, logger, containerized = False):
        super().__init__(
            config=config,
            logger=logger,
            containerized=containerized,
            sync_on_disk=True,            # We want to store jobs on disk
            worker_attempts=config['WORKER_ATTEMPTS'],
            worker_backoff_delay=config['WORKER_BACKOFF_DELAY'],
            worker_concurrency=config['ACTIVE_JOBS'],
            worker_lock_duration=config['LOCK_DURATION'],
            worker_autorun=False         # If that is what you originally had
        )

    async def worker_process(self, job, token):
        config = self._config_from_job_data(job.data)
        config.baseurl_v2 = job.data.get('baseurl_v2')
        or_token = job.data['token']
        openreview_client_v2 = openreview.api.OpenReviewClient(
            token=or_token,
            baseurl=config.baseurl_v2
        )
        try:
            # Create dataset
            execute_create_dataset(openreview_client_v2, config=config.to_json())
            await self._update_job_status(job, JobStatus.RUN_EXPERTISE)

            queue = multiprocessing.Queue()  # Queue for exception handling
            config_json = json.dumps(config.to_json())  # Serialize config
            process = multiprocessing.Process(target=BaseExpertiseService.expertise_worker, args=(config_json, queue))
            process.start()
            process.join()

            if not queue.empty():
                exception = queue.get()
                raise exception  # Re-raise the exception from the subprocess

            # Update job status
            await self._update_job_status(job, JobStatus.COMPLETED)

        except ExpectedDataError as e:
            # Expected data errors - mark as data error, don't re-raise, avoid triggering retries
            asyncio.run_coroutine_threadsafe(job.log(f'Job finished with expected data error: {e}'), self.queue_loop)
            await self._update_job_status(job, JobStatus.DATA_ERROR, str(e), error=str(e))
        except Exception as e:
            await self._update_job_status(job, JobStatus.ERROR, str(e), error=str(e))
            # Re raise exception so that it appears in the queue
            exception = e.with_traceback(e.__traceback__)
            raise exception
        finally:
            # Cleanup resources
            torch.cuda.empty_cache()
            gc.collect()

    def start_expertise(self, request, client):
        descriptions = JobDescription.VALS.value

        job_name = self._get_job_name(request)
        request_log = self._get_log_from_request(request)

        request_key = self.get_key_from_request(request)

        with self._submit_lock:
            if request_key in self._pending_request_keys:
                raise openreview.OpenReviewException("Request already in process")
            self._pending_request_keys.add(request_key)

        try:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.queue.getJobs([
                        'active',
                        'delayed',
                        'paused',
                        'waiting',
                        'waiting-children',
                        'prioritized',
                    ]),
                    self.queue_loop,
                )
                jobs = future.result()
            except Exception as e:
                jobs = []

            for job in jobs:
                if job.data.get('status') == JobStatus.COMPLETED:
                    continue
                if job.data.get('request_key') == request_key:
                    raise openreview.OpenReviewException("Request already in queue")

            config = self._validate_request(client, request)
            self._save_config(config)
            job_id = config.job_id

            config_log = self._get_log_from_config(config)

            self.logger.info('just before submitting')
            self.logger.info(f"\nconf: {config.to_json()}\n")

            future = asyncio.run_coroutine_threadsafe(
                self.queue.add(
                    job_name,
                    {
                        "job_id": job_id,
                        "request_key": request_key,
                        "user_id": config.user_id,
                        "token": client.token,
                        "status": JobStatus.QUEUED,
                        "description": descriptions[JobStatus.QUEUED],
                        "config": config.to_json(),
                        "api_request": config.api_request.to_json(),
                        "baseurl_v2": config.baseurl_v2,
                    },
                    {
                        'jobId': job_id,
                        'attempts': self.worker_attempts,
                        'backoff': {
                            'delay': self.worker_backoff_delay,
                            'type': 'exponential',
                        },
                        'removeOnComplete': {
                            'age': self.bullmq_remove_on_complete_age
                        },
                        'removeOnFail': {
                            'age': self.bullmq_remove_on_fail_age
                        },
                    }
                ),
                self.queue_loop
            )
            job = future.result()

            future = asyncio.run_coroutine_threadsafe(job.log(request_log), self.queue_loop)
            future.result()

            future = asyncio.run_coroutine_threadsafe(job.log(config_log), self.queue_loop)
            future.result()

            return job_id
        finally:
            with self._submit_lock:
                self._pending_request_keys.discard(request_key)

    def get_expertise_results(self, user_id, job_id, delete_on_get=False):
        """
        Gets the scores of a given job
        If delete_on_get is set, delete the directory after the scores are fetched

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param job_id: ID of the specific job to fetch
        :type job_id: str

        :param delete_on_get: A flag indicating whether or not to clean up the directory after it is fetched
        :type delete_on_get: bool

        :returns: A dictionary that contains the calculated scores and metadata
        """
        result = {'results': []}

        job = self._load_job_from_queue(user_id, job_id)
        config = self._config_from_job_data(job.data)
        status = job.data.get('status')
        description = job.data.get('description')
        if status is None:
            raise openreview.OpenReviewException(f"Job {job_id} not found in queue")

        self.logger.info(f"{user_id} able to access job at {job_id} - checking if scores are found")
        # Assemble scores
        if status != JobStatus.COMPLETED:
            raise openreview.OpenReviewException(f"Scores not found - status: {status} | description: {description}")

        # Search for scores files (if sparse scores exist, retrieve by default)
        ret_list = []

        # Check for output format
        group_group_matching = config.alternate_match_group is not None
        paper_paper_matching = config.api_request.entityA.get('type') == 'Note' and config.api_request.entityB.get('type') == 'Note'

        self.logger.info(f"Retrieving scores from {config.job_dir}")
        file_dir, metadata_dir = self._get_score_and_metadata_dir(config.job_dir)
        with open(file_dir, 'r') as csv_file:
            data_reader = reader(csv_file)
            for row in data_reader:
                if not group_group_matching and not paper_paper_matching and row[0] == 'dummy':
                    continue
                ret_list.append({
                    'entityA': row[0] if group_group_matching or paper_paper_matching else row[1],
                    'entityB': row[1] if group_group_matching or paper_paper_matching else row[0],
                    'score': float(row[2])
                })
        result['results'] = ret_list

        # Gather metadata
        with open(metadata_dir, 'r') as metadata:
            result['metadata'] = json.load(metadata)

        # Clear directory
        if delete_on_get:
            self.logger.info(f'Deleting {config.job_dir}')
            shutil.rmtree(config.job_dir)

        return result

    def get_expertise_metadata(self, user_id, job_id):
        """
        Gets the dataset metadata for a given job (submission/archive counts,
        missing profiles and publications). Lighter than get_expertise_results
        since it doesn't read the score file.
        """
        job = self._load_job_from_queue(user_id, job_id)
        config = self._config_from_job_data(job.data)
        status = job.data.get('status')
        description = job.data.get('description')
        if status is None:
            raise openreview.OpenReviewException(f"Job {job_id} not found in queue")

        if status != JobStatus.COMPLETED:
            raise openreview.OpenReviewException(
                f"Metadata not available - status: {status} | description: {description}"
            )

        metadata_path = os.path.join(config.job_dir, 'metadata.json')
        if not os.path.isfile(metadata_path):
            raise openreview.OpenReviewException(
                f"Metadata file not found for job {job_id}"
            )

        with open(metadata_path, 'r') as f:
            return json.load(f)

class ExpertiseCloudService(BaseExpertiseService):

    def __init__(self, config, logger, containerized = False):
        super().__init__(
            config=config,
            logger=logger,
            containerized=containerized,
            sync_on_disk=True,            # We want to store jobs on disk
            worker_attempts=config['WORKER_ATTEMPTS'],
            worker_backoff_delay=config['WORKER_BACKOFF_DELAY'],
            worker_concurrency=config['ACTIVE_JOBS'],
            worker_lock_duration=config['LOCK_DURATION'],
            worker_autorun=False         # If that is what you originally had
        )
        self.poll_interval = config['POLL_INTERVAL']
        self.max_attempts = config['POLL_MAX_ATTEMPTS']
        self.cloud = GCPInterface(
            config=config,
            logger=logger
        )

    def compute_machine_type_from_dataset(self, config):
        """Compute machine type from the already-created dataset on disk.

        Reads submission_count and archives_count from metadata.json written by execute_create_dataset().
        """
        metadata_path = os.path.join(config.job_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        note_count = metadata.get('submission_count', 0) + metadata.get('archives_count', 0)

        self.logger.info(f"Machine type selection: {note_count} submissions in dataset")

        if note_count < self.server_config.get('PIPELINE_MEDIUM_THRESHOLD'):
            return self.server_config.get('SMALL_NAME')
        elif note_count < self.server_config.get('PIPELINE_LARGE_THRESHOLD'):
            return self.server_config.get('MEDIUM_NAME')
        else:
            return self.server_config.get('LARGE_NAME')


    async def _create_dataset_step(self, job, config, openreview_client_v2):
        asyncio.run_coroutine_threadsafe(job.log('Task 1: fetching data from OpenReview and building dataset'), self.queue_loop)
        await self._update_job_status(job, JobStatus.FETCHING_DATA)
        try:
            execute_create_dataset(openreview_client_v2, config=config.to_json())
            return True
        except ExpectedDataError as e:
            asyncio.run_coroutine_threadsafe(job.log(f'Job finished with expected data error: {e}'), self.queue_loop)
            await self._update_job_status(job, JobStatus.DATA_ERROR, str(e), error=str(e))
            return False
        except Exception as e:
            self.logger.error(f"Error creating dataset for {job.id}: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            if job.data.get('status') != JobStatus.ERROR:
                await self._update_job_status(job, JobStatus.ERROR, str(e), error=str(e))
            raise e.with_traceback(e.__traceback__)

    async def _upload_dataset_step(self, job, config):
        asyncio.run_coroutine_threadsafe(job.log(f'Uploading dataset to gs://{self.cloud.bucket_name}/{self.cloud.jobs_folder}/{config.cloud_id}/dataset'), self.queue_loop)
        try:
            return self.cloud.upload_dataset(config, vertex_id=config.cloud_id)
        except ExpectedDataError as e:
            asyncio.run_coroutine_threadsafe(job.log(f'Job finished with expected data error: {e}'), self.queue_loop)
            await self._update_job_status(job, JobStatus.DATA_ERROR, str(e), error=str(e))
            return None

    def _create_pipeline_job_step(self, job, request, user_id, machine_type, dataset_gcs_path, config, gcp_regions):
        submit_error = None
        for region_index, region in enumerate(gcp_regions):
            config.cloud_region = region
            self._save_config(config)
            self.logger.info(f"Trying region {region} for job {job.id} with cloud_id {config.cloud_id}")
            asyncio.run_coroutine_threadsafe(job.log(f'Trying region {region} with cloud_id {config.cloud_id}'), self.queue_loop)

            try:
                # Submit the job and return the actual cloud job object
                cloud_job = self.cloud.create_job(
                    deepcopy(request),
                    job_id=job.id,
                    user_id=user_id,
                    machine_type=machine_type,
                    dataset_gcs_path=dataset_gcs_path,
                    vertex_id=config.cloud_id,
                    region=region
                )
                asyncio.run_coroutine_threadsafe(job.log(f'Submitted PipelineJob {config.cloud_id} in region {region}'), self.queue_loop)
                return cloud_job
            except google_exceptions.ResourceExhausted as e:
                submit_error = e
                msg = f"ResourceExhausted creating cloud job for {job.id} in region {region}: {e}"
                self.logger.error(msg)
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
            except google_exceptions.ServiceUnavailable as e:
                submit_error = e
                msg = f"ServiceUnavailable creating cloud job for {job.id} in region {region}: {e}"
                self.logger.error(msg)
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
            except google_exceptions.InvalidArgument as e:
                submit_error = e
                msg = f"InvalidArgument creating cloud job for {job.id} in region {region}: {e}"
                self.logger.error(msg)
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
            except ValueError as e:
                submit_error = e
                msg = f"ValueError creating cloud job for {job.id} in region {region}: {e}"
                self.logger.error(msg)
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
            except google_exceptions.AlreadyExists as e:
                msg = f"PipelineJob {config.cloud_id} already exists in {region}, polling existing job: {e}"
                self.logger.info(msg)
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
                # If it already exists, retrieve and return the existing cloud job
                return self.cloud.get_pipeline_job(config.cloud_id, region)
            except google_exceptions.PermissionDenied as e:
                asyncio.run_coroutine_threadsafe(job.log(f'Permission denied creating cloud job in region {region}: {e}'), self.queue_loop)
                raise e.with_traceback(e.__traceback__)
            except Exception as e:
                msg = f"Error creating cloud job for {job.id} in region {region}: {e}"
                self.logger.error(msg)
                self.logger.error(f"Error details: {traceback.format_exc()}")
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
                raise e.with_traceback(e.__traceback__)

            if submit_error is not None:
                if region_index + 1 < len(gcp_regions):
                    asyncio.run_coroutine_threadsafe(job.log(f'create_job failed in {region}, falling back to {gcp_regions[region_index + 1]}'), self.queue_loop)
                    continue
                msg = f"Error creating cloud job in all regions: {submit_error}"
                asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
                raise Exception(msg)
                
    async def _poll_pipeline_job_step(self, job, user_id, config, region):
        asyncio.run_coroutine_threadsafe(job.log(f'Polling PipelineJob {config.cloud_id} in region {region}'), self.queue_loop)
        for attempt in range(self.max_attempts):
            self.logger.info(f"{job.id} - attempt {attempt + 1} of {self.max_attempts}...")
            status = self.cloud.get_job_status_by_job_id(user_id, config)
            self.logger.info(f"Status for {job.id} in region {region}: {status}")
            dt = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            if not (status and isinstance(status, dict) and 'status' in status and 'description' in status):
                asyncio.run_coroutine_threadsafe(job.log(f'Invalid status received, retrying at {dt}'), self.queue_loop)
                await asyncio.sleep(self.poll_interval)
                continue

            current_status = job.data.get('status')
            current_description = job.data.get('description')
            
            if current_status != status['status'] or current_description != status['description']:
                if current_status == JobStatus.FETCHING_DATA and status['status'] in (JobStatus.QUEUED, JobStatus.INITIALIZED):
                    await asyncio.sleep(self.poll_interval)
                    continue
                await self._update_job_status(job, status['status'], status['description'])
                asyncio.run_coroutine_threadsafe(job.log(f'Status updated to {status["status"]}: {status["description"]} at {dt}'), self.queue_loop)

            if status['status'] == JobStatus.RUN_EXPERTISE:
                asyncio.run_coroutine_threadsafe(job.log(f'Pipeline {config.cloud_id} is running in region {region} at {dt}'), self.queue_loop)

            if status['status'] == JobStatus.COMPLETED:
                asyncio.run_coroutine_threadsafe(job.log(f'Pipeline {config.cloud_id} completed in region {region} at {dt}'), self.queue_loop)
                return True

            if status['status'] == JobStatus.DATA_ERROR:
                asyncio.run_coroutine_threadsafe(job.log(f'Pipeline {config.cloud_id} data error in region {region} at {dt}'), self.queue_loop)
                return True

            if status['status'] == JobStatus.ERROR:
                description = status.get('description', '')
                is_resource_error = status.get('errorCode') == code_pb2.RESOURCE_EXHAUSTED
                is_resource_text = 'insufficient' in description.lower() or \
                                   'Resources are insufficient in region:' in description
                if is_resource_error or is_resource_text:
                    msg = f"Pipeline {config.cloud_id} failed with resource exhaustion in region {region}: {description} at {dt}"
                    self.logger.error(msg)
                    asyncio.run_coroutine_threadsafe(job.log(msg), self.queue_loop)
                    return False
                asyncio.run_coroutine_threadsafe(job.log(f'Job failed in region {region}: {description}'), self.queue_loop)
                raise Exception(f"Job {job.id} failed in region {region}: {description}")

            asyncio.run_coroutine_threadsafe(job.log(f'Job status {status["status"]} in region {region}, waiting {self.poll_interval}s at {dt}'), self.queue_loop)
            await asyncio.sleep(self.poll_interval)
        else:
            self.logger.warning(f"Polling timed out after {self.max_attempts} attempts for job {job.id}.")
            if job.data.get('status') != JobStatus.ERROR:
                await self._update_job_status(job, JobStatus.ERROR, f"Polling timed out after {self.max_attempts} attempts.", error=f"Polling timed out after {self.max_attempts} attempts.")
            raise TimeoutError(f"Polling timed out for job {job.id} after {self.max_attempts} attempts.")


    async def worker_process(self, job, token):
        descriptions = JobDescription.VALS.value
        user_id = job.data['user_id']
        request = job.data['request']
        or_token = job.data['token']

        config = self._config_from_job_data(job.data)
        config.baseurl_v2 = job.data.get('baseurl_v2')
        openreview_client_v2 = openreview.api.OpenReviewClient(token=or_token, baseurl=config.baseurl_v2)

        # Step 1: Create dataset
        if not await self._create_dataset_step(job, config, openreview_client_v2):
            return

        config.cloud_id = f"{job.id}-{int(time.time() * 1000)}"
        machine_type = self.compute_machine_type_from_dataset(config)
        self.logger.info(f"Machine type for {job.id}: {machine_type}")

        # Persist cloud_id back into the BullMQ job data so endpoints can locate the GCS artifact.
        job.data = {**job.data, 'config': config.to_json()}

        # Step 2: Upload dataset
        dataset_gcs_path = await self._upload_dataset_step(job, config)
        if dataset_gcs_path is None:
            return

        asyncio.run_coroutine_threadsafe(job.log(f'Task 2: submitting Vertex AI PipelineJob (tier={machine_type})'), self.queue_loop)

        gcp_regions = config.regions or self.server_config.get('GCP_REGIONS', [self.cloud.region])

        # Step 3: Submit cloud job (the region loop is fully handled inside this step function)
        try:
            cloud_job = self._create_pipeline_job_step(
                job, request, user_id, machine_type, dataset_gcs_path, config, gcp_regions
            )
            active_region = config.cloud_region
        except Exception as e:
            await self._update_job_status(job, JobStatus.ERROR, str(e), error=str(e))
            raise e

        # Step 4: Poll the cloud job to completion
        success = await self._poll_pipeline_job_step(job, user_id, config, active_region)
        if not success:
            msg = f"Pipeline job {config.cloud_id} failed in region {active_region}"
            await self._update_job_status(job, JobStatus.ERROR, msg, error=msg)
            raise Exception(msg)    
    def start_expertise(self, request, client):
        descriptions = JobDescription.VALS.value

        job_name = self._get_job_name(request)
        request_log = self._get_log_from_request(request)

        request_key = self.get_key_from_request(request)

        with self._submit_lock:
            if request_key in self._pending_request_keys:
                raise openreview.OpenReviewException("Request already in process")
            self._pending_request_keys.add(request_key)

        try:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.queue.getJobs([
                        'active',
                        'delayed',
                        'paused',
                        'waiting',
                        'waiting-children',
                        'prioritized',
                    ]),
                    self.queue_loop,
                )
                jobs = future.result()
            except Exception as e:
                jobs = []

            for job in jobs:
                if job.data.get('status') == JobStatus.COMPLETED:
                    continue
                if job.data.get('request_key') == request_key:
                    raise openreview.OpenReviewException("Request already in queue")

            config = self._validate_request(client, deepcopy(request))
            config.mdate = int(time.time() * 1000)
            config.cloud_id = f"{config.job_id}-{int(time.time() * 1000)}"
            self._save_config(config)

            config_log = self._get_log_from_config(config)
            self.logger.info(f"Adding job {config.job_id} to queue with cloud_id {config.cloud_id}")

            future = asyncio.run_coroutine_threadsafe(
                self.queue.add(
                    job_name,
                    {
                        "request": request,
                        "request_key": request_key,
                        "user_id": config.user_id,
                        "token": client.token,
                        "status": JobStatus.QUEUED,
                        "description": descriptions[JobStatus.QUEUED],
                        "config": config.to_json(),
                        "api_request": config.api_request.to_json(),
                        "baseurl_v2": config.baseurl_v2,
                    },
                    {
                        'jobId': config.job_id,
                        'attempts': self.worker_attempts,
                        'backoff': {
                            'delay': self.worker_backoff_delay,
                            'type': 'exponential',
                        },
                        'removeOnComplete': {
                            'age': self.bullmq_remove_on_complete_age
                        },
                        'removeOnFail': {
                            'age': self.bullmq_remove_on_fail_age
                        },
                    }
                ),
                self.queue_loop
            )
            self.logger.info(f"Job {job_name} queued")
            job = future.result()

            future = asyncio.run_coroutine_threadsafe(job.log(request_log), self.queue_loop)
            future.result()

            future = asyncio.run_coroutine_threadsafe(job.log(config_log), self.queue_loop)
            future.result()

            return config.job_id
        finally:
            with self._submit_lock:
                self._pending_request_keys.discard(request_key)

    def get_expertise_results(self, user_id, job_id, delete_on_get=False):
        """
        Gets the scores of a given job
        If delete_on_get is set, delete the directory after the scores are fetched

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param job_id: ID of the specific job to fetch
        :type job_id: str

        :param delete_on_get: A flag indicating whether or not to clean up the directory after it is fetched
        :type delete_on_get: bool

        :returns: A dictionary that contains the calculated scores and metadata
        """
        job = self._load_job_from_queue(user_id, job_id)
        cloud_id = job.data.get('config', {}).get('cloud_id')
        if not cloud_id:
            raise openreview.OpenReviewException(f"Cloud ID not found for job {job_id}")
        return self.cloud.get_job_results(user_id, cloud_id, delete_on_get)

    def get_expertise_metadata(self, user_id, job_id):
        """
        Gets the dataset metadata for a given job (submission/archive counts,
        missing profiles and publications) by reading metadata.json from GCS.
        """
        job = self._load_job_from_queue(user_id, job_id)
        cloud_id = job.data.get('config', {}).get('cloud_id')
        if not cloud_id:
            raise openreview.OpenReviewException(f"Cloud ID not found for job {job_id}")
        return self.cloud.get_job_metadata(user_id, cloud_id)

    def get_expertise_signed_url(self, user_id, job_id, sparse=False):
        """Return a signed URL for the results file of a cloud job."""
        job = self._load_job_from_queue(user_id, job_id)
        cloud_id = job.data.get('config', {}).get('cloud_id')
        if not cloud_id:
            raise openreview.OpenReviewException(f"Cloud ID not found for job {job_id}")
        return self.cloud.get_job_results_signed_url(user_id, cloud_id, sparse=sparse)
