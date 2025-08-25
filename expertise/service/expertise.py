import shortuuid
import shutil
import time
import os
import json
import torch
import gc
from csv import reader
import openreview
from openreview import OpenReviewException
from enum import Enum
from threading import Lock
from pathlib import Path
import multiprocessing
from bullmq import Queue, Worker
from expertise.execute_expertise import execute_create_dataset, execute_expertise
from expertise.service.utils import GCPInterface
from expertise.create_dataset import OpenReviewExpertise
from expertise.config import ModelConfig
from copy import deepcopy
import asyncio
import threading
import traceback

from .utils import JobConfig, APIRequest, JobDescription, JobStatus, SUPERUSER_IDS, RedisDatabase, get_user_id

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
        :param sync_on_disk:   Whether RedisDatabase writes files to disk or purely memory
        :param worker_attempts: (Optional) number of attempts for the BullMQ worker
        :param worker_backoff_delay: (Optional) backoff delay 2 ^ attempts * delay (ms) for the BullMQ worker
        :param worker_concurrency: (Optional) concurrency for the BullMQ worker
        :param worker_lock_duration: (Optional) lock duration (ms) for the BullMQ worker
        :param worker_autorun: (Optional) whether the worker should start automatically
        """
        self.logger = logger
        self.server_config = config
        self.containerized = containerized
        self.sync_on_disk = sync_on_disk  # Whether to actually save jobs on disk (for Redis usage)
        self.default_expertise_config = config.get('DEFAULT_CONFIG')
        self.worker_attempts = worker_attempts
        self.worker_backoff_delay = worker_backoff_delay
        self.working_dir = config.get('WORKING_DIR')
        self.specter_dir = config.get('SPECTER_DIR')
        self.mfr_feature_vocab_file = config.get('MFR_VOCAB_DIR')
        self.mfr_checkpoint_dir = config.get('MFR_CHECKPOINT_DIR')

        # If using Redis to store job configs, initialize it (unless containerized means "no local Redis")
        if not containerized:
            self.redis = RedisDatabase(
                host=config['REDIS_ADDR'],
                port=config['REDIS_PORT'],
                db=config['REDIS_CONFIG_DB'],
                sync_on_disk=self.sync_on_disk
            )

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

        worker_settings = {
            'prefix': 'bullmq:expertise',
            'connection': {
                "host": config['REDIS_ADDR'],
                "port": config['REDIS_PORT'],
                "db": config['REDIS_CONFIG_DB'],
            },
            'autorun': worker_autorun
        }
        if worker_concurrency is not None:
            worker_settings['concurrency'] = worker_concurrency
        if worker_lock_duration is not None:
            worker_settings['lockDuration'] = worker_lock_duration

        self.worker = Worker(
            'Expertise',
            self.worker_process,
            worker_settings
        )
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

    def set_client(self, client):
        self.client = client

    def set_client_v2(self, client_v2):
        self.client_v2 = client_v2

    def start_queue_in_thread(self):
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.queue_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_event_loop, args=(self.queue_loop,), daemon=True)
        thread.start()

    def start_worker_in_thread(self):
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_event_loop, args=(loop,), daemon=True)
        thread.start()

        # Actually schedule the worker to run
        asyncio.run_coroutine_threadsafe(self.worker.run(), loop)

    async def close(self):
        await self.worker.close()
        await self.queue.close()

    def worker_process(self, job, token):
        """
        Override in child classes
        """
        raise NotImplementedError("worker_process must be implemented in a child class.")

    def update_status(self, config, new_status, desc=None):
        """
        Common logic for updating a job's status in Redis (if not containerized).
        """
        # from .utils import JobDescription, JobStatus  # Typically you’d import these at top
        descriptions = JobDescription.VALS.value
        config.status = new_status

        # Check for paper-paper-scoring
        paper_scoring = config.api_request.entityA.get('type') == 'Note' and config.api_request.entityB.get('type') == 'Note'

        if desc is None:
            config.description = descriptions[new_status]
        else:
            # Example: special text for certain known exceptions
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
            config.description = desc

        config.mdate = int(time.time() * 1000)

        # Save job if we have a Redis instance
        if not self.containerized:
            self.redis.save_job(config)

    def get_expertise_all_status(self, user_id, query_params):
        """
        Searches the server for all jobs submitted by a user that satisfies
        the HTTP GET query parameters
        """
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

        def check_result():
            return False not in [
                check_status(),
                check_member(),
                check_invitation(),
                check_paper_id()
            ]

        result = {'results': []}
        query_obj = {}
        '''
        {
            'paperId': value,
            'entityA': {
                'id': value
            }
        }
        '''

        for query, value in query_params.items():
            if query.find('.') < 0: ## If no entity, store value
                query_obj[query] = value
            else:
                entity, query_by = query.split('.') ## If entity, store value in entity obj
                if entity not in query_obj.keys():
                    query_obj[entity] = {}
                query_obj[entity][query_by] = value

        self.logger.info(f"Searching for jobs with query: {query_obj}")
        for config in self.redis.load_all_jobs(user_id):
            self.logger.info(f"{config.job_id} - {config.to_json()}")
            status = config.status
            description = config.description

            if check_result():
                # Append filtered config to the status
                self._filter_config(config)
                result['results'].append(
                    {
                        'name': config.name,
                        'tauthor': config.user_id,
                        'jobId': config.job_id,
                        'status': status,
                        'description': description,
                        'cdate': config.cdate,
                        'mdate': config.mdate,
                        'request': config.api_request.to_json()
                    }
                )

        # Sort results by cdate
        result['results'] = sorted(result['results'], key=lambda x: x['cdate'], reverse=True)

        return result

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

    def _prepare_config(self, request, job_id=None, client_v1=None, client=None) -> dict:
        """
        Overwrites/add specific key-value pairs in the submitted job config
        :param request: Contains the initial request from the user
        :type request: dict

        :param job_id: If provided, use this job ID instead of generating a new one
        :type job_id: str

        :returns config: A modified version of config with the server-required fields

        :raises Exception: Raises exceptions when a required field is missing, or when a parameter is provided
                        when it is not expected
        """

        if job_id:
            try:
                job = self.redis.load_job(job_id, get_user_id(self.client_v2))
                return job, self.client.token
            except Exception as e:
                if 'not found' not in str(e):
                    raise e

        # Validate fields
        or_client_v1 = client_v1 if client_v1 else self.client
        or_client = client if client else self.client_v2

        self.logger.info(f"Incoming request - {request}")
        validated_request = APIRequest(request)
        config = JobConfig.from_request(
            api_request = validated_request,
            job_id=job_id,
            starting_config = self.default_expertise_config,
            openreview_client= or_client_v1,
            openreview_client_v2= or_client,
            server_config = self.server_config,
            working_dir = self.working_dir
        )
        self.logger.info(f"Config validation passed - {config.to_json()}")

        # Create directory and config file
        if not os.path.isdir(config.dataset['directory']):
            os.makedirs(config.dataset['directory'])
        with open(os.path.join(config.job_dir, 'config.json'), 'w+') as f:
            json.dump(config.to_json(), f, ensure_ascii=False, indent=4)
        if not self.containerized:
            self.logger.info(f"Saving processed config to {os.path.join(config.job_dir, 'config.json')}")
            self.redis.save_job(config)

        return config, or_client.token

    def _get_subdirs(self, user_id):
        """
        Returns the direct children directories of the given root directory

        :returns: A list of subdirectories not prefixed by the given root directory
        """
        subdirs = [name for name in os.listdir(self.working_dir) if os.path.isdir(os.path.join(self.working_dir, name))]
        if user_id.lower() in SUPERUSER_IDS:
            return subdirs

        # Search all directories for matching user ID
        filtered_dirs = []
        for job_dir in subdirs:
            with open(os.path.join(self.working_dir, job_dir, 'config.json')) as f:
                config = JobConfig.from_json(json.load(f))
            if config.user_id == user_id:
                filtered_dirs.append(job_dir)

        return filtered_dirs

    def _get_score_and_metadata_dir(self, search_dir, group_scoring=False, paper_scoring=False):
        """
        Searches the given directory for a possible score file and the metadata file

        :param search_dir: The root directory to search in
        :type search_dir: str

        :param group_scoring: Indicate if scoring between groups, if so skip sparse scores
        :type group_scoring: bool

        :param paper_scoring: Indicate if scoring between papers, if so skip sparse scores
        :type paper_scoring: bool

        :returns file_dir: The directory of the score file, if it exists, starting from the given directory
        :returns metadata_dir: The directory of the metadata file, if it exists, starting from the given directory
        """
        # Search for scores files (if sparse scores exist, retrieve by default)
        file_dir, metadata_dir = None, None
        skip_sparse = group_scoring or paper_scoring
        with open(os.path.join(search_dir, 'config.json'), 'r') as f:
            config = JobConfig.from_json(json.load(f))

        # Look for files
        if os.path.isfile(os.path.join(search_dir, f"{config.name}.csv")):
            file_dir = os.path.join(search_dir, f"{config.name}.csv")
            if not skip_sparse:
                if 'sparse_value' in config.model_params.keys() and os.path.isfile(os.path.join(search_dir, f"{config.name}_sparse.csv")):
                    file_dir = os.path.join(search_dir, f"{config.name}_sparse.csv")
                else:
                    raise OpenReviewException("Sparse score file not found for job {job_id}".format(job_id=config.job_id))    
        else:
            raise OpenReviewException("Score file not found for job {job_id}".format(job_id=config.job_id))

        if os.path.isfile(os.path.join(search_dir, 'metadata.json')):
            metadata_dir = os.path.join(search_dir, 'metadata.json')
        else:
            raise OpenReviewException("Metadata file not found for job {job_id}".format(job_id=config.job_id))

        return file_dir, metadata_dir

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
        job_id = job.data['job_id']
        user_id = job.data['user_id']
        config = self.redis.load_job(job_id, user_id)
        or_token = job.data['token']
        openreview_client = openreview.Client(
            token=or_token,
            baseurl=config.baseurl
        )
        openreview_client_v2 = openreview.api.OpenReviewClient(
            token=or_token,
            baseurl=config.baseurl_v2
        )
        try:
            # Create dataset
            execute_create_dataset(openreview_client, openreview_client_v2, config=config.to_json())
            self.update_status(config, JobStatus.RUN_EXPERTISE)

            queue = multiprocessing.Queue()  # Queue for exception handling
            config_json = json.dumps(config.to_json())  # Serialize config
            process = multiprocessing.Process(target=BaseExpertiseService.expertise_worker, args=(config_json, queue))
            process.start()
            process.join()

            if not queue.empty():
                exception = queue.get()
                raise exception  # Re-raise the exception from the subprocess

            # Update job status
            self.update_status(config, JobStatus.COMPLETED)

        except Exception as e:
            self.update_status(config, JobStatus.ERROR, str(e))
            # Re raise exception so that it appears in the queue
            exception = e.with_traceback(e.__traceback__)
            raise exception
        finally:
            # Cleanup resources
            torch.cuda.empty_cache()
            gc.collect()

    def start_expertise(self, request, client_v1, client):
        descriptions = JobDescription.VALS.value

        job_name = self._get_job_name(request)
        request_log = self._get_log_from_request(request)

        request_key = self.get_key_from_request(request)

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
            if job.data.get('request_key') == request_key:
                raise openreview.OpenReviewException("Request already in queue")

        config, token = self._prepare_config(request, client_v1=client_v1, client=client)
        job_id = config.job_id

        config_log = self._get_log_from_config(config)

        config.mdate = int(time.time() * 1000)
        config.status = JobStatus.QUEUED
        config.description = descriptions[JobStatus.QUEUED]

        # Config has passed validation - add it to the user index
        self.logger.info('just before submitting')

        self.logger.info(f"\nconf: {config.to_json()}\n")
        self.redis.save_job(config)

        future = asyncio.run_coroutine_threadsafe(
            self.queue.add(
                job_name,
                {
                    "job_id": job_id,
                    "request_key": request_key,
                    "user_id": config.user_id,
                    "token": token
                },
                {
                    'jobId': job_id,
                    'attempts': self.worker_attempts,
                    'backoff': {
                        'delay': self.worker_backoff_delay,
                        'type': 'exponential', # Exponential backoff: 2 ^ attempts * delay milliseconds
                    },
                    'removeOnComplete': {
                        'count': 100,
                    },
                    'removeOnFail': {
                        'age': 2592000
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
        config = self.redis.load_job(job_id, user_id)
        status = config.status
        description = config.description
        
        # Append filtered config to the status
        self._filter_config(config)
        return {
            'name': config.name,
            'tauthor': config.user_id,
            'jobId': config.job_id,
            'status': status,
            'description': description,
            'cdate': config.cdate,
            'mdate': config.mdate,
            'request': config.api_request.to_json()
        }

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

        # Get and validate profile ID
        config = self.redis.load_job(job_id, user_id)

        # Fetch status
        status = config.status
        description = config.description

        self.logger.info(f"{user_id} able to access job at {job_id} - checking if scores are found")
        # Assemble scores
        if status != JobStatus.COMPLETED:
            raise openreview.OpenReviewException(f"Scores not found - status: {status} | description: {description}")
        else:
            # Search for scores files (if sparse scores exist, retrieve by default)
            ret_list = []

            # Check for output format
            group_group_matching = config.alternate_match_group is not None
            paper_paper_matching = config.api_request.entityA.get('type') == 'Note' and config.api_request.entityB.get('type') == 'Note'

            self.logger.info(f"Retrieving scores from {config.job_dir}")
            if group_group_matching:
                # If group-group matching, report results using "*_member" keys
                file_dir, metadata_dir = self._get_score_and_metadata_dir(config.job_dir, group_scoring=True)
                with open(file_dir, 'r') as csv_file:
                    data_reader = reader(csv_file)
                    for row in data_reader:
                        ret_list.append({
                            'match_member': row[0],
                            'submission_member': row[1],
                            'score': float(row[2])
                        })
                result['results'] = ret_list
            elif paper_paper_matching:
                # If paper-paper matching, report results using submission keywords
                file_dir, metadata_dir = self._get_score_and_metadata_dir(config.job_dir, paper_scoring=True)
                with open(file_dir, 'r') as csv_file:
                    data_reader = reader(csv_file)
                    for row in data_reader:
                        ret_list.append({
                            'match_submission': row[0],
                            'submission': row[1],
                            'score': float(row[2])
                        })
                result['results'] = ret_list
            else:
                # If reviewer-paper matching, use standard 'user' and 'score' keys
                file_dir, metadata_dir = self._get_score_and_metadata_dir(config.job_dir)
                with open(file_dir, 'r') as csv_file:
                    data_reader = reader(csv_file)
                    for row in data_reader:
                        # For single paper retrieval, filter out scores against the dummy submission
                        if row[0] == 'dummy':
                            continue

                        ret_list.append({
                            'submission': row[0],
                            'user': row[1],
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
            self.redis.remove_job(user_id, job_id)

        return result

    def del_expertise_job(self, user_id, job_id):
        """
        Returns the filtered config of a job and deletes the job directory

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param job_id: ID of the specific job to look up
        :type job_id: str

        :returns: Filtered config of the job to be deleted
        """
        config = self.redis.load_job(job_id, user_id)
        
        # Clear directory and Redis entry
        self.logger.info(f"Deleting {config.job_dir} for {user_id}")
        if os.path.isdir(config.job_dir):
            shutil.rmtree(config.job_dir)
        else:
            self.logger.info(f"No files found - only removing Redis entry")
        self.redis.remove_job(user_id, job_id)

        # Return filtered config
        self._filter_config(config)
        return config.to_json()

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

    def set_client_v2(self, client_v2):
        self.client_v2 = client_v2
        self.cloud.set_client(client_v2)

    def compute_machine_type(self, client, client_v2, api_request):
        config, _ = self._prepare_config(deepcopy(api_request), client_v1=client, client=client_v2)
        if config.machine_type is not None:
            return config.machine_type
        config = config.to_json()
        dataset_config = ModelConfig(config_dict=config)
        expertise = OpenReviewExpertise(
            client,
            client_v2,
            dataset_config
        )
        note_count = 0

        # Counts submissions (from one venue or both if doing paper-paper scoring)
        # and/or alternate group publications if doing group-group scoring
        # TODO: Decide on what count for what threshold

        #if 'match_group' in config or 'reviewer_ids' in self.config:
        #    expertise = self.retrieve_expertise()
        #    for pubs in expertise.values():
        #        note_count += len(pubs)

        if 'match_paper_invitation' in config or 'match_paper_id' in config or 'match_paper_venueid' in config or 'match_paper_content' in config:
            invitation_ids = expertise.convert_to_list(expertise.config.get('match_paper_invitation', []))
            paper_id = expertise.config.get('match_paper_id')
            paper_venueid = expertise.config.get('match_paper_venueid', None)
            paper_content = expertise.config.get('match_paper_content', None)

            reduced_submissions = expertise.get_submissions_helper(
                invitation_ids=invitation_ids,
                paper_id=paper_id,
                paper_venueid=paper_venueid,
                paper_content=paper_content
            )

            note_count += len(reduced_submissions)

        # Retrieve match groups to detect group-group matching
        group_group_matching = 'alternate_match_group' in config.keys()

        # if invitation ID is supplied, collect records for each submission
        if 'paper_invitation' in config or 'csv_submissions' in config or 'paper_id' in config or 'paper_venueid' in config or group_group_matching:
            invitation_ids = expertise.convert_to_list(expertise.config.get('paper_invitation', []))
            paper_id = expertise.config.get('paper_id')
            paper_venueid = expertise.config.get('paper_venueid', None)
            paper_content = expertise.config.get('paper_content', None)
            submission_groups = expertise.convert_to_list(expertise.config.get('alternate_match_group', []))

            reduced_submissions = expertise.get_submissions_helper(
                invitation_ids=invitation_ids,
                paper_id=paper_id,
                paper_venueid=paper_venueid,
                paper_content=paper_content,
                submission_groups=submission_groups
            )

            note_count += len(reduced_submissions)

        if note_count < self.server_config.get('PIPELINE_MEDIUM_THRESHOLD'):
            return 'small'
        elif note_count < self.server_config.get('PIPELINE_LARGE_THRESHOLD'):
            return 'medium'
        else:
            return 'large'

    async def worker_process(self, job, token):
        descriptions = JobDescription.VALS.value
        user_id = job.data['user_id']
        request = job.data['request']
        redis_id = job.data['redis_id']
        or_token = job.data['token']

        config = self.redis.load_job(redis_id, user_id)
        openreview_client_v1 = openreview.Client(
            token=or_token,
            baseurl=config.baseurl
        )
        openreview_client_v2 = openreview.api.OpenReviewClient(
            token=or_token,
            baseurl=config.baseurl_v2
        )
        try:
            machine_type = self.compute_machine_type(
                openreview_client_v1,
                openreview_client_v2,
                deepcopy(request)
            )

            cloud_id = self.cloud.create_job(
                deepcopy(request),
                client=openreview_client_v2,
                user_id = user_id,
                machine_type=machine_type
            )
        except Exception as e:
            self.logger.error(f"Error creating cloud job for {redis_id}: {e} tr={e.__traceback__}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            config = self.redis.load_job(redis_id, user_id)
            if config.status != JobStatus.ERROR:
                self.update_status(config, JobStatus.ERROR, f"Error creating cloud job: {e}")
            # If we fail to create the job, we should not proceed with polling
            # Re-raise exception to appear in the queue
            raise e.with_traceback(e.__traceback__)
        config.mdate = int(time.time() * 1000)
        config.status = JobStatus.QUEUED
        config.description = descriptions[JobStatus.QUEUED]
        config.cloud_id = cloud_id
        self.redis.save_job(config)

        try:
            self.logger.info(f"In polling worker...")
            for attempt in range(self.max_attempts):
                self.logger.info(f"{redis_id} - attempt {attempt + 1} of {self.max_attempts}...")
                status = self.cloud.get_job_status_by_job_id(user_id, cloud_id)
                self.logger.info(f"Invoked get_job_status_by_job_id for {redis_id} - status: {status}")

                # Check status validity
                self.logger.info(f"INFO: before status check")
                if status and isinstance(status, dict) and 'status' in status and 'description' in status:
                    self.logger.info(f"INFO: after status check")
                    config = self.redis.load_job(redis_id, user_id)
                    self.logger.info(f"INFO: after load job")
                    # Only update non-stale status
                    if config.status != status['status'] or config.description != status['description']:
                        self.logger.info(f"INFO: before update status")
                        self.update_status(config, status['status'], status['description'])
                        self.logger.info(f"INFO: after update status")

                    if status['status'] == JobStatus.COMPLETED:
                        self.logger.info(f"Job {redis_id} completed successfully.")
                        break # Exit the loop on successful completion

                    elif status['status'] == JobStatus.ERROR:
                        self.logger.error(f"Job {redis_id} encountered an error: {status['description']}")
                        raise Exception(f"Job {redis_id} failed: {status['description']}")
                    self.logger.info(f"Job {redis_id} status: {status['status']}. Waiting {self.poll_interval} seconds before next poll...")

                else:
                    self.logger.warning(f"Invalid or missing status received for job {redis_id}. Retrying...")

                self.logger.info(f"INFO: before sleep")
                await asyncio.sleep(self.poll_interval)
                self.logger.info(f"INFO: after sleep")

            # If the loop completes without a break, raise timeout
            else:
                self.logger.warning(f"Polling timed out after {self.max_attempts} attempts for job {redis_id}.")
                config = self.redis.load_job(redis_id, user_id)
                if config.status != JobStatus.ERROR:
                    self.update_status(config, JobStatus.ERROR, f"Polling timed out after {self.max_attempts} attempts.")
                raise TimeoutError(f"Polling timed out for job {redis_id} after {self.max_attempts} attempts.")

            self.logger.info(f"Polling loop finished for job {redis_id}.")

        except Exception as e:
            # Re-raise exception to appear in the queue
            raise e.with_traceback(e.__traceback__)

    def start_expertise(self, request, client_v1, client):
        descriptions = JobDescription.VALS.value

        job_name = self._get_job_name(request)
        request_log = self._get_log_from_request(request)

        request_key = self.get_key_from_request(request)

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
            if job.data.get('request_key') == request_key:
                raise openreview.OpenReviewException("Request already in queue")

        config, _ = self._prepare_config(deepcopy(request), client_v1=client_v1, client=client)
        config.mdate = int(time.time() * 1000)
        config.status = JobStatus.QUEUED
        config.description = descriptions[JobStatus.QUEUED]
        self.redis.save_job(config)

        config_log = self._get_log_from_config(config)
        self.logger.info(f"Adding job {config.job_id} to queue")

        future = asyncio.run_coroutine_threadsafe(
            self.queue.add(
                job_name,
                {
                    "request": request,
                    "request_key": request_key,
                    "user_id": config.user_id,
                    "redis_id": config.job_id,
                    "token": client.token
                },
                {
                    'jobId': config.job_id,
                    'attempts': self.worker_attempts,
                    'backoff': {
                        'delay': self.worker_backoff_delay,
                        'type': 'exponential', # Exponential backoff: 2 ^ attempts * delay milliseconds
                    },
                    'removeOnComplete': {
                        'count': 100,
                    },
                    'removeOnFail': {
                        'age': 2592000
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
        redis_job = self.redis.load_job(job_id, user_id)
        if redis_job.cloud_id is None:
            raise openreview.OpenReviewException(f"Not Found error: Job {job_id} has not requested resources")
        cloud_return = self.cloud.get_job_status_by_job_id(user_id, redis_job.cloud_id)
        cloud_return['name'] = redis_job.name
        cloud_return['jobId'] = redis_job.job_id
        return cloud_return

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
        redis_job = self.redis.load_job(job_id, user_id)
        return self.cloud.get_job_results(user_id, redis_job.cloud_id, delete_on_get)

    def del_expertise_job(self, user_id, job_id):
        """
        Returns the filtered config of a job and deletes the job directory

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param job_id: ID of the specific job to look up
        :type job_id: str

        :returns: Filtered config of the job to be deleted
        """
        config = self.redis.load_job(job_id, user_id)
        
        # Clear directory and Redis entry
        self.logger.info(f"Deleting {config.job_dir} for {user_id}")
        if os.path.isdir(config.job_dir):
            shutil.rmtree(config.job_dir)
        else:
            self.logger.info(f"No files found - only removing Redis entry")
        self.redis.remove_job(user_id, job_id)

        # Return filtered config
        self._filter_config(config)
        return config.to_json()