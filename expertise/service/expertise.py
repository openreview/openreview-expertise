import shortuuid
import shutil
import time
import os
import json
from csv import reader
import openreview
from openreview import OpenReviewException
from enum import Enum

SUPERUSER_IDS = ['openreview.net']

class JobStatus(str, Enum):
    INITIALIZED = 'Initialized'
    QUEUED = 'Queued'
    FETCHING_DATA  = 'Fetching Data'
    EXPERTISE_QUEUED = 'Queued for Expertise'
    RUN_EXPERTISE = 'Running Expertise'
    COMPLETED = 'Completed'
    ERROR = 'Error'

class JobDescription(dict, Enum):
    VALS = {
        JobStatus.INITIALIZED: 'Server received config and allocated space',
        JobStatus.QUEUED: 'Job is waiting to start fetching OpenReview data',
        JobStatus.FETCHING_DATA: 'Job is currently fetching data from OpenReview',
        JobStatus.EXPERTISE_QUEUED: 'Job has assembled the data and is waiting in queue for the expertise model',
        JobStatus.RUN_EXPERTISE: 'Job is running the selected expertise model to compute scores',
        JobStatus.COMPLETED: 'Job is complete and the computed scores are ready',
    }

class ExpertiseService(object):

    def __init__(self, client, config, logger):
        self.client = client
        self.logger = logger
        self.working_dir = config['WORKING_DIR']
        self.specter_dir = config['SPECTER_DIR']
        self.mfr_feature_vocab_file = config['MFR_VOCAB_DIR']
        self.mfr_checkpoint_dir = config['MFR_CHECKPOINT_DIR']


    def _prepare_config(self, request):
        """
        Overwrites/add specific key-value pairs in the submitted job config

        :param config: Configuration fields for creating the dataset and executing the expertise model
        :type config: dict

        :param job_id: The ID for the job to be submitted
        :type job_id: str

        :param profile_id: The OpenReview profile ID associated with the job
        :type profile_id: str

        :returns new_config: A modified version of config with the server-required fields

        :raises Exception: Raises exceptions when a required field is missing, or when a parameter is provided
                        when it is not expected
        """
        config = {
            "dataset": {},
            "model": "specter+mfr",
            "model_params": {
                "use_title": True,
                "batch_size": 4,
                "use_abstract": True,
                "average_score": False,
                "max_score": True,
                "skip_specter": False,
                "use_cuda": False
            }
        }
        # Define expected/required API fields
        req_fields = ['name', 'match_group', 'paper_invitation', 'user_id', 'job_id']
        optional_model_params = ['use_title', 'use_abstract', 'average_score', 'max_score', 'skip_specter']
        optional_fields = ['model', 'model_params', 'exclusion_inv', 'token', 'baseurl']
        path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']

        # Populate fields
        failed_request = False
        error_fields = {
            'required': [],
            'unexpected': [],
            'model_params': []
        }
        for field in req_fields:
            if field not in request:
                error_fields['required'].append(field)
                failed_request = True
                continue
            config[field] = request[field]
        for field in request.keys():
            if field not in optional_fields and field not in req_fields:
                error_fields['unexpected'].append(field)
                failed_request = True
                continue
            if field != 'model_params':
                config[field] = request[field]
        if 'model_params' in request.keys():
            for field in request['model_params']:
                if field not in optional_model_params:
                    error_fields['model_params'].append(field)
                    failed_request = True
                    continue
                config['model_params'][field] = request['model_params'][field]

        # Validate fields
        error_string = 'Bad request: '
        if len(error_fields['required']) > 0:
            error_string += 'missing required field: ' + ' '.join(error_fields['required']) + '\n'
        if len(error_fields['unexpected']) > 0:
            error_string += 'unexpected field: ' + ' '.join(error_fields['unexpected']) + '\n'
        if len(error_fields['model_params']) > 0:
            error_string += 'unexpected model param: ' + ' '.join(error_fields['model_params']) + '\n'

        if failed_request:
            raise OpenReviewException(error_string.strip())

        # Populate with server-side fields
        root_dir = os.path.join(self.working_dir, request['job_id'])
        descriptions = JobDescription.VALS.value
        config['dataset']['directory'] = root_dir
        for field in path_fields:
            config['model_params'][field] = root_dir
        config['job_dir'] = root_dir
        config['cdate'] = int(time.time())
        config['status'] = JobStatus.INITIALIZED.value
        config['description'] = descriptions[JobStatus.INITIALIZED]

        # Set SPECTER+MFR paths
        if config.get('model', 'specter+mfr') == 'specter+mfr':
            config['model_params']['specter_dir'] = self.specter_dir
            config['model_params']['mfr_feature_vocab_file'] = self.mfr_feature_vocab_file
            config['model_params']['mfr_checkpoint_dir'] = self.mfr_checkpoint_dir

        # Create directory and config file
        if not os.path.isdir(config['dataset']['directory']):
            os.makedirs(config['dataset']['directory'])
        with open(os.path.join(root_dir, 'config.json'), 'w+') as f:
            ## Remove the token before saving this in the file system
            token = config.get('token', None)
            if token is not None:
                del config['token']
                json.dump(config, f, ensure_ascii=False, indent=4)
                config['token'] = token
            else:
                json.dump(config, f, ensure_ascii=False, indent=4)

        return config

    def _get_subdirs(self, user_id=None):
        """
        Returns the direct children directories of the given root directory

        :returns: A list of subdirectories not prefixed by the given root directory
        """
        subdirs = [name for name in os.listdir(self.working_dir) if os.path.isdir(os.path.join(self.working_dir, name))]
        if user_id is None:
            return subdirs
        else:
            # If given a profile ID, assume looking for job dirs that contain a config with the
            # matching profile id
            filtered_dirs = []
            for subdir in subdirs:
                config_dir = os.path.join(self.working_dir, subdir, 'config.json')
                with open(config_dir, 'r') as f:
                    config = json.load(f)
                    if user_id == config['user_id'] or user_id.lower() in SUPERUSER_IDS:
                        filtered_dirs.append(subdir)
            return filtered_dirs

    def _get_score_and_metadata_dir(self, search_dir):
        """
        Searches the given directory for a possible score file and the metadata file

        :param search_dir: The root directory to search in
        :type search_dir: str

        :returns file_dir: The directory of the score file, if it exists, starting from the given directory
        :returns metadata_dir: The directory of the metadata file, if it exists, starting from the given directory
        """
        # Search for scores files (only non-sparse scores)
        file_dir, metadata_dir = None, None
        with open(os.path.join(search_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        # Look for score files
        for root, dirs, files in os.walk(search_dir, topdown=False):
            for name in files:
                if name == f"{config['name']}.csv":
                    file_dir = os.path.join(root, name)
                if 'metadata' in name:
                    metadata_dir = os.path.join(root, name)
        return file_dir, metadata_dir

    def start_expertise(self, request):
        descriptions = JobDescription.VALS.value
        job_id = shortuuid.ShortUUID().random(length=5)
        request['job_id'] = job_id

        from .celery_tasks import run_userpaper
        config = self._prepare_config(request)

        self.logger.info(f'Config: {config}')
        config['status'] = JobStatus.QUEUED
        config['description'] = descriptions[JobStatus.QUEUED]
        run_userpaper.apply_async(
            (config, self.logger),
            queue='userpaper',
            task_id=job_id
        )
        with open(os.path.join(config['job_dir'], 'config.json'), 'w+') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        return job_id

    def get_expertise_status(self, user_id, job_id=None):
        """
        Searches the server for all jobs submitted by a user
        If a job ID is provided, only fetch the status of this job

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param job_id: Optional ID of the specific job to look up
        :type job_id: str

        :returns: A dictionary with the key 'results' containing a list of job statuses
        """
        # Perform a walk of all job sub-directories for score files
        # TODO: This walks through all submitted jobs and requires reading a file per job
        # TODO: is this what we want to do?

        result = {}
        result['results'] = []
        descriptions = JobDescription.VALS.value

        job_subdirs = self._get_subdirs(user_id)
        # If given an ID, only get the status of the single job
        if job_id is not None:
            job_subdirs = [name for name in job_subdirs if name == job_id]

        for job_dir in job_subdirs:
            search_dir = os.path.join(self.working_dir, job_dir)

            # Load the config file to fetch the job name and status
            with open(os.path.join(search_dir, 'config.json'), 'r') as f:
                config = json.load(f)
            status = config['status']
            description = config['description']

            result['results'].append(
                {
                    'job_id': job_dir,
                    'name': config['name'],
                    'status': status,
                    'description': description
                }
            )
        return result

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
        result = {}
        result['results'] = []
        descriptions = JobDescription.VALS.value

        search_dir = os.path.join(self.working_dir, job_id)
        # Check for directory existence
        if not os.path.isdir(search_dir):
            raise openreview.OpenReviewException('Job not found')

        # Validate profile ID
        with open(os.path.join(search_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        if user_id != config['user_id'] and user_id.lower() not in SUPERUSER_IDS:
            raise OpenReviewException("Forbidden: Insufficient permissions to access job")

        # Fetch status
        status = config['status']
        description = config['description']

        # Assemble scores
        if status != JobStatus.COMPLETED:
            ## TODO: change it to Job not found
            raise openreview.OpenReviewException(f"Scores not found - status: {status} | description: {description}")
        else:
            # Search for scores files (only non-sparse scores)
            file_dir, metadata_dir = self._get_score_and_metadata_dir(search_dir)

            ret_list = []
            with open(file_dir, 'r') as csv_file:
                data_reader = reader(csv_file)
                for row in data_reader:
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
            self.logger.error(f'Deleting {search_dir}')
            shutil.rmtree(search_dir)

        return result