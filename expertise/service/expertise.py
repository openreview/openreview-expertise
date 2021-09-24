import shortuuid
import shutil
import time
import os
import json
from csv import reader
import openreview
from openreview import OpenReviewException
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = 'Queued'
    PROCESSING = 'Processing'
    COMPLETED = 'Completed'
    ERROR = 'Error'

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

        # Validate + populate fields
        for field in req_fields:
            if field not in request:
                raise OpenReviewException(f"Bad request: missing required field {field}")
            config[field] = request[field]

        for field in request.keys():
            if field not in optional_fields and field not in req_fields:
                raise OpenReviewException(f"Bad request: unexpected field {field}")
            if field != 'model_params':
                config[field] = request[field]

        if 'model_params' in request.keys():
            for field in request['model_params']:
                if field not in optional_model_params:
                    raise OpenReviewException(f"Bad request: unexpected model param: {field}")
                config['model_params'][field] = request['model_params'][field]

        # Populate with server-side fields
        root_dir = os.path.join(self.working_dir, request['job_id'])
        config['dataset']['directory'] = root_dir
        for field in path_fields:
            config['model_params'][field] = root_dir
        config['job_dir'] = root_dir
        config['cdate'] = int(time.time())

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
            if 'token' in config.keys():
                del config['token']
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
                    if user_id == config['user_id']:
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

        job_id = shortuuid.ShortUUID().random(length=5)
        request['job_id'] = job_id

        from .celery_tasks import run_userpaper

        config = self._prepare_config(request)

        self.logger.info(f'Config: {config}')

        run_userpaper.apply_async(
            (config, self.logger),
            queue='userpaper',
            task_id=job_id
        )

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

        job_subdirs = self._get_subdirs(user_id)
        # If given an ID, only get the status of the single job
        self.logger.info(f'check filtering | value of job_ID: {job_id}')
        if job_id is not None:
            self.logger.info(f'performing filtering')
            job_subdirs = [name for name in job_subdirs if name == job_id]
        self.logger.info(f'Subdirs: {job_subdirs}')

        for job_dir in job_subdirs:
            search_dir = os.path.join(self.working_dir, job_dir)
            self.logger.info(f'Looking at {search_dir}')
            file_dir, _ = self._get_score_and_metadata_dir(search_dir)
            err_dir = os.path.join(search_dir, 'err.log')

            # Load the config file to fetch the job name
            with open(os.path.join(search_dir, 'config.json'), 'r') as f:
                config = json.load(f)

            # Check if there has been an error - read the error and continue
            if os.path.isfile(err_dir):
                with open(err_dir, 'r') as f:
                    err = f.readline()
                err = list(err.strip().split(','))
                id, name, err = err[0], err[1], err[2]
                result['results'].append(
                    {
                        'job_id': id,
                        'name': name,
                        'status': JobStatus.ERROR.value,
                        'error': f'{err}'
                    }
                )
                continue

            self.logger.info(f'Current score status {file_dir}')
            # If found a non-sparse, non-data file CSV, job has completed
            if file_dir is None:
                status = JobStatus.PROCESSING.value
            else:
                status = JobStatus.COMPLETED.value

            # If there are no other directories, then the dataset has not been created
            # so the job is still queued
            subdirs = [name for name in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, name))]
            if len(subdirs) <= 0:
                status = JobStatus.QUEUED.value

            result['results'].append(
                {
                    'job_id': job_dir,
                    'name': config['name'],
                    'status': status
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

        # Validate profile ID
        search_dir = os.path.join(self.working_dir, job_id)
        with open(os.path.join(search_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        assert user_id == config['user_id'], "Forbidden: Insufficient permissions to access job"
        # Search for scores files (only non-sparse scores)
        file_dir, metadata_dir = self._get_score_and_metadata_dir(search_dir)

        # Assemble scores
        if file_dir is None:
            ## TODO: change it to Job not found
            raise openreview.OpenReviewException('Either job is still processing, has crashed, or does not exist')
        else:
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