import shortuuid
import shutil
import time
import os
import json
from csv import reader
import openreview
from openreview import OpenReviewException
from enum import Enum
from threading import Lock

SUPERUSER_IDS = ['openreview.net']
user_index_file_lock = Lock()

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
        self.server_config = config
        self.working_dir = config['WORKING_DIR']
        self.specter_dir = config['SPECTER_DIR']
        self.mfr_feature_vocab_file = config['MFR_VOCAB_DIR']
        self.mfr_checkpoint_dir = config['MFR_CHECKPOINT_DIR']

        # Define expected/required API fields
        self.req_fields = ['name', 'match_group', 'paper_invitation', 'user_id', 'job_id']
        self.optional_model_params = ['use_title', 'use_abstract', 'average_score', 'max_score', 'skip_specter']
        self.optional_fields = ['model', 'model_params', 'exclusion_inv', 'token', 'baseurl']
        self.path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']

    def _get_default_config(self):
        return self.server_config['DEFAULT_CONFIG']

    def _filter_config(self, running_config):
        """
        Filters out certain server-side fields of a config file in order to
        form a presentable config to the user

        :param running_config: Contains the config JSON as read from the servver
        :type running_config: dict

        :returns config: A modified version of config without the server fields
        """
        remove_fields = ['baseurl', 'token', 'user_id']
        for key in remove_fields:
            del running_config[key]

        return running_config

    def _prepare_config(self, request) -> dict:
        """
        Overwrites/add specific key-value pairs in the submitted job config
        :param request: Contains the initial request from the user
        :type request: dict

        :returns config: A modified version of config with the server-required fields

        :raises Exception: Raises exceptions when a required field is missing, or when a parameter is provided
                        when it is not expected
        """
        # Validate fields
        config = self._validate_fields(request)
        self.logger.info(f"Config validation passed - setting server-side fields")

        # Populate with server-side fields
        root_dir = os.path.join(self.working_dir, request['job_id'])
        descriptions = JobDescription.VALS.value
        config['dataset']['directory'] = root_dir
        for field in self.path_fields:
            config['model_params'][field] = root_dir
        config['job_dir'] = root_dir
        config['cdate'] = int(time.time())
        config['status'] = JobStatus.INITIALIZED.value
        config['description'] = descriptions[JobStatus.INITIALIZED]

        # Set SPECTER+MFR paths
        if 'specter' in config.get('model', 'specter+mfr'):
            config['model_params']['specter_dir'] = self.specter_dir
        if 'mfr' in config.get('model', 'specter+mfr'):
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

    def _validate_fields(self, request) -> dict:
        """
        If the server has detected any errors in the proposed config, assemble the error and raise the exception
        Otherwise, this function returns the properly augmented config

        :param request: Contains the initial request from the user
        :type request: dict

        :returns config: A server-compatible version of the user request

        :raises Exception: If the request is missing a required field, contains an unexpected field or an
                           unexpected model param
        """
        config = self._get_default_config()

        # Populate fields
        failed_request = False
        error_fields = {
            'required': [],
            'unexpected': [],
            'model_params': []
        }
        for field in self.req_fields:
            if field not in request:
                error_fields['required'].append(field)
                failed_request = True
                continue
            config[field] = request[field]
        for field in request.keys():
            if field not in self.optional_fields and field not in self.req_fields:
                error_fields['unexpected'].append(field)
                failed_request = True
                continue
            if field != 'model_params':
                # Only write to config if 1) overwriting a default with non-None value or 2) if the field is not in the default config
                if (field in config.keys() and request[field] is not None) or field not in config.keys():
                    config[field] = request[field]
        if 'model_params' in request.keys():
            for field in request['model_params']:
                if field not in self.optional_model_params:
                    error_fields['model_params'].append(field)
                    failed_request = True
                    continue
                # Only write to config if 1) overwriting a default with non-None value or 2) if the field is not in the default config
                if (field in config['model_params'].keys() and request['model_params'][field] is not None) or field not in config['model_params'].keys():
                    config['model_params'][field] = request['model_params'][field]

        if failed_request:
            error_string = 'Bad request: '
            if len(error_fields['required']) > 0:
                error_string += 'missing required field: ' + ' '.join(error_fields['required']) + '\n'
            if len(error_fields['unexpected']) > 0:
                error_string += 'unexpected field: ' + ' '.join(error_fields['unexpected']) + '\n'
            if len(error_fields['model_params']) > 0:
                error_string += 'unexpected model param: ' + ' '.join(error_fields['model_params']) + '\n'
            raise OpenReviewException(error_string.strip())
        else:
            return config

    def _get_subdirs(self, user_id):
        """
        Returns the direct children directories of the given root directory

        :returns: A list of subdirectories not prefixed by the given root directory
        """
        subdirs = [name for name in os.listdir(self.working_dir) if os.path.isdir(os.path.join(self.working_dir, name))]
        if user_id.lower() in SUPERUSER_IDS:
            return subdirs
        else:
            # If given a profile ID, assume looking for job dirs that contain a config with the
            # matching profile id
            filtered_dirs = self._get_from_user_index(user_id)
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

        # Look for files
        if os.path.isfile(os.path.join(search_dir, f"{config['name']}.csv")):
            file_dir = os.path.join(search_dir, f"{config['name']}.csv")
        if file_dir is None:
            raise OpenReviewException("Score file not found for job {job_id}".format(job_id=config["job_id"]))

        if os.path.isfile(os.path.join(search_dir, 'metadata.json')):
            metadata_dir = os.path.join(search_dir, 'metadata.json')
        if metadata_dir is None:
            raise OpenReviewException("Metadata file not found for job {job_id}".format(job_id=config["job_id"]))

        return file_dir, metadata_dir

    def _add_to_user_index(self, user_id, job_id):
        """
        Records that a valid job has been submitted under the given user ID

        :param user_id: The user ID that the job was submitted by
        :type user_id: str

        :param job_id: The ID of the job to be added to the record
        :type job_id: str
        """
        # Load existing index, otherwise initialize and empty index
        with user_index_file_lock:
            index_path = os.path.join(self.working_dir, 'index.json')
            if os.path.isfile(index_path):
                with open(os.path.join(self.working_dir, 'index.json'), 'r') as f:
                    index = json.load(f)
            else:
                index = {}

            # Add job_id to the user_id list in the index dict
            if user_id not in index.keys():
                index[user_id] = [job_id]
            else:
                index[user_id].append(job_id)
        
        # Write out the index
        with open(os.path.join(self.working_dir, 'index.json'), 'w+') as f:
            json.dump(index, f, ensure_ascii=False, indent=4)
    
    def _get_from_user_index(self, user_id):
        """
        Fetch a list of submitted job IDs for a given user ID

        :param user_id: The user ID that the jobs were submitted by
        :type user_id: str

        :param job_id: The ID of the job to be added to the record
        :type job_id: str

        :returns jobs: A list of strings, each of which is an ID for a job submitted by the user
        """
        # Load existing index
        with user_index_file_lock:
            index_path = os.path.join(self.working_dir, 'index.json')
            if os.path.isfile(index_path):
                with open(os.path.join(self.working_dir, 'index.json'), 'r') as f:
                    index = json.load(f)
            else:
                raise OpenReviewException('Bad request: no jobs have been submitted yet')

            # Return the entire list of job IDs
            if user_id in index.keys():
                return index[user_id]
            else:
                raise OpenReviewException('User not found: no jobs submitted with this user ID')
    
    def _del_from_user_index(self, user_id, job_id):
        """
        Removes a job ID from the record

        :param user_id: The user ID that the job was submitted by
        :type user_id: str

        :param job_id: The ID of the job to be added to the record
        :type job_id: str
        """
        # Load existing index, otherwise throw an error
        with user_index_file_lock:
            index_path = os.path.join(self.working_dir, 'index.json')
            if os.path.isfile(index_path):
                with open(os.path.join(self.working_dir, 'index.json'), 'r') as f:
                    index = json.load(f)
            else:
                raise OpenReviewException('Bad request: no jobs have been submitted yet')

            # Remove the job ID from the list
            if user_id in index.keys():
                index[user_id].remove(job_id)
            else:
                raise OpenReviewException('User not found: no jobs submitted with this user ID')
        
        # Write out the index
        with open(os.path.join(self.working_dir, 'index.json'), 'w+') as f:
            json.dump(index, f, ensure_ascii=False, indent=4)

    def start_expertise(self, request):
        descriptions = JobDescription.VALS.value
        job_id = shortuuid.ShortUUID().random(length=5)
        request['job_id'] = job_id

        from .celery_tasks import run_userpaper
        config = self._prepare_config(request)

        self.logger.info(f'Config: {config}')
        config['status'] = JobStatus.QUEUED
        config['description'] = descriptions[JobStatus.QUEUED]

        # Config has passed validation - add it to the user index
        self._add_to_user_index(config['user_id'], config['job_id'])
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
        result = {'results': []}

        job_subdirs = self._get_subdirs(user_id)
        self.logger.info(f"Searching {job_subdirs} for user {user_id}")
        # If given an ID, only get the status of the single job
        if job_id is not None:
            job_subdirs = [name for name in job_subdirs if name == job_id]

        for job_dir in job_subdirs:
            search_dir = os.path.join(self.working_dir, job_dir)

            # Load the config file to fetch the job name and status
            self.logger.info(f"Attempting to load {search_dir}/config.json")
            with open(os.path.join(search_dir, 'config.json'), 'r') as f:
                s = f"{''.join(f.readlines())}"
                config = json.loads(s)
            status = config['status']
            description = config['description']
            
            # Append filtered config to the status
            filtered_config = self._filter_config(config)
            result['results'].append(
                {
                    'job_id': job_dir,
                    'name': config['name'],
                    'status': status,
                    'description': description,
                    'config': filtered_config
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
        result = {'results': []}

        search_dir = os.path.join(self.working_dir, job_id)
        self.logger.info(f"Checking if {job_id} belongs to {user_id}")
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

        self.logger.info(f"Able to access job at {job_id} - checking if scores are found")
        # Assemble scores
        if status != JobStatus.COMPLETED:
            ## TODO: change it to Job not found
            raise openreview.OpenReviewException(f"Scores not found - status: {status} | description: {description}")
        else:
            # Search for scores files (only non-sparse scores)
            file_dir, metadata_dir = self._get_score_and_metadata_dir(search_dir)
            self.logger.info(f"Retrieving scores from {search_dir}")
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
            self._del_from_user_index(config['user_id'], config['job_id'])
            self.logger.info(f'Deleting {search_dir}')
            shutil.rmtree(search_dir)

        return result