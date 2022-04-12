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

from .utils import JobConfig, APIRequest, JobDescription, JobStatus, SUPERUSER_IDS

user_index_file_lock = Lock()
class ExpertiseService(object):

    def __init__(self, client, config, logger):
        self.client = client
        self.logger = logger
        self.server_config = config
        self.default_expertise_config = config['DEFAULT_CONFIG']
        self.working_dir = config['WORKING_DIR']
        self.specter_dir = config['SPECTER_DIR']
        self.mfr_feature_vocab_file = config['MFR_VOCAB_DIR']
        self.mfr_checkpoint_dir = config['MFR_CHECKPOINT_DIR']
        self.redis_args = {
            'host': config['REDIS_ADDR'],
            'port': config['REDIS_PORT'],
            'db': config['REDIS_DB']
        }

        # Define expected/required API fields
        self.req_fields = ['name', 'match_group', 'user_id', 'job_id']
        self.optional_model_params = ['use_title', 'use_abstract', 'average_score', 'max_score', 'skip_specter']
        self.optional_fields = ['model', 'model_params', 'exclusion_inv', 'token', 'baseurl', 'baseurl_v2', 'paper_invitation', 'paper_id']
        self.path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']

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
        validated_request = APIRequest(request)
        config = JobConfig.from_request(
            api_request = validated_request,
            starting_config = self.default_expertise_config,
            openreview_client= self.client,
            server_config = self.server_config,
            working_dir = self.working_dir
        )
        self.logger.info(f"Config validation passed - {config.to_json()}")

        # Create directory and config file
        if not os.path.isdir(config.dataset['directory']):
            os.makedirs(config.dataset['directory'])
        with open(os.path.join(config.job_dir, 'config.json'), 'w+') as f:
            json.dump(config.to_json(), f, ensure_ascii=False, indent=4)
        self.logger.info(f"Saving processed config to {os.path.join(config.job_dir, 'config.json')}")
        config.save(self.redis_args)

        return config, self.client.token

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
            config = JobConfig.from_json(json.load(f))

        # Look for files
        if os.path.isfile(os.path.join(search_dir, f"{config.name}.csv")):
            file_dir = os.path.join(search_dir, f"{config.name}.csv")
        else:
            raise OpenReviewException("Score file not found for job {job_id}".format(job_id=config.job_id))

        if os.path.isfile(os.path.join(search_dir, 'metadata.json')):
            metadata_dir = os.path.join(search_dir, 'metadata.json')
        else:
            raise OpenReviewException("Metadata file not found for job {job_id}".format(job_id=config.job_id))

        return file_dir, metadata_dir

    def start_expertise(self, request):
        descriptions = JobDescription.VALS.value

        from .celery_tasks import run_userpaper
        config, token = self._prepare_config(request)
        job_id = config.job_id

        config.mdate = int(time.time() * 1000)
        config.status = JobStatus.QUEUED
        config.description = descriptions[JobStatus.QUEUED]

        # Config has passed validation - add it to the user index
        self.logger.info('just before submitting')
        run_userpaper.apply_async(
            (config, token, self.logger, self.redis_args),
            queue='userpaper',
            task_id=job_id
        )
        self.logger.info(f"\nconf: {config.to_json()}\n")
        config.save(self.redis_args)

        return job_id

    def get_expertise_all_status(self, user_id, query_params):
        """
        Searches the server for all jobs submitted by a user

        :param user_id: The ID of the user accessing the data
        :type user_id: str

        :param query_params: Query parameters of the GET request
        :type query_params: dict

        :returns: A dictionary with the key 'results' containing a list of job statuses
        """
        result = {'results': []}
        search_status = query_params.get('status')

        for config in JobConfig.load_all_jobs(user_id, self.redis_args):
            status = config.status
            description = config.description

            if not search_status or status.lower().startswith(search_status.lower()):
                # Append filtered config to the status
                self._filter_config(config)
                result['results'].append(
                    {
                        'job_id': job_dir,
                        'name': config.name,
                        'status': status,
                        'description': description,
                        'cdate': config.cdate,
                        'mdate': config.mdate,
                        'config': config.to_json()
                    }
                )
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
        config = JobConfig.load_job(job_id, user_id, self.redis_args)
        status = config.status
        description = config.description
        
        # Append filtered config to the status
        self._filter_config(config)
        return {
            'job_id': config.job_dir,
            'name': config.name,
            'status': status,
            'description': description,
            'cdate': config.cdate,
            'mdate': config.mdate,
            'config': config.to_json()
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
        config = JobConfig.load_job(job_id, user_id, self.redis_args)

        # Fetch status
        status = config.status
        description = config.description

        self.logger.info(f"{user_id} able to access job at {job_id} - checking if scores are found")
        # Assemble scores
        if status != JobStatus.COMPLETED:
            raise openreview.OpenReviewException(f"Scores not found - status: {status} | description: {description}")
        else:
            # Search for scores files (only non-sparse scores)
            file_dir, metadata_dir = self._get_score_and_metadata_dir(config.job_dir)
            self.logger.info(f"Retrieving scores from {config.job_dir}")
            ret_list = []

            # Check for output format
            group_group_matching = config.alternate_match_group is not None

            if not group_group_matching:
                # If reviewer-paper matching, use standard 'user' and 'score' keys
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
            else:
                # If group-group matching, report results using "*_member" keys
                with open(file_dir, 'r') as csv_file:
                    data_reader = reader(csv_file)
                    for row in data_reader:
                        ret_list.append({
                            'match_member': row[0],
                            'submission_member': row[1],
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
            JobConfig.remove_job(user_id, job_id, self.redis_args)

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
        config = JobConfig.load_job(job_id, user_id, self.redis_args)
        
        # Clear directory and Redis entry
        self.logger.info(f"Deleting {config.job_dir} for {user_id}")
        if os.path.isdir(config.job_dir):
            shutil.rmtree(config.job_dir)
        else:
            self.logger.info(f"No files found - only removing Redis entry")
        JobConfig.remove_job(user_id, job_id, self.redis_args)

        # Return filtered config
        self._filter_config(config)
        return config.to_json()