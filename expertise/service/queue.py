import hashlib, json, threading, queue, os, openreview, shutil, logging
from typing import *
from dataclasses import dataclass, field
from multiprocessing import Process, TimeoutError, ProcessError
from ..execute_expertise import *
from csv import reader
@dataclass
class JobData:
    """Keeps track of job information and status"""
    id: str = field(
        metadata={"help": "The profile id at the time of submission"},
    )
    job_name: str = field(
        metadata={"help": "The name of the job specified in the submitted config file"},
    )
    config: dict = field(
        metadata={"help": "The submitted configuration file as a dictionary"},
    )
    job_id: str = field(
        default='',
        metadata={"help": "The unique id for this job"},
    )
    status: str = field(
        default='queued',
        metadata={"help": "The current status of this job"},
    )
    timeout: int = field(
        default=0,
        metadata={"help": "The maximum amount of time to run this job"},
    )

    def __post_init__(self) -> None:
        # Generate job id
        config_string = json.dumps(self.config)
        self.job_id = hashlib.md5(config_string.encode('utf-8')).hexdigest()

    def to_json(self) -> dict:
        """
        Converts JobData instance to a dictionary. The instance variable names are the keys and their values the values of the dictinary.

        :return: Dictionary containing all the parameters of a JobData instance
        :rtype: dict
        """
        return {
            'id': self.id,
            'job_name': self.job_name,
            'job_id': self.job_id,
            'config': self.config,
            'status': self.status,
            'timeout': self.timeout
        }

@dataclass
class ExpertiseInfo(JobData):
    """
    Keeps track of the create_expertise queue information and status. Dataset directory will be overwritten by the server.
    """
    def __post_init__(self) -> None:
        super().__post_init__()
        # Overwrite dataset -> directory in config
        # TODO: Overwrite all other possible directory variables with the job id
        if 'dataset' not in self.config.keys():
            self.config['dataset'] = {}
        self.config['dataset']['directory'] = f"./{self.job_id}"

@dataclass
class DatasetInfo(ExpertiseInfo):
    """
    Keeps track of the create_dataset queue information and status. Dataset directory will be overwritten by the server.
    Same information as expertise info but requires an authenticated token
    """
    token: str = field(
        default='',
        metadata={"help": "The authenticated token of the user client"}
    )
    baseurl: str = field(
        default='',
        metadata={"help": "The base URL of the API to call to log in"}
    )

class JobQueue:
    """
    Keeps track of queue metadata in-memory and is responsible for queuing jobs when given a config

    Create a subclass of a JobQueue and implement "get_result" and "run_job" with user-defined logic

    Status semantics:
        "queued" -- The job is currently awaiting processing by a worker
        "processing" -- The job is currently being worked on a by a worder
        "completed" -- The job has finished and the results are stored on the server
        "stale" -- The job has been cancelled before it arrived at processing
        "timeout" -- The job has exceeded the specified/default timeout
        "error" -- The job has run into an error 

    Important attributes:
        q -- The Python queue which from which the daemon thread pulls JobData objects
        submitted -- A list of JobData objects which have been submitted (to be updated to a redundant database like redis)
    """
    def __init__(self, max_jobs: int = 1) -> None:
        """
        Instantiates a JobQueue object using a max_jobs parameter which determines the amount of concurrent jobs that can be run which depends the type of computation
        and system resources. If no max_jobs is provided, default to 1.

        :param max_jobs: Integer of the amount of concurrent jobs
        :type max_jobs: int
        """
        self.q = queue.Queue()
        self.max_jobs: int = max_jobs
        self.submitted: List[JobData] = []
        self.lock_submitted = threading.Lock()
        self.running_semaphore = threading.BoundedSemaphore(value = max_jobs)

        # create logger with 'job_queue'
        self.logger = logging.getLogger('job_queue')
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        self.fh = logging.FileHandler('queue.log')
        self.fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(thread)d - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        self.logger.info('JobQueue successfully created')

        # Kickstart queue daemon thread
        threading.Thread(target=self._daemon, daemon=True).start()
    
    def put_job(self, request: JobData) -> None:
        """
        Adds a JobData object to the queue to be processed asynchronously
        
        :param request: A JobData object containing the metadata of the job to be executed
        :type request: JobData
        """
        self.logger.info('Putting job on queue...')
        self.submitted.append(request)
        self.q.put(request)
        self.logger.info(f'Job successfully submitted with information: {request}')
    
    def cancel_job(self, user_id: str, job_id: str = '', job_name: str = '') -> None:
        """
        For a job that is still queued, sets the status to stale to ensure that it does not get processed
        Currently, cannot cancel a job that is already in processing
        Identify the job to be canceled with a combination of the user_id and the optional arguments
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str
        """
        # Get list of job data and ensure a length of 1
        self.logger.info('Attempting to cancel job...')
        job_list: List[JobData] = self._get_job_data(user_id=user_id)
        assert len(job_list) == 1, 'Error: Multiple job matches'

        job_list[0].status = 'stale'
        self.logger.info(f'Successfully cancelled job from {user_id} with either job ID [{job_id}] or job name [{job_name}]')

    def get_jobs(self, user_id: str) -> List[dict]:
        """
        Returns a list of job names and ids of all jobs associated with the user id, and their statuses
        If no jobs found, return an empty list corresponding to no jobs

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :rtype: List[dict]
        """
        self.logger.info(f'Retrieving all jobs...')
        ret_list: List[dict] = []

        # Get list of job data
        try:
            job_list: List[JobData] = self._get_job_data(user_id=user_id)
        except:
            job_list = []
        for job in job_list:
            # Construct dicts and append to list
            ret_list.append(
                {
                    'job_name': job.job_name,
                    'job_id': job.job_id,
                    'status': job.status
                }
            )
        self.logger.info(f'Returning gathered jobs submitted by {user_id}')
        return ret_list

    def get_status(self, user_id: str, job_id: str = '', job_name: str = '') -> str:
        """
        Return the status of the job submitted by user_id with either the given job_id or job_name
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str

        :rtype: str
        """
        self.logger.info(f'Fetching status...')
        # Get list of job data and ensure a length of 1
        job_list: List[JobData] = self._get_job_data(user_id=user_id)
        assert len(job_list) == 1, 'Error: Multiple job matches'

        self.logger.info(f'Successfully fetched status from {user_id} with either job ID [{job_id}] or job name [{job_name}]')
        return job_list[0].status

    def get_result(self, user_id: str, delete_on_get: bool = True, job_id: str = '', job_name: str = '') -> List[dict]:
        """
        Return the result of the job submitted by user_id with either the given job_id or job_name
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id
        By default, deletes the data and metadata with the associated job 

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param delete_on_get: A boolean flag that decides whether or not to maintain the data and metadata from the job
        :type delete_on_get: bool

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str

        :rtype: List[dict]
        """
        raise Exception('Not Implemented Yet')

    def run_job(self, config: dict) -> None:
        """The actual work, set of functions to be run in a subprocess from the _handle_job thread"""
        raise Exception('Not Implemented Yet')
    
    # ------------ PRIVATE FUNCTIONS ------------
    def _daemon(self) -> None:
        """Job queue daemon function that continuously attempts to consume from the queue"""
        self.logger.info('Starting queue daemon - listening for queue insertions')
        while True:
            self.logger.info('Listening for queue insertions')
            self.running_semaphore.acquire()    # Blocks when the max jobs has been met
            self.logger.info('Semaphore acquired - starting thread to execute job')
            next_job_info: JobData = self.q.get()   # Blocks when queue is empty
            threading.Thread(target=self._handle_job, args=(next_job_info,)).start()
            self.logger.info('Thread spawned')
            
    def _handle_job(self, job_info: JobData) -> None:
        """Creates a process to perform the job, sleeps and kills process on wake up if process is still alive"""
        # TODO: If error or timedout, clean up working directory
        self.logger.info('Job handler thread started')
        if job_info.status != 'stale':
            # Create job directory if it doesn't exist
            self.logger.info('Making directory...')
            job_dir = os.path.join(os.getcwd(), job_info.job_id)
            if not os.path.exists(job_dir):
                os.mkdir(job_dir)

            # Spawn process to perform the job
            self.logger.info('Spawning job process from handler thread...')
            job_info.status = 'processing'
            p = Process(target=self.run_job, args=(job_info.config,))
            p.start()
            self.logger.info('Job process spawned')

            # Check timeout and execute sleep/join
            try:
                if job_info.timeout > 0:
                    p.join(job_info.timeout)
                else:
                    p.join()
                # Release semaphore and indicate job done
                self.running_semaphore.release()
                self.q.task_done()
                # If no errors detected, check exit codes
                if p.exitcode is None:
                    job_info.status = 'timeout'
                elif p.exitcode != 0:
                    job_info.status = 'error'
                else:
                    job_info.status = 'completed'
            except TimeoutError:
                job_info.status = 'timeout'
            except ProcessError:
                job_info.status = 'error'

            # If status isn't completed, delete directory
            if job_info.status != 'completed':
                shutil.rmtree(job_dir, ignore_errors=False)

            self.logger.info(f'Job process has terminated with error code {p.exitcode}')
            # If not exited, terminate the process
            if p.exitcode is None:
                p.terminate()

    def _get_job_data(self, user_id: str, job_id: str = '', job_name: str = '') -> List[JobData]:
        """
        Fetches a list of JobData objects that have been submitted by user_id with either the given job_id or job_name
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id
        If neither is provided, return all jobs associated with the user_id

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str

        :rtype: List[JobData]
        """
        ret_list: List[JobData] = []

        self.logger.info('Retrieving job data...')
        # Validate arguments
        aggregate_by_user: bool = not job_id and not job_name
        
        # Lock the submitted history and search
        self.lock_submitted.acquire()
        self.logger.info('Lock acquired on submitted datastore')
        for data in self.submitted:
            if data.id == user_id:
                if (job_id and job_id == data.job_id) or (job_name and job_name == data.job_name) or (aggregate_by_user):
                    ret_list.append(data)
        self.lock_submitted.release()
        self.logger.info('Lock released on submitted datastore')

        # If no item found, raise exception
        if not len(ret_list):
            raise Exception('No matching jobs')

        self.logger.info(f'Job data retrieved from user {user_id} with either job ID [{job_id}] or job name [{job_name}]')
        return ret_list

class ExpertiseQueue(JobQueue):
    """
    Keeps track of queue metadata and is responsible for queuing jobs when given a config for running the expertise model
    """
    def get_result(self, user_id: str, delete_on_get: bool = True, job_id: str = '', job_name: str = '') -> List[dict]:
        """
        Return the result of the job submitted by user_id with either the given job_id or job_name
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id
        By default, deletes the data and metadata with the associated job

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param delete_on_get: A boolean flag that decides whether or not to maintain the data and metadata from the job
        :type delete_on_get: bool

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str

        :rtype: dict
        """
        ret_list: List[dict] = []
        # Retrieve the single job data object
        self.logger.info('ExpertiseQueue: Retrieving results from an expertise job')
        matching_jobs: List[ExpertiseInfo] = self._get_job_data(user_id, job_id=job_id, job_name=job_name)
        assert len(matching_jobs) == 1
        current_job = matching_jobs[0]

        # Build the return list by reading the csv under the job_name.csv file
        cwd = os.getcwd()
        job_path = os.path.join(cwd, current_job.job_id)
        csv_path = os.path.join(job_path, f'{current_job.job_name}.csv')
        self.logger.info('ExpertiseQueue: Reading data from generated by the job')
        with open(csv_path, 'r') as csv_file:
            data_reader = reader(csv_file)
            for row in data_reader:
                ret_list.append({
                    'submission': row[0],
                    'user': row[1],
                    'score': float(row[2])
                })
        
        # Check flag and clear directory
        self.logger.info('ExpertiseQueue: Checking delete on get flag')
        if delete_on_get:
            shutil.rmtree(job_path)
        
        self.logger.info(f'ExpertiseQueue: Returning results from job user {user_id} with either job ID [{job_id}] or job name [{job_name}]')
        return ret_list
            
    def run_job(self, config: dict) -> None:
        """The actual work, set of functions to be run in a subprocess from the _handle_job thread"""
        execute_expertise(config_file=config)

class TwoStepQueue(JobQueue):
    """
    Manage 2 concurrent queues which consist of an outer (this object) and an inner queue
    The outer queue defers to the inner queue for get_results and must implement its own run_job

    Converts return types of certain get functions to dictionaries to allow access to information from the inner
    and outer queues simultaneously

    Jobs put into this queue perform the outer queue's task and if the task successfully completes,
    put the same job information into the inner queue
    """
    def __init__(self, max_jobs: int = 1, inner_queue = None, inner_key: str = 'inner', outer_key: str = 'outer') -> None:
        """
        Instantiates a TwoStepQueue object using a max_jobs parameter which determines the amount of concurrent jobs that can be run which depends the type of computation
        and system resources. If no max_jobs is provided, default to 1.

        Accepts a class of type JobQueue and keys that define the inner and outer queues

        :param max_jobs: Integer of the amount of concurrent jobs
        :type max_jobs: int

        :param inner_queue: A subclass of JobQueue that implements get_results and run_job
        :type inner_queue: class

        :param inner_key: The key to use to access the inner queue's information
        :type inner_key: str

        :param outer_key: The key to use to access the outer queue's information
        :type outer_key: str
        """
        super().__init__(max_jobs=max_jobs)
        self.logger.info('Initializing as TwoStepQueue...')
        self.inner_queue: JobQueue = inner_queue(max_jobs)
        self.inner_key = inner_key
        self.outer_key = outer_key

    def cancel_job(self, user_id: str, job_id: str = '', job_name: str = '') -> None:
        """
        For a job that is still queued, sets the status to stale to ensure that it does not get processed
        Currently, cannot cancel a job that is already in processing
        Identify the job to be canceled with a combination of the user_id and the optional arguments
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str
        """
        # Check job status on dataset queue and decide on action
        job_list: List[DatasetInfo] = self._get_job_data(user_id=user_id)
        assert len(job_list) == 1, 'Error: Multiple job matches'
        current_job = job_list[0]
        self.logger.info('TwoStepQueue: checking for outer queue completion')
        if current_job.status == 'completed':
            self.logger.info('TwoStepQueue: canceling inner queue job')
            self.inner_queue.cancel_job(user_id, job_id, job_name)
        else:
            self.logger.info('TwoStepQueue: canceling outer queue job')
            super().cancel_job(user_id, job_id, job_name)
        self.logger.info('TwoStepQueue: job cancelled')

    def get_jobs(self, user_id: str) -> dict:
        """
        Returns a dict of both inner and outer queue jobs created by the user

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :rtype: dict
        """
        # Query outer queue job statuses
        self.logger.info('TwoStepQueue: querying outer queue job list')
        outer_list = super().get_jobs(user_id)
        
        # Attempt to gather inner queue jobs
        self.logger.info('TwoStepQueue: querying inner queue job list')
        try:
            inner_list = self.inner_queue.get_jobs(user_id)
        except:
            inner_list = []

        # Gather queue queries
        self.logger.info(f'TwoStepQueue: gathering and returning job queries for user {user_id}')
        return {
            self.outer_key: outer_list,
            self.inner_key: inner_list
        }

    def get_status(self, user_id: str, job_id: str = '', job_name: str = '') -> dict:
        """
        Return a dict of statuses of a single job
        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str

        :rtype: dict
        """
        # Get the outer queue job status and if completed, query the inner queue status
        self.logger.info('TwoStepQueue: retrieving outer status')
        outer_status = super().get_status(user_id, job_id, job_name)
        inner_status = ''
        if outer_status == 'completed':
            self.logger.info('TwoStepQueue: outer status completed - retrieving inner status')
            inner_status = self.inner_queue.get_status(user_id, job_id, job_name)
        self.logger.info('TwoStepQueue: returning status results')
        return {
            self.inner_key: inner_status,
            self.outer_key: outer_status
        }


    def get_result(self, user_id: str, delete_on_get: bool = True, job_id: str = '', job_name: str = '') -> List[dict]:
        """
        Return the result of the job submitted by user_id with either the given job_id or job_name
        If neither the dataset nor the expertise step is not completed yet, return an empty list, otherwise query the expertise queue

        If no job_id is provided, uses job_name
        if no job_name is provided, uses job_id
        By default, deletes the data and metadata with the associated job 

        :param user_id: A string containing the user id that has submitted jobs
        :type user_id: str

        :param delete_on_get: A boolean flag that decides whether or not to maintain the data and metadata from the job
        :type delete_on_get: bool

        :param job_id: A string containing the submitted job id
        :type job_id: str

        :param job_name: A string containing the user specified name for the job
        :type job_name: str

        :rtype: List[dict]
        """
        # Return an empty list if the outer job is not completed
        self.logger.info('TwoStepQueue: fetching results - checking outer status')
        outer_status = super().get_status(user_id, job_id, job_name)
        if outer_status == 'completed':
            self.logger.info('TwoStepQueue: outer status completed - checking inner queue')
            inner_status = self.inner_queue.get_status(user_id, job_id, job_name)
            if inner_status == 'completed':
                return self.inner_queue.get_result(user_id, delete_on_get, job_id, job_name)
        self.logger.info('TwoStepQueue: No results found')
        return []
    
    def _handle_job(self, job_info: JobData) -> None:
        """Handle job - if completed, put a copy of the config on the expertise queue"""
        super()._handle_job(job_info)
        self.logger.info('TwoStepQueue: checking job status before enqueuing...')
        if job_info.status == 'completed':
            self.logger.info('TwoStepQueue: outer queue job process complete, enqueuing into inner queue')
            self.expertise_queue.put_job(job_info)

class DatasetQueue(TwoStepQueue):
    """
    Keeps track of queue metadata and is responsible for queuing jobs when given a config for getting the data for the expertise model
    """
    def put_job(self, request: DatasetInfo) -> None:
        """
        Adds a DatasetInfo object to the queue to be processed asynchronously
        Augments the request's config with the authenticated token
        
        :param request: A DatasetInfo object containing the metadata of the job to be executed
        :type request: DatasetInfo
        """
        # Update the config with the token and base URL
        self.logger.info('DatasetQueue: augmenting the request config with credentials...')
        request.config.update({
            'token': request.token,
            'baseurl': request.baseurl
        })
        self.logger.info('DatasetQueue: enqueuing the request')
        super().put_job(request)

    def run_job(self, config: dict) -> None:
        """The actual work, set of functions to be run in a subprocess from the _handle_job thread"""
        openreview_client = openreview.Client(
            token=config['token'],
            baseurl=config['baseurl']
        )
        execute_create_dataset(openreview_client, config_file=config)