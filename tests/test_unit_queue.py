import time, os, random
from pytest import *
from dataclasses import dataclass
from expertise.service.queue import *

@dataclass
class SleepInfo(JobData):
    """
    Keeps track of the sleep queue information and status
    Overwrites filename for server handling
    """
    def __post_init__(self) -> None:
        super().__post_init__()
        self.config['filename'] = f"./{self.job_id}/{self.config['filename']}"

class SleepQueue(JobQueue):
    """A queue that runs a job that sleeps for specified seconds from the config"""
    def run_job(self, config: dict) -> None:
        # Expect key in config: filename
        if config['sleep'] < 0:
            raise Exception('Negative sleep time not allowed')
        time.sleep(config['sleep'])
        fname = config['filename']
        with open(fname, 'w') as f:
            f.write('job done!')

    def get_result(self, user_id: str, delete_on_get: bool = False, job_id: str = '', job_name: str = '') -> List[dict]:
        ret_val: str = ''
        # Retrieve the single job data object
        matching_jobs: List[JobData] = self._get_job_data(user_id, job_id=job_id, job_name=job_name)
        assert len(matching_jobs) == 1
        current_job = matching_jobs[0]

        # Build the return list
        path = current_job.config['filename']
        with open(path, 'r') as csv_file:
            data_reader = reader(csv_file)
            for row in data_reader:
                ret_val = row[0]
        
        # Check flag and clear directory
        job_path = f'./{current_job.job_id}'
        if delete_on_get:
            shutil.rmtree(job_path)
        return ret_val

# Single job unit tests
def test_single_job_remove():
    '''Single job added, run and deleted after getting results'''
    short_queue = SleepQueue()
    # Sleep for 3 seconds then get the result
    conf = {'sleep': 3, 'filename': 'single_job_remove_job.txt'}
    id = 'test_single_job_remove'
    name = 'single_job_remove'
    next_job = SleepInfo(id, name, conf)
    short_queue.put_job(next_job)
    time.sleep(5)

    # Check directory made and output file made
    job_filename = next_job.config['filename']
    assert os.path.isdir(f'./{next_job.job_id}') == True
    assert os.path.isfile(f'{job_filename}') == True
    assert short_queue.get_result(id, job_name=name) == 'job done!'

    # Check directory cleanup
    job_filename = next_job.config['filename']
    result_and_delete: str = short_queue.get_result(id, delete_on_get=True, job_name=name)
    assert os.path.isdir(f'./{next_job.job_id}') == False
    assert os.path.isfile(f'{job_filename}') == False
    assert result_and_delete == 'job done!'

    # Now check for exception on retrieving results
    try:
        short_queue.get_result(id, delete_on_get=True, job_name=name)
        raise Exception('Retrieved results when no results should exist')
    except:
        print('Correctly raised exception')

    # Clean up queue log
    os.remove('queue.log')

def test_get_status():
    '''Queue a long running job, and query its status at several stages'''
    short_queue = SleepQueue()
    # Sleep for 3 seconds
    conf = {'sleep': 3, 'filename': 'get_status_job.txt'}
    id = 'test_get_status'
    name = 'get_status'
    next_job = SleepInfo(id, name, conf)

    # Try to get status of non-existant job
    try:
        assert short_queue.get_status(id, job_name=name) == 'queued'
        raise Exception('Detected job when no job has been enqueued')
    except:
        print('Correctly threw exception for non-existant job')
    short_queue.put_job(next_job)

    # Sleep to ensure job is scheduled and get status
    time.sleep(0.5)
    assert short_queue.get_status(id, job_name=name) == 'processing'

    # Sleep to finish job and get its status
    time.sleep(3)
    assert short_queue.get_status(id, job_name=name) == 'completed'

    # Clean up queue log and directory
    short_queue.get_result(id, delete_on_get=True, job_name=name)
    os.remove('queue.log')

def test_get_jobs():
    '''Several scenarios for get jobs'''
    short_queue = SleepQueue()
    # Sleep for 2 seconds
    conf = {'sleep': 2, 'filename': 'get_status_job.txt'}
    id = 'test_get_status'
    name = 'get_status'
    next_job = SleepInfo(id, name, conf)

    # Try to get a list of jobs for this user
    current_jobs = short_queue.get_jobs(id)
    assert len(current_jobs) == 0

    # There should now be a job in processing
    short_queue.put_job(next_job)
    current_jobs = short_queue.get_jobs(id)
    assert len(current_jobs) == 1
    current_job_data = current_jobs[0]
    assert current_job_data['job_name'] == name
    assert current_job_data['job_id'] == next_job.job_id
    assert current_job_data['status'] == 'processing'

    # Sleep and expect to find a single completed job
    time.sleep(5)
    current_jobs = short_queue.get_jobs(id)
    assert len(current_jobs) == 1
    current_job_data = current_jobs[0]
    assert current_job_data['job_name'] == name
    assert current_job_data['job_id'] == next_job.job_id
    assert current_job_data['status'] == 'completed'

    # Clean up queue log and directory
    short_queue.get_result(id, delete_on_get=True, job_name=name)
    os.remove('queue.log')

def test_timeout_job():
    '''Add a job with a timeout and check status for timeout'''
    short_queue = SleepQueue()
    # Sleep for 5 seconds and set a timeout for shorter than 5 seconds
    conf = {'sleep': 5, 'filename': 'set_timeout_job.txt'}
    id = 'test_set_timeout'
    name = 'set_timeout'
    next_job = SleepInfo(id, name, conf, timeout=3)
    short_queue.put_job(next_job)

    # Get jobs and check for timeout
    time.sleep(5)
    current_jobs = short_queue.get_jobs(id)
    assert len(current_jobs) == 1
    current_job_data = current_jobs[0]
    assert current_job_data['job_name'] == name
    assert current_job_data['job_id'] == next_job.job_id
    assert current_job_data['status'] == 'timeout'

    # Check for cleanup
    assert os.path.isdir(f'./{next_job.job_id}') == False

    # Clean up queue log and directory
    os.remove('queue.log')

def test_error_job():
    '''Add a job that is bound to return an exception'''
    short_queue = SleepQueue()
    # Sleep for negative seconds, which will throw an error for the underlying job code
    conf = {'sleep': -5, 'filename': 'illegal_config_job.txt'}
    id = 'test_illegal_config'
    name = 'illegal_config'
    next_job = SleepInfo(id, name, conf, timeout=3)
    short_queue.put_job(next_job)

    # Get jobs and check for error
    time.sleep(1)
    current_jobs = short_queue.get_jobs(id)
    assert len(current_jobs) == 1
    current_job_data = current_jobs[0]
    assert current_job_data['job_name'] == name
    assert current_job_data['job_id'] == next_job.job_id
    assert current_job_data['status'] == 'error'

    # Clean up queue log and directory
    os.remove('queue.log')

# Two job unit test
def test_two_jobs_remove():
    '''
    Enqueue two jobs for a queue that only runs 1 job at a time
    Query their statuses along the way and check for proper outputting
    '''
    short_queue = SleepQueue()
    # Create two jobs
    conf_one = {'sleep': 3, 'filename': 'job_one.txt'}
    id_one = 'test_job_one'
    name_one = 'job_one'
    job_one = SleepInfo(id_one, name_one, conf_one)
    conf_two = {'sleep': 2, 'filename': 'job_two.txt'}
    id_two = 'test_job_two'
    name_two = 'job_two'
    job_two = SleepInfo(id_two, name_two, conf_two)

    # Put jobs on queue and check initial statuses
    short_queue.put_job(job_one)
    short_queue.put_job(job_two)
    time.sleep(0.5)
    assert short_queue.get_status(id_one, job_name=name_one) == 'processing'
    assert short_queue.get_status(id_two, job_name=name_two) == 'queued'

    # Sleep for total job time and check created directories
    time.sleep(6)
    assert os.path.isdir(f'./{job_one.job_id}') == True
    assert os.path.isdir(f'./{job_two.job_id}') == True
    assert short_queue.get_status(id_one, job_name=name_one) == 'completed'
    assert short_queue.get_status(id_two, job_name=name_two) == 'completed'

    # Clean up queue log and directory
    short_queue.get_result(id_one, delete_on_get=True, job_name=name_one)
    short_queue.get_result(id_two, delete_on_get=True, job_name=name_two)
    os.remove('queue.log')

def test_cancel_job():
    '''Add a single job to sleep for a long time, cancel it, and check its status'''
    short_queue = SleepQueue()
    # Create two jobs
    conf_one = {'sleep': 3, 'filename': 'job_one.txt'}
    id_one = 'test_job_one'
    name_one = 'job_one'
    job_one = SleepInfo(id_one, name_one, conf_one)
    conf_two = {'sleep': 2, 'filename': 'job_two.txt'}
    id_two = 'test_job_two'
    name_two = 'job_two'
    job_two = SleepInfo(id_two, name_two, conf_two)

    # Put jobs on queue and check initial statuses
    short_queue.put_job(job_one)
    short_queue.put_job(job_two)
    time.sleep(0.5)
    assert short_queue.get_status(id_one, job_name=name_one) == 'processing'
    assert short_queue.get_status(id_two, job_name=name_two) == 'queued'

    # Send the cancel signal and confirm the status is changed to stale
    short_queue.cancel_job(id_two, job_name=name_two)
    assert short_queue.get_status(id_two, job_name=name_two) == 'stale'

    # Sleep for total job time and check created directories
    time.sleep(6)
    assert os.path.isdir(f'./{job_one.job_id}') == True
    assert os.path.isdir(f'./{job_two.job_id}') == False
    assert short_queue.get_status(id_one, job_name=name_one) == 'completed'
    assert short_queue.get_status(id_two, job_name=name_two) == 'stale'

    # Clean up queue log and directory
    short_queue.get_result(id_one, delete_on_get=True, job_name=name_one)
    os.remove('queue.log')

# Many job unit test
def test_many_jobs_singlethread():
    '''Enqueue many jobs to check ability to handle long workloads'''
    NUM_JOBS = 30
    many_queue = SleepQueue()
    configs: List[dict] = []
    ids: List[str] = []
    names: List[str] = []
    jobs: List[SleepInfo] = []
    for job_num in range(NUM_JOBS):
        configs.append({
            'sleep': random.random(),
            'filename': f'job_{job_num}.txt'
        })
        ids.append(f'test_job_{job_num}')
        names.append(f'job_{job_num}')
        jobs.append(SleepInfo(
            ids[job_num],
            names[job_num],
            configs[job_num]
        ))

    for job in jobs:
        many_queue.put_job(job)
    
    time.sleep(NUM_JOBS)
    for job in jobs:
        job_filepath = job.config['filename']
        assert many_queue.get_status(job.id, job_name = job.job_name) == 'completed'
        assert os.path.isdir(f'./{job.job_id}') == True
        assert os.path.isfile(f'{job_filepath}') == True
        assert many_queue.get_result(job.id, delete_on_get=True, job_name=job.job_name) == 'job done!'

    # Clean up queue log and directory
    os.remove('queue.log')

def test_many_jobs_multithread():
    '''Enqueue many jobs to check ability to handle long workloads with multithreading'''
    NUM_JOBS = 30
    many_queue = SleepQueue(max_jobs=5)
    configs: List[dict] = []
    ids: List[str] = []
    names: List[str] = []
    jobs: List[SleepInfo] = []
    for job_num in range(NUM_JOBS):
        configs.append({
            'sleep': random.random(),
            'filename': f'job_{job_num}.txt'
        })
        ids.append(f'test_job_{job_num}')
        names.append(f'job_{job_num}')
        jobs.append(SleepInfo(
            ids[job_num],
            names[job_num],
            configs[job_num]
        ))

    for job in jobs:
        many_queue.put_job(job)
    
    time.sleep(NUM_JOBS)
    for job in jobs:
        job_filepath = job.config['filename']
        assert many_queue.get_status(job.id, job_name = job.job_name) == 'completed'
        assert os.path.isdir(f'./{job.job_id}') == True
        assert os.path.isfile(f'{job_filepath}') == True
        assert many_queue.get_result(job.id, delete_on_get=True, job_name=job.job_name) == 'job done!'

    # Clean up queue log and directory
    os.remove('queue.log')