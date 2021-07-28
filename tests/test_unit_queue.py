import sys, time, os
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