import time, os, random, json, sys
from unittest.mock import patch, MagicMock
from pytest import *
from dataclasses import dataclass
from expertise.service.queue import *
from expertise.service.or_queue import *

# Import mock client from test_create_dataset
def mock_client():
    client = MagicMock(openreview.Client)

    def get_group(group_id):
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(confirmedEmails=None, ids=None, term=None):
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        profiles = data['profiles']
        profiles_dict_emails = {}
        profiles_dict_tilde = {}
        for profile in profiles:
            profile = openreview.Profile.from_json(profile)
            if profile.content.get('emails') and len(profile.content.get('emails')):
                profiles_dict_emails[profile.content['emails'][0]] = profile
            profiles_dict_tilde[profile.id] = profile
        if confirmedEmails:
            return_value = {}
            for email in confirmedEmails:
                if profiles_dict_emails.get(email, False):
                    return_value[email] = profiles_dict_emails[email]

        if ids:
            return_value = []
            for tilde_id in ids:
                return_value.append(profiles_dict_tilde[tilde_id])
        return return_value

    client.get_group = MagicMock(side_effect=get_group)
    client.search_profiles = MagicMock(side_effect=search_profiles)

    return client

class SleepQueue(JobQueue):
    """A queue that runs a job that sleeps for 3 seconds from the config"""
    def run_job(self, config: dict) -> None:
        # Expect key in config: filename
        # DEBUG: add error throwing when recognizing key
        if 'throw_error_inner' in config.keys():
            raise Exception('Debug Error')
        time.sleep(3)
        fname = os.path.join(config['dataset']['directory'], 'job_done.txt')
        with open(fname, 'a+') as f:
            f.write('job done!')

    def get_result(self, user_id: str, delete_on_get: bool = False, job_id: str = '', job_name: str = '') -> List[dict]:
        ret_val: str = ''
        # Retrieve the single job data object
        matching_jobs: List[JobData] = self._get_job_data(user_id, job_id=job_id, job_name=job_name)
        assert len(matching_jobs) == 1
        current_job = matching_jobs[0]

        # Build the return list
        path = os.path.join(current_job.config['dataset']['directory'], 'job_done.txt')
        with open(path, 'r') as csv_file:
            data_reader = reader(csv_file)
            for row in data_reader:
                ret_val = row[0]
        
        # Check flag and clear directory
        job_path = current_job.config['dataset']['directory']
        if delete_on_get:
            shutil.rmtree(job_path)
        return ret_val

# Create a new TwoStepQueue for testing
class TestPaperQueue(TwoStepQueue):
    """
    Keeps track of queue metadata and is responsible for queuing jobs when given a config for getting the data for the expertise model
    """
    def _prepare_job(self, job_info: JobData) -> None:
        """
        Given a job object, modify its config to work with the server after its job ID is assigned

        :param job_info: A JobData object whose fields will be modified with respect to the job ID
        :type job_info: JobData
        """
        # Overwrite certain keys in the config
        filepath_keys = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
        file_keys = ['csv_expertise', 'csv_submissions']

        # Filter keys by membership in the config
        if 'model_params' not in job_info.config.keys():
            job_info.config['model_params'] = {}
        filepath_keys = [key for key in filepath_keys if key in job_info.config['model_params'].keys()]
        file_keys = [key for key in file_keys if key in job_info.config.keys()]

        # First handle dataset -> directory
        if 'dataset' not in job_info.config.keys():
            job_info.config['dataset'] = {}
            job_info.config['dataset']['directory'] = f"./{job_info.job_id}"
        else:
            if 'directory' not in job_info.config['dataset'].keys():
                job_info.config['dataset']['directory'] = f"./{job_info.job_id}"
            else:
                job_info.config['dataset']['directory'] = JobQueue._augment_path(job_info, job_info.config['dataset']['directory'])

        if not os.path.isdir(job_info.config['dataset']['directory']):
            os.makedirs(job_info.config['dataset']['directory'])

        # Next handle other file paths
        for key in filepath_keys:
            job_info.config['model_params'][key] = JobQueue._augment_path(job_info, job_info.config['model_params'][key])
        
        # Now, write data stored in the file keys to disk
        for key in file_keys:
            output_file = key + '.csv'
            write_to_dir = os.path.join(job_info.config['dataset']['directory'], output_file)

            # Add newline characters, write to file and set the field in the config to the directory of the file
            for idx, data in enumerate(job_info.config[key]):
                job_info.config[key][idx] = data.strip() + '\n'
            with open(write_to_dir, 'w') as csv_out:
                csv_out.writelines(job_info.config[key])
            
            job_info.config[key] = output_file
        
        # Set SPECTER+MFR paths
        job_info.config['model_params']['specter_dir'] = '../expertise-utils/specter/'
        job_info.config['model_params']['mfr_feature_vocab_file'] = '../expertise-utils/multifacet_recommender/feature_vocab_file'
        job_info.config['model_params']['mfr_checkpoint_dir'] = '../expertise-utils/multifacet_recommender/mfr_model_checkpoint/'


    def run_job(self, config: dict) -> None:
        """The actual work, set of functions to be run in a subprocess from the _handle_job thread"""
        # DEBUG: add error throwing when recognizing key
        if 'throw_error' in config.keys():
            raise Exception('Debug Error')
        openreview_client = mock_client()
        execute_create_dataset(openreview_client, config_file=config)

def test_create_dataset_and_sleep():
    """Replicate test_get_submissions from test_create_dataset"""
    config = {
        'dataset': {
            'directory': 'tests/data/'
        },
        'csv_submissions': 'csv_submissions.csv'
    }
    # Parse csv_submissions into list of csv strings
    csv_list = []
    with open(os.path.join(config['dataset']['directory'], config['csv_submissions']), 'r') as f:
        csv_list = f.readlines()
    config['csv_submissions'] = csv_list
    id = 'test_create_dataset_and_sleep'
    name = 'create_dataset_and_sleep'
    next_job = JobData(id, name, config)
    server_queue = TestPaperQueue(max_jobs = 1, inner_queue = SleepQueue)
    server_queue.put_job(next_job)
    # Needs to keep running otherwise daemon threads will shutdown
    time.sleep(4)

    # Check create dataset results created
    result_dir = os.path.join(os.getcwd(), config['dataset']['directory'])
    assert os.path.isfile(os.path.join(result_dir, 'submissions.json'))
    with open(os.path.join(result_dir, 'submissions.json'), 'r') as output:
        submissions_json = json.load(output)
    assert json.dumps(submissions_json) == json.dumps({
        'GhJKSuij': {
            "id": "GhJKSuij",
            "content": {
                "title": "Manual & mechan traction",
                "abstract":"Etiam vel augue. Vestibulum rutrum rutrum neque. Aenean auctor gravida sem."
                }
            },
        'KAeiq76y': {
            "id": "KAeiq76y",
            "content": {
                "title": "Aorta resection & anast",
                "abstract":"Morbi non lectus. Aliquam sit amet diam in magna bibendum imperdiet. Nullam orci pede, venenatis non, sodales sed, tincidunt eu, felis.Fusce posuere felis sed lacus. Morbi sem mauris, laoreet ut, rhoncus aliquet, pulvinar sed, nisl. Nunc rhoncus dui vel sem."
                }
            }
    })

    # Clean up queue log and clean up job one
    os.remove('queue.log')
    shutil.rmtree(os.path.join(os.getcwd(), next_job.job_id))

def test_create_dataset_with_outer_error():
    """Pass in a faulty config (error in outer queue) and expect directory to automatically cleanup"""
    config = {'dataset': {
            'directory': 'tests/data/'
        },
        'throw_error': True
    }
    id = 'test_create_dataset_with_outer_error'
    name = 'create_dataset_with_outer_error'
    next_job = JobData(id, name, config)
    server_queue = TestPaperQueue(max_jobs = 1, inner_queue = SleepQueue)
    server_queue.put_job(next_job)
    # Needs to keep running otherwise daemon threads will shutdown
    time.sleep(4)

    # Check create dataset results created
    result_dir = os.path.join(os.getcwd(), config['dataset']['directory'])
    assert os.path.isfile(os.path.join(result_dir)) == False
    statuses = server_queue.get_status(id, job_name = name)
    assert statuses['inner'] == 'blocked'
    assert statuses['outer'] == 'error'

    # Clean up queue log and clean up job one
    os.remove('queue.log')

def test_create_dataset_with_inner_error():
    """Pass in a faulty config (error in inner queue) and expect directory to automatically cleanup"""
    config = {'dataset': {
            'directory': 'tests/data/'
        },
        'throw_error_inner': True
    }
    id = 'test_create_dataset_with_inner_error'
    name = 'create_dataset_with_inner_error'
    next_job = JobData(id, name, config)
    server_queue = TestPaperQueue(max_jobs = 1, inner_queue = SleepQueue)
    server_queue.put_job(next_job)
    # Needs to keep running otherwise daemon threads will shutdown
    time.sleep(4)

    # Check create dataset results created
    result_dir = os.path.join(os.getcwd(), config['dataset']['directory'])
    assert os.path.isfile(os.path.join(result_dir)) == False
    statuses = server_queue.get_status(id, job_name = name)
    assert statuses['inner'] == 'error'
    assert statuses['outer'] == 'completed'

    # Clean up queue log and clean up job one
    os.remove('queue.log')

def test_create_dataset_and_specter_mfr():
    """Submit a job to a create_dataset and specter+mfr queue"""
    config = {
        'name': 'test_run',
        'match_group': ["ABC.cc"],
        'dataset': {
            'directory': 'tests/data/'
        },
        'csv_submissions': 'csv_submissions.csv',
        'model': 'specter+mfr',
        "model_params": {
            "sparse_value": 300,
            "specter_dir": "../../specter/",
            "average_score": False,
            "max_score": True,
            "specter_batch_size": 16,
            "publications_path": "tests/data/",
            "submissions_path": "tests/data/",
            "mfr_feature_vocab_file": "multifacet_recommender/multifacet_recommender_data/feature_vocab_file",
            "mfr_checkpoint_dir": "multifacet_recommender/multifacet_recommender_data/mfr_model_checkpoint/",
            "mfr_epochs": 5,
            "mfr_batch_size": 50,
            "merge_alpha": 0.8,
            "work_dir": "tests/data/",
            "use_cuda": False,
            "scores_path": "tests/data/"
        }
    }
    # Filesystem setup - Parse csv_submissions into list of csv strings
    csv_list = []
    with open(os.path.join(config['dataset']['directory'], config['csv_submissions']), 'r') as f:
        csv_list = f.readlines()
    config['csv_submissions'] = csv_list
    shutil.copytree('tests/data/archives', '1/tests/data/archives') # This will get generated by a real API call
    id = 'test_create_dataset_and_sleep'
    name = 'test_run'
    next_job = JobData(id, name, config)
    server_queue = TestPaperQueue(max_jobs = 1, inner_queue = ExpertiseQueue)
    server_queue.put_job(next_job)

    # Needs to keep running otherwise daemon threads will shutdown
    time.sleep(150)

    # Check results
    statuses = server_queue.get_status(id, job_name = name)
    assert statuses['inner'] == 'completed'
    assert statuses['outer'] == 'completed'

    res = server_queue.get_result(id, False, job_id = next_job.job_id)
    ground_truth = []
    with open('tests/data/specter_mfr.csv', 'r') as f:
        ground_truth = f.readlines()
    assert 'results' in res.keys()

    for idx, item in enumerate(res['results']):
        gt_item = ground_truth[idx].split(',')
        assert item['submission'] == gt_item[0]
        assert item['user'] == gt_item[1]
        assert item['score'] == float(gt_item[2])
    
    # Check for cleanup
    res = server_queue.get_result(id, True, job_id = next_job.job_id)
    os.remove('queue.log')

def test_two_create_dataset_and_specter_mfr():
    """Submit multiple jobs, check statuses and canceling"""
    config = {
        'name': 'test_run',
        'match_group': ["ABC.cc"],
        'dataset': {
            'directory': 'tests/data/'
        },
        'csv_submissions': 'csv_submissions.csv',
        'model': 'specter+mfr',
        "model_params": {
            "sparse_value": 300,
            "specter_dir": "../../specter/",
            "average_score": False,
            "max_score": True,
            "specter_batch_size": 16,
            "publications_path": "tests/data/",
            "submissions_path": "tests/data/",
            "mfr_feature_vocab_file": "multifacet_recommender/multifacet_recommender_data/feature_vocab_file",
            "mfr_checkpoint_dir": "multifacet_recommender/multifacet_recommender_data/mfr_model_checkpoint/",
            "mfr_epochs": 5,
            "mfr_batch_size": 50,
            "merge_alpha": 0.8,
            "work_dir": "tests/data/",
            "use_cuda": False,
            "scores_path": "tests/data/"
        }
    }
    # Filesystem setup - Parse csv_submissions into list of csv strings
    csv_list = []
    with open(os.path.join(config['dataset']['directory'], config['csv_submissions']), 'r') as f:
        csv_list = f.readlines()
    config['csv_submissions'] = csv_list
    shutil.copytree('tests/data/archives', '1/tests/data/archives') # This will get generated by a real API call
    shutil.copytree('tests/data/archives', '2/tests/data/archives') # This will get generated by a real API call
    server_queue = TestPaperQueue(max_jobs = 1, inner_queue = ExpertiseQueue)
    ids = [f'user{i}' for i in range(3)]
    names = [f'job{i}' for i in range(3)]
    jobs = [JobData(ids[i], names[i], deepcopy(config)) for i in range(3)]
    for job in jobs:
        server_queue.put_job(job)

    # Check queued statuses
    statuses = server_queue.get_status(ids[0], job_name = names[0])
    assert statuses['inner'] == 'blocked'
    assert statuses['outer'] == 'processing'
    statuses = server_queue.get_status(ids[1], job_name = names[1])
    assert statuses['inner'] == 'blocked'
    assert statuses['outer'] == 'queued'

    # Send cancel signal
    server_queue.cancel_job(ids[2], job_name = names[2])

    # Needs to keep running otherwise daemon threads will shutdown
    time.sleep(150)

    assert os.path.isdir(f'./{jobs[0].job_id}')
    assert os.path.isdir(f'./{jobs[1].job_id}')
    assert not os.path.isdir(f'./{jobs[2].job_id}')

    res_0 = server_queue.get_result(ids[0], True, job_id = jobs[0].job_id)
    res_1 = server_queue.get_result(ids[1], True, job_id = jobs[1].job_id)
    for idx in range(len((res_0['results']))):
        item_0, item_1 = res_0['results'][idx], res_1['results'][idx]
        assert item_0['submission'] == item_1['submission']
        assert item_0['user'] == item_0['user']
        assert item_0['score'] == item_0['score']
    
    # Check for cleanup
    assert not os.path.isdir(f'./{jobs[0].job_id}')
    assert not os.path.isdir(f'./{jobs[1].job_id}')
    assert not os.path.isdir(f'./{jobs[2].job_id}')

    os.remove('queue.log')
