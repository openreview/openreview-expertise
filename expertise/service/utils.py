import openreview, json, os, time, shutil, json, shortuuid, threading
from unittest.mock import MagicMock
from csv import reader

# -----------------
# -- Mock Client --
# -----------------
def mock_client():
    client = MagicMock(openreview.Client)

    def get_profile():
        mock_profile = {
            "id": "~Test_User1",
            "content": {
                "preferredEmail": "Test_User1@mail.com",
                "emails": [
                    "Test_User1@mail.com"
                ]
            }
        }
        return openreview.Profile.from_json(mock_profile)

    def get_notes(id = None,
        paperhash = None,
        forum = None,
        original = None,
        invitation = None,
        replyto = None,
        tauthor = None,
        signature = None,
        writer = None,
        trash = None,
        number = None,
        content = None,
        limit = None,
        offset = None,
        mintcdate = None,
        details = None,
        sort = None):

        if offset != 0:
            return []

        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        if invitation:
            notes=data['notes'][invitation]
            return [openreview.Note.from_json(note) for note in notes]

        if 'authorids' in content:
            authorid = content['authorids']
            profiles = data['profiles']
            for profile in profiles:
                if authorid == profile['id']:
                    return [openreview.Note.from_json(note) for note in profile['publications']]

        return []

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

    client.get_notes = MagicMock(side_effect=get_notes)
    client.get_group = MagicMock(side_effect=get_group)
    client.search_profiles = MagicMock(side_effect=search_profiles)
    client.get_profile = MagicMock(side_effect=get_profile)

    return client
# -----------------
# -- Mock Client --
# -----------------

# -------------------------------
# -- Endpoint Helper Functions --
# -------------------------------
def preprocess_config(config, job_id, profile_id, server_config):
    """
    Overwrites/add specific key-value pairs in the submitted job config

    :param config: Configuration fields for creating the dataset and executing the expertise model
    :type config: dict

    :param job_id: The ID for the job to be submitted
    :type job_id: str

    :param profile_id: The OpenReview profile ID associated with the job
    :type profile_id: str

    :param server_config: The Flask server's configuration
    :type server_config: dict

    :returns new_config: A modified version of config with the server-required fields

    :raises Exception: Raises exceptions when a required field is missing, or when a parameter is provided
                       when it is not expected
    """
    new_config = {
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
    req_fields = ['name', 'match_group', 'paper_invitation']
    optional_model_params = ['use_title', 'use_abstract', 'average_score', 'max_score', 'skip_specter']
    optional_fields = ['model', 'model_params', 'exclusion_inv', 'token', 'baseurl']
    path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
    # Validate + populate fields
    for field in req_fields:
        assert field in config, f'Missing required field: {field}'
        new_config[field] = config[field]
    for field in config.keys():
        assert field in optional_fields or field in req_fields, f'Unexpected field: {field}'
        if field != 'model_params':
            new_config[field] = config[field]
    if 'model_params' in config.keys():
        for field in config['model_params']:
            assert field in optional_model_params, f'Unexpected model param: {field}'
            new_config['model_params'][field] = config['model_params'][field]

    # Populate with server-side fields
    root_dir = os.path.join(server_config['WORKING_DIR'], job_id)
    new_config['dataset']['directory'] = root_dir
    for field in path_fields:
        new_config['model_params'][field] = root_dir    
    new_config['job_id'] = job_id
    new_config['job_dir'] = root_dir
    new_config['profile'] = profile_id
    new_config['cdate'] = int(time.time())

    # Set SPECTER+MFR paths
    if config.get('model', 'specter+mfr') == 'specter+mfr':
        new_config['model_params']['specter_dir'] = server_config['SPECTER_DIR']
        new_config['model_params']['mfr_feature_vocab_file'] = server_config['MFR_VOCAB_DIR']
        new_config['model_params']['mfr_checkpoint_dir'] = server_config['MFR_CHECKPOINT_DIR']

    # Create directory and config file
    if not os.path.isdir(new_config['dataset']['directory']):
        os.makedirs(new_config['dataset']['directory'])
    with open(os.path.join(root_dir, 'config.cfg'), 'w+') as f:
        json.dump(new_config, f, ensure_ascii=False, indent=4)
    
    return new_config   

def get_subdirs(root_dir, profile_id=None):
    """
    Returns the direct children directories of the given root directory
    
    :param root_dir: The relative directory to be searched for sub-directories
    :type root_dir: str

    :param profile_id: If given, only return subdirectories authorized with this user ID
    :type profile_id: str

    :returns: A list of subdirectories not prefixed by the given root directory
    """
    subdirs = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    if profile_id is None:
        return subdirs
    else:
        # If given a profile ID, assume looking for job dirs that contain a config with the
        # matching profile id
        filtered_dirs = []
        for subdir in subdirs:
            config_dir = os.path.join(root_dir, subdir, 'config.cfg')
            with open(config_dir, 'r') as f:
                config = json.load(f)
                if profile_id == config['profile']:
                    filtered_dirs.append(subdir)
        return filtered_dirs


def get_score_and_metadata_dir(search_dir):
    """
    Searches the given directory for a possible score file and the metadata file
    
    :param search_dir: The root directory to search in
    :type search_dir: str

    :returns file_dir: The directory of the score file, if it exists, starting from the given directory
    :returns metadata_dir: The directory of the metadata file, if it exists, starting from the given directory
    """
    # Search for scores files (only non-sparse scores)
    file_dir, metadata_dir = None, None
    with open(os.path.join(search_dir, 'config.cfg'), 'r') as f:
        config = json.load(f)
    # Look for score files
    for root, dirs, files in os.walk(search_dir, topdown=False):
        for name in files:
            if name == f"{config['name']}.csv":
                file_dir = os.path.join(root, name)
            if 'metadata' in name:
                metadata_dir = os.path.join(root, name)
    return file_dir, metadata_dir

def get_profile(openreview_client):
    """
    Returns the OpenReview profile object given a client
    
    :param openreview_client: A logged in client with the user credentials
    :type openreview_client: openreview.Client

    :returns profile: An OpenReview profile object for the user
    """
    profile = openreview_client.get_profile()
    if profile is None:
        default_client = openreview.Client()
        profile = default_client.get_profile()
    return profile

def get_user_id(openreview_client):
    """
    Returns the user id from an OpenReview client for authenticating access
    
    :param openreview_client: A logged in client with the user credentials
    :type openreview_client: openreview.Client

    :returns id: The id of the logged in user 
    """
    profile = get_profile(openreview_client)
    return profile.id

def cleanup_thread(server_config, logger):
    """
    This function is meant to be called as a daemon thread on server start
    (before first request)
    Browses jobs in the server's working directory and checks the cdate of each config
    If it is time for the job to be cleaned up, wipe the entire job directory

    Expects server_config to contain keys "WORKING_DIR", "CHECK_EVERY", "DELETE_AFTER"

    :param server_config: The Flask server's configuration and expect certain fields from here
    :type server_config: dict

    :param logger: The Flask server's logger, or some form of logger
    :type logger: logging.Logger
    """
    while True:
        logger.info('Running eviction check')
        job_subdirs = get_subdirs(server_config['WORKING_DIR'])
        logger.info(f"Looking to evict {job_subdirs}")
        check_every = int(server_config['CHECK_EVERY'])
        delete_after = int(server_config['DELETE_AFTER'])
        current_time = int(time.time())
        for job_dir in job_subdirs:
            search_dir = os.path.join(server_config['WORKING_DIR'], job_dir)
            logger.info(f"Checking {search_dir}")
            # Load the config file to fetch the job name
            with open(os.path.join(search_dir, 'config.cfg'), 'r') as f:
                config = json.load(f)
            cdate = int(config['cdate'])
            logger.info(f"Current time {current_time}")
            logger.info(f"Expiration time {cdate + delete_after}")
            if cdate + delete_after <= current_time:
                logger.info(f"Evicting {search_dir}")
                shutil.rmtree(search_dir)
        time.sleep(check_every)
        

# -------------------------------
# -- Endpoint Helper Functions --
# -------------------------------

# ------------------------
# -- Endpoint Functions --
# ------------------------

def before_first_request(server_config, logger):
    """
    Performs the following steps:
        1) Starts the eviction thread for stale jobs
        2) Writes an error log for jobs that were interrupted

    :param server_config: The Flask server's configuration and expect certain fields from here
    :type server_config: dict
    
    :param logger: The Flask server's logger, or some form of logger
    :type logger: logging.Logger
    """
    # Start cleanup thread
    threading.Thread(target=cleanup_thread, args=(
        server_config,
        logger),
        daemon=True
    ).start()

    # Get all profile directories
    root_dir = server_config['WORKING_DIR']
    if os.path.isdir(root_dir):
        job_ids = get_subdirs(root_dir)
        # If no score file is present, write to error log
        for job_id in job_ids:
            with open(os.path.join(root_dir, job_id, 'config.cfg'), 'r') as f:
                config = json.load(f)
            error_dir = os.path.join(root_dir, job_id, 'err.log')
            job_dir = os.path.join(root_dir, job_id)
            score_dir, _ = get_score_and_metadata_dir(job_dir)
            if score_dir is None:
                with open(error_dir, 'a+') as f:
                    f.write(f"{job_id},{config['name']},Interrupted before completed")


def post_expertise(json_request, profile_id, server_config, logger):
    """
    Puts the incoming request on the 'userpaper' queue - which runs creates the dataset followed by executing the expertise

    :param json_request: This is entire body of the json request, possibly augmented by the server
    :type json_request: dict

    :param profile_id: The OpenReview profile ID associated with the job
    :type profile_id: str

    :param server_config: The Flask server's configuration and expect certain fields from here
    :type server_config: dict
    
    :param logger: The Flask server's logger, or some form of logger
    :type logger: logging.Logger

    :param job_id: Optional ID of the specific job to look up
    :type job_id: str

    :returns job_id: A unique string assigned to this job
    """
    job_id = shortuuid.ShortUUID().random(length=5)

    from .celery_tasks import run_userpaper
    config = preprocess_config(json_request, job_id, profile_id, server_config)
    logger.info(f'Config: {config}')
    run_userpaper.apply_async(
        (config, logger),
        queue='userpaper',
        task_id=job_id
    )

    return job_id

def get_jobs(user_id, server_config, logger, job_id=None):
    """
    Searches the server for all jobs submitted by a user
    If a job ID is provided, only fetch the status of this job

    :param user_id: The ID of the user accessing the data
    :type user_id: str

    :param server_config: The Flask server's configuration and expect certain fields from here
    :type server_config: dict

    :param logger: The Flask server's logger, or some form of logger
    :type logger: logging.Logger

    :param job_id: Optional ID of the specific job to look up
    :type job_id: str
    
    :returns: A dictionary with the key 'results' containing a list of job statuses
    """
    # Perform a walk of all job sub-directories for score files
    # TODO: This walks through all submitted jobs and requires reading a file per job
    # TODO: is this what we want to do?

    result = {}
    result['results'] = []

    job_subdirs = get_subdirs(server_config['WORKING_DIR'], user_id)
    # If given an ID, only get the status of the single job
    logger.info(f'check filtering | value of job_ID: {job_id}')
    if job_id is not None:
        logger.info(f'performing filtering')
        job_subdirs = [name for name in job_subdirs if name == job_id]
    logger.info(f'Subdirs: {job_subdirs}')

    for job_dir in job_subdirs:
        search_dir = os.path.join(server_config['WORKING_DIR'], job_dir)
        logger.info(f'Looking at {search_dir}')
        file_dir, _ = get_score_and_metadata_dir(search_dir)
        err_dir = os.path.join(search_dir, 'err.log')

        # Load the config file to fetch the job name
        with open(os.path.join(search_dir, 'config.cfg'), 'r') as f:
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
                    'status': f'Error: {err}'
                }
            )
            continue

        logger.info(f'Current score status {file_dir}')
        # If found a non-sparse, non-data file CSV, job has completed
        if file_dir is None:
            status = 'Processing'
        else:
            status = 'Completed'
        result['results'].append(
            {
                'job_id': job_dir,
                'name': config['name'],
                'status': status
            }
        )
    
    return result

def get_results(user_id, job_id, delete_on_get, server_config, logger):
    """
    Gets the scores of a given job
    If delete_on_get is set, delete the directory after the scores are fetched

    :param user_id: The ID of the user accessing the data
    :type user_id: str

    :param job_id: ID of the specific job to fetch
    :type job_id: str

    :param delete_on_get: A flag indicating whether or not to clean up the directory after it is fetched
    :type delete_on_get: bool

    :param server_config: The Flask server's configuration and expect certain fields from here
    :type server_config: dict

    :param logger: The Flask server's logger, or some form of logger
    :type logger: logging.Logger
    
    :returns: A dictionary that contains the calculated scores and metadata
    """
    result = {}
    result['results'] = []

    # Validate profile ID
    search_dir = os.path.join(server_config['WORKING_DIR'], job_id)
    with open(os.path.join(search_dir, 'config.cfg'), 'r') as f:
        config = json.load(f)
    assert user_id == config['profile'], "Forbidden: Insufficient permissions to access job"
    # Search for scores files (only non-sparse scores)
    file_dir, metadata_dir = get_score_and_metadata_dir(search_dir)

    # Assemble scores
    if file_dir is None:
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
        logger.error(f'Deleting {search_dir}')
        shutil.rmtree(search_dir)
    
    return result
# ------------------------
# -- Endpoint Functions --
# ------------------------