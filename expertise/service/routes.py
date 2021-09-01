'''
Implements the Flask API endpoints.
'''
import flask, os, shutil, random, string, json, shortuuid
from copy import deepcopy
from flask_cors import CORS
from multiprocessing import Value
from csv import reader
import openreview
from openreview.openreview import OpenReviewException
from .utils import mock_client


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

# TODO: Fault tolerance - on server start, for each profile dir, wipe error log and re-populate with crashed jobs
#     : and clear the directories of the crashed jobs


def preprocess_config(config: dict, job_id: int, profile_id: str, test_mode: bool = False):
    """Overwrites/add specific keywords in the submitted job config"""
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
    optional_fields = ['model', 'model_params', 'exclusion_inv']
    path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
    # Validate + populate fields
    for field in req_fields:
        assert field in flask.request.json, f'Missing required field: {field}'
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
    root_dir = os.path.join(flask.current_app.config['WORKING_DIR'], profile_id, job_id)
    new_config['dataset']['directory'] = root_dir
    for field in path_fields:
        new_config['model_params'][field] = root_dir
    new_config['job_id'] = job_id
    new_config['profile_dir'] = os.path.join(flask.current_app.config['WORKING_DIR'], profile_id)
    # Set SPECTER+MFR paths
    if config.get('model', 'specter+mfr') == 'specter+mfr':
        new_config['model_params']['specter_dir'] = flask.current_app.config['SPECTER_DIR']
        new_config['model_params']['mfr_feature_vocab_file'] = flask.current_app.config['MFR_VOCAB_DIR']
        new_config['model_params']['mfr_checkpoint_dir'] = flask.current_app.config['MFR_CHECKPOINT_DIR']

    if not os.path.isdir(new_config['dataset']['directory']):
        os.makedirs(new_config['dataset']['directory'])
    
    return new_config   

def enqueue_expertise(json_request, profile_id, in_test_mode):
    job_id = shortuuid.ShortUUID().random(length=5)

    from .celery_tasks import run_userpaper
    config = preprocess_config(json_request, job_id, profile_id, in_test_mode)
    flask.current_app.logger.info(f'Config: {config}')
    run_userpaper.apply_async(
        (config, flask.current_app.logger),
        queue='userpaper',
        task_id=job_id
    )

    return job_id

@BLUEPRINT.route('/test')
def test():
    """Test endpoint."""
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'

@BLUEPRINT.route('/expertise', methods=['POST'])
def expertise():
    """
    Requires authentication
    Required fields:
        name
        match_group
        paper_invitation
    All other fields are optional and will be populated by the server
    """
    result = {}

    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()
    
    if not token and not in_test_mode:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    
    try:
        user_config = flask.request.json
        flask.current_app.logger.info('Received expertise request')
        
        if not in_test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
            user_config['token'] = token
            user_config['baseurl'] = flask.current_app.config['OPENREVIEW_BASEURL']
        else:
            openreview_client = mock_client()

        profile = openreview_client.get_profile()
        if profile is None:
            raise OpenReviewException('Forbidden: Profile does not exist')
        profile_id = profile.id

        job_id = enqueue_expertise(user_config, profile_id, in_test_mode)

        result['job_id'] = job_id
        flask.current_app.logger.info('Returning from request')
        
    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle))

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403

        result['error'] = error_type
        return flask.jsonify(result), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        result['error'] = 'Internal server error: {}'.format(error_handle)
        return flask.jsonify(result), 500
        
    else:
        flask.current_app.logger.debug('POST returns ' + str(result))
        return flask.jsonify(result), 200

@BLUEPRINT.route('/jobs', methods=['GET'])
def jobs():
    """
    Requires authentication
    Required fields: none
    """
    result = {}

    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()
    
    if not token and not in_test_mode:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    
    try:
        result['results'] = []
        job_id = flask.request.args.get('id', None)
        if not in_test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
        else:
            openreview_client = mock_client()

        profile = openreview_client.get_profile()
        if profile is None:
            raise OpenReviewException('Forbidden: Profile does not exist')
        profile_id = profile.id

        profile_dir = os.path.join(flask.current_app.config['WORKING_DIR'], profile_id)
        flask.current_app.logger.info(f'Looking at {profile_dir}')

        # Check for profile directory
        if not os.path.isdir(profile_dir):
            raise OpenReviewException('No jobs submitted since last server reboot')

        # Check for error log
        flask.current_app.logger.info(f'Checking error log')
        err_jobs = []
        err_dir = os.path.join(profile_dir, 'err.log')
        if os.path.isfile(err_dir):
            with open(err_dir, 'r') as f:
                err_jobs = f.readline().split(',')[:-1]
        for job in err_jobs:
            result['results'].append(
                {
                    'job_id': job,
                    'status': 'Error'
                }
            )

        # Perform a walk of all job sub-directories for score files
        job_subdirs = [name for name in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, name))]
        # If given an ID, only get the status of the single job
        if job_id is not None:
            job_subdirs = [name for name in job_subdirs if name == job_id]
        flask.current_app.logger.info(f'Subdirs: {job_subdirs}')

        for job_dir in job_subdirs:
            search_dir = os.path.join(profile_dir, job_dir)
            flask.current_app.logger.info(f'Looking at {search_dir}')
            # Look for score files
            file_dir = None
            for root, dirs, files in os.walk(search_dir, topdown=False):
                for name in files:
                    if '.csv' in name and 'sparse' not in name:
                        file_dir = os.path.join(root, name)

            flask.current_app.logger.info(f'Current score status {file_dir}')
            # If found a non-sparse, non-data file CSV, job has completed
            if file_dir is None:
                status = 'Processing'
            else:
                status = 'Completed'
            result['results'].append(
                {
                    'job_id': job_dir,
                    'status': status
                }
            )
                        
    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle))

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403

        result['error'] = error_type
        return flask.jsonify(result), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        result['error'] = 'Internal server error: {}'.format(error_handle)
        return flask.jsonify(result), 500
        
    else:
        flask.current_app.logger.debug('POST returns ' + str(result))
        return flask.jsonify(result), 200

@BLUEPRINT.route('/results', methods=['GET'])
def results():
    """
    Requires authentication
    Required fields:
        job_id
    Optional field:
        delete_on_get
    """
    result = {}

    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()
    
    if not token and not in_test_mode:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    
    try:
        job_id = flask.request.args['job_id']
        delete_on_get = flask.request.args.get('delete_on_get', 'False')

        if delete_on_get.lower() == 'true':
            delete_on_get = True
        else:
            delete_on_get = False
    
        if not in_test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
        else:
            openreview_client = mock_client()

        profile = openreview_client.get_profile()
        if profile is None:
            raise OpenReviewException('Forbidden: Profile does not exist')
        profile_id = profile.id

        profile_dir = os.path.join(flask.current_app.config['WORKING_DIR'], profile_id)
        if not os.path.isdir(profile_dir):
            raise OpenReviewException('No jobs submitted since last server reboot')

        # Search for scores files (only non-sparse scores)
        file_dir, metadata_dir = None, None
        search_dir = os.path.join(profile_dir, job_id)
        # Look for score files
        for root, dirs, files in os.walk(search_dir, topdown=False):
            for name in files:
                if '.csv' in name and 'sparse' not in name:
                    file_dir = os.path.join(root, name)
                if 'metadata' in name:
                    metadata_dir = os.path.join(root, name)

        # Assemble scores
        if file_dir is None:
            raise OpenReviewException('Either job is still processing, has crashed, or does not exist')
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
            flask.current_app.logger.error(f'Deleting {search_dir}')
            shutil.rmtree(search_dir)
                
    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle))

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403

        result['error'] = error_type
        return flask.jsonify(result), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        result['error'] = 'Internal server error: {}'.format(error_handle)
        return flask.jsonify(result), 500
        
    else:
        flask.current_app.logger.debug('POST returns code 200')
        return flask.jsonify(result), 200