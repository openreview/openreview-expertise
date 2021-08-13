'''
Implements the Flask API endpoints.
'''
import flask, os, shutil
from flask_cors import CORS
from multiprocessing import Value
from csv import reader
import openreview
from openreview.openreview import OpenReviewException


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

# task_queue = UserPaperQueue(max_jobs = 1, inner_queue=ExpertiseQueue, inner_key='expertise', outer_key='dataset')
global_id: Value = Value('i', 0)

class ExpertiseStatusException(Exception):
    '''Exception wrapper class for errors related to the status of the Expertise model'''
    pass

# TODO: Fault tolerance - on server start, wipe all profile directories

def preprocess_config(config: dict, job_id: int, profile_id: str, test_mode: bool = False):
    """Overwrites/add specific keywords in the submitted job config"""
    # Overwrite certain keys in the config
    filepath_keys = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
    file_keys = ['csv_expertise', 'csv_submissions']
    root_dir = f"./{profile_id}/{job_id}"

    # Filter some keys
    if 'model_params' not in config.keys():
        config['model_params'] = {}
    file_keys = [key for key in file_keys if key in config.keys()]

    # First handle dataset -> directory
    if 'dataset' not in config.keys():
        config['dataset'] = {}
    config['dataset']['directory'] = root_dir

    if not os.path.isdir(config['dataset']['directory']):
        os.makedirs(config['dataset']['directory'])

    # Next handle other file paths
    for key in filepath_keys:
        config['model_params'][key] = root_dir
    
    # Now, write data stored in the file keys to disk
    if test_mode:
        for key in file_keys:
            output_file = key + '.csv'
            write_to_dir = os.path.join(config['dataset']['directory'], output_file)

            # Add newline characters, write to file and set the field in the config to the directory of the file
            for idx, data in enumerate(config[key]):
                config[key][idx] = data.strip() + '\n'
            with open(write_to_dir, 'w') as csv_out:
                csv_out.writelines(config[key])
            
            config[key] = output_file
    
    # For error handling, add job_id and profile_dir to config
    config['job_id'] = job_id
    config['profile_dir'] = f'./{profile_id}'

    # Set SPECTER+MFR paths
    config['model_params']['specter_dir'] = '../expertise-utils/specter/'
    config['model_params']['mfr_feature_vocab_file'] = '../expertise-utils/multifacet_recommender/feature_vocab_file'
    config['model_params']['mfr_checkpoint_dir'] = '../expertise-utils/multifacet_recommender/mfr_model_checkpoint/'
    return f'{job_id};{profile_id}'


@BLUEPRINT.route('/test')
def test():
    """Test endpoint."""
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'

@BLUEPRINT.route('/expertise', methods=['POST'])
def expertise():
    result = {}

    token = flask.request.headers.get('Authorization')
    test_num = flask.request.headers.get('TEST_NUM')
    test_mode = False
    
    if 'TEST_NUM' in flask.current_app.config.keys():
        if test_num != flask.current_app.config['TEST_NUM']:
            flask.current_app.logger.error('[TEST] Error in test num match')
            result['error'] = 'Error in test num match'
            return flask.jsonify(result), 400
        else:
            test_mode = True
    elif not token:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    
    try:
        config = flask.request.json['config']
        flask.current_app.logger.info('Received expertise request')
        if not test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
            profile_id = openreview_client.profile.id
            config['token'] = token
            config['baseurl'] = flask.current_app.config['OPENREVIEW_BASEURL']
        else:
            profile_id = f'~{test_num}'
        from .celery_tasks import run_userpaper
        with global_id.get_lock():
            job_id = preprocess_config(config, global_id.value, profile_id, test_mode)
            run_userpaper.apply_async(
                (config, flask.current_app.logger, test_mode),
                queue='userpaper',
                task_id=job_id
            )
            result['job_id'] = global_id.value
            global_id.value += 1
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
    result = {}

    token = flask.request.headers.get('Authorization')
    test_num = flask.request.headers.get('TEST_NUM')
    test_mode = False
    
    if 'TEST_NUM' in flask.current_app.config.keys():
        if test_num != flask.current_app.config['TEST_NUM']:
            flask.current_app.logger.error('[TEST] Error in test num match')
            result['error'] = 'Error in test num match'
            return flask.jsonify(result), 400
        else:
            test_mode = True
    elif not token:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    
    try:
        data_files = ['csv_expertise.csv', 'csv_submissions.csv']
        if not test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
            profile_id = openreview_client.get_profile().id
        else:
            profile_id = f'~{test_num}'
        profile_dir = f'./{profile_id}'

        # Perform a walk of all job sub-directories for score files
        job_subdirs = [name for name in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, name))]
        result['results'] = []
        for job_dir in job_subdirs:
            search_dir = os.path.join(profile_dir, job_dir)
            # Look for score files
            file_dir = None
            for root, dirs, files in os.walk(search_dir, topdown=False):
                for name in files:
                    if '.csv' in name and 'sparse' not in name and name not in data_files:
                        file_dir = os.path.join(root, name)

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
    result = {}

    token = flask.request.headers.get('Authorization')
    test_num = flask.request.headers.get('TEST_NUM')
    test_mode = False
    
    if 'TEST_NUM' in flask.current_app.config.keys():
        if test_num != flask.current_app.config['TEST_NUM']:
            flask.current_app.logger.error('[TEST] Error in test num match')
            result['error'] = 'Error in test num match'
            return flask.jsonify(result), 400
        else:
            test_mode = True
    elif not token:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    
    try:
        data_files = ['csv_expertise.csv', 'csv_submissions.csv']
        job_id = flask.request.args['job_id']
        delete_on_get = flask.request.args.get('delete_on_get', False)

        # Check type of delete_on_get
        if isinstance(delete_on_get, str):
            if delete_on_get.lower() == 'true':
                delete_on_get = True
            else:
                delete_on_get = False
        if not test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
            profile_id = openreview_client.get_profile().id
        else:
            profile_id = f'~{test_num}'
        profile_dir = f'./{profile_id}'

        # Search for scores files (only non-sparse scores)
        # TODO: checking for error by reading profile id directory and error log
        file_dir = None
        search_dir = os.path.join(profile_dir, job_id)
        # Look for score files
        for root, dirs, files in os.walk(search_dir, topdown=False):
            for name in files:
                if '.csv' in name and 'sparse' not in name and name not in data_files:
                    file_dir = os.path.join(root, name)

        # Assemble scores
        if file_dir is None:
            raise OpenReviewException('Either job is still processing or this job id does not exist')
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
        
        # Clear directory
        if delete_on_get:
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