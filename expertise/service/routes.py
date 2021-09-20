'''
Implements the Flask API endpoints.
'''
import flask, os, shutil, threading, json
from copy import deepcopy
from flask_cors import CORS
from multiprocessing import Value
from csv import reader
import openreview
from openreview.openreview import OpenReviewException
from .utils import (
    mock_client,
    get_user_id,
    get_subdirs,
    get_score_and_metadata_dir,
    post_expertise,
    get_jobs,
    get_results,
    before_first_request
)


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

@BLUEPRINT.before_app_first_request
def start_server():
    """
    On server start, check if there is a working directory
    If so, free the space from all incomplete jobs and mark them in the error log
    """
    before_first_request(
        flask.current_app.config,
        flask.current_app.logger
    )

@BLUEPRINT.route('/test')
def test():
    """Test endpoint."""
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'

@BLUEPRINT.route('/expertise', methods=['POST'])
def expertise():
    """
    Submit a job to create a dataset and execute an expertise model based on the submitted configuration

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param name: A name describing the job being submitted
    :type name: str

    :param match_group: A group whose profiles will be used to compute expertise
    :type match_group: str

    :param paper_invitation: An invitation containing submissions used to compute expertise
    :type paper_invitation: str
    """
    result = {}

    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()
    
    if not token and not in_test_mode:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'Forbidden: No Authorization token in headers'
        return flask.jsonify(result), 403
    
    try:
        flask.current_app.logger.info('Received expertise request')

        # Parse request args
        user_config = flask.request.json
        
        # Fetch OR client
        if not in_test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
            user_config['token'] = token
            user_config['baseurl'] = flask.current_app.config['OPENREVIEW_BASEURL']
        else:
            openreview_client = mock_client()

        # Perform server work
        user_id = get_user_id(openreview_client)
        job_id = post_expertise(
            user_config,
            user_id,
            flask.current_app.config,
            flask.current_app.logger
        )

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
    Query all submitted jobs associated with the logged in user
    If provided with a job_id field, only retrieve the status of the job with that job_id

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    result = {}

    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()
    
    if not token and not in_test_mode:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'Forbidden: No Authorization token in headers'
        return flask.jsonify(result), 403
    
    try:
        # Parse query parameters
        job_id = flask.request.args.get('id', None)

        # Fetch OR client
        if not in_test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
        else:
            openreview_client = mock_client()

        # Perform server work
        user_id = get_user_id(openreview_client)
        result = get_jobs(
            user_id,
            flask.current_app.config,
            flask.current_app.logger,
            job_id
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
    Get the results of a single submitted job with the associated job_id
    If provided with a delete_on_get field, delete the job from the server after retrieving results

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    
    :param delete_on_get: Decide whether to keep the data on the server after getting the results
    :type delete_on_get: bool
    """
    result = {}

    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()
    
    if not token and not in_test_mode:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'Forbidden: No Authorization token in headers'
        return flask.jsonify(result), 403
    
    try:
        # Parse query parameters
        job_id = flask.request.args['job_id']
        delete_on_get = flask.request.args.get('delete_on_get', 'False')

        if delete_on_get.lower() == 'true':
            delete_on_get = True
        else:
            delete_on_get = False

        # Fetch OR client
        if not in_test_mode:
            openreview_client = openreview.Client(
                token=token,
                baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
            )
        else:
            openreview_client = mock_client()

        # Perform server work
        user_id = get_user_id(openreview_client)
        result = get_results(
            user_id,
            job_id,
            delete_on_get,
            flask.current_app.config,
            flask.current_app.logger
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
        flask.current_app.logger.debug('POST returns code 200')
        return flask.jsonify(result), 200