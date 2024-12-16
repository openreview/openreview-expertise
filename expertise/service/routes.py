'''
Implements the Flask API endpoints.
'''
from expertise.service.expertise import ExpertiseService
import openreview
from openreview.openreview import OpenReviewException
from .utils import get_user_id
import flask
from copy import deepcopy
from flask_cors import CORS
from multiprocessing import Value
from csv import reader

BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

def run_once(f):
    """
    Decorator to run a function only once and return its output for any subsequent call to the function without running
    it again
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            wrapper.to_return = f(*args, **kwargs)
        return wrapper.to_return
    wrapper.has_run = False
    return wrapper

@run_once
def get_expertise_service(config, logger):
    return ExpertiseService(config, logger)

def get_client():
    token = flask.request.headers.get('Authorization')
    return (
        openreview.Client(token=token, baseurl=flask.current_app.config['OPENREVIEW_BASEURL']),
        openreview.api.OpenReviewClient(token=token, baseurl=flask.current_app.config['OPENREVIEW_BASEURL_V2']),
    )

def format_error(status_code, description):
    '''
    Formulates an error that is in the same format as the OpenReview API errors

    :param status_code: The status code determined by looking at the description
    :type status_code: int

    :param description: Useful information about the error
    :type description: str

    :returns template: A dictionary that zips all the information into a proper format
    '''
    # Parse status code
    error_name = ''
    if status_code == 400:
        error_name = 'BadRequestError'
    elif status_code == 403:
        error_name = 'ForbiddenError'
    elif status_code == 404:
        error_name = 'NotFoundError'
    elif status_code == 500:
        error_name = 'InternalServerError'

    template = {
        'name': error_name,
        'message': description,
    }

    return template

@BLUEPRINT.route('/expertise/test')
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

    openreview_client, openreview_client_v2 = get_client()

    user_id = get_user_id(openreview_client)
    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

    try:
        flask.current_app.logger.info('Received expertise request')

        # Parse request args
        user_request = flask.request.json

        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        expertise_service.set_client_v2(openreview_client_v2)

        request_key = expertise_service.get_key_from_request(user_request)

        if expertise_service.redis.db.incr(request_key) > 1:
            raise openreview.OpenReviewException("Request already in process")

        try:
            job_id = expertise_service.start_expertise(user_request)
            expertise_service.redis.db.delete(request_key)
        except Exception as error_handle:
            expertise_service.redis.db.delete(request_key)
            raise error_handle

        result = {'jobId': job_id }
        flask.current_app.logger.info('Returning from request')

        flask.current_app.logger.debug('POST returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        import traceback
        traceback.print_exc()
        traceback.print_tb(error_handle.__traceback__)
        flask.current_app.logger.error(str(error_handle), exc_info=True)

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify(format_error(status, error_type)), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        import traceback
        traceback.print_exc()
        traceback.print_tb(error_handle.__traceback__)
        flask.current_app.logger.error(str(error_handle), exc_info=True)
        return flask.jsonify(format_error(500, 'Internal server error: {}'.format(error_handle))), 500

@BLUEPRINT.route('/expertise/legacy', methods=['POST'])
def expertise_legacy():
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

    openreview_client, openreview_client_v2 = get_client()

    user_id = get_user_id(openreview_client)
    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

    try:
        flask.current_app.logger.info('Received expertise request')

        # Parse request args
        user_request = flask.request.json

        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        expertise_service.set_client_v2(openreview_client_v2)

        job_id = expertise_service.start_expertise_legacy(user_request)

        result = {'jobId': job_id }
        flask.current_app.logger.info('Returning from request')

        flask.current_app.logger.debug('POST returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify(format_error(status, error_type)), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)
        return flask.jsonify(format_error(500, 'Internal server error: {}'.format(error_handle))), 500

@BLUEPRINT.route('/expertise/status', methods=['GET'])
def jobs():
    """
    Only retrieves the status of the job with that job_id

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    openreview_client, _ = get_client()

    user_id = get_user_id(openreview_client)

    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

    try:
        # Parse query parameters
        job_id = flask.request.args.get('jobId', None)
        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        expertise_service.set_client_v2(openreview_client)
        if job_id is None or len(job_id) == 0:
            result = expertise_service.get_expertise_all_status(user_id, flask.request.args)
        else:
            result = expertise_service.get_expertise_status(user_id, job_id)
        flask.current_app.logger.debug('GET returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify(format_error(status, error_type)), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)
        return flask.jsonify(format_error(500, 'Internal server error: {}'.format(error_handle))), 500

@BLUEPRINT.route('/expertise/status/all', methods=['GET'])
def all_jobs():
    """
    Query all submitted jobs associated with the logged in user

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    openreview_client, _ = get_client()

    user_id = get_user_id(openreview_client)

    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

    try:
        # Parse query parameters
        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        result = expertise_service.get_expertise_all_status(user_id, flask.request.args)
        flask.current_app.logger.debug('GET returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        # import traceback
        # traceback.print_exc()
        flask.current_app.logger.error(str(error_handle), exc_info=True)

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify(format_error(status, error_type)), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        # import traceback
        # traceback.print_exc()
        flask.current_app.logger.error(str(error_handle), exc_info=True)
        return flask.jsonify(format_error(500, 'Internal server error: {}'.format(error_handle))), 500

@BLUEPRINT.route('/expertise/delete', methods=['GET'])
def delete_job():
    """
    Retrieves the config of a job to be deleted, and removes the job by deleting the job directory.

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    openreview_client, _ = get_client()

    user_id = get_user_id(openreview_client)

    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

    try:
        # Parse query parameters
        job_id = flask.request.args.get('jobId', None)
        if job_id is None or len(job_id) == 0:
            raise openreview.OpenReviewException('Bad request: jobId is required')
        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        result = expertise_service.del_expertise_job(user_id, job_id)
        flask.current_app.logger.debug('GET returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify(format_error(status, error_type)), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)
        return flask.jsonify(format_error(500, 'Internal server error: {}'.format(error_handle))), 500

@BLUEPRINT.route('/expertise/results', methods=['GET'])
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
    openreview_client, _ = get_client()

    user_id = get_user_id(openreview_client)

    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

    try:
        # Parse query parameters
        job_id = flask.request.args.get('jobId', None)
        if job_id is None or len(job_id) == 0:
            raise openreview.OpenReviewException('Bad request: jobId is required')
        delete_on_get = flask.request.args.get('deleteOnGet', 'False').lower() == 'true'

        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        result = expertise_service.get_expertise_results(user_id, job_id, delete_on_get)

        flask.current_app.logger.debug('GET returns code 200')
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify(format_error(status, error_type)), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle), exc_info=True)
        return flask.jsonify(format_error(500, 'Internal server error: {}'.format(error_handle))), 500
