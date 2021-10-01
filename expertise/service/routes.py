'''
Implements the Flask API endpoints.
'''
from expertise.service.expertise import ExpertiseService
import openreview
from openreview.openreview import OpenReviewException
from .utils import mock_client
from .utils import get_user_id
import flask
from copy import deepcopy
from flask_cors import CORS
from multiprocessing import Value
from csv import reader


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)


def get_client():
    token = flask.request.headers.get('Authorization')
    in_test_mode = 'IN_TEST' in flask.current_app.config.keys()

    if in_test_mode:
        return mock_client()

    return openreview.Client(
        token=token,
        baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
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

    openreview_client = get_client()

    user_id = get_user_id(openreview_client)
    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify({ 'error': 'Forbidden: No Authorization token in headers'}), 403

    try:
        flask.current_app.logger.info('Received expertise request')

        # Parse request args
        user_request = flask.request.json
        user_request['token'] = openreview_client.token
        user_request['baseurl'] = flask.current_app.config['OPENREVIEW_BASEURL']
        user_request['user_id'] = user_id

        job_id = ExpertiseService(openreview_client, flask.current_app.config, flask.current_app.logger).start_expertise(user_request)

        result = {'job_id': job_id }
        flask.current_app.logger.info('Returning from request')

        flask.current_app.logger.debug('POST returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle))

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify({'error': error_type}), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle))
        return flask.jsonify({'error': 'Internal server error: {}'.format(error_handle)}), 500


@BLUEPRINT.route('/expertise/status', methods=['GET'])
def jobs():
    """
    Query all submitted jobs associated with the logged in user
    If provided with a job_id field, only retrieve the status of the job with that job_id

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    openreview_client = get_client()

    user_id = get_user_id(openreview_client)

    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify({ 'error': 'Forbidden: No Authorization token in headers'}), 403

    try:
        # Parse query parameters
        job_id = flask.request.args.get('id', None)

        result = ExpertiseService(openreview_client, flask.current_app.config, flask.current_app.logger).get_expertise_status(user_id, job_id)

        flask.current_app.logger.debug('GET returns ' + str(result))
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle))

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403

        return flask.jsonify({'error': error_type}), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle))
        return flask.jsonify({'error': 'Internal server error: {}'.format(error_handle)}), 500

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
    openreview_client = get_client()

    user_id = get_user_id(openreview_client)

    if not user_id:
        flask.current_app.logger.error('No Authorization token in headers')
        return flask.jsonify({ 'error': 'Forbidden: No Authorization token in headers'}), 403

    try:
        # Parse query parameters
        job_id = flask.request.args.get('job_id', None)
        if job_id is None:
            raise openreview.OpenReviewException('Bad request: job_id is required')
        delete_on_get = flask.request.args.get('delete_on_get', 'False').lower() == 'true'

        result = ExpertiseService(openreview_client, flask.current_app.config, flask.current_app.logger).get_expertise_results(user_id, job_id, delete_on_get)

        flask.current_app.logger.debug('GET returns code 200')
        return flask.jsonify(result), 200

    except openreview.OpenReviewException as error_handle:
        flask.current_app.logger.error(str(error_handle))

        error_type = str(error_handle)
        status = 500

        if 'not found' in error_type.lower():
            status = 404
        elif 'forbidden' in error_type.lower():
            status = 403
        elif 'bad request' in error_type.lower():
            status = 400

        return flask.jsonify({'error': error_type}), status

    # pylint:disable=broad-except
    except Exception as error_handle:
        flask.current_app.logger.error(str(error_handle))
        return flask.jsonify({'error', 'Internal server error: {}'.format(error_handle)}), 500