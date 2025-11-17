'''
Implements the Flask API endpoints.
'''
from expertise.service.expertise import ExpertiseService, ExpertiseCloudService
from expertise.service import model_ready, artifact_loading_started
from expertise.service.utils import GCPInterface
import openreview
import json
from .utils import get_user_id
import flask
from flask import g
from copy import deepcopy
from flask_cors import CORS
from multiprocessing import Value
from csv import reader

from .auth import require_auth
from .responses import format_error

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
    if config.get('USE_GCP', False):
        flask.current_app.logger.info('Using GCP')
        return ExpertiseCloudService(config, logger)
    flask.current_app.logger.info('Using local')
    return ExpertiseService(config, logger)

@BLUEPRINT.route('/expertise/test')
def test():
    """Test endpoint."""
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'

@BLUEPRINT.route('/expertise', methods=['POST'])
@require_auth
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

    try:
        openreview_client = g.or_client
        openreview_client_v2 = g.or_client_v2

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
            job_id = expertise_service.start_expertise(user_request, openreview_client, openreview_client_v2)
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

@BLUEPRINT.route('/expertise/status', methods=['GET'])
@require_auth
def jobs():
    """
    Only retrieves the status of the job with that job_id

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    try:
        openreview_client = g.or_client
        openreview_client_v2 = g.or_client_v2
        user_id = g.user_id

        flask.current_app.logger.debug('GET receives ' + str(flask.request.args))
        # Parse query parameters
        job_id = flask.request.args.get('jobId', None)
        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        expertise_service.set_client_v2(openreview_client_v2)
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
@require_auth
def all_jobs():
    """
    Query all submitted jobs associated with the logged in user

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    try:
        openreview_client = g.or_client
        user_id = g.user_id
        # Parse query parameters
        flask.current_app.logger.debug('GET receives ' + str(flask.request.args))
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

@BLUEPRINT.route('/expertise/<job_id>', methods=['DELETE'])
@require_auth
def delete_job(job_id):
    """
    Retrieves the config of a job to be deleted, and removes the job by deleting the job directory.

    :param token: Authorization from a logged in user, which defines the set of accessible data
    :type token: str

    :param job_id: The ID of a submitted job
    :type job_id: str
    """
    try:
        openreview_client = g.or_client
        user_id = g.user_id

        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        result = expertise_service.del_expertise_job(user_id, job_id)

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
@require_auth
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
    try:
        openreview_client = g.or_client
        openreview_client_v2 = g.or_client_v2
        user_id = g.user_id
        # Parse query parameters
        flask.current_app.logger.debug('GET receives ' + str(flask.request.args))
        job_id = flask.request.args.get('jobId', None)
        if job_id is None or len(job_id) == 0:
            raise openreview.OpenReviewException('Bad request: jobId is required')
        delete_on_get = flask.request.args.get('deleteOnGet', 'False').lower() == 'true'
        
        # Check Accept header to determine response format
        accept_header = flask.request.headers.get('Accept', '')
        flask.current_app.logger.debug(f'Accept header: {accept_header}')
        # Check if Accept header contains text/csv or application/csv
        return_csv = 'text/csv' in accept_header or 'application/csv' in accept_header
        if return_csv:
            flask.current_app.logger.debug('Format selection: CSV (from Accept header)')
        else:
            flask.current_app.logger.debug('Format selection: JSONL (default or from Accept header)')

        expertise_service = get_expertise_service(flask.current_app.config, flask.current_app.logger)
        expertise_service.set_client(openreview_client)
        expertise_service.set_client_v2(openreview_client_v2)
        result = expertise_service.get_expertise_results(user_id, job_id, delete_on_get, return_csv=return_csv)
        
        # Check if result is a generator (for streaming) or a regular dict
        if hasattr(result, '__iter__') and not isinstance(result, (dict, list)):
            # It's a generator - use streaming response
            flask.current_app.logger.debug('Using streaming response')
            
            def generate():
                try:
                    # Branch streaming format based on return_csv
                    if return_csv:
                        # Stream a single JSON string by concatenating CSV chunks
                        yield '{"results":"'

                        metadata = None
                        for chunk in result:
                            # Save metadata for later
                            if chunk.get('metadata') is not None:
                                metadata = chunk['metadata']

                            # Stream CSV chunks (strings) safely escaped for JSON
                            if chunk.get('results'):
                                csv_chunk = chunk['results']
                                if isinstance(csv_chunk, str):
                                    # Append without outer quotes
                                    yield json.dumps(csv_chunk)[1:-1]
                                else:
                                    # Fallback: stringify non-str results
                                    yield json.dumps(str(csv_chunk))[1:-1]

                        # Close the results string and add metadata
                        yield '",'
                        yield f'"metadata":{json.dumps(metadata or {})}'
                        yield '}'
                    else:
                        # JSONL: Stream an array of JSON objects
                        yield '{"results":['

                        first_chunk = True
                        metadata = None

                        for chunk in result:
                            # Save metadata for later
                            if chunk.get('metadata') is not None:
                                metadata = chunk['metadata']

                            # Stream results as individual JSON items
                            if chunk.get('results'):
                                for result_item in chunk['results']:
                                    if not first_chunk:
                                        yield ','
                                    yield json.dumps(result_item)
                                    first_chunk = False

                        # Close the results array and add metadata
                        yield '],'
                        yield f'"metadata":{json.dumps(metadata or {})}'
                        yield '}'

                except Exception as e:
                    flask.current_app.logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                    # If an error occurs during streaming, we need to yield a valid JSON
                    yield '[],"error":"Error during streaming"}'

            flask.current_app.logger.debug('Streaming response started')
            return flask.Response(generate(), mimetype='application/json')
        else:
            # It's a regular dictionary - use standard JSON response
            flask.current_app.logger.debug('Using standard JSON response')
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

@BLUEPRINT.route('/startup')
def startup():
    """An endpoint for checking availability for predictions"""
    flask.current_app.logger.info('In startup')
    if not model_ready.is_set():
        return {'status': 'Service unavailable: Artifact loading in progress'}, 503
    return {'status': 'Available for predictions'}, 200

@BLUEPRINT.route('/health')
def health():
    """An endpoint for server uptime"""
    flask.current_app.logger.info('In health')
    return {'status': 'Healthy instance'}, 200
