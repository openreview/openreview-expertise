'''
Implements the Flask API endpoints.
'''
import flask
from flask_cors import CORS
import threading
from multiprocessing import Value
import openreview
from .or_queue import ExpertiseQueue, UserPaperQueue, DatasetInfo


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

task_queue = UserPaperQueue(max_jobs = 1, inner_queue=ExpertiseQueue, inner_key='expertise', outer_key='dataset')
global_id: Value = Value('i', 0)

class ExpertiseStatusException(Exception):
    '''Exception wrapper class for errors related to the status of the Expertise model'''
    pass

@BLUEPRINT.route('/expertise/test')
def test():
    '''Test endpoint.'''
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'

@BLUEPRINT.route('/expertise', methods=['POST'])
def expertise():
    result = {}

    token = flask.request.headers.get('Authorization')
    
    if not token:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    try:
        config = flask.request.json['config']
        openreview_client = openreview.Client(
            token=token,
            baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
        )
        config['token'] = token
        config['baseurl'] = flask.current_app.config['OPENREVIEW_BASEURL']
        
        from .celery_tasks import run_userpaper
        with global_id.get_lock():
            run_userpaper.apply_async(
                (config, flask.current_app.logger),
                queue='userpaper',
                task_id=global_id.value
            )
            global_id.value += 1
        
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
    
    if not token:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    try:
        openreview_client = openreview.Client(
            token=token,
            baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
        )
        profile_id = openreview_client.get_profile().id
        result = task_queue.get_jobs(profile_id)
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
    
    if not token:
        flask.current_app.logger.error('No Authorization token in headers')
        result['error'] = 'No Authorization token in headers'
        return flask.jsonify(result), 400
    try:
        job_id = flask.request.args['job_id']
        delete_on_get = flask.request.args.get('delete_on_get', False)

        # Check type of delete_on_get
        if isinstance(delete_on_get, str):
            if delete_on_get.lower() == 'true':
                delete_on_get = True
            else:
                delete_on_get = False
            
        openreview_client = openreview.Client(
            token=token,
            baseurl=flask.current_app.config['OPENREVIEW_BASEURL']
        )
        profile_id = openreview_client.get_profile().id
        result = task_queue.get_result(
            profile_id,
            delete_on_get = delete_on_get,
            job_id = job_id
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