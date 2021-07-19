'''
Implements the Flask API endpoints.
'''
import flask
from flask_cors import CORS
import threading
import openreview
from ..execute_expertise import *


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

class ExpertiseStatusException(Exception):
    '''Exception wrapper class for errors related to the status of the Expertise model'''
    pass

def expertise_pipeline(client, config):
    execute_create_dataset(client, config_file=config)
    execute_expertise(config_file=config)

@BLUEPRINT.route('/expertise/test')
def test():
    '''Test endpoint.'''
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'

@BLUEPRINT.route('/expertise', methods=['POST'])
def expertise():
    flask.current_app.logger.debug('Create dataset request received')
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
        
        thread = threading.Thread(
            target = expertise_pipeline(openreview_client, flask.request.json['config'])
        )
        thread.start()

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