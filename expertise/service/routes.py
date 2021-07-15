'''
Implements the Flask API endpoints.
'''
import flask
from flask_cors import CORS
import threading
import openreview


BLUEPRINT = flask.Blueprint('expertise', __name__)
CORS(BLUEPRINT, supports_credentials=True)

class ExpertiseStatusException(Exception):
    '''Exception wrapper class for errors related to the status of the Expertise model'''
    pass

@BLUEPRINT.route('/expertise/test')
def test():
    '''Test endpoint.'''
    flask.current_app.logger.info('In test')
    return 'OpenReview Expertise (test)'