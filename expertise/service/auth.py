"""Authentication helpers for the expertise service."""

from functools import wraps

import openreview
from flask import current_app, g, jsonify, request

from .responses import format_error
from .utils import get_user_id


def get_client(token):
    """Return OpenReview v1/v2 clients for the provided token."""
    return (
        openreview.Client(token=token, baseurl=current_app.config['OPENREVIEW_BASEURL']),
        openreview.api.OpenReviewClient(token=token, baseurl=current_app.config['OPENREVIEW_BASEURL_V2']),
    )

def require_auth(func):
    """Decorator that enforces Authorization header and populates flask.g clients."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token is None:
            current_app.logger.error('No Authorization token in headers')
            return jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

        try:
            or_client, or_client_v2 = get_client(token)
            user_id = get_user_id(or_client)
        except openreview.OpenReviewException as error_handle:
            current_app.logger.error(str(error_handle), exc_info=True)
            return jsonify(format_error(403, str(error_handle))), 403
        except Exception as error_handle:  # pylint: disable=broad-except
            current_app.logger.error(str(error_handle), exc_info=True)
            return jsonify(format_error(500, f'Internal server error: {error_handle}')), 500

        if not user_id:
            current_app.logger.error('No Authorization token in headers')
            return jsonify(format_error(403, 'Forbidden: No Authorization token in headers')), 403

        g.or_client = or_client
        g.or_client_v2 = or_client_v2
        g.user_id = user_id

        return func(*args, **kwargs)

    return wrapper

