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