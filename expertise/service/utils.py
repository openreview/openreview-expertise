import openreview
import json
import re
from unittest.mock import MagicMock

# -----------------
# -- Mock Client --
# -----------------
def mock_client(version=1):
    client = MagicMock(openreview.Client)

    def get_user():
        return {
            'user': {
                'id': 'test_user1@mail.com'
            }
        }

    def get_token():
        return None

    def get_note(id):
        if version == 1:
            with open('tests/data/fakeData.json') as json_file:
                data = json.load(json_file)
        elif version == 2:
            with open('tests/data/api2Data.json') as json_file:
                data = json.load(json_file)
        else:
            raise openreview.OpenReviewException('Version number not supported')

        for invitation in data['notes'].keys():
            for note in data['notes'][invitation]:
                if note['id'] == id:
                    return openreview.Note.from_json(note)
        raise openreview.OpenReviewException({'name': 'NotFoundError', 'message': f"The Note {id} was not found", 'status': 404, 'details': {'path': 'id', 'value': id}})

    def get_profile():
        mock_profile = {
            "id": "~Test_User1",
            "content": {
                "preferredEmail": "test_user1@mail.com",
                "emails": [
                    "test_user1@mail.com"
                ]
            }
        }
        return openreview.Profile.from_json(mock_profile)

    def get_notes(id = None,
        paperhash = None,
        forum = None,
        original = None,
        invitation = None,
        replyto = None,
        tauthor = None,
        signature = None,
        writer = None,
        trash = None,
        number = None,
        content = None,
        limit = None,
        offset = None,
        mintcdate = None,
        details = None,
        sort = None):

        if offset != 0:
            return []
        if version == 1:
            with open('tests/data/expertiseServiceData.json') as json_file:
                data = json.load(json_file)
        elif version == 2:
            with open('tests/data/api2Data.json') as json_file:
                data = json.load(json_file)
        else:
            raise openreview.OpenReviewException('Version number not supported')

        if invitation:
            notes=data['notes'][invitation]
            return [openreview.Note.from_json(note) for note in notes]

        if 'authorids' in content:
            authorid = content['authorids']
            profiles = data['profiles']
            for profile in profiles:
                if authorid == profile['id']:
                    return [openreview.Note.from_json(note) for note in profile['publications']]

        return []

    def get_group(group_id):
        if version == 1:
            with open('tests/data/expertiseServiceData.json') as json_file:
                data = json.load(json_file)
        elif version == 2:
            with open('tests/data/api2Data.json') as json_file:
                data = json.load(json_file)
        else:
            raise openreview.OpenReviewException('Version number not supported')
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(confirmedEmails=None, ids=None, term=None):
        if version == 1:
            with open('tests/data/expertiseServiceData.json') as json_file:
                data = json.load(json_file)
        elif version == 2:
            with open('tests/data/api2Data.json') as json_file:
                data = json.load(json_file)
        else:
            raise openreview.OpenReviewException('Version number not supported')
        profiles = data['profiles']
        profiles_dict_emails = {}
        profiles_dict_tilde = {}
        for profile in profiles:
            profile = openreview.Profile.from_json(profile)
            if profile.content.get('emails') and len(profile.content.get('emails')):
                profiles_dict_emails[profile.content['emails'][0]] = profile
            profiles_dict_tilde[profile.id] = profile
        if confirmedEmails:
            return_value = {}
            for email in confirmedEmails:
                if profiles_dict_emails.get(email, False):
                    return_value[email] = profiles_dict_emails[email]

        if ids:
            return_value = []
            for tilde_id in ids:
                return_value.append(profiles_dict_tilde[tilde_id])
        return return_value

    client.get_notes = MagicMock(side_effect=get_notes)
    client.get_note = MagicMock(side_effect=get_note)
    client.get_group = MagicMock(side_effect=get_group)
    client.search_profiles = MagicMock(side_effect=search_profiles)
    client.get_profile = MagicMock(side_effect=get_profile)
    client.get_user = MagicMock(side_effect=get_user)
    client.user = {
        'user': {
            'id': 'test_user1@mail.com'
        }
    }
    client.token = None

    return client
# -----------------
# -- Mock Client --
# -----------------

def get_user_id(openreview_client):
    """
    Returns the user id from an OpenReview client for authenticating access

    :param openreview_client: A logged in client with the user credentials
    :type openreview_client: openreview.Client

    :returns id: The id of the logged in user
    """
    user = openreview_client.user
    return user.get('user', {}).get('id') if user else None

def _get_required_field(req, superkey, key):
    try:
        field = req.pop(key)
    except KeyError:
        raise openreview.OpenReviewException(f"Bad request: required field missing in {superkey}: {key}")
    return field

class APIRequest(object):
    """
    Validates and load objects and fields from POST requests
    """
    def __init__(self,
            name = None):
            
        self.name = name
        self.entityA = {}
        self.entityB = {}
        self.model = {}

    def from_json(self, request = {}):
        root_key = 'request'
        def _get_field_from_request(field):
            return _get_required_field(request, root_key, field)

        def _load_entity_a(entity):
            self._load_entity('entityA', entity, self.entityA)

        def _load_entity_b(entity):
            self._load_entity('entityB', entity, self.entityB)

        # Get the name of the job
        self.name = _get_field_from_request('name')

        # Validate entityA and entityB
        entity_a = _get_field_from_request('entityA')
        entity_b = _get_field_from_request('entityB')

        _load_entity_a(entity_a)
        _load_entity_b(entity_b)

        # Optionally check for model object
        self.model = request.pop('model', {})

        # Check for empty request
        if len(request.keys()) > 0:
            raise openreview.OpenReviewException(f"Bad request: unexpected fields in {root_key}: {list(request.keys())}")
    
    def _load_entity(self, entity_id, source_entity, target_entity):
        '''Load information from an entity into the config'''
        def _get_from_entity(key):
            return _get_required_field(source_entity, entity_id, key)

        type = _get_from_entity('type')
        target_entity['type'] = type
        # Handle type group
        if type == 'Group':
            if 'memberOf' in source_entity.keys():
                target_entity['memberOf'] = _get_from_entity('memberOf')
                # Check for optional expertise field
                if 'expertise' in source_entity.keys():
                    target_entity['expertise'] = source_entity.pop('expertise')
            else:
                raise openreview.OpenReviewException(f"Bad request: no valid {type} properties in {entity_id}")
        # Handle type note
        elif type == 'Note':
            if 'invitation' in source_entity.keys() and 'id' in source_entity.keys():
                raise openreview.OpenReviewException(f"Bad request: only provide a single id or single invitation in {entity_id}")

            if 'invitation' in source_entity.keys():
                target_entity['invitation'] = _get_from_entity('invitation')
            elif 'id' in source_entity.keys():
                target_entity['id'] = _get_from_entity('id')
            else:
                raise openreview.OpenReviewException(f"Bad request: no valid {type} properties in {entity_id}")
        else:
            raise openreview.OpenReviewException(f"Bad request: invalid type in {entity_id}")

        # Check for extra entity fields
        if len(source_entity.keys()) > 0:
            raise openreview.OpenReviewException(f"Bad request: unexpected fields in {entity_id}: {list(source_entity.keys())}")


class JobConfig(object):
    """
    Helps translate fields from API requests to fields usable by the expertise system
    """
    def __init__(self, starting_config = {}, api_request: APIRequest = None):
        """
        Sets default fields from the starting_config and attempts to override from api_request fields
        """
        def _camel_to_snake(camel_str):
            camel_str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', camel_str).lower()
    
        # Set metadata fields from request
        self.name = api_request.name
        self.dataset = starting_config.get('dataset', {})

        # Handle Group cases
        # (for now, only single match group)
        self.match_group = starting_config.get('match_group', None)
        if api_request.entityA['type'] == 'Group':
            self.match_group = api_request.entityA['memberOf']
        elif api_request.entityB['type'] == 'Group':
            self.match_group = api_request.entityB['memberOf']

        # Handle Note cases
        self.paper_invitation = None
        self.paper_id = None
        self.exclusion_inv = None

        if api_request.entityA['type'] == 'Note':
            inv, id = api_request.entityA.get('invitation', None), api_request.entityA.get('id', None)
            excl_inv = api_request.entityA.get('expertise', None)

            if inv:
                self.paper_invitation = inv
            if id:
                self.paper_id = id
            if excl_inv:
                self.exclusion_inv = excl_inv.get('exclusion', {}).get('invitation', None)
        elif api_request.entityB['type'] == 'Note':
            inv, id = api_request.entityB.get('invitation', None), api_request.entityB.get('id', None)
            excl_inv = api_request.entityB.get('expertise', None)

            if inv:
                self.paper_invitation = inv
            if id:
                self.paper_id = id
            if excl_inv:
                self.exclusion_inv = excl_inv.get('exclusion', {}).get('invitation', None)


        # Load optional model params from default config
        self.allowed_model_params = [
            'name',
            'sparseValue',
            'useTitle',
            'useAbstract',
            'scoreComputation',
            'skipSpecter'
        ]
        self.model = starting_config.get('model', None)
        self.model_params = starting_config.get('model_params', {})
        model_params = starting_config.get('model_params', {})
        self.model_params = {}
        self.model_params['use_title'] = model_params.get('use_title', None)
        self.model_params['use_abstract'] = model_params.get('use_abstract', None)
        self.model_params['average_score'] = model_params.get('average_score', None)
        self.model_params['max_score'] = model_params.get('max_score', None)
        self.model_params['skip_specter'] = model_params.get('skip_specter', None)
        self.model_params['batch_size'] = model_params.get('batch_size', 1)
        self.model_params['use_cuda'] = model_params.get('use_cuda', False)

        # Attempt to load any API request model params
        api_model = api_request.model
        if api_model:
            for param in api_model.keys():
                # Handle special cases
                if param == 'scoreComputation':
                    compute_with = api_model.get('scoreComputation', None)
                    if compute_with == 'max':
                        self.model_params['max_score'] = True
                        self.model_params['average_score'] = False
                    elif compute_with == 'avg':
                        self.model_params['max_score'] = False
                        self.model_params['average_score'] = True
                    else:
                        raise openreview.OpenReviewException("Bad request: invalid value in field 'scoreComputation' in 'model' object")
                    continue
                
                # Handle general case
                if param not in self.allowed_model_params:
                    raise openreview.OpenReviewException(f"Bad request: unexpected fields in model: {[param]}")

                snake_param = _camel_to_snake(param)
                self.model_params[snake_param] = api_model[param]

    def to_json(self):
        pre_body = {
            'name': self.name,
            'match_group': self.match_group,
            'dataset': self.dataset,
            'model': self.model,
            'exclusion_inv': self.exclusion_inv,
            'paper_invitation': self.paper_invitation,
            'paper_id': self.paper_id,
            'model_params': self.model_params
        }

        # Remove objects that are none
        body = {}
        body_items = pre_body.items()
        for key, val in body_items:
            # Allow a None token
            if val is not None or key == 'token':
                body[key] = val

        return body