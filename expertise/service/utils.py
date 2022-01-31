import openreview
import json
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

class ServerConfig(object):
    '''
    Helps translate fields from API requests to fields usable by the expertise system
    '''
    def __init__(self, starting_config = {}):
        # Loads all fields from the staring config
        # Required fields get None by default
        self.name = None
        self.match_group = None
        self.user_id = None
        self.job_id = None

        # Optional fields
        self.model = starting_config.get('model', None)
        self.model_params = starting_config.get('model_params', {})
        self.exclusion_inv = starting_config.get('exclusion_inv', None)
        self.token = starting_config.get('token', None)
        self.baseurl = starting_config.get('baseurl', None)
        self.baseurl_v2 = starting_config.get('baseurl_v2', None)
        self.paper_invitation = starting_config.get('paper_invitation', None)
        self.paper_id = starting_config.get('paper_id', None)

        # Optional model params
        model_params = starting_config.get('model_params', {})
        self.model_params = {}
        self.model_params['use_title'] = model_params.get('use_title', None)
        self.model_params['use_abstract'] = model_params.get('use_abstract', None)
        self.model_params['average_score'] = model_params.get('average_score', None)
        self.model_params['max_score'] = model_params.get('max_score', None)
        self.model_params['skip_specter'] = model_params.get('skip_specter', None)

    def from_request(self, request):
        pass

    def from_json(self, config):
        pass

    def to_json(self):
        pass