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

        for invitation in data['notes'].keys():
            for note in data['notes'][invitation]:
                if note['id'] == id:
                    return openreview.Note.from_json(note)

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

        if invitation:
            notes=data['notes'][invitation]
            return [openreview.Note.from_json(note) for note in notes]

        if 'authorids' in content:
            authorid = content['authorids']
            if isinstance(authorid, dict):
                authorid = authorid['value'][0]
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
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(confirmedEmails=None, ids=None, term=None):
        if version == 1:
            with open('tests/data/expertiseServiceData.json') as json_file:
                data = json.load(json_file)
        elif version == 2:
            with open('tests/data/api2Data.json') as json_file:
                data = json.load(json_file)
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
