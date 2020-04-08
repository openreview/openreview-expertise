from expertise import create_dataset
from unittest.mock import patch, MagicMock
from collections import defaultdict
import openreview
import json

def test_convert_to_list():
    groupList = create_dataset.convert_to_list('group.cc')
    assert groupList == ['group.cc']

    groupList = create_dataset.convert_to_list(['group.cc', 'group.aa'])
    assert groupList == ['group.cc', 'group.aa']

@patch('openreview.tools.iterget_notes')
def test_get_publications(mock_iterget_notes):
    mock_iterget_notes.return_value = iter([1,2,3])
    publications = create_dataset.get_publications(MagicMock(openreview.Client), '~Carlos_Mondragon1')
    assert publications == [1,2,3]

def mock_client():
    client = MagicMock(openreview.Client)

    def get_group(group_id):
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(emails=None, ids=None, term=None):
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        profiles = data['profiles']
        profiles_dict_emails = {}
        profiles_dict_tilde = {}
        for profile in profiles:
            profile = openreview.Profile.from_json(profile)
            profiles_dict_emails[profile.content['emails'][0]] = profile
            profiles_dict_tilde[profile.id] = profile
        if emails:
            return_value = {}
            for email in emails:
                return_value[email] = profiles_dict_emails[email]

        if ids:
            return_value = []
            for tilde_id in ids:
                return_value.append(profiles_dict_tilde[tilde_id])
        return return_value

    client.get_group = MagicMock(side_effect=get_group)
    client.search_profiles = MagicMock(side_effect=search_profiles)

    return client

def test_get_profile_ids():
    openreview_client = mock_client()
    ids = create_dataset.get_profile_ids(openreview_client, ['ABC.cc'])
    assert len(ids) == 100
    for tilde_id, email_id in ids:
        assert '~' in tilde_id
        assert '@' in email_id

def iterget_notes(openreview_client, content):
    author_id = content['authorids']
    with open('tests/data/fakeData.json') as json_file:
        data = json.load(json_file)
    profiles = data['profiles']
    for profile in profiles:
        if profile['id'] == author_id:
            return [openreview.Note.from_json(publication) for publication in profile['publications']]

def get_paperhash(prefix, title):
    return prefix + title

@patch('openreview.tools.get_paperhash', side_effect=get_paperhash)
@patch('openreview.tools.iterget_notes', side_effect=iterget_notes)
def test_retrieve_expertise(iterget_notes, get_paperhash, tmp_path):
    openreview_client = mock_client()
    config = {
        'use_email_ids': False,
        'match_group': 'ABC.cc'
    }
    metadata = {
        "no_publications_count": 0,
        "no_publications": [],
        "archive_counts": defaultdict(lambda: {'arx': 0, 'bid': 0})
    }
    archive_dir = tmp_path / 'archives'
    archive_dir.mkdir()
    create_dataset.retrieve_expertise(openreview_client, config, defaultdict(list), archive_dir, metadata)

