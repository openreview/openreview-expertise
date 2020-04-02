from expertise import create_dataset
from unittest.mock import patch, MagicMock
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

def search_profiles(emails=None, ids=None):
    if emails:
        with open('fakeData.json') as json_file:
            data = json.load(json_file)
        data['profiles']

def get_group(group_id):
    with open('fakeData.json') as json_file:
        data = json.load(json_file)
    return data['groups'][group_id]

@patch('openreview_client.search_profiles', side_effect=)
@patch('openreview_client.get_group', side_effect=get_group)
def test_get_profile_ids(mock_get_group):

