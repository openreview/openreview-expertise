from expertise import create_dataset
from unittest.mock import patch, MagicMock
from collections import defaultdict
from expertise.dataset import ArchivesDataset
import openreview
import json

def test_convert_to_list():
    groupList = create_dataset.convert_to_list('group.cc')
    assert groupList == ['group.cc']

    groupList = create_dataset.convert_to_list(['group.cc', 'group.aa'])
    assert groupList == ['group.cc', 'group.aa']

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
            if profile.content.get('emails') and len(profile.content.get('emails')):
                profiles_dict_emails[profile.content['emails'][0]] = profile
            profiles_dict_tilde[profile.id] = profile
        if emails:
            return_value = {}
            for email in emails:
                if profiles_dict_emails.get(email, False):
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
    ids, _ = create_dataset.get_profile_ids(openreview_client, group_ids=['ABC.cc'])
    assert len(ids) == 100
    for tilde_id, email_id in ids:
        # ~Arianna_Daugherty3 does not have emails, so both fields should have her tilde ID
        if tilde_id == '~Arianna_Daugherty3':
            assert '~' in tilde_id
            assert '~' in email_id
        else :
            assert '~' in tilde_id
            assert '@' in email_id

    ids, _ = create_dataset.get_profile_ids(openreview_client, reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de'])
    assert len(ids) == 3
    assert sorted(ids) == sorted([('~Romeo_Mraz2', 'hkinder2b@army.mil'), ('~Stacee_Powlowski9', 'mdagg5@1und1.de'), ('~Stanley_Bogisich4', 'cchippendale26@smugmug.com')])

    ids, _ = create_dataset.get_profile_ids(openreview_client, group_ids=['ABC.cc'], reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de'])
    assert len(ids) == 100

    ids, inv_ids = create_dataset.get_profile_ids(openreview_client, reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de', 'mondragon@email.com'])
    assert len(ids) == 3
    assert sorted(ids) == sorted([('~Romeo_Mraz2', 'hkinder2b@army.mil'), ('~Stacee_Powlowski9', 'mdagg5@1und1.de'), ('~Stanley_Bogisich4', 'cchippendale26@smugmug.com')])
    assert len(inv_ids) == 1
    assert inv_ids[0] == 'mondragon@email.com'

def iterget_notes(openreview_client, content):
    author_id = content['authorids']
    with open('tests/data/fakeData.json') as json_file:
        data = json.load(json_file)
    profiles = data['profiles']
    for profile in profiles:
        if profile['id'] == author_id:
            return [openreview.Note.from_json(publication) for publication in profile['publications']]
    return []

@patch('openreview.tools.iterget_notes', side_effect=iterget_notes)
def test_get_publications(mock_iterget_notes):
    config = {}
    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Carlos_Mondragon1')
    assert publications == []

    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Perry_Volkman3')
    assert len(publications) == 3

    minimum_pub_date = 1554819115
    config = {
        'dataset': {
            'minimum_pub_date': minimum_pub_date
        }
    }
    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication.cdate > minimum_pub_date

    top_recent_pubs = 2
    config = {
        'dataset': {
            'top_recent_pubs': top_recent_pubs
        }
    }
    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication.cdate > minimum_pub_date

    top_recent_pubs = 1
    config = {
        'dataset': {
            'top_recent_pubs': top_recent_pubs,
            'minimum_pub_date': minimum_pub_date
        }
    }
    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Perry_Volkman3')
    assert len(publications) == 1
    assert publications[0].cdate > minimum_pub_date

    top_recent_pubs = 1
    config = {
        'dataset': {
            'or': {
                'top_recent_pubs': top_recent_pubs,
                'minimum_pub_date': minimum_pub_date
            }
        }
    }
    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication.cdate > minimum_pub_date

    top_recent_pubs = '10%'
    config = {
        'dataset': {
            'or': {
                'top_recent_pubs': top_recent_pubs,
                'minimum_pub_date': minimum_pub_date
            }
        }
    }
    publications = create_dataset.get_publications(MagicMock(openreview.Client), config, '~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication.cdate > minimum_pub_date

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

    archives_dataset = ArchivesDataset(archives_path=archive_dir)

    with open('tests/data/fakeData.json') as json_file:
        data = json.load(json_file)
    profiles = data['profiles']
    for profile in profiles:
        if len(profile['publications']) > 0:
            assert len(archives_dataset[profile['id']]) == len(profile['publications'])
