from expertise.create_dataset import OpenReviewExpertise
from unittest.mock import patch, MagicMock
from collections import defaultdict
import openreview
import json
import sys

def mock_client():
    client = MagicMock(openreview.Client)

    def get_profile():
        mock_profile = {
            "id": "~Test_User1",
            "content": {
                "preferredEmail": "Test_User1@mail.com",
                "emails": [
                    "Test_User1@mail.com"
                ]
            }
        }
        return openreview.Profile.from_json(mock_profile)

    def get_note(id):
        with open('tests/data/api2Data.json') as json_file:
            data = json.load(json_file)
        for invitation in data['notes'].keys():
            for note in data['notes'][invitation]:
                if note['id'] == id:
                    return openreview.Note.from_json(note)

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
        with open('tests/data/api2Data.json') as json_file:
            data = json.load(json_file)
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(confirmedEmails=None, ids=None, term=None):
        with open('tests/data/api2Data.json') as json_file:
            data = json.load(json_file)
        profiles = data['profiles']
        profiles_dict_emails = {}
        profiles_dict_tilde = {}
        for profile in profiles:
            profile = openreview.Profile.from_json(profile)
            if profile.content.get('emails') and len(profile.content.get('emails')):
                profile.content['emailsConfirmed'] = profile.content.get('emails')
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

    return client

def test_get_publications():
    openreview_client = mock_client()
    config = {
        'dataset': {
            'top_recent_pubs': 3,
        }
    }
    or_expertise = OpenReviewExpertise(openreview_client, config)
    publications = or_expertise.get_publications('~Carlos_Mondragon1')
    assert publications == []

    publications = or_expertise.get_publications('~Harold_Rice8')
    assert len(publications) == 3
    for pub in publications:
        content = pub['content']
        assert 'value' in content['title'].keys()
        assert 'value' in content['abstract'].keys()
    
    config = {
        'dataset': {
            'top_recent_pubs': 3,
        },
        'version': 2
    }
    or_expertise = OpenReviewExpertise(openreview_client, config)
    publications = or_expertise.get_publications('~Harold_Rice8')
    assert len(publications) == 3
    for pub in publications:
        content = pub['content']
        assert isinstance(content['title'], str)
        assert isinstance(content['abstract'], str)


def test_get_submissions_from_invitation():
    openreview_client = mock_client()
    config = {
        'use_email_ids': False,
        'match_group': 'ABC.cc',
        'paper_invitation': 'ABC.cc/-/Submission',
        'version': 1
    }
    or_expertise = OpenReviewExpertise(openreview_client, config)
    submissions = or_expertise.get_submissions()
    print(submissions)
    assert 'value' in submissions['KHnr1r7H']['content']['title'].keys()
    assert 'value' in submissions['KHnr1r7H']['content']['abstract'].keys()

    config = {
        'use_email_ids': False,
        'match_group': 'ABC.cc',
        'paper_invitation': 'ABC.cc/-/Submission',
        'version': 2
    }
    or_expertise = OpenReviewExpertise(openreview_client, config)
    submissions = or_expertise.get_submissions()
    print(submissions)
    assert not isinstance(submissions['KHnr1r7H']['content']['title'], dict)
    assert isinstance(submissions['KHnr1r7H']['content']['title'], str)
    assert not isinstance(submissions['KHnr1r7H']['content']['abstract'], dict)
    assert isinstance(submissions['KHnr1r7H']['content']['abstract'], str)
    assert json.dumps(submissions) == json.dumps({
        'KHnr1r7H': {
            "id": "KHnr1r7H",
            "content": {
                "title": "Repair Right Metatarsal, Percutaneous Endoscopic Approach",
                "abstract": "Nam ultrices, libero non mattis pulvinar, nulla pede ullamcorper augue, a suscipit nulla elit ac nulla. Sed vel enim sit amet nunc viverra dapibus. Nulla suscipit ligula in lacus.\n\nCurabitur at ipsum ac tellus semper interdum. Mauris ullamcorper purus sit amet nulla. Quisque arcu libero, rutrum ac, lobortis vel, dapibus at, diam."
            }
        },
        'YQtWeE8P': {
            "id": "YQtWeE8P",
            "content": {
                "title": "Bypass L Com Iliac Art to B Com Ilia, Perc Endo Approach",
                "abstract": "Nullam sit amet turpis elementum ligula vehicula consequat. Morbi a ipsum. Integer a nibh.\n\nIn quis justo. Maecenas rhoncus aliquam lacus. Morbi quis tortor id nulla ultrices aliquet.\n\nMaecenas leo odio, condimentum id, luctus nec, molestie sed, justo. Pellentesque viverra pede ac diam. Cras pellentesque volutpat dui."
            }
        }
    })

def test_get_by_submissions_from_paper_id():
    openreview_client = mock_client()
    config = {
        'paper_id': 'KHnr1r7H',
        'version': 1
    }
    or_expertise = OpenReviewExpertise(openreview_client, config)
    submissions = or_expertise.get_submissions()
    print(submissions)
    assert 'value' in submissions['KHnr1r7H']['content']['title'].keys()
    assert 'value' in submissions['KHnr1r7H']['content']['abstract'].keys()

    config = {
        'paper_id': 'KHnr1r7H',
        'version': 2
    }
    or_expertise = OpenReviewExpertise(openreview_client, config)
    submissions = or_expertise.get_submissions()
    print(submissions)
    assert not isinstance(submissions['KHnr1r7H']['content']['title'], dict)
    assert not isinstance(submissions['KHnr1r7H']['content']['abstract'], dict)
    assert json.dumps(submissions) == json.dumps({
        'KHnr1r7H': {
            "id": "KHnr1r7H",
            "content": {
                "title": "Repair Right Metatarsal, Percutaneous Endoscopic Approach",
                "abstract": "Nam ultrices, libero non mattis pulvinar, nulla pede ullamcorper augue, a suscipit nulla elit ac nulla. Sed vel enim sit amet nunc viverra dapibus. Nulla suscipit ligula in lacus.\n\nCurabitur at ipsum ac tellus semper interdum. Mauris ullamcorper purus sit amet nulla. Quisque arcu libero, rutrum ac, lobortis vel, dapibus at, diam."
            }
        }
    })