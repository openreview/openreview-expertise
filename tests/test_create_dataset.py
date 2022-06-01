from expertise.create_dataset import OpenReviewExpertise
from expertise.service.utils import mock_client as mock_v2
from unittest.mock import patch, MagicMock
from collections import defaultdict
import openreview
import json, re, shutil, os

def mock_client():
    client = MagicMock(openreview.Client)

    def get_profile(email_or_id = None):
        mock_profile = {
            "id": "~Test_User1",
            "content": {
                "preferredEmail": "Test_User1@mail.com",
                "emails": [
                    "Test_User1@mail.com"
                ]
            }
        }
        if email_or_id:
            tildematch = re.compile('~.+')
            if tildematch.match(email_or_id):
                att = 'id'
            else:
                att = 'email'
            with open('tests/data/fakeData.json') as json_file:
                data = json.load(json_file)
            profiles = data['profiles']
            for profile in profiles:
                profile = openreview.Profile.from_json(profile)
                if att == 'id':
                    if profile.id == email_or_id:
                        return profile
                else:
                    if email_or_id in profile.content.get('emails'):
                        return profile
        return openreview.Profile.from_json(mock_profile)

    def get_note(id):
        with open('tests/data/fakeData.json') as json_file:
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
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
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
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(confirmedEmails=None, ids=None, term=None):
        with open('tests/data/fakeData.json') as json_file:
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
    client.token = None

    return client

def test_convert_to_list(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    groupList = or_expertise.convert_to_list('group.cc')
    assert groupList == ['group.cc']

    groupList = or_expertise.convert_to_list(['group.cc', 'group.aa'])
    assert groupList == ['group.cc', 'group.aa']

def test_get_papers_from_group(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    all_papers = or_expertise.get_papers_from_group('DEF.cc')
    assert len(all_papers) == 145
    if os.path.isfile('publications_by_profile_id.json'):
        os.remove('publications_by_profile_id.json')

def test_get_profile_ids(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    ids, _ = or_expertise.get_profile_ids(group_ids=['DEF.cc'])
    assert len(ids) == 99
    for tilde_id, email_id in ids:
        assert '~' in tilde_id
        assert '@' in email_id

    ids, _ = or_expertise.get_profile_ids(reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de'])
    assert len(ids) == 3
    assert sorted(ids) == sorted([('~Romeo_Mraz2', 'hkinder2b@army.mil'), ('~Stacee_Powlowski9', 'mdagg5@1und1.de'), ('~Stanley_Bogisich4', 'cchippendale26@smugmug.com')])

    ids, _ = or_expertise.get_profile_ids(group_ids=['DEF.cc'], reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de'])
    assert len(ids) == 99

    ids, inv_ids = or_expertise.get_profile_ids(reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de', 'mondragon@email.com'])
    assert len(ids) == 3
    assert sorted(ids) == sorted([('~Romeo_Mraz2', 'hkinder2b@army.mil'), ('~Stacee_Powlowski9', 'mdagg5@1und1.de'), ('~Stanley_Bogisich4', 'cchippendale26@smugmug.com')])
    assert len(inv_ids) == 1
    assert inv_ids[0] == 'mondragon@email.com'


def test_get_publications(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    publications = or_expertise.get_publications('~Carlos_Mondragon1')
    assert publications == []

    publications = or_expertise.get_publications('~Perry_Volkman3')
    assert len(publications) == 3

    minimum_pub_date = 1554819115
    config = {
        'dataset': {
            'minimum_pub_date': minimum_pub_date
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication['cdate'] > minimum_pub_date

    top_recent_pubs = 2
    config = {
        'dataset': {
            'top_recent_pubs': top_recent_pubs
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication['cdate'] > minimum_pub_date

    top_recent_pubs = 1
    config = {
        'dataset': {
            'top_recent_pubs': top_recent_pubs,
            'minimum_pub_date': minimum_pub_date
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman3')
    assert len(publications) == 1
    assert publications[0]['cdate'] > minimum_pub_date

    top_recent_pubs = 1
    config = {
        'dataset': {
            'or': {
                'top_recent_pubs': top_recent_pubs,
                'minimum_pub_date': minimum_pub_date
            }
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication['cdate'] > minimum_pub_date

    top_recent_pubs = '10%'
    config = {
        'dataset': {
            'or': {
                'top_recent_pubs': top_recent_pubs,
                'minimum_pub_date': minimum_pub_date
            }
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman3')
    assert len(publications) == 2
    for publication in publications:
        assert publication['cdate'] > minimum_pub_date

def test_get_submissions(client, openreview_client):
    config = {
        'dataset': {
            'directory': 'tests/data/'
        },
        'csv_submissions': 'csv_submissions.csv'
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    submissions = or_expertise.get_submissions()
    print(submissions)
    assert json.dumps(submissions) == json.dumps({
        'GhJKSuij': {
            "id": "GhJKSuij",
            "content": {
                "title": "Manual & mechan traction",
                "abstract":"Etiam vel augue. Vestibulum rutrum rutrum neque. Aenean auctor gravida sem."
                }
            },
        'KAeiq76y': {
            "id": "KAeiq76y",
            "content": {
                "title": "Aorta resection & anast",
                "abstract":"Morbi non lectus. Aliquam sit amet diam in magna bibendum imperdiet. Nullam orci pede, venenatis non, sodales sed, tincidunt eu, felis.Fusce posuere felis sed lacus. Morbi sem mauris, laoreet ut, rhoncus aliquet, pulvinar sed, nisl. Nunc rhoncus dui vel sem."
                }
            }
    })

def get_paperhash(prefix, title):
    return prefix + title

@patch('openreview.tools.get_paperhash', side_effect=get_paperhash)
def test_retrieve_expertise(get_paperhash, client, openreview_client):
    config = {
        'use_email_ids': False,
        'match_group': 'DEF.cc'
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    expertise = or_expertise.retrieve_expertise()

    with open('tests/data/fakeData.json') as json_file:
        data = json.load(json_file)
    profiles = data['profiles']
    for profile in profiles:
        if len(profile['publications']) > 0:
            if profile['id'] == '~Perry_Volkman3':
                assert len(expertise[profile['id']]) < len(profile['publications'])
            else:
                assert len(expertise[profile['id']]) == len(profile['publications'])

def test_get_submissions_from_invitation(client, openreview_client):
    config = {
        'use_email_ids': False,
        'match_group': 'DEF.cc',
        'paper_invitation': 'DEF.cc/-/Submission'
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    submissions = or_expertise.get_submissions()
    retrieved_titles = [sub['content']['title'] for sub in submissions.values()]
    assert len(retrieved_titles) == 2
    assert retrieved_titles[0] != retrieved_titles[1]
    assert "Repair Right Metatarsal, Percutaneous Endoscopic Approach" in retrieved_titles
    assert "Bypass L Com Iliac Art to B Com Ilia, Perc Endo Approach" in retrieved_titles
    print(submissions)

def test_get_by_submissions_from_paper_id(client, openreview_client):
    # Get a paper ID
    target_paper = list(openreview.tools.iterget_notes(client, invitation='DEF.cc/-/Submission'))[0]
    config = {
        'paper_id': f"{target_paper.id}"
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    submissions = or_expertise.get_submissions()
    print(submissions)
    assert target_paper.id in submissions.keys()
    retrieved_paper = submissions[target_paper.id]
    assert retrieved_paper['content']['title'] == target_paper.content['title']
    assert retrieved_paper['content']['abstract'] == target_paper.content['abstract']
    assert retrieved_paper['id'] == target_paper.id

def test_get_profile():
    openreview_client = mock_client()
    user_profile = openreview_client.get_profile()
    assert user_profile.id == '~Test_User1'
    assert user_profile.content['preferredEmail'] == 'Test_User1@mail.com'