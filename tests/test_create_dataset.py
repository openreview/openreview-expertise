from expertise.create_dataset import OpenReviewExpertise
from unittest.mock import patch, MagicMock
from collections import defaultdict
import openreview
import json, re, shutil, os

def test_convert_to_list(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    groupList = or_expertise.convert_to_list('group.cc')
    assert groupList == ['group.cc']

    groupList = or_expertise.convert_to_list(['group.cc', 'group.aa'])
    assert groupList == ['group.cc', 'group.aa']

def test_get_papers_from_group(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    all_papers = or_expertise.get_papers_from_group('DEF.cc/Reviewers')
    assert len(all_papers) == 147
    if os.path.isfile('publications_by_profile_id.json'):
        os.remove('publications_by_profile_id.json')

def test_get_profile_ids(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    ids, _ = or_expertise.get_profile_ids(group_ids=['DEF.cc/Reviewers'])
    assert len(ids) == 100
    for tilde_id, email_id in ids:
        assert '~' in tilde_id
        assert '@' in email_id

    ids, _ = or_expertise.get_profile_ids(reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de'])
    assert len(ids) == 3
    assert sorted(ids) == sorted([('~Romeo_Mraz1', 'hkinder2b@army.mil'), ('~Stacee_Powlowski1', 'mdagg5@1und1.de'), ('~Stanley_Bogisich1', 'cchippendale26@smugmug.com')])

    ids, _ = or_expertise.get_profile_ids(group_ids=['DEF.cc/Reviewers'], reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de'])
    assert len(ids) == 100

    ids, inv_ids = or_expertise.get_profile_ids(reviewer_ids=['hkinder2b@army.mil', 'cchippendale26@smugmug.com', 'mdagg5@1und1.de', 'mondragon@email.com'])
    assert len(ids) == 3
    assert sorted(ids) == sorted([('~Romeo_Mraz1', 'hkinder2b@army.mil'), ('~Stacee_Powlowski1', 'mdagg5@1und1.de'), ('~Stanley_Bogisich1', 'cchippendale26@smugmug.com')])
    assert len(inv_ids) == 1
    assert inv_ids[0] == 'mondragon@email.com'


def test_get_publications(client, openreview_client):
    or_expertise = OpenReviewExpertise(client, openreview_client, {})
    publications = or_expertise.get_publications('~Carlos_Mondragon1')
    assert publications == []

    publications = or_expertise.get_publications('~Perry_Volkman1')
    assert len(publications) == 3

    minimum_pub_date = 1554819115
    config = {
        'dataset': {
            'minimum_pub_date': minimum_pub_date
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman1')
    assert len(publications) == 3
    for publication in publications:
        assert publication['cdate'] > minimum_pub_date

    top_recent_pubs = 2
    config = {
        'dataset': {
            'top_recent_pubs': top_recent_pubs
        }
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    publications = or_expertise.get_publications('~Perry_Volkman1')
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
    publications = or_expertise.get_publications('~Perry_Volkman1')
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
    publications = or_expertise.get_publications('~Perry_Volkman1')
    assert len(publications) == 3
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
    publications = or_expertise.get_publications('~Perry_Volkman1')
    assert len(publications) == 3
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
        'match_group': 'DEF.cc/Reviewers'
    }
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    expertise = or_expertise.retrieve_expertise()
    # Exclude users whose expertise will be posted in API2
    exclude_ids = ['~Kyunghyun_Cho1', '~Raia_Hadsell1']

    with open('tests/data/fakeData.json') as json_file:
        data = json.load(json_file)
    profiles = data['profiles']
    for profile in profiles:
        if len(profile['publications']) > 0:
            if profile['id'] == '~Perry_Volkman1':
                assert len(expertise[profile['id']]) < len(profile['publications'])
            elif profile['id'] in exclude_ids:
                assert len(expertise[profile['id']]) == 0
            else:
                assert len(expertise[profile['id']]) == len(profile['publications'])

def test_get_submissions_from_invitation(client, openreview_client):
    config = {
        'use_email_ids': False,
        'match_group': 'DEF.cc/Reviewers',
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

def test_deduplication(client, openreview_client, helpers):
    author_id = '~Harold_Rice1'
    original_note = list(openreview.tools.iterget_notes(client, content={'authorids': author_id}))[0]
    or_expertise = OpenReviewExpertise(client, openreview_client, {})

    publications = or_expertise.get_publications('~Harold_Rice1')
    assert len(publications) == 3

    note = openreview.Note(
        invitation = 'openreview.net/-/paper',
        readers = ['everyone'],
        writers = ['~SomeTest_User1'],
        signatures = ['~SomeTest_User1'],
        content = original_note.content,
        original = original_note.id
    )
    test_user_client = openreview.Client(username='test@google.com', password=helpers.strong_password)
    note = test_user_client.post_note(note)

    publications = or_expertise.get_publications('~Harold_Rice1')
    assert len(publications) == 3

def test_expertise_selection(client, openreview_client, helpers):
    config = {
        'use_email_ids': False,
        'exclusion_inv': 'DEF.cc/-/Expertise_Selection',
        'match_group': 'DEF.cc/Reviewers'
    }
    author_id = '~Harold_Rice1'
    original_note = list(openreview.tools.iterget_notes(client, content={'authorids': author_id}))[0]
    or_expertise = OpenReviewExpertise(client, openreview_client, config)

    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~Harold_Rice1']) == 3

    note = openreview.api.Note(
        content = {
            "title": { 'value': "test_exclude_def" },
            "abstract": { 'value': original_note.content['abstract'] },
            "authors": { 'value': original_note.content['authorids'] },
            "authorids": { 'value': original_note.content['authorids'] },
        },
        pdate = 1554819115,
        license = 'CC BY-SA 4.0'
    )

    note_edit = openreview_client.post_note_edit(
        invitation='openreview.net/Archive/-/Direct_Upload',
        signatures = ['~SomeTest_User1'],
        note=note
    )
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~Harold_Rice1']) == 4 # New note added
    
    user_client = openreview.Client(username='strevino0@ox.ac.uk', password=helpers.strong_password)
    edge = openreview.Edge(
                        invitation='DEF.cc/-/Expertise_Selection',
                        head=note_edit['note']['id'],
                        tail='~Harold_Rice1',
                        label='Exclude',
                        readers=['DEF.cc', '~Harold_Rice1'],
                        writers=['~Harold_Rice1'],
                        signatures=['~Harold_Rice1']
                    )
    edge = user_client.post_edge(edge)

    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    or_expertise.excluded_ids_by_user = or_expertise.exclude()
    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~Harold_Rice1']) == 3

    # Test API2 expertise selection
def test_expertise_selection_api2(client, openreview_client, helpers):
    config = {
        'use_email_ids': False,
        'exclusion_inv': 'API2/Reviewers/-/Expertise_Selection',
        'match_group': 'API2/Reviewers'
    }
    author_id = '~C.V._Lastname1'
    original_note = client.get_all_notes(content={'authorids': author_id})[0]
    print(original_note)
    or_expertise = OpenReviewExpertise(client, openreview_client, config)

    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~C.V._Lastname1']) == 1, expertise

    test_user_client = openreview.api.OpenReviewClient(username='test@google.com', password=helpers.strong_password)
    note_1 = test_user_client.post_note_edit(
            invitation=f'API2/-/Submission',
            signatures= ['~SomeTest_User1'],
            note=openreview.api.Note(
                content={
                    'title': { 'value': 'test_exclude' },
                    'abstract': { 'value': original_note.content['abstract'] },
                    'authors': { 'value': ['C.V Lastname']},
                    'authorids': { 'value': original_note.content['authorids']},
                    'pdf': {'value': '/pdf/' + 'p' * 40 +'.pdf' },
                    'keywords': {'value': ['aa'] }
                }
            ))

    helpers.await_queue_edit(openreview_client, edit_id=note_1['id'])

    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~C.V._Lastname1']) == 1
    
    user_client = openreview.api.OpenReviewClient(username='testdots@google.com', password=helpers.strong_password)
    user_client.post_edge(
        openreview.api.Edge(invitation = f'API2/Reviewers/-/Expertise_Selection',
            readers = ['API2', '~C.V._Lastname1'],
            writers = ['API2', '~C.V._Lastname1'],
            signatures = ['~C.V._Lastname1'],
            head = note_1['note']['id'],
            tail = '~C.V._Lastname1',
            label = 'Exclude'
    ))

    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    or_expertise.excluded_ids_by_user = or_expertise.exclude()
    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~C.V._Lastname1']) == 1


def test_expertise_inclusion(client, openreview_client, helpers):
    config = {
        'use_email_ids': False,
        'inclusion_inv': 'ABC.cc/-/Expertise_Selection',
        'match_group': 'ABC.cc/Reviewers'
    }
    author_id = '~Harold_Rice1'
    original_note = list(openreview.tools.iterget_notes(client, content={'authorids': author_id}))[0]
    or_expertise = OpenReviewExpertise(client, openreview_client, config)

    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~Harold_Rice1']) == 4 # No edges use all publications

    note = openreview.api.Note(
        content = {
            "title": { 'value': "test_include_abchij" },
            "abstract": { 'value': original_note.content['abstract'] },
            "authors": { 'value': original_note.content['authorids'] },
            "authorids": { 'value': original_note.content['authorids'] },
        },
        pdate = 1554819115,
        license = 'CC BY-SA 4.0'
    )
    exclude_note = openreview.api.Note(
        content = {
            "title": { 'value': "test_exclude_abchij" },
            "abstract": { 'value': original_note.content['abstract'] },
            "authors": { 'value': original_note.content['authorids'] },
            "authorids": { 'value': original_note.content['authorids'] },
        },
        pdate = 1554819115,
        license = 'CC BY-SA 4.0'
    )

    note_edit = openreview_client.post_note_edit(
        invitation='openreview.net/Archive/-/Direct_Upload',
        signatures = ['~SomeTest_User1'],
        note=note
    )
    exclude_note_edit = openreview_client.post_note_edit(
        invitation='openreview.net/Archive/-/Direct_Upload',
        signatures = ['~SomeTest_User1'],
        note=exclude_note
    )
    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~Harold_Rice1']) == 6 # New notes added
    
    # Post this edge to both ABC and HIJ, ABC will be deleted, HIJ will be used for the API tests
    edge = openreview.Edge(
                        invitation='HIJ.cc/-/Expertise_Selection',
                        head=note_edit['note']['id'],
                        tail='~Harold_Rice1',
                        label='Include',
                        readers=['HIJ.cc', '~Harold_Rice1'],
                        writers=['~Harold_Rice1'],
                        signatures=['~Harold_Rice1']
                    )
    edge = client.post_edge(edge)

    edge = openreview.Edge(
        invitation='ABC.cc/-/Expertise_Selection',
        head=note_edit['note']['id'],
        tail='~Harold_Rice1',
        label='Include',
        readers=['ABC.cc', '~Harold_Rice1'],
        writers=['~Harold_Rice1'],
        signatures=['~Harold_Rice1']
    )
    edge = client.post_edge(edge)

    # Use this edge to test that 'Exclude' label edges are not included
    inv = client.get_invitation('ABC.cc/-/Expertise_Selection')
    inv.reply = {
        "readers": {
            "values-copied": [
                "ABC.cc",
                "{signatures}"
            ]
        },
        "writers": {
            "values-copied": [
                "ABC.cc",
                "{signatures}"
            ]
        },
        "signatures": {
            "values-regex": "~.*"
        },
        "content": {
            "head": {
                "type": "Note"
            },
            "tail": {
                "type": "Profile"
            },
            "label": {
                "value-radio": [
                    "Exclude"
                ],
                "required": True
            }
        }
    }
    client.post_invitation(inv)
    edge = openreview.Edge(
                        invitation='ABC.cc/-/Expertise_Selection',
                        head=exclude_note_edit['note']['id'],
                        tail='~Harold_Rice1',
                        label='Exclude',
                        readers=['ABC.cc', '~Harold_Rice1'],
                        writers=['~Harold_Rice1'],
                        signatures=['~Harold_Rice1']
                    )
    edge = client.post_edge(edge)

    inv = client.get_invitation('ABC.cc/-/Expertise_Selection')
    inv.reply = {
        "readers": {
            "values-copied": [
                "ABC.cc",
                "{signatures}"
            ]
        },
        "writers": {
            "values-copied": [
                "ABC.cc",
                "{signatures}"
            ]
        },
        "signatures": {
            "values-regex": "~.*"
        },
        "content": {
            "head": {
                "type": "Note"
            },
            "tail": {
                "type": "Profile"
            },
            "label": {
                "value-radio": [
                    "Include"
                ],
                "required": True
            }
        }
    }
    client.post_invitation(inv)

    assert client.get_edges_count(invitation='ABC.cc/-/Expertise_Selection') == 2

    or_expertise = OpenReviewExpertise(client, openreview_client, config)
    or_expertise.included_ids_by_user = or_expertise.include()
    assert len(or_expertise.included_ids_by_user['~Harold_Rice1']) == 1
    expertise = or_expertise.retrieve_expertise()
    assert len(expertise['~Harold_Rice1']) == 1

    # Clear the inclusion edges for ABC
    client.delete_edges(invitation='ABC.cc/-/Expertise_Selection')