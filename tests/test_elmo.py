from unittest.mock import patch, MagicMock
from pathlib import Path
import openreview
import json
import pytest
import numpy as np
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import elmo

@pytest.fixture
def create_elmo():
    def simple_elmo(config):
        archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
        submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

        return elmo.Model(
            archives_dataset,
            submissions_dataset,
            use_title=config['model_params'].get('use_title'),
            use_abstract=config['model_params'].get('use_abstract'),
            use_cuda=config['model_params'].get('use_cuda'),
            batch_size=config['model_params'].get('batch_size'),
            knn=config['model_params'].get('knn')
        )
    return simple_elmo

def test_elmo_scores(tmp_path, create_elmo):
    config = {
        'name': 'test_elmo',
        'model_params': {
            'use_title': False,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'knn': None,
            'skip_elmo': False
        }
    }

    elmoModel = create_elmo(config)

    if not config['model_params'].get('skip_elmo', False):
        publications_path = tmp_path / 'publications'
        publications_path.mkdir()
        submissions_path = tmp_path / 'submissions'
        submissions_path.mkdir()
        elmoModel.embed_publications(publications_path=publications_path.joinpath('pub2vec.pkl'))
        elmoModel.embed_submssions(submissions_path=submissions_path.joinpath('sub2vec.pkl'))

    scores_path = tmp_path / 'scores'
    scores_path.mkdir()
    all_scores = elmoModel.all_scores(
        publications_path=publications_path.joinpath('pub2vec.pkl'),
        submissions_path=submissions_path.joinpath('sub2vec.pkl'),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )

def test_normalize_scores(create_elmo):
    config = {
        'name': 'test_elmo',
        'model_params': {
            'use_title': False,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'knn': None,
            'skip_elmo': False
        }
    }

    elmoModel = create_elmo(config)

    score_matrix = np.array([
        [1,2,3],
        [5,5,5],
        [1,0,1]
    ])

    normalized_matrix = elmoModel.normalize_scores(score_matrix)

    print(normalized_matrix)

    result_array = np.array([
        [0.,  0.5, 1. ],
        [0.5,  0.5,  0.5 ],
        [1.,  0.,  1. ]
    ])
    assert np.array_equal(result_array, normalized_matrix)
