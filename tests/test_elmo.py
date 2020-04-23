from unittest.mock import patch, MagicMock
from pathlib import Path
import openreview
import json
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import elmo

def test_elmo(tmp_path):
    archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
    submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

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

    elmoModel = elmo.Model(
        archives_dataset,
        submissions_dataset,
        use_title=config['model_params'].get('use_title'),
        use_abstract=config['model_params'].get('use_abstract'),
        use_cuda=config['model_params'].get('use_cuda'),
        batch_size=config['model_params'].get('batch_size'),
        knn=config['model_params'].get('knn')
    )

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
