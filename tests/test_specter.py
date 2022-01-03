from unittest.mock import patch, MagicMock
from pathlib import Path
import openreview
import json
import pytest
import numpy as np
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import multifacet_recommender

@pytest.fixture
def create_spec():
    def simple_spec(config):
        archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
        submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

        spc_predictor = multifacet_recommender.SpecterPredictor(
            specter_dir=config['model_params'].get('specter_dir', "../expertise-utils/specter/"),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            batch_size=config['model_params'].get('batch_size', 16),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value')
        )

        spc_predictor.set_archives_dataset(archives_dataset)
        spc_predictor.set_submissions_dataset(submissions_dataset)
        return spc_predictor
    return simple_spec

def test_duplicate_detection(tmp_path, create_spec):
    submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

    config = {
        'name': 'test_specter',
        'model_params': {
            'use_title': False,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'work_dir': tmp_path,
            'sparse_value': 1
        }
    }
    specModel = create_spec(config)

    other_submissions_dataset = False

    submissions_path = tmp_path / 'submissions'
    submissions_path.mkdir()
    specModel.set_submissions_dataset(submissions_dataset)
    specModel.embed_submissions(submissions_path.joinpath('sub2vec.jsonl'))

    scores_path = tmp_path / 'scores'
    scores_path.mkdir()
    duplicates = specModel.find_duplicates(
        submissions_path=submissions_path.joinpath('sub2vec.jsonl'),
        other_submissions_path=(Path(config['model_params']['other_submissions_path']).joinpath('osub2vec.jsonl') if other_submissions_dataset else None),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )
    for sub_1, sub_2, score in duplicates:
        if score > 0.99:
            assert sub_1 == 'duplicate' or sub_2 == 'duplicate'