from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy
import openreview
import json
import pytest
import numpy as np
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import multifacet_recommender
import redisai
from expertise.utils.utils import generate_sparse_scores

@pytest.fixture
def create_smfr():
    def simple_smfr(config):
        archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
        submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

        ens_predictor = multifacet_recommender.EnsembleModel(
            specter_dir=config['model_params'].get('specter_dir', "../expertise-utils/specter/"),
            mfr_feature_vocab_file=config['model_params'].get('mfr_feature_vocab_file', "../expertise-utils/multifacet_recommender/feature_vocab_file"),
            mfr_checkpoint_dir=config['model_params'].get('mfr_checkpoint_dir', "../expertise-utils/multifacet_recommender/mfr_model_checkpoint/"),
            mfr_epochs=config['model_params'].get('mfr_epochs', 100),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            specter_batch_size=config['model_params'].get('specter_batch_size', 16),
            mfr_batch_size=config['model_params'].get('mfr_batch_size', 50),
            merge_alpha=config['model_params'].get('merge_alpha', 0.8),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False)
        )

        ens_predictor.set_archives_dataset(archives_dataset)
        ens_predictor.set_submissions_dataset(submissions_dataset)
        return ens_predictor
    return simple_smfr

@pytest.fixture
def create_specter():
    def simple_specter(config, archives_path=None):
        archives_dataset = ArchivesDataset(archives_path=Path(archives_path) if archives_path else Path('tests/data/archives'))
        submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

        spcter_predictor = multifacet_recommender.SpecterPredictor(
            specter_dir=config['model_params'].get('specter_dir', "../expertise-utils/specter/"),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            batch_size=config['model_params'].get('specter_batch_size', 16),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False)
        )

        spcter_predictor.set_archives_dataset(archives_dataset)
        spcter_predictor.set_submissions_dataset(submissions_dataset)
        return spcter_predictor
    return simple_specter


def test_smfr_scores(tmp_path, create_smfr, create_specter):
    config = {
        'name': 'test_spectermfr',
        'model_params': {
            'use_title': False,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'work_dir': tmp_path
        }
    }

    specterModel = create_specter(config)
    smfrModel = create_smfr(config)

    publications_path = tmp_path / 'publications'
    publications_path.mkdir()
    submissions_path = tmp_path / 'submissions'
    submissions_path.mkdir()
    smfrModel.embed_publications(specter_publications_path=publications_path.joinpath('pub2vec.jsonl'),
                                 mfr_publications_path=None,
                                 skip_specter=config['model_params'].get('skip_specter', False))
    smfrModel.embed_submissions(submissions_path.joinpath('sub2vec.jsonl'),
            mfr_submissions_path=None, skip_specter=config['model_params'].get('skip_specter', False))

    scores_path = tmp_path / 'scores'
    scores_path.mkdir()
    all_scores = smfrModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec.jsonl'),
        mfr_publications_path=None,
        specter_submissions_path=submissions_path.joinpath('sub2vec.jsonl'),
        mfr_submissions_path=None,
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )

    specterModel.embed_publications(publications_path.joinpath('pub2vec.jsonl'))

def test_sparse_scores(tmp_path, create_smfr):
    config = {
        'name': 'test_spectermfr',
        'model_params': {
            'use_title': False,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'work_dir': tmp_path,
            'sparse_value': 1,
            'use_redis': False
        }
    }
    smfrModel = create_smfr(config)

    publications_path = tmp_path / 'publications'
    publications_path.mkdir()
    submissions_path = tmp_path / 'submissions'
    submissions_path.mkdir()
    smfrModel.embed_publications(specter_publications_path=publications_path.joinpath('pub2vec.jsonl'),
                                 mfr_publications_path=None,
                                 skip_specter=config['model_params'].get('skip_specter', False))
    smfrModel.embed_submissions(submissions_path.joinpath('sub2vec.jsonl'),
            mfr_submissions_path=None, skip_specter=config['model_params'].get('skip_specter', False))

    scores_path = tmp_path / 'scores'
    scores_path.mkdir()
    all_scores = smfrModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec.jsonl'),
        mfr_publications_path=None,
        specter_submissions_path=submissions_path.joinpath('sub2vec.jsonl'),
        mfr_submissions_path=None,
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )
    all_users, all_submissions = set(), set()
    for score in all_scores:
        all_users.add(score[1])
        all_submissions.add(score[0])

    if config['model_params'].get('sparse_value'):
        all_scores = generate_sparse_scores(all_scores, config['model_params']['sparse_value'], scores_path.joinpath(config['name'] + '_sparse.csv'))
    
    sparse_users, sparse_submissions = set(), set()
    for score in all_scores:
        sparse_users.add(score[1])
        sparse_submissions.add(score[0])

    assert all_users == sparse_users
    assert all_submissions == sparse_submissions
