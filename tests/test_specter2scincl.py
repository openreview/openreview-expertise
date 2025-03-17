from unittest.mock import patch, MagicMock
from pathlib import Path

import time
import numpy
import openreview
import json
import pytest
import numpy as np
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import specter2_scincl
import redisai

@pytest.fixture
def create_specncl():
    def simple_specncl(config):
        archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
        submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))

        ens_predictor = specter2_scincl.EnsembleModel(
            specter_dir=config['model_params'].get('specter_dir', "../expertise-utils/specter/"),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            specter_batch_size=config['model_params'].get('specter_batch_size', 16),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False)
        )

        ens_predictor.set_archives_dataset(archives_dataset)
        ens_predictor.set_submissions_dataset(submissions_dataset)
        return ens_predictor
    return simple_specncl


def test_specncl_scores(tmp_path, create_specncl):
    config = {
        'name': 'test_specncl',
        'model_params': {
            'use_title': True,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'work_dir': tmp_path
        }
    }

    specnclModel = create_specncl(config)

    publications_path = tmp_path / 'publications'
    publications_path.mkdir()
    submissions_path = tmp_path / 'submissions'
    submissions_path.mkdir()
    start = time.time()
    specnclModel.embed_publications(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl')
    )
    specnclModel.embed_submissions(
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
    )

    scores_path = tmp_path / 'scores'
    scores_path.mkdir()
    all_scores = specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )
    from_scratch = time.time() - start
    print(f"From scratch: {from_scratch}")

    # Cache scores
    with open(scores_path.joinpath(config['name'] + '.csv'), 'r') as f:
        scores_cache = [line for line in f.readlines()]

    # Cache sub2vec_specter.jsonl and pub2vec_scincl.jsonl
    with open(submissions_path.joinpath('sub2vec_specter.jsonl'), 'r') as f:
        submissions_cache = [line for line in f.readlines()]
    with open(publications_path.joinpath('pub2vec_scincl.jsonl'), 'r') as f:
        publications_cache = [line for line in f.readlines()]

    # Delete some submissions
    with open(submissions_path.joinpath('sub2vec_specter.jsonl'), 'r') as f:
        # Filter out the first 2 embeddings
        submissions = [json.loads(line) for line in f.readlines()]
        submissions = submissions[2:]
    with open(submissions_path.joinpath('sub2vec_specter.jsonl'), 'w') as f:
        for submission in submissions:
            f.write(json.dumps(submission) + '\n')

    # Delete some publications
    with open(publications_path.joinpath('pub2vec_scincl.jsonl'), 'r') as f:
        # Filter out the first 2 embeddings
        publications = [json.loads(line) for line in f.readlines()]
        publications = publications[2:]
    with open(publications_path.joinpath('pub2vec_scincl.jsonl'), 'w') as f:
        for publication in publications:
            f.write(json.dumps(publication) + '\n')

    config = {
        'name': 'test_specncl_chk',
        'model_params': {
            'use_title': True,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'work_dir': tmp_path,
            'emb_checkpoint': True
        }
    }

    specnclModel = create_specncl(config)

    start = time.time()
    specnclModel.embed_publications(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl')
    )
    specnclModel.embed_submissions(
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
    )
    all_scores = specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )
    from_checkpoint = time.time() - start
    print(f"From checkpoint: {from_checkpoint}")

    assert from_scratch > from_checkpoint
    # Check all embeddings are the same
    ## Publications
    with open(publications_path.joinpath('pub2vec_scincl.jsonl'), 'r') as f:
        publications = [line for line in f.readlines()]
        assert set(publications) == set(publications_cache)

    ## Submissions  
    with open(submissions_path.joinpath('sub2vec_specter.jsonl'), 'r') as f:
        submissions = [line for line in f.readlines()]
        assert set(submissions) == set(submissions_cache)

    # Check all score
    with open(scores_path.joinpath(config['name'] + '.csv'), 'r') as f:
        scores = [line for line in f.readlines()]
        assert set(scores) == set(scores_cache)


def test_sparse_scores(tmp_path, create_specncl):
    config = {
        'name': 'test_specncl',
        'model_params': {
            'use_title': True,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'sparse_value': 1,
            'work_dir': tmp_path
        }
    }

    specnclModel = create_specncl(config)

    publications_path = tmp_path / 'publications'
    publications_path.mkdir()
    submissions_path = tmp_path / 'submissions'
    submissions_path.mkdir()
    specnclModel.embed_publications(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl')
    )
    specnclModel.embed_submissions(
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
    )

    scores_path = tmp_path / 'scores'
    scores_path.mkdir()
    all_scores = specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )

    if config['model_params'].get('sparse_value'):
        all_scores = specnclModel.sparse_scores(
            scores_path=scores_path.joinpath(config['name'] + '_sparse.csv')
        )

    assert len(all_scores) == 8
    for row in all_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])
        assert len(submission_id) >= 1
        assert len(profile_id) >= 1
        assert profile_id.startswith('~')
        assert score >= 0 and score <= 1