import pytest
import numpy as np
import csv
from pathlib import Path
from expertise.models import specter2_scincl, multifacet_recommender
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.utils.utils import generate_sparse_scores


@pytest.fixture
def create_specncl_model_for_precision_test(tmp_path):
    """
    Fixture to create a specter2_scincl EnsembleModel instance with minimal dummy data
    for ArchivesDataset and SubmissionsDataset within a temporary path,
    specifically for testing score precision.
    """
    def _create_specncl(config):
        # Create minimal dummy data files for datasets
        dummy_archives_path = tmp_path / "dummy_archives_specncl"
        dummy_archives_path.mkdir()
        # CORRECTED: Added 'content' key
        (dummy_archives_path / "pub1.jsonl").write_text('{"id": "pub1", "content": {"title": "Test Publication One", "abstract": "Abstract for test pub one."}}')
        (dummy_archives_path / "pub2.jsonl").write_text('{"id": "pub2", "content": {"title": "Test Publication Two", "abstract": "Abstract for test pub two."}}')

        dummy_submissions_path = tmp_path / "dummy_submissions_specncl"
        dummy_submissions_path.mkdir()
        # CORRECTED: Added 'content' key
        (dummy_submissions_path / "sub1.jsonl").write_text('{"id": "sub1", "content": {"title": "Test Submission One", "abstract": "Abstract for test sub one."}}')
        (dummy_submissions_path / "sub2.jsonl").write_text('{"id": "sub2", "content": {"title": "Test Submission Two", "abstract": "Abstract for test sub two."}}')

        archives_dataset = ArchivesDataset(archives_path=dummy_archives_path)
        submissions_dataset = SubmissionsDataset(submissions_path=dummy_submissions_path)

        # Placeholder for specter_dir, as we're not actually running specter embedding here,
        # but the model constructor requires it.
        specter_dir = tmp_path / "specter_model_placeholder"
        specter_dir.mkdir(exist_ok=True)
        
        ens_predictor = specter2_scincl.EnsembleModel(
            specter_dir=str(specter_dir),
            work_dir=str(tmp_path),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            specter_batch_size=config['model_params'].get('specter_batch_size', 16),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False),
            normalize_scores=config['model_params'].get('normalize_scores', True),
        )

        ens_predictor.set_archives_dataset(archives_dataset)
        ens_predictor.set_submissions_dataset(submissions_dataset)
        return ens_predictor
    return _create_specncl

@pytest.fixture
def create_smfr_model_for_precision_test(tmp_path):
    """
    Fixture to create a multifacet_recommender EnsembleModel instance with minimal dummy data
    for ArchivesDataset and SubmissionsDataset within a temporary path,
    specifically for testing score precision.
    """
    def _create_smfr(config):
        # Create minimal dummy data files for datasets
        dummy_archives_path = tmp_path / "dummy_archives_smfr"
        dummy_archives_path.mkdir()
        # CORRECTED: Added 'content' key
        (dummy_archives_path / "pub1.jsonl").write_text('{"id": "pub1", "content": {"title": "Test Publication One", "abstract": "Abstract for test pub one."}}')
        (dummy_archives_path / "pub2.jsonl").write_text('{"id": "pub2", "content": {"title": "Test Publication Two", "abstract": "Abstract for test pub two."}}')

        dummy_submissions_path = tmp_path / "dummy_submissions_smfr"
        dummy_submissions_path.mkdir()
        # CORRECTED: Added 'content' key
        (dummy_submissions_path / "sub1.jsonl").write_text('{"id": "sub1", "content": {"title": "Test Submission One", "abstract": "Abstract for test sub one."}}')
        (dummy_submissions_path / "sub2.jsonl").write_text('{"id": "sub2", "content": {"title": "Test Submission Two", "abstract": "Abstract for test sub two."}}')

        archives_dataset = ArchivesDataset(archives_path=dummy_archives_path)
        submissions_dataset = SubmissionsDataset(submissions_path=dummy_submissions_path)

        # Placeholder paths for external model directories required by MFR constructor
        specter_dir = tmp_path / "specter_model_placeholder_smfr"
        specter_dir.mkdir(exist_ok=True)
        mfr_feature_vocab_file = tmp_path / "mfr_feature_vocab_file_placeholder"
        mfr_feature_vocab_file.touch() # Create an empty file
        mfr_checkpoint_dir = tmp_path / "mfr_model_checkpoint_placeholder"
        mfr_checkpoint_dir.mkdir(exist_ok=True)

        ens_predictor = multifacet_recommender.EnsembleModel(
            specter_dir=str(specter_dir),
            mfr_feature_vocab_file=str(mfr_feature_vocab_file),
            mfr_checkpoint_dir=str(mfr_checkpoint_dir),
            mfr_epochs=config['model_params'].get('mfr_epochs', 100),
            work_dir=str(tmp_path),
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
    return _create_smfr

# --- New tests for scoring precision (unchanged, as the issue was in fixtures) ---

def test_specter2_scincl_all_scores_precision(tmp_path, create_specncl_model_for_precision_test):
    """
    Verify that scores generated by the specter2_scincl.EnsembleModel.all_scores method
    adhere to the precision rule (approximately <= 4 decimal places) and are within the [0, 1] range.
    This test directly calls all_scores with dummy embedding files.
    """
    config = {
        'name': 'test_specncl_precision',
        'model_params': {
            'use_title': True,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'work_dir': tmp_path,
            'normalize_scores': True
        }
    }

    specnclModel = create_specncl_model_for_precision_test(config)

    # Create dummy embedding files, which are the *inputs* to all_scores
    publications_path = tmp_path / 'publications_specncl'
    publications_path.mkdir()
    submissions_path = tmp_path / 'submissions_specncl'
    submissions_path.mkdir()

    # These are simplified dummy embeddings; in a real scenario, they would come from embed_publications/submissions
    (publications_path / 'pub2vec_specter.jsonl').write_text('{"paper_id": "pub1", "embedding": [0.1, 0.2, 0.3]}\n{"paper_id": "pub2", "embedding": [0.4, 0.5, 0.6]}')
    (publications_path / 'pub2vec_scincl.jsonl').write_text('{"paper_id": "pub1", "embedding": [0.1, 0.2, 0.3]}\n{"paper_id": "pub2", "embedding": [0.4, 0.5, 0.6]}')
    (submissions_path / 'sub2vec_specter.jsonl').write_text('{"paper_id": "sub1", "embedding": [0.7, 0.8, 0.9]}\n{"paper_id": "sub2", "embedding": [0.1, 0.3, 0.5]}')
    (submissions_path / 'sub2vec_scincl.jsonl').write_text('{"paper_id": "sub1", "embedding": [0.7, 0.8, 0.9]}\n{"paper_id": "sub2", "embedding": [0.1, 0.3, 0.5]}')

    scores_output_path = tmp_path / 'scores_specncl'
    scores_output_path.mkdir()
    
    # Directly call the all_scores method
    all_scores = specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        scores_path=scores_output_path.joinpath(config['name'] + '.csv')
    )

    assert len(all_scores) > 0, "No scores were generated by specter2_scincl model."

    for row in all_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])
        
        assert score >= 0.0 and score <= 1.0, f"Score {score} is out of expected [0, 1] range."
        
        # Check score precision (max 4 decimal places using numpy for robust float comparison)
        score_rounded = round(score, 4)
        assert np.isclose(score, score_rounded, atol=1e-5), f"Score {score} has more than 4 decimal places of precision."

def test_multifacet_recommender_all_scores_precision(tmp_path, create_smfr_model_for_precision_test):
    """
    Verify that scores generated by the multifacet_recommender.EnsembleModel.all_scores method
    adhere to the precision rule (approximately <= 4 decimal places) and are within the [0, 1] range.
    This test directly calls all_scores with dummy embedding files.
    """
    config = {
        'name': 'test_smfr_precision',
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
    smfrModel = create_smfr_model_for_precision_test(config)

    # Create dummy embedding files, which are the *inputs* to all_scores
    publications_path = tmp_path / 'publications_smfr'
    publications_path.mkdir()
    submissions_path = tmp_path / 'submissions_smfr'
    submissions_path.mkdir()

    # These are simplified dummy embeddings
    (publications_path / 'pub2vec.jsonl').write_text('{"paper_id": "pub1", "embedding": [0.1, 0.2, 0.3]}\n{"paper_id": "pub2", "embedding": [0.4, 0.5, 0.6]}')
    (submissions_path / 'sub2vec.jsonl').write_text('{"paper_id": "sub1", "embedding": [0.7, 0.8, 0.9]}\n{"paper_id": "sub2", "embedding": [0.1, 0.3, 0.5]}')

    scores_output_path = tmp_path / 'scores_smfr'
    scores_output_path.mkdir()

    # Directly call the all_scores method
    all_scores = smfrModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec.jsonl'),
        mfr_publications_path=None, # Assuming MFR specific embeddings are not used for this test
        specter_submissions_path=submissions_path.joinpath('sub2vec.jsonl'),
        mfr_submissions_path=None, # Assuming MFR specific embeddings are not used for this test
        scores_path=scores_output_path.joinpath(config['name'] + '.csv')
    )

    assert len(all_scores) > 0, "No scores were generated by multifacet_recommender model."

    for row in all_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])
        
        assert score >= 0.0 and score <= 1.0, f"Score {score} is out of expected [0, 1] range."
        
        # Check score precision (max 4 decimal places using numpy for robust float comparison)
        score_rounded = round(score, 4)
        assert np.isclose(score, score_rounded, atol=1e-5), f"Score {score} has more than 4 decimal places of precision."

def test_generate_sparse_scores_precision(tmp_path):
    """
    Verify that scores generated by generate_sparse_scores adhere to the precision rule
    (approximately <= 4 decimal places) and are within the [0, 1] range.
    This test directly calls generate_sparse_scores with dummy input scores.
    """
    # Create dummy scores with varying precision as input
    initial_scores = [
        ["sub1", "~user1", 0.99999999], # Should be rounded to 1.0000
        ["sub1", "~user2", 0.12345678], # Should be rounded to 0.1235
        ["sub2", "~user1", 0.50000000],
        ["sub2", "~user3", 0.00000123], # Should be rounded to 0.0000
        ["sub3", "~user4", 1.00000000],
        ["sub4", "~user5", 0.00000000]
    ]

    # CORRECTED: Removed output_file argument
    sparse_scores = generate_sparse_scores(initial_scores, sparse_value=0.0001)

    assert len(sparse_scores) > 0, "No sparse scores were generated."

    for row in sparse_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])

        assert score >= 0.0 and score <= 1.0, f"Sparse score {score} is out of expected [0, 1] range."

        # Check score precision (max 4 decimal places using numpy for robust float comparison)
        score_rounded = round(score, 4)
        assert np.isclose(score, score_rounded, atol=1e-5), f"Sparse score {score} has more than 4 decimal places of precision."
