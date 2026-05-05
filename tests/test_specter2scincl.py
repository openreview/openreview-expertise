from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy
import openreview
import json
import pytest
import numpy as np
import torch
from expertise.dataset import ArchivesDataset, SubmissionsDataset
from expertise.models import specter2_scincl
import redisai
from expertise.utils.utils import generate_sparse_scores


def compute_score_statistics(scores, label=""):
    """Compute histogram-like statistics for affinity scores."""
    score_values = [float(row[2]) for row in scores]
    stats = {
        'count': len(score_values),
        'mean': np.mean(score_values),
        'std': np.std(score_values),
        'min': np.min(score_values),
        'max': np.max(score_values),
        'median': np.median(score_values),
        'q25': np.percentile(score_values, 25),
        'q75': np.percentile(score_values, 75),
    }
    
    # Histogram bins
    hist, bin_edges = np.histogram(score_values, bins=10)
    
    print(f"\n=== Score Statistics {label} ===")
    print(f"Count: {stats['count']}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Q25: {stats['q25']:.4f}")
    print(f"Q75: {stats['q75']:.4f}")
    print("\nHistogram:")
    for i in range(len(hist)):
        print(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]} ({hist[i]/len(score_values)*100:.1f}%)")
    
    return stats

def assert_scores_have_max_4_decimals(csv_path):
    """Scores rounded to 4 decimals is a contract — CSVs exposed to the API
    should never expose more precision than the model actually provides."""
    violations = []
    with open(csv_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            score_str = parts[-1]
            try:
                float(score_str)
            except ValueError:
                continue
            if '.' in score_str:
                decimals = len(score_str.split('.')[1])
                if decimals > 4:
                    violations.append(f"{csv_path}:{line_num}: score={score_str} has {decimals} decimals")
    assert not violations, "Scores with more than 4 decimal places found:\n" + "\n".join(violations)


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
            use_redis=config['model_params'].get('use_redis', False),
            normalize_scores=config['model_params'].get('normalize_scores', True),
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

    assert_scores_have_max_4_decimals(scores_path.joinpath(config['name'] + '.csv'))


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
        all_scores = generate_sparse_scores(all_scores, config['model_params']['sparse_value'], scores_path.joinpath(config['name'] + '_sparse.csv'))

    assert_scores_have_max_4_decimals(scores_path.joinpath(config['name'] + '.csv'))
    assert_scores_have_max_4_decimals(scores_path.joinpath(config['name'] + '_sparse.csv'))

    assert len(all_scores) == 8
    for row in all_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])
        assert len(submission_id) >= 1
        assert len(profile_id) >= 1
        assert profile_id.startswith('~')
        assert score >= 0 and score <= 1

def test_normalization(tmp_path, create_specncl):
    # Test unnormalized scores
    unnorm_path = tmp_path / '-unnorm'
    config = {
        'name': 'test_specncl_unnormalized',
        'model_params': {
            'use_title': True,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'sparse_value': 1,
            'work_dir': unnorm_path,
            'normalize_scores': False
        }
    }

    specnclModel = create_specncl(config)

    publications_path = unnorm_path / 'publications'
    publications_path.mkdir()
    submissions_path = unnorm_path / 'submissions'
    submissions_path.mkdir()
    specnclModel.embed_publications(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl')
    )
    specnclModel.embed_submissions(
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
    )

    scores_path = unnorm_path / 'scores'
    scores_path.mkdir()
    all_scores = specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )

    for row in all_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])
        assert len(submission_id) >= 1
        assert len(profile_id) >= 1
        assert profile_id.startswith('~')
        assert score >= 0 and score <= 1
    
    # Compute statistics for unnormalized scores
    unnorm_stats = compute_score_statistics(all_scores, "(Unnormalized)")
    
    # Test normalization
    norm_path = tmp_path / '-norm'
    config = {
        'name': 'test_specncl_normalized',
        'model_params': {
            'use_title': True,
            'use_abstract': True,
            'use_cuda': False,
            'batch_size': 1,
            'average_score': True,
            'max_score': False,
            'sparse_value': 1,
            'work_dir': norm_path
        }
    }

    specnclModel = create_specncl(config)

    publications_path = norm_path / 'publications'
    publications_path.mkdir()
    submissions_path = norm_path / 'submissions'
    submissions_path.mkdir()
    specnclModel.embed_publications(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl')
    )
    specnclModel.embed_submissions(
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
    )

    scores_path = norm_path / 'scores'
    scores_path.mkdir()
    all_scores = specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        scores_path=scores_path.joinpath(config['name'] + '.csv')
    )

    for row in all_scores:
        submission_id, profile_id, score = row[0], row[1], float(row[2])
        assert len(submission_id) >= 1
        assert len(profile_id) >= 1
        assert profile_id.startswith('~')
        assert score >= 0 and score <= 1    
    # Compute statistics for normalized scores
    norm_stats = compute_score_statistics(all_scores, "(Normalized)")

    # Unnormalized mean tends to be around 0.8
    # Normalization by definition should be closer to 0.5
    assert norm_stats['mean'] < unnorm_stats['mean']

    ## Perform epsilon neighborhood check
    epsilon = 0.05  # Define a small epsilon value for neighborhood check
    assert abs(norm_stats['mean'] - 0.5) < epsilon, f"Normalized mean {norm_stats['mean']} is not within epsilon neighborhood of 0.5"
    assert abs(unnorm_stats['mean'] - 0.8) < epsilon, f"Unnormalized mean {unnorm_stats['mean']} is not within epsilon neighborhood of 0.8"

    # Unnormalized std should be smaller than normalized std
    ## Unnormalized scores tend to be more concentrated and higher
    assert unnorm_stats['std'] < norm_stats['std'], f"Unnormalized std {unnorm_stats['std']} should be less than normalized std {norm_stats['std']}"


def _find_embedding_with_self_dot_above_1(dim=768, seed=0):
    """
    Find a raw float32 embedding whose self-dot-product exceeds 1.0 after the
    normalization used inside load_emb_file:
        normed = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
        score  = dot(normed, normed)   # can be > 1.0 in float32
    Returns the raw (pre-normalization) embedding as a Python list and the
    expected self-dot value.
    """
    torch.manual_seed(seed)
    for _ in range(100_000):
        emb = torch.randn(1, dim)
        normed = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
        dot = torch.sum(normed[0] * normed[0]).item()
        if dot > 1.0:
            return emb[0].tolist(), dot
    raise RuntimeError(
        "Could not find an embedding that produces a self-dot > 1.0. "
        "This is unexpected given ~23% empirical hit rate."
    )


def test_self_similarity_score_within_bounds(tmp_path):
    """
    Reproduce issue #296: scoring a paper against itself produces a value
    slightly above 1.0 (e.g. 1.000000298023224) when normalize_scores=False.

    Root cause: load_emb_file in specter.py normalizes embeddings as
        normed = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
    Due to float32 accumulation errors, the dot product of a near-unit vector
    with itself can marginally exceed 1.0. This violates the [0, 1] range
    expected for all affinity scores.

    A crafted raw embedding is injected directly into the jsonl files so the
    self-dot > 1.0 condition is triggered deterministically without running
    the neural model. normalize_scores=False is used so the raw dot product
    is returned unchanged, directly exposing the bug.
    """
    raw_embedding, expected_raw_dot = _find_embedding_with_self_dot_above_1()
    assert expected_raw_dot > 1.0, (
        f"Precondition: expected raw self-dot > 1.0, got {expected_raw_dot}"
    )

    paper_id = "SelfSim01"

    # Only one reviewer/one submission so every score is a self-similarity
    # score, avoiding legitimate negative cosine values between unrelated
    # embeddings from confusing the assertion.
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive_entry = {
        "id": paper_id,
        "content": {"title": "Self-Similarity Paper", "abstract": "Abstract A."}
    }
    (archive_dir / "~Self_Author1.jsonl").write_text(json.dumps(archive_entry) + "\n")

    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    submission_entry = {
        "id": paper_id,
        "content": {"title": "Self-Similarity Paper", "abstract": "Abstract A."}
    }
    (submissions_dir / f"{paper_id}.jsonl").write_text(json.dumps(submission_entry) + "\n")

    archives_dataset = ArchivesDataset(archives_path=archive_dir)
    submissions_dataset = SubmissionsDataset(submissions_path=submissions_dir)

    # normalize_scores=False so the raw dot product is returned directly,
    # making the > 1.0 bug visible without any min-max mapping.
    model = specter2_scincl.EnsembleModel(
        specter_dir="../expertise-utils/specter/",
        work_dir=str(tmp_path),
        average_score=True,
        max_score=False,
        use_cuda=False,
        normalize_scores=False,
        use_redis=False,
    )
    model.set_archives_dataset(archives_dataset)
    model.set_submissions_dataset(submissions_dataset)

    pub_dir = tmp_path / "publications"
    pub_dir.mkdir()
    sub_dir = tmp_path / "sub_embeddings"
    sub_dir.mkdir()

    specter_pub_path = pub_dir / "pub2vec_specter.jsonl"
    scincl_pub_path  = pub_dir / "pub2vec_scincl.jsonl"
    specter_sub_path = sub_dir / "sub2vec_specter.jsonl"
    scincl_sub_path  = sub_dir / "sub2vec_scincl.jsonl"

    # Inject the same crafted embedding on both the reviewer (pub) and
    # submission (sub) side so the model scores the paper against itself.
    emb_line = json.dumps({"paper_id": paper_id, "embedding": raw_embedding}) + "\n"
    for path in (specter_pub_path, scincl_pub_path, specter_sub_path, scincl_sub_path):
        path.write_text(emb_line)

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()

    all_scores = model.all_scores(
        specter_publications_path=specter_pub_path,
        scincl_publications_path=scincl_pub_path,
        specter_submissions_path=specter_sub_path,
        scincl_submissions_path=scincl_sub_path,
        scores_path=scores_dir / "test_self_similarity.csv",
    )

    assert len(all_scores) > 0, "No scores were produced — check dataset setup."

    self_scores = [
        (sid, rid, score)
        for sid, rid, score in all_scores
        if sid == paper_id and rid == "~Self_Author1"
    ]
    assert len(self_scores) == 1, (
        f"Expected exactly one self-similarity score entry, got: {self_scores}"
    )

    _, _, self_score = self_scores[0]

    print(f"\nExpected raw self-dot (post load_emb_file normalization): {expected_raw_dot!r}")
    print(f"Final self-similarity score:                                {self_score!r}")

    # Core assertion for issue #296.
    # With normalize_scores=False the raw dot product of a paper with itself
    # must be clamped to [0, 1]. Due to float32 imprecision in
    # 'emb / (norm + epsilon)', the self-dot can marginally exceed 1.0
    # (e.g. 1.0000001192092896). The fix is to clamp scores before returning.
    assert 0.0 <= self_score <= 1.0, (
        f"Self-similarity score {self_score} is outside [0, 1]. "
        f"This reproduces issue #296: float32 imprecision in "
        f"'emb / (norm + epsilon)' normalization causes the self-dot "
        f"product to marginally exceed 1.0 (observed = {expected_raw_dot}). "
        f"Fix: clamp scores to [0, 1] before returning from all_scores()."
    )


def test_cached_embeddings_produce_identical_results(tmp_path):
    """
    Verify that embeddings loaded from cache produce byte-for-byte identical
    output compared to fresh embeddings. This prevents precision loss from
    JSON round-tripping (see PR #367 discussion).
    """
    # Setup test data
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive_entries = [
        {"id": "Paper1", "content": {"title": "Test Paper One", "abstract": "Abstract one."}},
        {"id": "Paper2", "content": {"title": "Test Paper Two", "abstract": "Abstract two."}},
    ]
    (archive_dir / "~Author1.jsonl").write_text(
        "\n".join(json.dumps(e) for e in archive_entries) + "\n"
    )

    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    submission_entries = [
        {"id": "Sub1", "content": {"title": "Submission One", "abstract": "Submission abstract one."}},
    ]
    (submissions_dir / "Sub1.jsonl").write_text(json.dumps(submission_entries[0]) + "\n")

    archives_dataset = ArchivesDataset(archives_path=archive_dir)
    submissions_dataset = SubmissionsDataset(submissions_path=submissions_dir)

    work_dir_fresh = tmp_path / "fresh"
    work_dir_cached = tmp_path / "cached"
    work_dir_fresh.mkdir()
    work_dir_cached.mkdir()

    # First run: generate fresh embeddings
    model_fresh = specter2_scincl.EnsembleModel(
        specter_dir="../expertise-utils/specter/",
        work_dir=str(work_dir_fresh),
        average_score=True,
        max_score=False,
        use_cuda=False,
        normalize_scores=False,  # Disable normalization to avoid min-max effects
        use_redis=False,
    )
    model_fresh.set_archives_dataset(archives_dataset)
    model_fresh.set_submissions_dataset(submissions_dataset)

    fresh_pub_path = work_dir_fresh / "pub2vec_specter.jsonl"
    fresh_sub_path = work_dir_fresh / "sub2vec_specter.jsonl"
    fresh_pub_path.parent.mkdir(exist_ok=True)
    fresh_sub_path.parent.mkdir(exist_ok=True)

    model_fresh.specter_predictor.embed_publications(fresh_pub_path)
    model_fresh.specter_predictor.embed_submissions(fresh_sub_path)

    # Copy fresh embeddings to create "cached" version
    cached_pub_path = work_dir_cached / "cached_pub2vec_specter.jsonl"
    cached_pub_path.parent.mkdir(exist_ok=True)
    cached_pub_path.write_text(fresh_pub_path.read_text())

    # Second run: use cached embeddings
    model_cached = specter2_scincl.EnsembleModel(
        specter_dir="../expertise-utils/specter/",
        work_dir=str(work_dir_cached),
        average_score=True,
        max_score=False,
        use_cuda=False,
        normalize_scores=False,
        use_redis=False,
    )
    model_cached.set_archives_dataset(archives_dataset)
    model_cached.set_submissions_dataset(submissions_dataset)

    # Create the cached file in the right location for the predictor to find it
    # The cached file should be at: <dir>/cached_<basename> next to <dir>/<basename>
    output_pub_path = work_dir_cached / "pub2vec_specter.jsonl"
    output_sub_path = work_dir_cached / "sub2vec_specter.jsonl"

    # Run embed_publications - it should use the cached embeddings
    model_cached.specter_predictor.embed_publications(output_pub_path)
    model_cached.specter_predictor.embed_submissions(output_sub_path)

    # Compare the output files byte-for-byte
    fresh_content = fresh_pub_path.read_text()
    cached_content = output_pub_path.read_text()

    assert fresh_content == cached_content, (
        f"Cached embeddings produced different output than fresh embeddings.\n"
        f"Fresh ({len(fresh_content)} chars): {fresh_content[:200]}...\n"
        f"Cached ({len(cached_content)} chars): {cached_content[:200]}..."
    )
