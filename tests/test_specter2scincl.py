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
from expertise.utils.utils import generate_sparse_scores_from_matrix


def _matrix_to_rows_and_csv(model, csv_path=None):
    """Test adapter: flatten the new matrix-based model output into the
    legacy [(test_id, reviewer_id, score), ...] tuple list, and optionally
    write the equivalent CSV file. Used by tests that previously consumed
    the per-row CSV/preliminary_scores produced by all_scores().
    Scores are rounded to 4 decimals at the row boundary — fp32 storage in
    the matrix preserves at most ~7 significant digits, so the per-cell
    Python round matches what the legacy code emitted in the CSV.
    """
    rows = []
    if model.scores_matrix is None:
        return rows
    scores = model.scores_matrix.tolist()
    for i, test_id in enumerate(model.test_id_list):
        for j, reviewer_id in enumerate(model.reviewer_ids):
            rows.append((test_id, reviewer_id, round(scores[i][j], 4)))
    if csv_path is not None:
        with open(csv_path, 'w') as f:
            for test_id, reviewer_id, score in rows:
                f.write(f'{test_id},{reviewer_id},{score}\n')
    return rows


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
    specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        matrix_path=scores_path.joinpath(config['name'] + '.pt')
    )

    csv_path = scores_path.joinpath(config['name'] + '.csv')
    _matrix_to_rows_and_csv(specnclModel, csv_path=csv_path)
    assert_scores_have_max_4_decimals(csv_path)


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
    specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        matrix_path=scores_path.joinpath(config['name'] + '.pt')
    )

    full_csv = scores_path.joinpath(config['name'] + '.csv')
    full_scores_snapshot = _matrix_to_rows_and_csv(specnclModel, csv_path=full_csv)

    sparse_csv = scores_path.joinpath(config['name'] + '_sparse.csv')

    if config['model_params'].get('sparse_value'):
        generate_sparse_scores_from_matrix(
            specnclModel.scores_matrix,
            specnclModel.test_id_list,
            specnclModel.reviewer_ids,
            config['model_params']['sparse_value'],
            sparse_csv,
        )

    assert_scores_have_max_4_decimals(full_csv)
    assert_scores_have_max_4_decimals(sparse_csv)

    # Read sparse CSV back into rows for assertions.
    sparse_rows = []
    with open(sparse_csv) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                sparse_rows.append((parts[0], parts[1], float(parts[2])))

    assert len(sparse_rows) == 8
    for submission_id, profile_id, score in sparse_rows:
        assert len(submission_id) >= 1
        assert len(profile_id) >= 1
        assert profile_id.startswith('~')
        assert score >= 0 and score <= 1

    # End-to-end ordering check on the sparse CSV: for each reviewer, the
    # papers selected for that reviewer (sorted by score desc) must match
    # the top-`sparse_value` papers for that reviewer from the full scores;
    # for each submission, vice versa. Runs whichever sparsifier the
    # pipeline calls (current generate_sparse_scores or a future
    # matrix/topk-based replacement), so it catches regressions
    # independently of the implementation.
    from collections import defaultdict
    sparse_value = config['model_params']['sparse_value']

    full_papers_by_reviewer = defaultdict(list)
    full_reviewers_by_paper = defaultdict(list)
    for sub, rev, score in full_scores_snapshot:
        full_papers_by_reviewer[rev].append((float(score), sub))
        full_reviewers_by_paper[sub].append((float(score), rev))

    sparse_papers_by_reviewer = defaultdict(list)
    sparse_reviewers_by_paper = defaultdict(list)
    with open(scores_path.joinpath(config['name'] + '_sparse.csv')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, r, sc = line.split(',')
            sparse_papers_by_reviewer[r].append((float(sc), s))
            sparse_reviewers_by_paper[s].append((float(sc), r))

    for rev, scored_subs in full_papers_by_reviewer.items():
        expected_top = [sub for _, sub in sorted(scored_subs, reverse=True)[:sparse_value]]
        actual_for_rev = sorted(sparse_papers_by_reviewer[rev], reverse=True)
        actual_top = [sub for _, sub in actual_for_rev[:sparse_value]]
        assert actual_top == expected_top, (
            f"Reviewer {rev}: top-{sparse_value} papers in sparse CSV "
            f"(by score desc) don't match top-{sparse_value} from full scores.\n"
            f"  expected: {expected_top}\n"
            f"  actual:   {actual_top}"
        )

    for sub, scored_revs in full_reviewers_by_paper.items():
        expected_top = [rev for _, rev in sorted(scored_revs, reverse=True)[:sparse_value]]
        actual_for_sub = sorted(sparse_reviewers_by_paper[sub], reverse=True)
        actual_top = [rev for _, rev in actual_for_sub[:sparse_value]]
        assert actual_top == expected_top, (
            f"Submission {sub}: top-{sparse_value} reviewers in sparse CSV "
            f"(by score desc) don't match top-{sparse_value} from full scores.\n"
            f"  expected: {expected_top}\n"
            f"  actual:   {actual_top}"
        )

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
    specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        matrix_path=scores_path.joinpath(config['name'] + '.pt')
    )
    all_scores = _matrix_to_rows_and_csv(specnclModel)

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
    specnclModel.all_scores(
        specter_publications_path=publications_path.joinpath('pub2vec_specter.jsonl'),
        scincl_publications_path=publications_path.joinpath('pub2vec_scincl.jsonl'),
        specter_submissions_path=submissions_path.joinpath('sub2vec_specter.jsonl'),
        scincl_submissions_path=submissions_path.joinpath('sub2vec_scincl.jsonl'),
        matrix_path=scores_path.joinpath(config['name'] + '.pt')
    )
    all_scores = _matrix_to_rows_and_csv(specnclModel)

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

    model.all_scores(
        specter_publications_path=specter_pub_path,
        scincl_publications_path=scincl_pub_path,
        specter_submissions_path=specter_sub_path,
        scincl_submissions_path=scincl_sub_path,
        matrix_path=scores_dir / "test_self_similarity.pt",
    )
    all_scores = _matrix_to_rows_and_csv(model)

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


def test_venue_specific_weights_with_weightless_cache(tmp_path):
    """Regression test: when venue_specific_weights is on and the publications
    file (e.g. reused from a prior cached job) has no 'weight' field per line,
    all_scores must still succeed by reading weights from the metadata file
    rather than the embedding line. Previously this raised KeyError: 'weight'."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive_entries = [
        {"id": "Paper1", "content": {"title": "Test Paper One", "abstract": "Abstract one.", "weight": 1.4}},
        {"id": "Paper2", "content": {"title": "Test Paper Two", "abstract": "Abstract two.", "weight": 1.0}},
    ]
    (archive_dir / "~Author1.jsonl").write_text(
        "\n".join(json.dumps(e) for e in archive_entries) + "\n"
    )

    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    submission_entry = {"id": "Sub1", "content": {"title": "Submission One", "abstract": "Submission abstract one."}}
    (submissions_dir / "Sub1.jsonl").write_text(json.dumps(submission_entry) + "\n")

    archives_dataset = ArchivesDataset(archives_path=archive_dir)
    submissions_dataset = SubmissionsDataset(submissions_path=submissions_dir)

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    model = specter2_scincl.EnsembleModel(
        specter_dir="../expertise-utils/specter/",
        work_dir=str(work_dir),
        average_score=True,
        max_score=False,
        use_cuda=False,
        normalize_scores=False,
        use_redis=False,
        venue_specific_weights=True,
    )
    model.set_archives_dataset(archives_dataset)
    model.set_submissions_dataset(submissions_dataset)

    specter_pub_path = work_dir / "pub2vec_specter.jsonl"
    scincl_pub_path = work_dir / "pub2vec_scincl.jsonl"
    specter_sub_path = work_dir / "sub2vec_specter.jsonl"
    scincl_sub_path = work_dir / "sub2vec_scincl.jsonl"

    # Embed once to get real embeddings, then strip the embedding lines into
    # weightless cached_pub2vec_*.jsonl files to simulate cache from a prior
    # job that ran without venue_specific_weights.
    model.embed_publications(specter_pub_path, scincl_pub_path)

    def strip_weight_into_cache(src, dst):
        with open(src) as f_in, open(dst, "w") as f_out:
            for line in f_in:
                entry = json.loads(line)
                entry.pop("weight", None)
                f_out.write(json.dumps(entry) + "\n")

    specter_cache = work_dir / "cached_pub2vec_specter.jsonl"
    scincl_cache = work_dir / "cached_pub2vec_scincl.jsonl"
    strip_weight_into_cache(specter_pub_path, specter_cache)
    strip_weight_into_cache(scincl_pub_path, scincl_cache)
    specter_pub_path.unlink()
    scincl_pub_path.unlink()

    # Re-run with weightless cache present. all_scores must not raise KeyError.
    # Spy on _batch_predict so we can prove the cache was the source of truth
    # rather than the model. If the cache wasn't consumed, embed_publications
    # would re-run inference and call _batch_predict at least once per predictor.
    with patch.object(model.specter_predictor, '_batch_predict', wraps=model.specter_predictor._batch_predict) as specter_spy, \
         patch.object(model.scincl_predictor, '_batch_predict', wraps=model.scincl_predictor._batch_predict) as scincl_spy:
        model.embed_publications(specter_pub_path, scincl_pub_path)
        assert specter_spy.call_count == 0, (
            f"Expected cache hit for all papers, but specter._batch_predict ran {specter_spy.call_count} times"
        )
        assert scincl_spy.call_count == 0, (
            f"Expected cache hit for all papers, but scincl._batch_predict ran {scincl_spy.call_count} times"
        )

    model.embed_submissions(specter_sub_path, scincl_sub_path)

    # Cache lines should pass through verbatim (byte-equal to what we wrote).
    assert specter_pub_path.read_text() == specter_cache.read_text()
    assert scincl_pub_path.read_text() == scincl_cache.read_text()

    matrix_path = work_dir / "scores.pt"
    model.all_scores(
        specter_publications_path=specter_pub_path,
        scincl_publications_path=scincl_pub_path,
        specter_submissions_path=specter_sub_path,
        scincl_submissions_path=scincl_sub_path,
        matrix_path=matrix_path,
    )

    assert model.scores_matrix is not None and model.scores_matrix.numel() > 0, "Expected at least one score row"
    assert matrix_path.exists()


def test_all_scores_skips_reviewer_with_only_bad_embeddings(tmp_path):
    """Regression: when every publication for a reviewer has a zero-length
    embedding (added to train_bad_id_set by load_emb_file), train_paper_idx
    becomes empty. The previous code would slice p2p_aff_norm[:, []] and crash
    on .max(dim=1)[0] / torch.quantile, or produce NaNs from .mean(dim=1).
    Expected: the reviewer is skipped and the run completes successfully
    with that reviewer omitted from scores_matrix's columns.

    Setup: two reviewers — one with a good paper, one whose only paper has
    a zero-length embedding. The bad-embedding reviewer must not appear in
    the resulting reviewer_ids; the good-embedding reviewer must.
    """
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "~GoodReviewer1.jsonl").write_text(
        json.dumps({"id": "good_paper", "content": {"title": "Good Paper", "abstract": "Abstract."}}) + "\n"
    )
    (archive_dir / "~BadReviewer1.jsonl").write_text(
        json.dumps({"id": "bad_paper", "content": {"title": "Bad Paper", "abstract": "Abstract."}}) + "\n"
    )

    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    (submissions_dir / "Sub1.jsonl").write_text(
        json.dumps({"id": "Sub1", "content": {"title": "Submission", "abstract": "Submission abstract."}}) + "\n"
    )

    archives_dataset = ArchivesDataset(archives_path=archive_dir)
    submissions_dataset = SubmissionsDataset(submissions_path=submissions_dir)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    model = specter2_scincl.EnsembleModel(
        specter_dir="../expertise-utils/specter/",
        work_dir=str(work_dir),
        average_score=True,
        max_score=False,
        use_cuda=False,
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

    # Write hand-crafted embeddings. The "bad_paper" gets an empty embedding
    # ([]), which load_emb_file detects (paper_emb_size == 0) and adds to
    # train_bad_id_set. ~BadReviewer1's only paper is bad — so their
    # train_paper_idx will be empty after filtering.
    good_embedding = [0.1] * 768
    pub_lines = (
        json.dumps({"paper_id": "good_paper", "embedding": good_embedding}) + "\n"
        + json.dumps({"paper_id": "bad_paper", "embedding": []}) + "\n"
    )
    specter_pub_path.write_text(pub_lines)
    scincl_pub_path.write_text(pub_lines)

    sub_emb = [0.2] * 768
    sub_line = json.dumps({"paper_id": "Sub1", "embedding": sub_emb}) + "\n"
    specter_sub_path.write_text(sub_line)
    scincl_sub_path.write_text(sub_line)

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    # Must not raise.
    model.all_scores(
        specter_publications_path=specter_pub_path,
        scincl_publications_path=scincl_pub_path,
        specter_submissions_path=specter_sub_path,
        scincl_submissions_path=scincl_sub_path,
        matrix_path=scores_dir / "scores.pt",
    )

    # The good reviewer is present; the bad-only reviewer is dropped.
    assert "~GoodReviewer1" in model.reviewer_ids
    assert "~BadReviewer1" not in model.reviewer_ids
    # Matrix shape: [num_test=1, num_reviewers=1].
    assert tuple(model.scores_matrix.shape) == (1, 1)
    # Score is finite (no NaN from a degenerate reduction).
    score = model.scores_matrix[0, 0].item()
    assert score == score, "Score must not be NaN"


@pytest.mark.parametrize("aggregation_mode,use_weights", [
    ("max", False),
    ("average", False),
    ("percentile", False),
    ("max", True),
    ("average", True),
    ("percentile", True),
])
def test_vectorized_aggregation_matches_reference_loop(tmp_path, aggregation_mode, use_weights):
    """Regression for PR #386: verify vectorized reviewer aggregation produces
    the same scores as the pre-change loop-based implementation.

    Uses hand-crafted embeddings so p2p_aff values are deterministic.
    Reviewers have varying paper counts (tests padding/masking).
    One paper has a zero-length embedding (tests bad_id filtering).
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    predictor = specter2_scincl.Specter2Predictor(
        specter_dir="../expertise-utils/specter/",
        work_dir=str(work_dir),
        use_cuda=False,
        normalize_scores=False,
        average_score=(aggregation_mode == "average"),
        max_score=(aggregation_mode == "max"),
        percentile_select=83 if aggregation_mode == "percentile" else None,
        venue_specific_weights=use_weights,
    )

    # --- 1. Archives: 3 reviewers with varying paper counts ---
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    # Rev1: 2 papers
    (archive_dir / "~Rev1.jsonl").write_text(
        json.dumps({"id": "p1", "content": {"title": "P1", "abstract": "A1"}}) + "\n"
        + json.dumps({"id": "p2", "content": {"title": "P2", "abstract": "A2"}}) + "\n"
    )
    # Rev2: 3 papers (one will have bad embedding)
    (archive_dir / "~Rev2.jsonl").write_text(
        json.dumps({"id": "p2", "content": {"title": "P2", "abstract": "A2"}}) + "\n"
        + json.dumps({"id": "p3", "content": {"title": "P3", "abstract": "A3"}}) + "\n"
        + json.dumps({"id": "p4", "content": {"title": "P4", "abstract": "A4"}}) + "\n"
    )
    # Rev3: 1 paper
    (archive_dir / "~Rev3.jsonl").write_text(
        json.dumps({"id": "p5", "content": {"title": "P5", "abstract": "A5"}}) + "\n"
    )

    archives_dataset = ArchivesDataset(archives_path=archive_dir)
    predictor.set_archives_dataset(archives_dataset)

    # --- 2. Submissions: 2 submissions ---
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir()
    (sub_dir / "Sub1.jsonl").write_text(
        json.dumps({"id": "Sub1", "content": {"title": "S1", "abstract": "S1"}}) + "\n"
    )
    (sub_dir / "Sub2.jsonl").write_text(
        json.dumps({"id": "Sub2", "content": {"title": "S2", "abstract": "S2"}}) + "\n"
    )

    submissions_dataset = SubmissionsDataset(submissions_path=sub_dir)
    predictor.set_submissions_dataset(submissions_dataset)

    # --- 3. Hand-crafted embeddings ---
    # Use 768-dim vectors where unit vectors produce predictable dot products.
    # After load_emb_file normalization, e_i stays e_i (already unit length).
    def unit_vec(i):
        v = [0.0] * 768
        v[i] = 1.0
        return v

    # Publications:
    #   p1 = e0, p2 = e1, p3 = e2, p4 = e3, p5 = e4
    # Submissions:
    #   Sub1 = (e0 + e1) / sqrt(2)
    #   Sub2 = (e1 + e2) / sqrt(2)
    #
    # Expected raw p2p_aff (before normalization, which is disabled):
    #           p1    p2    p3    p4    p5
    #   Sub1  1/√2  1/√2   0     0     0
    #   Sub2   0    1/√2  1/√2   0     0
    #
    # p4 has a zero-length embedding -> added to train_bad_id_set.
    # So Rev2 effectively has papers [p2, p3] after filtering.

    pub_lines = (
        json.dumps({"paper_id": "p1", "embedding": unit_vec(0)}) + "\n"
        + json.dumps({"paper_id": "p2", "embedding": unit_vec(1)}) + "\n"
        + json.dumps({"paper_id": "p3", "embedding": unit_vec(2)}) + "\n"
        + json.dumps({"paper_id": "p4", "embedding": []}) + "\n"  # bad embedding
        + json.dumps({"paper_id": "p5", "embedding": unit_vec(4)}) + "\n"
    )

    def sub_vec(i, j):
        v = [0.0] * 768
        v[i] = 1.0 / (2 ** 0.5)
        v[j] = 1.0 / (2 ** 0.5)
        return v

    sub_lines = (
        json.dumps({"paper_id": "Sub1", "embedding": sub_vec(0, 1)}) + "\n"
        + json.dumps({"paper_id": "Sub2", "embedding": sub_vec(1, 2)}) + "\n"
    )

    pub_path = work_dir / "pub2vec_specter.jsonl"
    sub_path = work_dir / "sub2vec_specter.jsonl"
    pub_path.write_text(pub_lines)
    sub_path.write_text(sub_lines)

    # --- 4. Venue weights metadata (if needed) ---
    if use_weights:
        weights_meta = {
            "p1": {"weight": 2.0},
            "p2": {"weight": 1.0},
            "p3": {"weight": 0.5},
            "p4": {"weight": 1.0},
            "p5": {"weight": 3.0},
        }
        with open(work_dir / "specter_reviewer_paper_data.json", "w") as f:
            json.dump(weights_meta, f)

    # --- 5. Run the vectorized code ---
    predictor.all_scores(publications_path=pub_path, submissions_path=sub_path)

    # --- 6. Compute reference scores with the old loop ---
    # Load embeddings to reconstruct p2p_aff
    with open(pub_path) as f:
        pub_embs = {}
        for line in f:
            d = json.loads(line)
            if d["embedding"]:
                e = torch.tensor(d["embedding"], dtype=torch.float32)
                e = e / (e.norm() + 1e-12)
                pub_embs[d["paper_id"]] = e

    with open(sub_path) as f:
        sub_embs = {}
        for line in f:
            d = json.loads(line)
            e = torch.tensor(d["embedding"], dtype=torch.float32)
            e = e / (e.norm() + 1e-12)
            sub_embs[d["paper_id"]] = e

    sub_ids = ["Sub1", "Sub2"]
    pub_ids = ["p1", "p2", "p3", "p4", "p5"]
    p2p = torch.zeros((2, 5), dtype=torch.float32)
    for i, sid in enumerate(sub_ids):
        for j, pid in enumerate(pub_ids):
            if pid in pub_embs:
                p2p[i, j] = sub_embs[sid].dot(pub_embs[pid]).item()

    # Reviewer paper indices after bad_id filtering
    reviewer_papers = {
        "~Rev1": ["p1", "p2"],
        "~Rev2": ["p2", "p3"],  # p4 filtered out
        "~Rev3": ["p5"],
    }
    pub_id_to_idx = {pid: j for j, pid in enumerate(pub_ids)}

    # Load weights if needed
    weights = None
    if use_weights:
        with open(work_dir / "specter_reviewer_paper_data.json") as f:
            weights_meta = json.load(f)
        weights = torch.tensor([weights_meta.get(pid, {}).get("weight", 1.0) for pid in pub_ids], dtype=torch.float32)

    ref_vectors = []
    ref_reviewer_ids = []
    for rev_id, papers in reviewer_papers.items():
        paper_indices = [pub_id_to_idx[p] for p in papers]
        train_paper_aff_j = p2p[:, paper_indices]

        if use_weights:
            train_weight_j = weights[paper_indices]
            epsilon = 1e-8
            logits = torch.logit(train_paper_aff_j.clamp(epsilon, 1 - epsilon), eps=epsilon)
            log_w = torch.log(torch.clamp(train_weight_j, min=epsilon))
            weighted_logits = logits + log_w.unsqueeze(0)
            train_paper_aff_j = torch.sigmoid(weighted_logits)

        if aggregation_mode == "percentile":
            q = 0.83
            all_paper_aff = torch.quantile(train_paper_aff_j, q, dim=1, interpolation='linear')
        elif aggregation_mode == "average":
            all_paper_aff = train_paper_aff_j.mean(dim=1)
        elif aggregation_mode == "max":
            all_paper_aff = train_paper_aff_j.max(dim=1)[0]

        ref_vectors.append(all_paper_aff)
        ref_reviewer_ids.append(rev_id)

    ref_matrix = torch.stack(ref_vectors, dim=1)  # [2, 3]

    # --- 7. Compare using the same dict-based pattern as compare_scores.py ---
    actual_scores = {}
    for i, sub_id in enumerate(predictor.test_id_list):
        for j, rev_id in enumerate(predictor.reviewer_ids):
            actual_scores[(sub_id, rev_id)] = round(float(predictor.scores_matrix[i, j]), 4)

    ref_scores = {}
    for i, sub_id in enumerate(sub_ids):
        for j, rev_id in enumerate(ref_reviewer_ids):
            ref_scores[(sub_id, rev_id)] = round(float(ref_matrix[i, j]), 4)

    common = set(actual_scores.keys()) & set(ref_scores.keys())
    diffs = []
    for key in common:
        if actual_scores[key] != ref_scores[key]:
            diffs.append({
                "submission_id": key[0],
                "reviewer_id": key[1],
                "score_actual": actual_scores[key],
                "score_expected": ref_scores[key],
                "delta": actual_scores[key] - ref_scores[key],
            })

    assert len(diffs) == 0, (
        f"Score mismatch in mode={aggregation_mode}, weights={use_weights}: "
        f"{len(diffs)} pairs differ out of {len(common)} total.\n"
        + "\n".join(
            f"  {d['submission_id']} / {d['reviewer_id']}: "
            f"actual={d['score_actual']:.4f} expected={d['score_expected']:.4f} "
            f"delta={d['delta']:+.4f}"
            for d in diffs[:10]
        )
    )
