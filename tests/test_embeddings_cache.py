import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from expertise.embeddings_cache import GlobalEmbeddingsCache


def _make_pq_table(rows):
    """Build a pyarrow Table from dict rows for local testing."""
    paper_ids = [r["paper_id"] for r in rows]
    embeddings = [r["embedding"] for r in rows]
    models = [r["model"] for r in rows]
    year_months = [r["year_month"] for r in rows]
    embedding_dates = [r["embedding_date"] for r in rows]
    return pa.table({
        "paper_id": pa.array(paper_ids, pa.string()),
        "embedding": pa.array(embeddings, pa.list_(pa.float32())),
        "model": pa.array(models, pa.string()),
        "year_month": pa.array(year_months, pa.string()),
        "embedding_date": pa.array(embedding_dates, pa.string()),
    })


def _write_local_dataset(tmp_path, table):
    """Write a Hive-partitioned dataset locally so ds.dataset works."""
    df = table.to_pandas()
    for (model, ym), grp in df.groupby(["model", "year_month"]):
        part_dir = tmp_path / f"model={model}" / f"year_month={ym}"
        part_dir.mkdir(parents=True, exist_ok=True)
        sub = grp.drop(columns=["model", "year_month"])
        pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), part_dir / "part-00000.parquet")
    return str(tmp_path)


class TestGlobalEmbeddingsCache:

    def test_get_embeddings_returns_embeddings(self, tmp_path):
        rows = [
            {"paper_id": "p1", "embedding": [0.1, 0.2], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
            {"paper_id": "p2", "embedding": [0.3, 0.4], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        result = cache.get_embeddings(["p1", "p2"], "specter")
        assert result["p1"] == pytest.approx([0.1, 0.2])
        assert result["p2"] == pytest.approx([0.3, 0.4])

    def test_get_embeddings_filters_by_model(self, tmp_path):
        rows = [
            {"paper_id": "p1", "embedding": [0.1], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
            {"paper_id": "p1", "embedding": [0.9], "model": "scincl", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        result = cache.get_embeddings(["p1"], "specter")
        assert result["p1"] == pytest.approx([0.1])

    def test_get_embeddings_for_models_multi_model(self, tmp_path):
        rows = [
            {"paper_id": "p1", "embedding": [0.1], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
            {"paper_id": "p1", "embedding": [0.9], "model": "scincl", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        result = cache.get_embeddings_for_models(["p1"], ["specter", "scincl"])
        assert result["specter"]["p1"] == pytest.approx([0.1])
        assert result["scincl"]["p1"] == pytest.approx([0.9])

    def test_get_embeddings_missing_paper_returns_empty(self, tmp_path):
        rows = [
            {"paper_id": "p1", "embedding": [0.1], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        result = cache.get_embeddings(["p99"], "specter")
        assert result == {}

    def test_get_embeddings_excludes_stale_embedding_when_mdate_newer(self, tmp_path):
        """If the paper's mdate is newer than the cached embedding_date, skip it."""
        rows = [
            {"paper_id": "p1", "embedding": [0.1], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        # mdate is AFTER embedding_date -> stale cache entry, should be excluded
        result = cache.get_embeddings(["p1"], "specter", paper_mdates={"p1": "2024-06-01T00:00:00Z"})
        assert "p1" not in result

    def test_get_embeddings_includes_fresh_embedding_when_mdate_older(self, tmp_path):
        """If the paper's mdate is older than the cached embedding_date, keep it."""
        rows = [
            {"paper_id": "p1", "embedding": [0.1], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-06-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        # mdate is BEFORE embedding_date -> fresh cache entry, should be included
        result = cache.get_embeddings(["p1"], "specter", paper_mdates={"p1": "2024-01-01T00:00:00Z"})
        assert result["p1"] == pytest.approx([0.1])

    def test_get_embeddings_includes_when_no_mdate_provided(self, tmp_path):
        """If no mdate is provided, the embedding should be returned regardless."""
        rows = [
            {"paper_id": "p1", "embedding": [0.1], "model": "specter", "year_month": "2024-01", "embedding_date": "2024-01-15T00:00:00Z"},
        ]
        dataset_path = _write_local_dataset(tmp_path, _make_pq_table(rows))
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        result = cache.get_embeddings(["p1"], "specter")
        assert result["p1"] == pytest.approx([0.1])

    def test_get_embeddings_handles_null_embedding_date(self, tmp_path):
        """If embedding_date is null, do not skip (cannot determine staleness)."""
        table = pa.table({
            "paper_id": pa.array(["p1"], pa.string()),
            "embedding": pa.array([[0.1]], pa.list_(pa.float32())),
            "model": pa.array(["specter"], pa.string()),
            "year_month": pa.array(["2024-01"], pa.string()),
            "embedding_date": pa.array([None], pa.string()),
        })
        dataset_path = _write_local_dataset(tmp_path, table)
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        cache._dataset = pa.dataset.dataset(dataset_path, partitioning="hive")

        result = cache.get_embeddings(["p1"], "specter", paper_mdates={"p1": "2024-06-01T00:00:00Z"})
        assert result["p1"] == pytest.approx([0.1])

    def test_get_embeddings_dataset_failure_returns_empty(self):
        """If reading the dataset fails, return empty dicts gracefully."""
        cache = GlobalEmbeddingsCache(bucket_name="fake", cache_prefix="fake")
        # _dataset is None, so _get_dataset will try GCS and fail
        result = cache.get_embeddings(["p1"], "specter")
        assert result == {}
