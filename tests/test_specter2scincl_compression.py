import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from expertise.execute_expertise import execute_expertise
from expertise.evaluation.evaluator import OpenReviewExpertiseEvaluation
from expertise.models.specter2_scincl import scincl as scincl_module
from expertise.models.specter2_scincl import specter as specter_module


def _build_predictor(predictor_cls, compression):
    predictor = predictor_cls.__new__(predictor_cls)
    predictor.embedding_compression = compression
    return predictor


def _expected_int8_restore(embedding_array):
    max_abs = float(np.max(np.abs(embedding_array)))
    if max_abs == 0.0:
        return np.zeros_like(embedding_array, dtype=np.float32)
    scale = max_abs / 127.0
    quantized = np.clip(np.rint(embedding_array / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized.astype(np.float32) * np.float32(scale), dtype=np.float32)


@pytest.mark.parametrize(
    ("predictor_cls", "_keep_dims"),
    [
        (specter_module.Specter2Predictor, specter_module.KEEP_DIMS),
        (scincl_module.SciNCLPredictor, scincl_module.KEEP_DIMS),
    ],
)
def test_float16_embedding_roundtrip(predictor_cls, _keep_dims):
    predictor = _build_predictor(predictor_cls, "float16")
    embedding = torch.linspace(-1.0, 1.0, steps=768)

    payload = json.loads(predictor._build_embedding_jsonl({"paper_id": "paper-1"}, embedding))
    restored = np.array(predictor._restore_embedding(payload, 768), dtype=np.float32)

    expected = embedding.detach().cpu().numpy().astype(np.float16).astype(np.float32)
    assert payload["embedding_compression"] == "float16"
    assert restored.shape == (768,)
    np.testing.assert_allclose(restored, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("predictor_cls", "_keep_dims"),
    [
        (specter_module.Specter2Predictor, specter_module.KEEP_DIMS),
        (scincl_module.SciNCLPredictor, scincl_module.KEEP_DIMS),
    ],
)
def test_int8_embedding_roundtrip(predictor_cls, _keep_dims):
    predictor = _build_predictor(predictor_cls, "int8_per_vector")
    embedding = torch.linspace(-0.75, 0.75, steps=768)

    payload = json.loads(predictor._build_embedding_jsonl({"paper_id": "paper-1"}, embedding))
    restored = np.array(predictor._restore_embedding(payload, 768), dtype=np.float32)

    expected = _expected_int8_restore(embedding.detach().cpu().numpy().astype(np.float32))
    assert payload["embedding_compression"] == "int8_per_vector"
    assert restored.shape == (768,)
    np.testing.assert_allclose(restored, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("predictor_cls", "keep_dims"),
    [
        (specter_module.Specter2Predictor, specter_module.KEEP_DIMS),
        (scincl_module.SciNCLPredictor, scincl_module.KEEP_DIMS),
    ],
)
def test_int8_keep_dims_embedding_roundtrip(predictor_cls, keep_dims):
    predictor = _build_predictor(predictor_cls, "int8_per_vector_keep_dims")
    embedding = torch.linspace(-0.5, 0.5, steps=768)

    payload = json.loads(predictor._build_embedding_jsonl({"paper_id": "paper-1"}, embedding))
    restored = np.array(predictor._restore_embedding(payload, 768), dtype=np.float32)

    expected = np.zeros(768, dtype=np.float32)
    expected_subset = _expected_int8_restore(embedding.detach().cpu().numpy().astype(np.float32)[keep_dims])
    expected[np.asarray(keep_dims, dtype=np.int64)] = expected_subset

    assert payload["embedding_compression"] == "int8_per_vector_keep_dims"
    assert restored.shape == (768,)
    np.testing.assert_allclose(restored, expected, rtol=0, atol=1e-6)

    dropped_dims = sorted(set(range(768)) - set(keep_dims))
    assert np.count_nonzero(restored[dropped_dims]) == 0


@pytest.mark.parametrize(
    ("model_name", "predictor_path", "compression_key", "compression_value"),
    [
        ("specter2", "expertise.models.specter2_scincl.Specter2Predictor", "specter2_compression", "float16"),
        ("scincl", "expertise.models.specter2_scincl.SciNCLPredictor", "scincl_compression", "int8_per_vector"),
    ],
)
def test_execute_expertise_passes_single_model_compression(
    tmp_path,
    model_name,
    predictor_path,
    compression_key,
    compression_value,
):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "archives").mkdir(parents=True)
    (dataset_dir / "submissions").mkdir()

    config = {
        "name": f"test_{model_name}",
        "dataset": {"directory": str(dataset_dir)},
        "model": model_name,
        "model_params": {
            compression_key: compression_value,
            "work_dir": str(tmp_path / "work"),
            "scores_path": str(tmp_path / "scores"),
            "publications_path": str(tmp_path / "publications"),
            "submissions_path": str(tmp_path / "submissions_out"),
            "use_cuda": False,
        },
    }

    with patch("expertise.execute_expertise.ArchivesDataset"), \
         patch("expertise.execute_expertise.SubmissionsDataset"), \
         patch(predictor_path) as mock_predictor:
        execute_expertise(config)

    assert mock_predictor.call_args.kwargs["embedding_compression"] == compression_value


def test_execute_expertise_passes_ensemble_compression(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "archives").mkdir(parents=True)
    (dataset_dir / "submissions").mkdir()

    config = {
        "name": "test_specter2_scincl",
        "dataset": {"directory": str(dataset_dir)},
        "model": "specter2+scincl",
        "model_params": {
            "specter2_compression": "int8_per_vector_keep_dims",
            "scincl_compression": "float16",
            "work_dir": str(tmp_path / "work"),
            "scores_path": str(tmp_path / "scores"),
            "publications_path": str(tmp_path / "publications"),
            "submissions_path": str(tmp_path / "submissions_out"),
            "use_cuda": False,
        },
    }

    with patch("expertise.execute_expertise.ArchivesDataset"), \
         patch("expertise.execute_expertise.SubmissionsDataset"), \
         patch("expertise.models.specter2_scincl.EnsembleModel") as mock_predictor:
        execute_expertise(config)

    assert mock_predictor.call_args.kwargs["specter2_compression"] == "int8_per_vector_keep_dims"
    assert mock_predictor.call_args.kwargs["scincl_compression"] == "float16"


def test_evaluator_model_config_includes_model_param_overrides():
    evaluator = OpenReviewExpertiseEvaluation({
        "datasets": ["goldstandard"],
        "configs": {"goldstandard": {}},
        "models": ["specter2"],
        "evaluate": {},
        "use_coda": False,
        "model_params": {
            "default": {"embedding_compression": "float16"},
            "specter2": {"specter2_compression": "int8_per_vector_keep_dims"},
        },
    })

    model_config = evaluator._build_model_config("specter2", "/tmp/run-dir", "d_20_1")

    assert model_config["name"] == "specter2_d_20_1"
    assert model_config["model_params"]["embedding_compression"] == "float16"
    assert model_config["model_params"]["specter2_compression"] == "int8_per_vector_keep_dims"
