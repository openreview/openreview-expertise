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
        archives_dataset = ArchivesDataset(archives_path=Path("tests/data/archives"))
        submissions_dataset = SubmissionsDataset(
            submissions_path=Path("tests/data/submissions")
        )

        elmoModel = elmo.Model(
            use_title=config["model_params"].get("use_title"),
            use_abstract=config["model_params"].get("use_abstract"),
            use_cuda=config["model_params"].get("use_cuda"),
            batch_size=config["model_params"].get("batch_size"),
            knn=config["model_params"].get("knn"),
            sparse_value=config["model_params"].get("sparse_value"),
        )
        elmoModel.set_archives_dataset(archives_dataset)
        elmoModel.set_submissions_dataset(submissions_dataset)
        return elmoModel

    return simple_elmo


def test_elmo_scores(tmp_path, create_elmo):
    config = {
        "name": "test_elmo",
        "model_params": {
            "use_title": False,
            "use_abstract": True,
            "use_cuda": False,
            "batch_size": 1,
            "average_score": True,
            "max_score": False,
            "knn": None,
            "normalize": False,
            "skip_elmo": False,
        },
    }

    elmoModel = create_elmo(config)

    if not config["model_params"].get("skip_elmo", False):
        publications_path = tmp_path / "publications"
        publications_path.mkdir()
        submissions_path = tmp_path / "submissions"
        submissions_path.mkdir()
        elmoModel.embed_publications(
            publications_path=publications_path.joinpath("pub2vec.pkl")
        )
        elmoModel.embed_submissions(
            submissions_path=submissions_path.joinpath("sub2vec.pkl")
        )

    scores_path = tmp_path / "scores"
    scores_path.mkdir()
    all_scores = elmoModel.all_scores(
        publications_path=publications_path.joinpath("pub2vec.pkl"),
        submissions_path=submissions_path.joinpath("sub2vec.pkl"),
        scores_path=scores_path.joinpath(config["name"] + ".csv"),
    )


def test_normalize_scores(create_elmo):
    config = {
        "name": "test_elmo",
        "model_params": {
            "use_title": False,
            "use_abstract": True,
            "use_cuda": False,
            "batch_size": 1,
            "average_score": True,
            "max_score": False,
            "knn": None,
            "skip_elmo": False,
        },
    }

    elmoModel = create_elmo(config)

    score_matrix = np.array([[1, 2, 3], [5, 5, 5], [1, 0, 1]])

    normalized_matrix = elmoModel.normalize_scores(score_matrix)

    print(normalized_matrix)

    result_array = np.array([[0.0, 0.5, 1.0], [0.5, 0.5, 0.5], [1.0, 0.0, 1.0]])
    assert np.array_equal(result_array, normalized_matrix)


def test_duplicate_detection(tmp_path):
    submissions_dataset = SubmissionsDataset(
        submissions_path=Path("tests/data/submissions")
    )

    config = {
        "name": "test_elmo_duplicates",
        "model_params": {
            "use_title": False,
            "use_abstract": True,
            "use_cuda": False,
            "batch_size": 1,
            "average_score": False,
            "max_score": True,
            "knn": 6,
            "skip_elmo": False,
        },
    }

    other_submissions_dataset = False

    elmoModel = elmo.Model(
        use_title=config["model_params"].get("use_title", False),
        use_abstract=config["model_params"].get("use_abstract", True),
        use_cuda=config["model_params"].get("use_cuda", False),
        batch_size=config["model_params"].get("batch_size", 4),
        knn=config["model_params"].get("knn", 10),
        skip_same_id=(not other_submissions_dataset),
    )

    if not config["model_params"].get("skip_elmo", False):
        submissions_path = tmp_path / "submissions"
        submissions_path.mkdir()
        elmoModel.set_submissions_dataset(submissions_dataset)
        elmoModel.embed_submissions(
            submissions_path=submissions_path.joinpath("sub2vec.pkl")
        )

    scores_path = tmp_path / "scores"
    scores_path.mkdir()
    duplicates = elmoModel.find_duplicates(
        submissions_path=submissions_path.joinpath("sub2vec.pkl"),
        other_submissions_path=(
            Path(config["model_params"]["other_submissions_path"]).joinpath(
                "osub2vec.pkl"
            )
            if other_submissions_dataset
            else None
        ),
        scores_path=scores_path.joinpath(config["name"] + ".csv"),
    )
    for sub_1, sub_2, score in duplicates:
        if score > 0.99:
            assert sub_1 == "duplicate" or sub_2 == "duplicate"


def test_sparse_scores(tmp_path, create_elmo):
    config = {
        "name": "test_elmo",
        "model_params": {
            "use_title": False,
            "use_abstract": True,
            "use_cuda": False,
            "batch_size": 1,
            "average_score": True,
            "max_score": False,
            "knn": None,
            "normalize": False,
            "skip_elmo": False,
            "sparse_value": 1,
        },
    }

    elmoModel = create_elmo(config)

    if not config["model_params"].get("skip_elmo", False):
        publications_path = tmp_path / "publications"
        publications_path.mkdir()
        submissions_path = tmp_path / "submissions"
        submissions_path.mkdir()
        elmoModel.embed_publications(
            publications_path=publications_path.joinpath("pub2vec.pkl")
        )
        elmoModel.embed_submissions(
            submissions_path=submissions_path.joinpath("sub2vec.pkl")
        )

    scores_path = tmp_path / "scores"
    scores_path.mkdir()
    all_scores = elmoModel.all_scores(
        publications_path=publications_path.joinpath("pub2vec.pkl"),
        submissions_path=submissions_path.joinpath("sub2vec.pkl"),
        scores_path=scores_path.joinpath(config["name"] + ".csv"),
    )

    if config["model_params"].get("sparse_value"):
        all_scores = elmoModel.sparse_scores(
            scores_path=scores_path.joinpath(config["name"] + "_sparse.csv")
        )

    assert len(all_scores) == 8
