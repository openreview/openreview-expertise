from expertise.dataset import ArchivesDataset, SubmissionsDataset, BidsDataset
from unittest.mock import patch, MagicMock
from pathlib import Path
import openreview
import json

def test_archives_dataset():
    archives_dataset = ArchivesDataset(archives_path=Path('tests/data/archives'))
    assert len(archives_dataset) == 3
    assert len(archives_dataset['~Jackson_Upton9']) == 3
    assert len(archives_dataset['~Jerald_Hackett2']) == 3
    assert len(archives_dataset['~Loreta_Tremblay3']) == 4

    archives_dataset.remove_publication('N6F1Vu82', '~Loreta_Tremblay3')
    assert len(archives_dataset['~Loreta_Tremblay3']) == 3

    archives_dataset.add_publication('{"id":"N6F1Vu82","content":{"title":"Insert esoph obtu airway","abstract":"Phasellus sit amet erat. Nulla tempus. Vivamus in felis eu sapien cursus vestibulum."} }', '~Loreta_Tremblay3')
    assert len(archives_dataset['~Loreta_Tremblay3']) == 4

def test_submissions_dataset():
    submissions_dataset = SubmissionsDataset(submissions_path=Path('tests/data/submissions'))
    assert len(submissions_dataset) == 6
