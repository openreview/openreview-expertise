import argparse
from pathlib import Path

from .dataset import ArchivesDataset, SubmissionsDataset, BidsDataset
from .config import ModelConfig
from .models import bm25

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a JSON file containing all other arguments')
    args = parser.parse_args()

    config = ModelConfig(config_file_path=args.config)
    archives_dataset = ArchivesDataset(archives_path=Path(config['dataset']['directory']).joinpath('archives'))
    submissions_dataset = SubmissionsDataset(submissions_path=Path(config['dataset']['directory']).joinpath('submissions'))

    if config['model'] == 'bm25':
        bm25Model = bm25.Model(archives_dataset, submissions_dataset, use_title=config['model_params']['use_title'], use_abstract=config['model_params']['use_abstract'], workers=config['model_params']['workers'])
        bm25Model.all_scores(Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv'))

