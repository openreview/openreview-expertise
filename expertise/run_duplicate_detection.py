import argparse
from pathlib import Path

from .dataset import ArchivesDataset, SubmissionsDataset, BidsDataset
from .config import ModelConfig
from .models import bm25, elmo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a JSON file containing all other arguments')
    args = parser.parse_args()

    config = ModelConfig(config_file_path=args.config)
    if Path(config['dataset']['directory']).joinpath('submissions').exists():
        submissions_dataset = SubmissionsDataset(submissions_path=Path(config['dataset']['directory']).joinpath('submissions'))
    elif Path(config['dataset']['directory']).joinpath('submissions.jsonl').exists():
        submissions_dataset = SubmissionsDataset(submissions_file=Path(config['dataset']['directory']).joinpath('submissions.jsonl'))

    if Path(config['dataset']['directory']).joinpath('other_submissions').exists():
        other_submissions_dataset = SubmissionsDataset(submissions_path=Path(config['dataset']['directory']).joinpath('other_submissions'))
    elif Path(config['dataset']['directory']).joinpath('other_submissions.jsonl').exists()
        other_submissions_dataset = SubmissionsDataset(submissions_file=Path(config['dataset']['directory']).joinpath('other_submissions.jsonl'))

    if config['model'] == 'elmo':
        elmoModel = elmo.Model(
            use_title=config['model_params'].get('use_title'),
            use_abstract=config['model_params'].get('use_abstract'),
            use_cuda=config['model_params'].get('use_cuda'),
            batch_size=config['model_params'].get('batch_size'),
            knn=config['model_params'].get('knn')
        )
        elmoModel.set_submissions_dataset(submissions_dataset)
        elmoModel.set_other_submissions_dataset(other_submissions_dataset)
        if config['model_params'].get('skip_elmo') is None or not config['model_params'].get('skip_elmo'):
            elmoModel.embed_submssions(submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'))
            elmoModel.embed_other_submssions(other_submissions_path=Path(config['model_params']['other_submissions_path']).joinpath('osub2vec.pkl'))
        elmoModel.find_duplicates(
            submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'),
            other_submissions_path=Path(config['model_params']['other_submissions_path']).joinpath('osub2vec.pkl'),
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )
