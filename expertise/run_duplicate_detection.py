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
    elif Path(config['dataset']['directory']).joinpath('other_submissions.jsonl').exists():
        other_submissions_dataset = SubmissionsDataset(submissions_file=Path(config['dataset']['directory']).joinpath('other_submissions.jsonl'))

    elmoModel = elmo.Model(
        use_title=config['model_params'].get('use_title', False),
        use_abstract=config['model_params'].get('use_abstract', True),
        use_cuda=config['model_params'].get('use_cuda', False),
        batch_size=config['model_params'].get('batch_size', 4),
        knn=config['model_params'].get('knn', 10),
        skip_same_id=config['model_params'].get('skip_same_id', True)
    )
    elmoModel.set_submissions_dataset(submissions_dataset)
    elmoModel.set_other_submissions_dataset(other_submissions_dataset)
    if not config['model_params'].get('skip_elmo', False):
        elmoModel.embed_submssions(submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'))
        elmoModel.embed_other_submssions(other_submissions_path=Path(config['model_params']['other_submissions_path']).joinpath('osub2vec.pkl'))
    elmoModel.find_duplicates(
        submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'),
        other_submissions_path=Path(config['model_params']['other_submissions_path']).joinpath('osub2vec.pkl'),
        scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
    )
