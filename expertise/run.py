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
    archives_dataset = ArchivesDataset(Path(config['dataset']['directory']).joinpath('archives'))
    submissions_dataset = SubmissionsDataset(Path(config['dataset']['directory']).joinpath('submissions'))

    if config['model'] == 'bm25':
        bm25Model = bm25.Model(archives_dataset, submissions_dataset, use_title=config['model_params'].get('use_title'), use_abstract=config['model_params'].get('use_abstract'), , workers=config['model_params'].get('workers'))
        bm25Model.all_scores(Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv'))

    if config['model'] == 'elmo':
        elmoModel = elmo.Model(archives_dataset, submissions_dataset, use_title=config['model_params'].get('use_title'), use_abstract=config['model_params'].get('use_abstract'), use_cuda=config['model_params'].get('use_cuda'), batch_size=config['model_params'].get('batch_size'), knn=config['model_params'].get('knn'))
        if config['model_params'].get('skip_elmo') is None or not config['model_params'].get('skip_elmo'):
            elmoModel.embed_publications(publications_path=Path(config['model_params']['publications_path']).joinpath('pub2vec.pkl'))
            elmoModel.embed_submssions(submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'))
        elmoModel.all_scores(
            publications_path=Path(config['model_params']['publications_path']).joinpath('pub2vec.pkl'),
            submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'),
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )
