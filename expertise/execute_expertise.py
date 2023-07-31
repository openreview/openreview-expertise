from pathlib import Path
import openreview, os, json, csv
from .create_dataset import OpenReviewExpertise
from .dataset import ArchivesDataset, SubmissionsDataset, BidsDataset
from .config import ModelConfig
from .utils.utils import aggregate_by_group

# Move run.py functionality to a function that accepts a config dict
def execute_expertise(config):

    config = ModelConfig(config_dict=config)

    archives_dataset = ArchivesDataset(archives_path=Path(config['dataset']['directory']).joinpath('archives'))
    if Path(config['dataset']['directory']).joinpath('submissions').exists():
        submissions_dataset = SubmissionsDataset(submissions_path=Path(config['dataset']['directory']).joinpath('submissions'))
    elif Path(config['dataset']['directory']).joinpath('submissions.json').exists():
        submissions_dataset = SubmissionsDataset(submissions_file=Path(config['dataset']['directory']).joinpath('submissions.json'))

    if config['model'] == 'bm25':
        from .models import bm25
        bm25Model = bm25.Model(
            use_title=config['model_params'].get('use_title', False),
            use_abstract=config['model_params'].get('use_abstract', True),
            workers=config['model_params'].get('workers', 1),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            sparse_value=config['model_params'].get('sparse_value')
        )
        bm25Model.set_archives_dataset(archives_dataset)
        bm25Model.set_submissions_dataset(submissions_dataset)

        if not config['model_params'].get('skip_bm25', False):
            bm25Model.all_scores(
                preliminary_scores_path=Path(config['model_params']['scores_path']).joinpath('preliminary_scores.pkl'),
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
            )
        if config['model_params'].get('sparse_value'):
            bm25Model.sparse_scores(
                preliminary_scores_path=Path(config['model_params']['scores_path']).joinpath('preliminary_scores.pkl'),
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '_sparse.csv')
            )

    if config['model'] == 'elmo':
        from .models import elmo
        elmoModel = elmo.Model(
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            use_title=config['model_params'].get('use_title', False),
            use_abstract=config['model_params'].get('use_abstract', True),
            use_cuda=config['model_params'].get('use_cuda', False),
            batch_size=config['model_params'].get('batch_size', 4),
            knn=config['model_params'].get('knn'),
            normalize=config['model_params'].get('normalize', False),
            sparse_value=config['model_params'].get('sparse_value')
        )
        elmoModel.set_archives_dataset(archives_dataset)
        elmoModel.set_submissions_dataset(submissions_dataset)
        if not config['model_params'].get('skip_elmo', False):
            elmoModel.embed_publications(publications_path=Path(config['model_params']['publications_path']).joinpath('pub2vec.pkl'))
            elmoModel.embed_submissions(submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'))
        elmoModel.all_scores(
            publications_path=Path(config['model_params']['publications_path']).joinpath('pub2vec.pkl'),
            submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.pkl'),
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )

        if config['model_params'].get('sparse_value'):
            elmoModel.sparse_scores(
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '_sparse.csv')
            )

    if config['model'] == 'specter':
        from .models import multifacet_recommender
        specter_predictor = multifacet_recommender.SpecterPredictor(
            specter_dir=config['model_params'].get('specter_dir', "./models/multifacet_recommender/specter/"),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            batch_size=config['model_params'].get('batch_size', 16),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False)
        )
        specter_predictor.set_archives_dataset(archives_dataset)
        specter_predictor.set_submissions_dataset(submissions_dataset)
        if not config['model_params'].get('skip_specter', False):
            publication_path = Path(config['model_params']['publications_path']).joinpath('pub2vec.jsonl')
            if config['model_params'].get('use_redis', False):
                publication_path = None
            specter_predictor.embed_publications(publications_path=publication_path)
            specter_predictor.embed_submissions(submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.jsonl'))
        specter_predictor.all_scores(
            publications_path=publication_path,
            submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.jsonl'),
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )

        if config['model_params'].get('sparse_value'):
            specter_predictor.sparse_scores(
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '_sparse.csv')
            )

    if config['model'] == 'specter2':
        from .models import specter2_scincl
        specter_predictor = specter2_scincl.Specter2Predictor(
            specter_dir=config['model_params'].get('specter_dir', "./models/multifacet_recommender/specter/"),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            batch_size=config['model_params'].get('batch_size', 16),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False)
        )
        specter_predictor.set_archives_dataset(archives_dataset)
        specter_predictor.set_submissions_dataset(submissions_dataset)
        if not config['model_params'].get('skip_specter', False):
            publication_path = Path(config['model_params']['publications_path']).joinpath('pub2vec.jsonl')
            if config['model_params'].get('use_redis', False):
                publication_path = None
            specter_predictor.embed_publications(publications_path=publication_path)
            specter_predictor.embed_submissions(submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.jsonl'))
        specter_predictor.all_scores(
            publications_path=publication_path,
            submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.jsonl'),
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )

        if config['model_params'].get('sparse_value'):
            specter_predictor.sparse_scores(
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '_sparse.csv')
            )

    if config['model'] == 'mfr':
        from .models import multifacet_recommender
        mfr_predictor = multifacet_recommender.MultiFacetRecommender(
            work_dir=config['model_params'].get('work_dir', "./"),
            feature_vocab_file=config['model_params'].get('feature_vocab_file', "./feature/dictionary_index"),
            model_checkpoint_dir=config['model_params'].get('model_checkpoint_dir', "./"),
            epochs=config['model_params'].get('epochs', 100),
            batch_size=config['model_params'].get('batch_size', 50),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value')
        )
        mfr_predictor.set_archives_dataset(archives_dataset)
        mfr_predictor.set_submissions_dataset(submissions_dataset)
        mfr_predictor.embed_publications(publications_path=None)
        mfr_predictor.embed_submissions(submissions_path=None)
        mfr_predictor.all_scores(
            publications_path=None,
            submissions_path=None,
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )

        if config['model_params'].get('sparse_value'):
            mfr_predictor.sparse_scores(
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '_sparse.csv')
            )

    if config['model'] == 'specter+mfr':
        from .models import multifacet_recommender
        ens_predictor = multifacet_recommender.EnsembleModel(
            specter_dir=config['model_params'].get('specter_dir', "./models/multifacet_recommender/specter/"),
            mfr_feature_vocab_file=config['model_params'].get('mfr_feature_vocab_file', "./feature/dictionary_index"),
            mfr_checkpoint_dir=config['model_params'].get('mfr_checkpoint_dir', "./"),
            mfr_epochs=config['model_params'].get('mfr_epochs', 100),
            work_dir=config['model_params'].get('work_dir', "./"),
            average_score=config['model_params'].get('average_score', False),
            max_score=config['model_params'].get('max_score', True),
            specter_batch_size=config['model_params'].get('specter_batch_size', 16),
            mfr_batch_size=config['model_params'].get('mfr_batch_size', 50),
            merge_alpha=config['model_params'].get('merge_alpha', 0.8),
            use_cuda=config['model_params'].get('use_cuda', False),
            sparse_value=config['model_params'].get('sparse_value'),
            use_redis=config['model_params'].get('use_redis', False)
        )
        ens_predictor.set_archives_dataset(archives_dataset)
        ens_predictor.set_submissions_dataset(submissions_dataset)
        specter_publication_path = Path(config['model_params']['publications_path']).joinpath('pub2vec.jsonl')
        if config['model_params'].get('use_redis', False):
            specter_publication_path = None
        ens_predictor.embed_publications(
            specter_publications_path=specter_publication_path,
            mfr_publications_path=None, skip_specter=config['model_params'].get('skip_specter', False)
        )
        ens_predictor.embed_submissions(
            specter_submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.jsonl'),
            mfr_submissions_path=None, skip_specter=config['model_params'].get('skip_specter', False))
        ens_predictor.all_scores(
            specter_publications_path=specter_publication_path,
            mfr_publications_path=None,
            specter_submissions_path=Path(config['model_params']['submissions_path']).joinpath('sub2vec.jsonl'),
            mfr_submissions_path=None,
            scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '.csv')
        )

        if config['model_params'].get('sparse_value'):
            ens_predictor.sparse_scores(
                scores_path=Path(config['model_params']['scores_path']).joinpath(config['name'] + '_sparse.csv')
            )

    if 'alternate_match_group' in config.keys():
        aggregate_by_group(config)

def execute_create_dataset(client, client_v2, config=None):

    config = ModelConfig(config_dict=config)
    
    print(config)

    expertise = OpenReviewExpertise(client, client_v2, config)
    expertise.run()