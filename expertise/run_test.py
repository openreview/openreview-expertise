import argparse
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict, defaultdict

from .dataset import ArchivesDataset, SubmissionsDataset, BidsDataset
from .config import ModelConfig
from .models import bm25

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a JSON file containing all other arguments')
    args = parser.parse_args()

    config = ModelConfig(config_file_path=args.config)
    archives_dataset = ArchivesDataset(archives_path=Path(config['dataset']['directory']).joinpath('archives'))
    # submissions_dataset = SubmissionsDataset(submissions_path=Path(config['dataset']['directory']).joinpath('submissions'))

    if config['model'] == 'bm25':
        submissions_dict = {}
        publication_id_to_profile_id = defaultdict(list)
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                submissions_dict[publication['id']] = publication
                publication_id_to_profile_id[publication['id']].append(profile_id)
        submissions_dataset = SubmissionsDataset(submissions_dict=submissions_dict)

        rank = 50
        good = []
        counter = 0
        for note_id, submission in tqdm(submissions_dataset.items(), total=len(submissions_dataset)):
            removed_publications = []
            for profile_id in publication_id_to_profile_id[note_id]:
                removed_publications.append(archives_dataset.remove_publication(note_id, profile_id))
            bm25Model = bm25.Model(archives_dataset, submissions_dataset, use_title=config['model_params']['use_title'], use_abstract=config['model_params']['use_abstract'])
            reviewer_scores = bm25Model.score(submission)

            sorted_profile_ids = [(profile_id, value) for profile_id, value in sorted(reviewer_scores.items(), key=lambda item: item[1], reverse=True)]
            for idx, (sorted_profile_id, value) in enumerate(sorted_profile_ids):
                if sorted_profile_id in publication_id_to_profile_id[note_id]:
                    if idx < rank:
                        good.append(profile_id)
                    break
                if idx >= rank:
                    break
            for removed_publication in removed_publications:
                archives_dataset.add_publication(removed_publication, profile_id)
            # counter += 1
            # if counter > 100:
            #     break

        print('{good}/{all}'.format(good=len(good), all=len(submissions_dataset)))
