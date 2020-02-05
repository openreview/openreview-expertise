import argparse
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import multiprocessing
import pickle
import copy
import sys

from .dataset import ArchivesDataset, SubmissionsDataset, BidsDataset
from .config import ModelConfig
from .models import bm25

def ranking(archives_dataset, submissions_dataset, publication_id_to_profile_id, worker):
    # counter = 0
    for note_id, submission in tqdm(submissions_dataset.items(), total=len(submissions_dataset), position=worker):
        removed_publication = None
        for profile_id in publication_id_to_profile_id[note_id]:
            removed_publication = archives_dataset.remove_publication(note_id, profile_id) or removed_publication
        bm25Model = bm25.Model(archives_dataset, submissions_dataset, use_title=config['model_params']['use_title'], use_abstract=config['model_params']['use_abstract'])
        reviewer_scores = bm25Model.score(submission)

        sorted_profile_ids = [(profile_id, value) for profile_id, value in sorted(reviewer_scores.items(), key=lambda item: item[1], reverse=True)]
        with open(Path('./test/' + note_id + '.pkl'), 'wb') as f:
            pickle.dump(sorted_profile_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        for profile_id in publication_id_to_profile_id[note_id]:
            archives_dataset.add_publication(removed_publication, profile_id)
        # counter += 1
        # if counter == 20:
        #     break

def evaluate_scores(scores_path, publication_id_to_profile_id, rank):
    within_rank = 0
    total = 0
    for submission_file in scores_path.iterdir():
        total += 1
        dot_location = str(submission_file.name).rindex('.')
        note_id = str(submission_file.name)[:dot_location]
        with open(submission_file, 'rb') as file_handle:
            sorted_profile_ids = pickle.load(file_handle)
            for idx, (sorted_profile_id, value) in enumerate(sorted_profile_ids):
                if sorted_profile_id in publication_id_to_profile_id[note_id]:
                    if idx < rank:
                        within_rank += 1
                    break
                if idx >= rank:
                    break
    return within_rank / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a JSON file containing all other arguments')
    args = parser.parse_args()

    config = ModelConfig(config_file_path=args.config)
    archives_dataset = ArchivesDataset(archives_path=Path(config['dataset']['directory']).joinpath('archives'))

    if config['model'] == 'bm25':
        workers = config['test_params']['workers']
        submissions_dicts = []
        submissions_dict = {}
        publication_id_to_profile_id = defaultdict(list)
        publication_id_set = set()
        for idx, (profile_id, publications) in enumerate(archives_dataset.items()):
            if idx % (len(archives_dataset) // (workers) + 1) >= len(archives_dataset) // (workers):
                submissions_dicts.append(submissions_dict)
                submissions_dict = {}
            for publication in publications:
                if config['model_params']['use_title'] and 'title' not in publication['content']:
                    continue
                if config['model_params']['use_abstract'] and 'abstract' not in publication['content']:
                    continue
                if publication['id'] not in publication_id_set:
                    submissions_dict[publication['id']] = publication
                    publication_id_set.add(publication['id'])
                publication_id_to_profile_id[publication['id']].append(profile_id)
        submissions_dicts.append(submissions_dict)

        only_evaluate = config['test_params']['only_evaluate']
        if only_evaluate:
            print(evaluate_scores(Path('./test'), publication_id_to_profile_id, config['test_params']['rank']))
            sys.exit(0)

        processes = []
        for worker in range(workers):
            processes.append(multiprocessing.Process(target=ranking, args=(copy.deepcopy(archives_dataset), SubmissionsDataset(submissions_dict=copy.deepcopy(submissions_dicts[worker])), copy.deepcopy(publication_id_to_profile_id), worker, )))

        for process in processes:
            process.start()

        for process in processes:
            process.join()
