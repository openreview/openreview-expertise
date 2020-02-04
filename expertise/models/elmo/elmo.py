import torch
from tqdm import tqdm
from collections import defaultdict

class Model(object):
    def __init__(self, archives_dataset, submissions_dataset, use_title=False, use_abstract=True):
        if not use_title and not use_abstract:
            raise ValueError('use_title and use_abstract cannot both be False')
        if use_title and use_abstract:
            raise ValueError('use_title and use_abstract cannot both be True')
        self.metadata = {
            'closest_match': {},
            'no_expertise': set()
        }
        self.use_title = use_title
        self.use_abstract = use_abstract
        self.submissions_dataset = submissions_dataset
        self.note_id_to_author_ids = defaultdict(list)
        self.note_id_to_abstract = {}
        self.note_id_to_title = {}
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if self.use_abstract and 'abstract' in publication['content']:
                    self.note_id_to_author_ids[publication['id']].append(profile_id)
                    self.note_id_to_abstract[publication['id']] = publication['content']['abstract']
                elif self.use_title and 'title' in publication['content']:
                    self.note_id_to_author_ids[publication['id']].append(profile_id)
                    self.note_id_to_abstract[publication['id']] = publication['content']['title']

    def score(self, submission):
        pass

    def all_scores(self, scores_path=None):
        print('Computing all scores...')
