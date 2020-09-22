import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import multiprocessing
import pickle

class Model(object):
    def __init__(self, use_title=False, use_abstract=True, average_score=False, max_score=True, workers=1, sparse_value=None):
        if not average_score and not max_score:
            raise ValueError('average_score and max_score cannot both be False')
        if not use_title and not use_abstract:
            raise ValueError('use_title and use_abstract cannot both be False')
        self.metadata = {
            'closest_match': {},
            'no_expertise': set()
        }
        self.workers = workers
        self.use_title = use_title
        self.use_abstract = use_abstract
        self.average_score = average_score
        self.max_score = max_score
        self.sparse_value = sparse_value

    def set_archives_dataset(self, archives_dataset):
        self.title_corpus = []
        self.abstract_corpus = []
        self.raw_publications = []
        self.profie_id_to_indices = {}
        start_index = 0
        counter = 0
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if self.use_abstract and self._is_valid_field(publication['content'], 'abstract'):
                    tokenized_abstract = publication['content']['abstract'].lower().split(' ')
                    self.abstract_corpus.append(tokenized_abstract)
                    self.raw_publications.append(publication)
                    counter += 1
                elif self.use_title and self._is_valid_field(publication['content'], 'title'):
                    tokenized_title = publication['content']['title'].lower().split(' ')
                    self.title_corpus.append(tokenized_title)
                    self.raw_publications.append(publication)
                    counter += 1
            self.profie_id_to_indices[profile_id] = (start_index, counter)
            start_index = counter

        if self.use_title:
            self.bm25_titles = BM25Okapi(self.title_corpus)
        if self.use_abstract:
            self.bm25_abstracts = BM25Okapi(self.abstract_corpus)

    def set_submissions_dataset(self, submissions_dataset):
        self.submissions_dataset = submissions_dataset

    def _is_valid_field(self, obj, field):
        return field in obj and obj.get(field)

    def normalize_tensor(self, tensor):
        max_value = tensor.max()
        min_value = tensor.min()
        if max_value == min_value:
            tensor[tensor!=0] = 1
            return tensor
        return (tensor - min_value) / (max_value - min_value)

    def score(self, submission):
        submission_scores = None
        reviewer_scores = {}
        if self.use_abstract and self._is_valid_field(submission['content'], 'abstract'):
            tokenized_abstract = submission['content']['abstract'].lower().split(' ')
            submission_scores = torch.tensor(self.bm25_abstracts.get_scores(tokenized_abstract), dtype=torch.float32)
        elif self.use_title and self._is_valid_field(submission['content'], 'title'):
            tokenized_title = submission['content']['title'].lower().split(' ')
            submission_scores = torch.tensor(self.bm25_titles.get_scores(tokenized_title), dtype=torch.float32)
        else:
            return None
        self.metadata['closest_match'][submission['id']] = (submission, self.raw_publications[submission_scores.max(dim=0)[1]])
        submission_scores = self.normalize_tensor(submission_scores)
        if self.average_score:
            for profile_id, (start_index, end_index) in self.profie_id_to_indices.items():
                if (start_index == end_index):
                    reviewer_scores[profile_id] = 0.
                    self.metadata['no_expertise'].add(profile_id)
                else:
                    reviewer_scores[profile_id] = submission_scores[start_index:end_index].mean().item()
        elif self.max_score:
            for profile_id, (start_index, end_index) in self.profie_id_to_indices.items():
                if (start_index == end_index):
                    reviewer_scores[profile_id] = 0.
                    self.metadata['no_expertise'].add(profile_id)
                else:
                    reviewer_scores[profile_id] = submission_scores[start_index:end_index].max().item()
        return reviewer_scores

    def all_scores_helper(self, submissions_dataset):
        csv_scores = []
        for note_id, submission in tqdm(submissions_dataset.items(), total=len(submissions_dataset)):
            reviewer_scores = self.score(submission)
            if not reviewer_scores:
                continue
            for profile_id, score in reviewer_scores.items():
                csv_line = (note_id, profile_id, score)
                csv_scores.append(csv_line)
        return csv_scores

    def all_scores(self, preliminary_scores_path=None, scores_path=None):
        print('Computing all scores...')
        submissions_dicts = []
        submissions_dict = {}
        for idx, (note_id, submission) in enumerate(self.submissions_dataset.items()):
            submissions_dict[note_id] = submission
            if idx % (len(self.submissions_dataset) // self.workers + 1) >= len(self.submissions_dataset) // self.workers:
                submissions_dicts.append(submissions_dict)
                submissions_dict = {}
        submissions_dicts.append(submissions_dict)
        pool = multiprocessing.Pool(processes=self.workers)
        scores_list = pool.map(self.all_scores_helper, submissions_dicts)

        self.preliminary_scores = []
        for scores in scores_list:
            self.preliminary_scores += scores

        if preliminary_scores_path:
            with open(preliminary_scores_path, 'wb') as f:
                pickle.dump(self.preliminary_scores, f, pickle.HIGHEST_PROTOCOL)

        if scores_path:
            with open(scores_path, 'w') as f:
                for note_id, profile_id, score in tqdm(self.preliminary_scores, desc='Saving preliminary scores'):
                    f.write('{0},{1},{2}\n'.format(note_id, profile_id, score))
        return self.preliminary_scores

    def _sparse_scores_helper(self, all_scores, id_index):
        counter = 0
        # Get the first note_id or profile_id
        current_id = self.preliminary_scores[0][id_index]
        if id_index == 0:
            desc = 'Note IDs'
        else:
            desc = 'Profiles IDs'
        for note_id, profile_id, score in tqdm(self.preliminary_scores, total=len(self.preliminary_scores), desc=desc):
            if counter < self.sparse_value:
                all_scores.add((note_id, profile_id, score))
            elif (note_id, profile_id)[id_index] != current_id:
                counter = 0
                all_scores.add((note_id, profile_id, score))
                current_id = (note_id, profile_id)[id_index]
            counter += 1
        return all_scores

    def sparse_scores(self, preliminary_scores_path=None, scores_path=None):
        with open(preliminary_scores_path, 'rb') as f:
            self.preliminary_scores = pickle.load(f)
        print('Sorting...')
        self.preliminary_scores.sort(key=lambda x: (x[0], x[2]), reverse=True)
        all_scores = set()
        # They are first sorted by note_id
        all_scores = self._sparse_scores_helper(all_scores, 0)

        # Sort by profile_id
        print('Sorting...')
        self.preliminary_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        all_scores = self._sparse_scores_helper(all_scores, 1)

        print('Final Sort...')
        all_scores = sorted(list(all_scores), key=lambda x: (x[0], x[2]), reverse=True)
        if scores_path:
            with open(scores_path, 'w') as f:
                for note_id, profile_id, score in all_scores:
                    f.write('{0},{1},{2}\n'.format(note_id, profile_id, score))

        return all_scores
