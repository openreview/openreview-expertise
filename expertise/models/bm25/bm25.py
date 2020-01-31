import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi

class Model(object):
    def __init__(self, archives_dataset, submissions_dataset, use_title=False, use_abstract=True, average_score=False, max_score=True):
        if not average_score and not max_score:
            raise ValueError('average_score and max_score cannot both be False')
        if not use_title and not use_abstract:
            raise ValueError('use_title and use_abstract cannot both be False')
        self.metadata = {
            'closest_match': {},
            'no_expertise': []
        }
        self.use_title = use_title
        self.use_abstract = use_abstract
        self.average_score = average_score
        self.max_score = max_score
        self.submissions_dataset = submissions_dataset
        self.title_corpus = []
        self.abstract_corpus = []
        self.raw_publications = []
        self.closest_match = []
        self.profie_id_to_indices = {}
        start_index = 0
        counter = 0
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if self.use_abstract and 'abstract' in publication['content']:
                    tokenized_abstract = publication['content']['abstract'].lower().split(' ')
                    self.abstract_corpus.append(tokenized_abstract)
                    self.raw_publications.append(publication)
                    counter += 1
                elif self.use_title and 'title' in publication['content']:
                    tokenized_title = publication['content']['title'].lower().split(' ')
                    self.title_corpus.append(tokenized_title)
                    self.raw_publications.append(publication)
                    counter += 1
            self.profie_id_to_indices[profile_id] = (start_index, counter)
            start_index = counter

        if use_title:
            self.bm25_titles = BM25Okapi(self.title_corpus)
        if use_abstract:
            self.bm25_abstracts = BM25Okapi(self.abstract_corpus)

    def normalize_tensor(self, tensor):
        maxValue = tensor.max()
        minValue = tensor.min()
        return (tensor - minValue) / (maxValue - minValue)

    def score(self, submission):
        submission_scores = None
        reviewer_scores = {}
        if self.use_abstract:
            tokenized_abstract = submission['content']['abstract'].lower().split(' ')
            submission_scores = torch.tensor(self.bm25_abstracts.get_scores(tokenized_abstract), dtype=torch.float32)
        elif self.use_title:
            tokenized_title = submission['content']['title'].lower().split(' ')
            submission_scores = torch.tensor(self.bm25_titles.get_scores(tokenized_title), dtype=torch.float32)
        self.metadata['closest_match'][submission['id']] = (submission, self.raw_publications[submission_scores.max(dim=0)[1]])
        submission_scores = self.normalize_tensor(submission_scores)
        if self.average_score:
            for profile_id, (start_index, end_index) in self.profie_id_to_indices.items():
                if (start_index == end_index):
                    reviewer_scores[profile_id] = 0.
                    self.metadata['no_expertise'].append(profile_id)
                else:
                    reviewer_scores[profile_id] = submission_scores[start_index:end_index].mean().item()
        if self.max_score:
            for profile_id, (start_index, end_index) in self.profie_id_to_indices.items():
                if (start_index == end_index):
                    reviewer_scores[profile_id] = 0.
                    self.metadata['no_expertise'].append(profile_id)
                else:
                    reviewer_scores[profile_id] = submission_scores[start_index:end_index].max().item()
        return reviewer_scores

    def all_scores(self, scores_path=None):
        print('Computing all scores...')
        csv_scores = []
        for note_id, submission in tqdm(self.submissions_dataset.items(), total=len(self.submissions_dataset)):
            reviewer_scores = self.score(submission)
            for profile_id, score in reviewer_scores.items():
                csv_line = '{reviewer},{note_id},{score}'.format(reviewer=profile_id, note_id=note_id, score=score)
                csv_scores.append(csv_line)
        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')
        return csv_scores
