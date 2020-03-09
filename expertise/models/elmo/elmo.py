import torch
from allennlp.commands.elmo import ElmoEmbedder
from sacremoses import MosesTokenizer
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
import math
from collections import defaultdict
import faiss

class Model(object):
    def __init__(self, archives_dataset, submissions_dataset, use_title=False, use_abstract=True, use_cuda=False, batch_size=8, average_score=False, max_score=True, knn=None, sparse_value=None):
        if not use_title and not use_abstract:
            raise ValueError('use_title and use_abstract cannot both be False')
        self.metadata = {
            'closest_match': {},
            'no_expertise': set()
        }
        self.use_title = use_title
        self.use_abstract = use_abstract
        self.submissions_dataset = submissions_dataset
        self.pub_note_id_to_author_ids = defaultdict(list)
        self.pub_note_id_to_abstract = {}
        self.pub_note_id_to_title = {}
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if self.use_abstract and self._is_valid_field(publication['content'], 'abstract'):
                    self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                    self.pub_note_id_to_abstract[publication['id']] = publication['content']['abstract']
                elif self.use_title and self._is_valid_field(publication['content'], 'title'):
                    self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                    self.pub_note_id_to_title[publication['id']] = publication['content']['title']

        self.sub_note_id_to_abstract = {}
        self.sub_note_id_to_title = {}
        for note_id, submission in submissions_dataset.items():
            if self.use_abstract and self._is_valid_field(submission['content'], 'abstract'):
                self.sub_note_id_to_abstract[submission['id']] = submission['content']['abstract']
            elif self.use_title and self._is_valid_field(submission['content'], 'title'):
                self.sub_note_id_to_title[submission['id']] = submission['content']['title']

        self.batch_size = batch_size
        # create tokenizer and ELMo objects
        self.tokenizer = MosesTokenizer()
        if use_cuda:
            self.elmo = ElmoEmbedder(cuda_device=0)
        else:
            self.elmo = ElmoEmbedder()

        self.average_score = average_score
        self.max_score = max_score
        self.sparse_value = sparse_value
        self.knn = knn

    def _is_valid_field(self, obj, field):
        return field in obj and len(obj.get(field)) > 0

    def _extract_elmo(self, papers, tokenizer, elmo):
        toks_list = []
        for p in papers:
            toks_list.append(tokenizer.tokenize(p, escape=False))
        vecs = elmo.embed_batch(toks_list)
        content_vecs = []
        for vec in vecs:
            new_vec = np.transpose(vec, (1,0,2)).reshape(-1, vec.shape[0]*vec.shape[2])
            content_vecs.append(new_vec.mean(0))
        return np.vstack(content_vecs)

    def _embed(self, uid2pub):
        all_pubs = [(uid, pub) for uid, pub in uid2pub.items()]
        batched_pubs = []
        for i in range(math.ceil(len(all_pubs) / self.batch_size)):
            batched_pubs.append(all_pubs[i * self.batch_size:(i + 1) * self.batch_size])

        uids = []
        vecs = []
        for batch in tqdm(batched_pubs, total=len(batched_pubs), desc='Embedding'):
            _uids = [x[0] for x in batch]
            _pubs = [x[1] for x in batch]
            try:
                _vecs = self._extract_elmo(_pubs, self.tokenizer, self.elmo)
            except:
                print(_uids)
            uids.extend(_uids)
            vecs.append(_vecs)

        vecs = np.vstack(vecs)
        uid2vec = {_uid : _vec for _uid, _vec in zip(uids, vecs)}

        uid_index = []
        all_paper_vecs = []
        for uid, paper_vec in uid2vec.items():
            if ~np.isnan(paper_vec).any():
                uid_index.append(uid)
                all_paper_vecs.append(paper_vec)

        all_papers_tensor = normalize(np.vstack(all_paper_vecs), axis=1)
        all_papers_tensor = all_papers_tensor.astype(np.float32)

        return uid_index, all_papers_tensor

    def embed_submssions(self, submissions_path=None):
        print('Embedding submissions...')
        if self.use_title:
            self.sub_note_id_to_vec = self._embed(self.sub_note_id_to_title)
        elif self.use_abstract:
            self.sub_note_id_to_vec = self._embed(self.sub_note_id_to_abstract)

        with open(submissions_path, 'wb') as f:
            pickle.dump(self.sub_note_id_to_vec, f, pickle.HIGHEST_PROTOCOL)

    def embed_publications(self, publications_path=None):
        print('Embedding publications...')
        if self.use_title:
            self.pub_note_id_to_vec = self._embed(self.pub_note_id_to_title)
        elif self.use_abstract:
            self.pub_note_id_to_vec = self._embed(self.pub_note_id_to_abstract)

        with open(publications_path, 'wb') as f:
            pickle.dump(self.pub_note_id_to_vec, f, pickle.HIGHEST_PROTOCOL)

    def score(self, submission):
        pass

    def all_scores(self, publications_path=None, submissions_path=None, scores_path=None):
        csv_scores = []
        print('Loading cached publications...')
        with open(publications_path, 'rb') as f:
            self.pub_note_id_to_vec = pickle.load(f)
        print('Loading cached submissions...')
        with open(submissions_path, 'rb') as f:
            self.sub_note_id_to_vec = pickle.load(f)

        print('Computing all scores...')
        index = faiss.IndexFlatL2(self.pub_note_id_to_vec[1].shape[1])
        index.add(self.pub_note_id_to_vec[1])

        if self.knn is None:
            self.knn = self.pub_note_id_to_vec[1].shape[0]

        print('Querying the index...')
        D, I = index.search(self.sub_note_id_to_vec[1], self.knn)
        preliminary_scores = (2 - D) / 2

        submission_scores = {}
        for row, publication_indexes in enumerate(I):
            note_id = self.sub_note_id_to_vec[0][row]
            submission_scores[note_id] = defaultdict(list)
            for col, publication_index in enumerate(publication_indexes):
                publication_id = self.pub_note_id_to_vec[0][publication_index]
                profile_ids = self.pub_note_id_to_author_ids[publication_id]
                for reviewer_id in profile_ids:
                    submission_scores[note_id][reviewer_id].append(preliminary_scores[row][col])

        self.preliminary_scores = []
        csv_scores = []
        if self.average_score:
            for note_id, reviewer_scores in submission_scores.items():
                for reviewer_id, scores in reviewer_scores.items():
                    average_score = np.average(scores)
                    self.preliminary_scores.append((note_id, reviewer_id, average_score))
                    csv_line = '{note_id},{reviewer},{score}'.format(reviewer=reviewer_id, note_id=note_id, score=average_score)
                    csv_scores.append(csv_line)

        if self.max_score:
            for note_id, reviewer_scores in submission_scores.items():
                for reviewer_id, scores in reviewer_scores.items():
                    max_score = np.max(scores)
                    self.preliminary_scores.append((note_id, reviewer_id, max_score))
                    csv_line = '{note_id},{reviewer},{score}'.format(reviewer=reviewer_id, note_id=note_id, score=max_score)
                    csv_scores.append(csv_line)

        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')
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

    def sparse_scores(self, scores_path=None):
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
