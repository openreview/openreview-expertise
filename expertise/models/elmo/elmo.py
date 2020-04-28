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
    def __init__(self, archives_dataset, submissions_dataset, use_title=False, use_abstract=True, use_cuda=False, batch_size=8, average_score=False, max_score=True, knn=None, normalize=False):
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
        self.pub_note_id_to_author_ids = defaultdict(list)
        self.pub_note_id_to_abstract = {}
        self.pub_note_id_to_title = {}
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if self.use_abstract and 'abstract' in publication['content']:
                    self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                    self.pub_note_id_to_abstract[publication['id']] = publication['content']['abstract']
                elif self.use_title and 'title' in publication['content']:
                    self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                    self.pub_note_id_to_title[publication['id']] = publication['content']['title']

        self.sub_note_id_to_abstract = {}
        self.sub_note_id_to_title = {}
        for note_id, submission in submissions_dataset.items():
            if self.use_abstract and 'abstract' in submission['content']:
                self.sub_note_id_to_abstract[submission['id']] = submission['content']['abstract']
            elif self.use_title and 'title' in submission['content']:
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
        self.knn = knn
        self.normalize = normalize

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
            _vecs = self._extract_elmo(_pubs, self.tokenizer, self.elmo)
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

    def normalize_scores(self, score_matrix, axis=1):
        '''
        If axis is 0, we normalize over the submissions
        If axis is 1, we normalize over the publications (recommended)
        '''
        min_values = np.nanmin(score_matrix, axis=axis, keepdims=True)
        max_values = np.nanmax(score_matrix, axis=axis, keepdims=True)
        normalized = (score_matrix - min_values) / (max_values - min_values)
        normalized[np.isnan(normalized)] = 0.5
        return normalized

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
        # D and I are 2D matrices. The row indexes correspond to the submission index
        # of self.sub_note_id_to_vec[1]. That means that row 0 in D corresponds to the submission
        # self.sub_note_id_to_vec[1][0], row 1 to the submission self.sub_note_id_to_vec[1][1], and
        # so on. The values in the D matrix contain the scores between a submission and a
        # publication. The scores in matrix D are sorted in descending order from left to right.
        # In order to know what publication a particular score is for, the I matrix is used.
        # The I matrix contains the indexes of the publications. Let us call the value in [0, 0] of
        # matrix I be v. Therefore the value in [0, 0] of matrix D contains the highest score for
        # submission in self.sub_note_id_to_vec[1][0] and publication self.pub_note_id_to_vec[1][v].
        D, I = index.search(self.sub_note_id_to_vec[1], self.knn)
        # The D matrix scores go from 0 to 2. When values are closer to 0, it means that the
        # similarity is greater. However, we need to have values closer to 1 to indicate more
        # similarity. Also, the D matrix values range from 0 to 2.
        if normalize:
            preliminary_scores = self.normalize_scores(2 - D, axis=1)
        else:
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

        csv_scores = []
        all_scores = []
        if self.average_score:
            for note_id, reviewer_scores in submission_scores.items():
                for reviewer_id, scores in reviewer_scores.items():
                    score = np.average(scores)
                    csv_line = '{note_id},{reviewer},{score}'.format(reviewer=reviewer_id, note_id=note_id, score=score)
                    csv_scores.append(csv_line)
                    all_scores.append((note_id, reviewer_id, score))

        if self.max_score:
            for note_id, reviewer_scores in submission_scores.items():
                for reviewer_id, scores in reviewer_scores.items():
                    score = np.max(scores)
                    csv_line = '{note_id},{reviewer},{score}'.format(reviewer=reviewer_id, note_id=note_id, score=score)
                    csv_scores.append(csv_line)
                    all_scores.append((note_id, reviewer_id, score))

        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')
        return all_scores
