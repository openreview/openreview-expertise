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
    def __init__(self, archives_dataset, submissions_dataset, use_title=False, use_abstract=True, use_cuda=False, batch_size=8, average_score=False, max_score=True, knn=None):
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
        if self.use_abstract:
            self.sub_note_id_to_vec = self._embed(self.sub_note_id_to_abstract)
        elif self.use_title:
            self.sub_note_id_to_vec = self._embed(self.sub_note_id_to_title)

        with open(submissions_path, 'wb') as f:
            pickle.dump(self.sub_note_id_to_vec, f, pickle.HIGHEST_PROTOCOL)

    def embed_publications(self, publications_path=None):
        print('Embedding publications...')
        if self.use_abstract:
            self.pub_note_id_to_vec = self._embed(self.pub_note_id_to_abstract)
        elif self.use_title:
            self.pub_note_id_to_vec = self._embed(self.pub_note_id_to_title)

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

        csv_scores = []
        if self.average_score:
            for note_id, reviewer_scores in submission_scores.items():
                for reviewer_id, scores in reviewer_scores.items():
                    csv_line = '{note_id},{reviewer},{score}'.format(reviewer=reviewer_id, note_id=note_id, score=np.average(scores))
                    csv_scores.append(csv_line)

        if self.max_score:
            for note_id, reviewer_scores in submission_scores.items():
                for reviewer_id, scores in reviewer_scores.items():
                    csv_line = '{note_id},{reviewer},{score}'.format(reviewer=reviewer_id, note_id=note_id, score=np.max(scores))
                    csv_scores.append(csv_line)

        all_scores = []
        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')
        return all_scores
