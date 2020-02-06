import torch
from allennlp.commands.elmo import ElmoEmbedder
from sacremoses import MosesTokenizer
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
import math
from collections import defaultdict

class Model(object):
    def __init__(self, archives_dataset, submissions_dataset, use_title=False, use_abstract=True, use_cuda=False, batch_size=8):
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
        for note_id, submission in submissions_dataset:
            if self.use_abstract and 'abstract' in submission['content']:
                self.sub_note_id_to_abstract[subsubmission['id']] =submission['content']['abstract']
            elif self.use_title and 'title' in submission['content']:
                self.sub_note_id_to_title[subsubmission['id']] =submission['content']['title']

        self.batch_size = batch_size
        # create tokenizer and ELMo objects
        self.tokenizer = MosesTokenizer()
        if use_cuda:
            self.elmo = ElmoEmbedder(cuda_device=0)
        else:
            self.elmo = ElmoEmbedder()

    def _embed(self, uid2pub):
        all_pubs = [(uid, pub) for uid, pub in uid2pub.items()]
        batched_pubs = []
        for i in range(math.ceil(len(all_pubs) / batch_size)):
            batched_pubs.append(all_pubs[i * batch_size:(i + 1) * batch_size])

        uids = []
        vecs = []
        for batch in tqdm(batched_pubs, total=len(batched_pubs), desc='Embedding'):
            _uids = [x[0] for x in batch]
            _pubs = [x[1] for x in batch]
            _vecs = extract_elmo(_pubs, tokenizer, elmo)
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

        return {note_id: vector for note_id, vector in zip(uid_index, all_papers_tensor)}

    def embed_submssions(self, submissions_path=None)
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

    def all_scores(self, scores_path=None):
        print('Computing all scores...')
