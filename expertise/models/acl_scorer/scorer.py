import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sacremoses import MosesTokenizer
from tqdm import tqdm

from .instance import Instance
from .models import Averaging, LSTM
from .utils import entok, unk_string, torchify_batch


class Model(object):
    def __init__(self, batch_size=128, max_score=True, weighted_topk: int = None,
                 sparse_value=None, normalize=False, use_cuda=False):
        # create tokenizer and ELMo objects
        self.tokenizer = MosesTokenizer()
        self.model = None
        self.device = 'cuda' if use_cuda else 'cpu'

        self.batch_size = batch_size
        self.max_score = max_score
        self.weighted_topk = weighted_topk
        self.sparse_value = sparse_value
        self.normalize = normalize

        self.pub_note_id_to_author_ids = defaultdict(list)
        self.pub_note_id_to_abstract = {}
        self.pub_note_id_to_vec = None

        self.n_pub_notes = 0
        self.n_authors = 0
        self.pub_note_author_mapping = np.zeros((self.n_pub_notes, self.n_authors))
        self.author_id_mapping = defaultdict(list)

        self.sub_note_id_to_abstract = {}
        self.sub_note_id_to_vec = None

        self.preliminary_scores = []

    def set_archives_dataset(self, archives_dataset):
        for i, archive in enumerate(archives_dataset.items()):
            profile_id, publications = archive
            self.author_id_mapping[profile_id].append(i)
            for publication in publications:
                if self._is_valid_field(publication['content'], 'abstract'):
                    self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                    self.pub_note_id_to_abstract[publication['id']] = publication['content']['abstract']

    def set_pub_note_author_mapping(self):
        self.n_authors = len(self.author_id_mapping)
        self.n_pub_notes = len(self.pub_note_id_to_author_ids)
        pub_note_author_mapping = np.zeros((self.n_pub_notes, self.n_authors))

        for i, pub_note_id in enumerate(self.pub_note_id_to_author_ids.keys()):
            for author_id in self.pub_note_id_to_author_ids[pub_note_id]:
                j = self.author_id_mapping[author_id]
                pub_note_author_mapping[i][j] = 1

        self.pub_note_author_mapping = pub_note_author_mapping

    def set_submissions_dataset(self, submissions_dataset):
        for note_id, submission in submissions_dataset.items():
            if self._is_valid_field(submission['content'], 'abstract'):
                self.sub_note_id_to_abstract[submission['id']] = submission['content']['abstract']

    def _is_valid_field(self, obj, field):
        return field in obj and obj.get(field)

    def load_model(self, data, model_dir):
        model = torch.load(Path(model_dir).joinpath('scratch/similarity-model.pt'), map_location=torch.device(self.device))

        state_dict = model['state_dict']
        model_args = model['args']
        vocab = model['vocab']
        vocab_fr = model['vocab_fr']
        optimizer = model['optimizer']

        if self.device == 'cpu':
            model_args.gpu = False

        if model_args.model == "avg":
            model = Averaging(data, model_args, vocab, vocab_fr, model_dir)
        elif model_args.model == "lstm":
            model = LSTM(data, model_args, vocab, vocab_fr, model_dir)

        model.load_state_dict(state_dict)
        model.optimizer.load_state_dict(optimizer)

        self.model = model

    def calc_similarity_matrix(self, similarity_scores_path=None):
        pub_emb = self.pub_note_id_to_vec
        sub_emb = self.sub_note_id_to_vec

        assert pub_emb is not None
        assert sub_emb is not None

        print(f'Performing similarity calculation')
        similarity_matrix = np.matmul(sub_emb, np.transpose(pub_emb))
        if similarity_scores_path is not None:
            np.save(similarity_scores_path, similarity_matrix)
        return similarity_matrix

    def embed_submissions(self, submissions_path=None):
        print('Embedding submissions...')
        self.sub_note_id_to_vec = self._embed(self.sub_note_id_to_abstract)

        with open(submissions_path, 'wb') as f:
            pickle.dump(self.sub_note_id_to_vec, f, pickle.HIGHEST_PROTOCOL)

    def embed_publications(self, publications_path=None):
        print('Embedding publications...')
        self.pub_note_id_to_vec = self._embed(self.pub_note_id_to_abstract)

        with open(publications_path, 'wb') as f:
            pickle.dump(self.pub_note_id_to_vec, f, pickle.HIGHEST_PROTOCOL)

    def normalize_scores(self, score_matrix, axis=1):
        """
        If axis is 0, we normalize over the submissions
        If axis is 1, we normalize over the publications (recommended)
        """
        min_values = np.nanmin(score_matrix, axis=axis, keepdims=True)
        max_values = np.nanmax(score_matrix, axis=axis, keepdims=True)
        normalized = (score_matrix - min_values) / (max_values - min_values)
        normalized[np.isnan(normalized)] = 0.5
        return normalized

    def all_scores(self, publications_path=None, submissions_path=None, scores_path=None, similarity_scores_path=None):
        print('Loading cached publications...')
        with open(publications_path, 'rb') as f:
            self.pub_note_id_to_vec = pickle.load(f)
        print('Loading cached submissions...')
        with open(submissions_path, 'rb') as f:
            self.sub_note_id_to_vec = pickle.load(f)

        print('Computing all scores...')
        paper_similarity_score_matrix = self.calc_similarity_matrix(similarity_scores_path)
        note_author_mapping = self.pub_note_author_mapping
        reviewer_scores = np.zeros((paper_similarity_score_matrix.shape[0], note_author_mapping.shape[1]))
        print('Calculating aggregate reviewer scores')

        for i in range(paper_similarity_score_matrix.shape[0]):
            scores = paper_similarity_score_matrix[i]
            invalid_score = 0

            mapping_scores = note_author_mapping * scores.reshape((len(scores), 1))
            mapping_scores += (1 - note_author_mapping) * invalid_score

            if self.max_score:
                reviewer_scores[i] = np.amax(mapping_scores, axis=0)
            elif self.weighted_topk is not None and self.weighted_topk > 0:
                k = self.weighted_topk
                weighting = np.reshape(1 / np.array(range(1, k + 1)), (k, 1))
                mapping_scores.sort(axis=0)
                top_k = mapping_scores[-k:, :]
                reviewer_scores[i] = (top_k * weighting).sum(axis=0)
            else:
                raise ValueError(
                    'Either max_score should be set to True or weighted_topk should be set to a positive integer.'
                )

        if self.normalize:
            reviewer_scores = self.normalize_scores(reviewer_scores)

        csv_scores = []
        sub_note_ids = list(self.sub_note_id_to_abstract.keys())
        reviewer_ids = list(self.author_id_mapping.keys())

        for i, sub_note_id in enumerate(sub_note_ids):
            for j, reviewer_id in enumerate(reviewer_ids):
                csv_line = '{note_id},{reviewer},{score}'.format(note_id=sub_note_id, reviewer=reviewer_id,
                                                                 score=reviewer_scores[i, j])
                csv_scores.append(csv_line)
                self.preliminary_scores.append((sub_note_ids[i], reviewer_id, reviewer_scores[i, j]))

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
        print('preliminary', self.preliminary_scores, len(self.preliminary_scores))
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

        print('ALL SCORES', all_scores)
        return all_scores

    def _embed(self, note_id_to_abs):
        print(
            f'Preprocessing notes',
        )
        abs_values = list(note_id_to_abs.values())
        abs_embs = []
        for i, line in enumerate(abs_values):
            tokens = " ".join(entok.tokenize(line, escape=False)).lower()
            if self.model.sp is not None:
                tokens = self.model.sp.EncodeAsPieces(tokens)
                tokens = " ".join(tokens)
            token_instances = Instance(tokens)
            token_instances.populate_embeddings(self.model.vocab, self.model.zero_unk, self.model.args.ngrams)
            if len(token_instances.embeddings) == 0:
                token_instances.embeddings.append(self.model.vocab[unk_string])
            abs_embs.append(token_instances)

        # Create embeddings
        print(
            'Creating Embeddings'
        )
        embeddings = np.zeros((len(abs_values), self.model.args.dim))
        for i in range(0, len(abs_embs), self.batch_size):
            max_idx = min(i + self.batch_size, len(abs_embs))
            curr_batch = abs_embs[i:max_idx]
            gpu = False if self.device == 'cpu' else True
            wx1, wl1 = torchify_batch(curr_batch, gpu)
            vecs = self.model.encode(wx1, wl1)
            vecs = vecs.detach().cpu().numpy()
            # Normalize for NN search
            vecs = vecs / np.sqrt((vecs * vecs).sum(axis=1))[:, None]
            embeddings[i:max_idx] = vecs
            # print_progress(i, self.batch_size)

        return embeddings
