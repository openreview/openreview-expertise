import time

from collections import defaultdict
import json
import os
import torch
import sys
import itertools
from tqdm import tqdm
from typing import Optional
import redisai
import numpy as np

from transformers import AutoTokenizer, AutoModel
from .predictor import Predictor

from expertise.service.server import redis_embeddings_pool

import logging
"""
archive_file: $SPECTER_FOLDER/model.tar.gz
input_file: $SAMPLE_ID_TRAIN
include-package: specter
predictor: specter_predictor
overrides:
    model:
        predict_mode: 'true'
        include_venue: 'false'
    dataset_reader:
        type: 'specter_data_reader'
        predict_mode: 'true'
        paper_features_path: $SPECTER_TRAIN_FILE
        included_text_fields: 'abstract title'
    vocabulary:
        directory_path: $SPECTER_FOLDER/data/vocab/
cuda-device: 0
output-file: $SPECTER_TRAIN_EMB_RAW
batch-size: 16
silent
"""
class SciNCLPredictor(Predictor):
    def __init__(self, specter_dir, work_dir, average_score=False, max_score=True, batch_size=16, use_cuda=True,
                 sparse_value=None, use_redis=False, dump_p2p=False, compute_paper_paper=False, venue_specific_weights=False):
        self.model_name = 'scincl'
        self.specter_dir = specter_dir
        self.model_archive_file = os.path.join(specter_dir, "model.tar.gz")
        self.vocab_dir = os.path.join(specter_dir, "data/vocab/")
        self.predictor_name = "specter_predictor"
        self.work_dir = work_dir
        self.average_score = average_score
        self.max_score = max_score
        assert max_score ^ average_score, "(Only) One of max_score or average_score must be True"
        self.batch_size = batch_size
        if use_cuda:
            self.cuda_device = torch.device("cuda:0")
        else:
            self.cuda_device = torch.device("cpu")
        self.preliminary_scores = None
        self.sparse_value = sparse_value
        if not os.path.exists(self.work_dir) and not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        self.use_redis = use_redis
        self.redis = None
        self.dump_p2p = dump_p2p
        self.compute_paper_paper = compute_paper_paper
        self.venue_specific_weights = venue_specific_weights

        self.tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
        #load base model
        self.model = AutoModel.from_pretrained('malteos/scincl')
        self.model.to(self.cuda_device)
        self.model.eval()

    def _fetch_batches(self, dict_data, batch_size):
        iterator = iter(dict_data.items())
        for _ in itertools.count():
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    def _batch_predict(self, batch_data):
        jsonl_out = []
        text_batch = [d[1]['title'] + self.tokenizer.sep_token + (d[1].get('abstract') or '') for d in batch_data]
        # preprocess the input
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = inputs.to(self.cuda_device)
        with torch.no_grad():
            output = self.model(**inputs)
        # take the first token in the batch as the embedding
        embeddings = output.last_hidden_state[:, 0, :]

        for paper, embedding in zip(batch_data, embeddings):
            paper = paper[1]
            jsonl_out.append(json.dumps({'paper_id': paper['paper_id'], 'embedding': embedding.detach().cpu().numpy().tolist()}) + '\n')

        # clean up batch data
        del embeddings
        del output
        del inputs
        torch.cuda.empty_cache()
        return jsonl_out

    def set_archives_dataset(self, archives_dataset):
        self.pub_note_id_to_author_ids = defaultdict(list)
        self.pub_author_ids_to_note_id = defaultdict(list)
        self.pub_note_id_to_abstract = {}
        self.pub_note_id_to_title = {}
        self.pub_note_id_to_cache_key = {}
        output_dict = {}
        paper_ids_list = []
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                if publication['content'].get('title').strip() or publication['content'].get('abstract').strip():
                    self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                    self.pub_author_ids_to_note_id[profile_id].append(publication['id'])
                    self.pub_note_id_to_title[publication['id']] = publication['content'].get('title').strip() if publication['content'].get('title').strip() else "."
                    self.pub_note_id_to_abstract[publication['id']] = publication['content'].get('abstract').strip() if publication['content'].get('abstract').strip() else "."
                    pub_mdate = publication.get('mdate', int(time.time()))
                    pub_cache_key = publication['id'] + "_" + str(pub_mdate)
                    pub_weight = publication.get('content', {}).get('weight', 0)
                    self.pub_note_id_to_cache_key[publication['id']] = pub_cache_key
                    if self.redis is None or not self.redis.exists(pub_cache_key):
                        if publication['id'] in output_dict:
                            output_dict[publication['id']]["authors"].append(profile_id)
                        else:
                            paper_ids_list.append(publication['id'])
                            output_dict[publication['id']] = {
                                "title": self.pub_note_id_to_title[publication['id']],
                                "abstract": self.pub_note_id_to_abstract[publication['id']],
                                "paper_id": publication["id"],
                                "authors": [profile_id],
                                "mdate": pub_mdate
                            }
                            if self.venue_specific_weights:
                                output_dict[publication['id']]['weight'] = pub_weight
                        self._remove_keys_from_cache(publication["id"])
                else:
                    print(f"Skipping publication {publication['id']}. Either title or abstract must be provided ")
        with open(os.path.join(self.work_dir, "scincl_reviewer_paper_data.json"), 'w') as f_out:
            json.dump(output_dict, f_out, indent=1)
        with open(os.path.join(self.work_dir, "scincl_reviewer_paper_ids.txt"), 'w') as f_out:
            f_out.write('\n'.join(paper_ids_list)+'\n')

    def set_submissions_dataset(self, submissions_dataset):
        self.sub_note_id_to_abstract = {}
        self.sub_note_id_to_title = {}
        output_dict = {}
        paper_ids_list = []
        for note_id, submission in submissions_dataset.items():
            self.sub_note_id_to_title[submission['id']] = submission['content'].get('title', "")
            self.sub_note_id_to_abstract[submission['id']] = submission['content'].get('abstract', "")
            paper_ids_list.append(submission['id'])
            output_dict[submission['id']] = {"title": self.sub_note_id_to_title[submission['id']],
                                             "abstract": self.sub_note_id_to_abstract[submission['id']],
                                             "paper_id": submission["id"],
                                             "authors": []}
        with open(os.path.join(self.work_dir, "scincl_submission_paper_data.json"), 'w') as f_out:
            json.dump(output_dict, f_out, indent=1)
        with open(os.path.join(self.work_dir, "scincl_submission_paper_ids.txt"), 'w') as f_out:
            f_out.write('\n'.join(paper_ids_list)+'\n')

    def embed_submissions(self, submissions_path=None):
        print('Embedding submissions...')
        metadata_file = os.path.join(self.work_dir, "scincl_submission_paper_data.json")
        ids_file = os.path.join(self.work_dir, "scincl_submission_paper_ids.txt")

        with open(metadata_file, 'r') as f:
            paper_data = json.load(f)

        sub_jsonl = []
        for batch_data in tqdm(self._fetch_batches(paper_data, self.batch_size), desc='Embedding Subs', total=int(len(paper_data.keys())/self.batch_size), unit="batches"):
            sub_jsonl.extend(self._batch_predict(batch_data))

        with open(submissions_path, 'w') as f:
            f.writelines(sub_jsonl)

    def embed_publications(self, publications_path=None):
        if not self.use_redis:
            assert publications_path, "Either publications_path must be given or use_redis must be set to true"
        print('Embedding publications...')
        metadata_file = os.path.join(self.work_dir, "scincl_reviewer_paper_data.json")
        ids_file = os.path.join(self.work_dir, "scincl_reviewer_paper_ids.txt")

        with open(metadata_file, 'r') as f:
            paper_data = json.load(f)

        pub_jsonl = []
        for batch_data in tqdm(self._fetch_batches(paper_data, self.batch_size), desc='Embedding Pubs', total=int(len(paper_data.keys())/self.batch_size), unit="batches"):
            pub_jsonl.extend(self._batch_predict(batch_data))

        with open(publications_path, 'w') as f:
            f.writelines(pub_jsonl)

    def all_scores(self, publications_path=None, submissions_path=None, scores_path=None, p2p_path=None):
        def load_emb_file(emb_file):
            paper_emb_size_default = 768
            id_list = []
            emb_list = []
            weight_list = []
            bad_id_set = set()
            for line in emb_file:
                paper_data = json.loads(line.rstrip())
                paper_id = paper_data['paper_id']
                paper_emb_size = len(paper_data['embedding'])
                assert paper_emb_size == 0 or paper_emb_size == paper_emb_size_default
                if paper_emb_size == 0:
                    paper_emb = [0] * paper_emb_size_default
                    bad_id_set.add(paper_id)
                else:
                    paper_emb = paper_data['embedding']
                id_list.append(paper_id)
                emb_list.append(paper_emb)
                if self.venue_specific_weights:
                    weight_list.append(paper_data.get('weight', 0))
            emb_tensor = torch.tensor(emb_list, device=torch.device('cpu'))
            emb_tensor = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 0.000000000001)
            weight_tensor = torch.tensor(weight_list, device=torch.device('cpu'))
            print(len(bad_id_set))
            return emb_tensor, id_list, bad_id_set, weight_tensor

        print('Loading cached publications...')
        with open(publications_path) as f_in:
            paper_emb_train, train_id_list, train_bad_id_set, train_weight_tensor = load_emb_file(f_in)
        paper_num_train = len(train_id_list)

        paper_id2train_idx = {}
        for idx, paper_id in enumerate(train_id_list):
            paper_id2train_idx[paper_id] = idx

        with open(submissions_path) as f_in:
            print('Loading cached submissions...')
            paper_emb_test, test_id_list, test_bad_id_set, _ = load_emb_file(f_in)
            paper_num_test = len(test_id_list)

        print('Computing all scores...')
        p2p_aff = torch.empty((paper_num_test, paper_num_train), device=torch.device('cpu'))
        for i in range(paper_num_test):
            p2p_aff[i, :] = torch.sum(paper_emb_test[i, :].unsqueeze(dim=0) * paper_emb_train, dim=1)

        if self.venue_specific_weights:
            print('Applying venue specific weights...')
            # - fetch the weights for each reviewer publication
            # - compute softmax across weights
            # - apply to the scores
            # First create a weight vector for all publications
            all_weights = torch.zeros(paper_num_train, device=torch.device('cpu'))
            for reviewer_id, train_note_id_list in self.pub_author_ids_to_note_id.items():
                if len(train_note_id_list) == 0:
                    continue
                train_paper_idx = []
                for paper_id in train_note_id_list:
                    if paper_id not in train_bad_id_set:
                        train_paper_idx.append(paper_id2train_idx[paper_id])
                train_weight_j = train_weight_tensor[train_paper_idx]
                softmax_weight_j = torch.softmax(train_weight_j, dim=0)
                all_weights[train_paper_idx] = softmax_weight_j

            # Then broadcast and multiply
            p2p_aff *= all_weights.unsqueeze(0)  # broadcast to match p2p_aff dimensions

        if self.dump_p2p:
            p2p_dict = {}
            for i in range(paper_num_test):
                p2p_dict[test_id_list[i]] = {}
                for j in range(paper_num_train):
                    p2p_dict[test_id_list[i]][train_id_list[j]] = float(p2p_aff[i, j])
            with open(p2p_path, 'w') as f:
                json.dump(p2p_dict, f, indent=4)

        # Normalize all scores
        min_val = p2p_aff.min()
        max_val = p2p_aff.max()
        p2p_aff_norm = (p2p_aff - min_val) / (max_val - min_val)

        csv_scores = []
        self.preliminary_scores = []

        if self.compute_paper_paper:
            for i in range(paper_num_train):
                for j in range(paper_num_test):
                    csv_line = '{match_id},{submission_id},{score}'.format(match_id=test_id_list[j], submission_id=train_id_list[i],
                                                                    score=p2p_aff[j, i].item())
                    csv_scores.append(csv_line)
                    self.preliminary_scores.append((test_id_list[j], train_id_list[i], p2p_aff[j, i].item()))
        else:
            for reviewer_id, train_note_id_list in self.pub_author_ids_to_note_id.items():
                if len(train_note_id_list) == 0:
                    continue
                train_paper_idx = []
                for paper_id in train_note_id_list:
                    if paper_id not in train_bad_id_set:
                        train_paper_idx.append(paper_id2train_idx[paper_id])
                train_paper_aff_j = p2p_aff_norm[:, train_paper_idx]

                if self.average_score:
                    all_paper_aff = train_paper_aff_j.mean(dim=1)
                elif self.max_score:
                    all_paper_aff = train_paper_aff_j.max(dim=1)[0]
                for j in range(paper_num_test):
                    csv_line = '{note_id},{reviewer},{score}'.format(note_id=test_id_list[j], reviewer=reviewer_id,
                                                                    score=all_paper_aff[j].item())
                    csv_scores.append(csv_line)
                    self.preliminary_scores.append((test_id_list[j], reviewer_id, all_paper_aff[j].item()))

        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')

        return self.preliminary_scores

    def sparse_scores(self, scores_path=None):
        if self.preliminary_scores is None:
            raise RuntimeError("Call all_scores before calling sparse_scores")

        print('Sorting...')
        self.preliminary_scores.sort(key=lambda x: (x[0], x[2]), reverse=True)
        print('Sort 1 complete')
        all_scores = set()
        # They are first sorted by note_id
        all_scores = self._sparse_scores_helper(all_scores, 0)

        # Sort by profile_id
        print('Sorting...')
        self.preliminary_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        print('Sort 2 complete')
        all_scores = self._sparse_scores_helper(all_scores, 1)

        print('Final Sort...')
        all_scores = sorted(list(all_scores), key=lambda x: (x[0], x[2]), reverse=True)
        if scores_path:
            with open(scores_path, 'w') as f:
                for note_id, profile_id, score in all_scores:
                    f.write('{0},{1},{2}\n'.format(note_id, profile_id, score))

        print('Sparse score computation complete')
        return all_scores

    def _remove_keys_from_cache(self, key):
        if self.redis:
            for key in self.redis.scan_iter(match=key+"*"):
                self.redis.delete(key)
