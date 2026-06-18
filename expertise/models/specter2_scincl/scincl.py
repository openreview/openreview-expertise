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
                 sparse_value=None, use_redis=False, dump_p2p=False, compute_paper_paper=False, percentile_select=None, venue_specific_weights=False,
                 normalize_scores=True, scincl_hf_dir=None):
        self.model_name = 'scincl'
        self.specter_dir = specter_dir
        self.model_archive_file = os.path.join(specter_dir, "model.tar.gz")
        self.vocab_dir = os.path.join(specter_dir, "data/vocab/")
        self.scincl_hf_dir = scincl_hf_dir or os.getenv('SCINCL_HF_DIR') or 'malteos/scincl'
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
        self.scores_matrix = None
        self.test_id_list = None
        self.reviewer_ids = None
        self.sparse_value = sparse_value
        if not os.path.exists(self.work_dir) and not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        self.use_redis = use_redis
        self.redis = None
        self.dump_p2p = dump_p2p
        self.compute_paper_paper = compute_paper_paper
        self.venue_specific_weights = venue_specific_weights
        self.normalize_scores = normalize_scores
        print(f"SciNCL venue_specific_weights: {venue_specific_weights}")
        self.percentile_select = percentile_select

        scincl_source = "BUCKET (local dir)" if os.path.isdir(self.scincl_hf_dir) else "HUGGINGFACE HUB (network)"
        print(f"[scincl] Loading tokenizer from '{self.scincl_hf_dir}' [source={scincl_source}]", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.scincl_hf_dir)
        print(f"[scincl] Loading model from '{self.scincl_hf_dir}' [source={scincl_source}]", flush=True)
        self.model = AutoModel.from_pretrained(self.scincl_hf_dir)
        print("Model loaded, moving to device...")
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
            jsonl_out.append(self._build_embedding_jsonl(paper, embedding))

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
                    pub_weight = publication.get('content', {}).get('weight', 1) ## Mention that default weights are 1
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

        cached, cached_lines = self._load_cached_publication_embeddings(publications_path)
        pub_jsonl = []
        if cached:
            remaining = {}
            for paper_id, paper in paper_data.items():
                line = cached_lines.get(paper_id)
                if line is not None:
                    pub_jsonl.append(line)
                else:
                    remaining[paper_id] = paper
            print(f"Reusing {len(pub_jsonl)} cached publication embeddings; computing {len(remaining)}.")
            paper_data = remaining

        for batch_data in tqdm(self._fetch_batches(paper_data, self.batch_size), desc='Embedding Pubs', total=int(len(paper_data.keys())/self.batch_size), unit="batches"):
            pub_jsonl.extend(self._batch_predict(batch_data))

        with open(publications_path, 'w') as f:
            f.writelines(pub_jsonl)

    def all_scores(self, publications_path=None, submissions_path=None, matrix_path=None, p2p_path=None):
        def load_emb_file(emb_file, paper_id_to_weight=None):
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
                if paper_id_to_weight is not None:
                    weight_list.append(paper_id_to_weight.get(paper_id, 1.0))
            emb_tensor = torch.tensor(emb_list, device=torch.device('cpu'))
            emb_tensor = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 0.000000000001)
            weight_tensor = torch.tensor(weight_list, device=torch.device('cpu'), dtype=torch.float32)
            print(len(bad_id_set))
            return emb_tensor, id_list, bad_id_set, weight_tensor

        train_paper_id_to_weight = None
        if self.venue_specific_weights:
            metadata_file = os.path.join(self.work_dir, "scincl_reviewer_paper_data.json")
            with open(metadata_file) as f:
                train_paper_id_to_weight = {
                    pid: paper.get('weight', 1.0)
                    for pid, paper in json.load(f).items()
                }

        print('Loading cached publications...')
        with open(publications_path) as f_in:
            paper_emb_train, train_id_list, train_bad_id_set, train_weight_tensor = load_emb_file(f_in, paper_id_to_weight=train_paper_id_to_weight)
        paper_num_train = len(train_id_list)

        paper_id2train_idx = {}
        for idx, paper_id in enumerate(train_id_list):
            paper_id2train_idx[paper_id] = idx

        with open(submissions_path) as f_in:
            print('Loading cached submissions...')
            paper_emb_test, test_id_list, test_bad_id_set, _ = load_emb_file(f_in)
            paper_num_test = len(test_id_list)

        print('Computing all scores...')
        p2p_aff = paper_emb_test @ paper_emb_train.T

        # Note: Venue-specific weights are now applied per-reviewer in the scoring loop below

        if self.dump_p2p:
            print('Dumping paper-to-paper scores...', flush=True)
            p2p_dict = {}
            for i in range(paper_num_test):
                p2p_dict[test_id_list[i]] = {}
                for j in range(paper_num_train):
                    p2p_dict[test_id_list[i]][train_id_list[j]] = float(p2p_aff[i, j])
            with open(p2p_path, 'w') as f:
                json.dump(p2p_dict, f, indent=4)

        # Normalize all scores in-place to avoid allocating a second full
        # paper_num_test x paper_num_train tensor (would double peak memory).
        if self.normalize_scores:
            print("Normalizing scores...")
            min_val = p2p_aff.min()
            max_val = p2p_aff.max()
            if max_val - min_val == 0:
                p2p_aff.clamp_(0.0, 1.0)
            else:
                p2p_aff.sub_(min_val).div_(max_val - min_val)
        else:
            print("Skipping normalization of scores...")
        p2p_aff_norm = p2p_aff

        print("Computing scincl per-reviewer scores...", flush=True)
        if self.compute_paper_paper:
            # Paper-paper similarity: matrix IS the full p2p_aff_norm.
            # Column ids are train paper ids (not reviewer ids).
            # TODO: at very large scale (e.g. 35k x 500k = ~70GB) this tensor
            # won't fit in memory for torch.save; needs chunked save.
            self.scores_matrix = p2p_aff_norm
            self.test_id_list = test_id_list
            self.reviewer_ids = train_id_list
        else:
            # Build a flat list of valid paper indices per reviewer.
            # This loop does NOT do any PyTorch compute; it only prepares the
            # index data so we can run a single bulk gather and reduction
            # outside the loop (instead of slicing and reducing per reviewer).
            reviewer_ids = []
            reviewer_paper_indices = []
            for reviewer_id, train_note_id_list in self.pub_author_ids_to_note_id.items():
                if len(train_note_id_list) == 0:
                    continue
                valid = [paper_id2train_idx[pid] for pid in train_note_id_list if pid not in train_bad_id_set]
                if not valid:
                    # Every publication for this reviewer had a bad embedding
                    # (length 0, added to train_bad_id_set in load_emb_file).
                    # Skip consistently with the "no publications" case.
                    continue
                reviewer_ids.append(reviewer_id)
                reviewer_paper_indices.append(valid)

            if reviewer_ids:
                max_papers = max(len(papers) for papers in reviewer_paper_indices)
                num_reviewers = len(reviewer_ids)
                # reviewer_paper_indices is a ragged Python list (reviewers have
                # different numbers of papers), so we pad it into a dense matrix.
                # No PyTorch compute here — just list-to-tensor formatting.
                paper_idx = torch.zeros((num_reviewers, max_papers), dtype=torch.long)
                mask = torch.zeros((num_reviewers, max_papers), dtype=torch.bool)
                for r, papers in enumerate(reviewer_paper_indices):
                    paper_idx[r, :len(papers)] = torch.tensor(papers, dtype=torch.long)
                    mask[r, :len(papers)] = True

                # Single bulk gather: all reviewer–paper scores at once.
                # Old code did p2p_aff_norm[:, train_paper_idx] once per reviewer
                # inside the loop above; now we do it once for everyone.
                scores = p2p_aff_norm[:, paper_idx]  # (num_test, num_reviewers, max_papers)

                # Apply venue weights in bulk across the entire 3D tensor.
                # Old code did this per reviewer inside the loop above.
                if self.venue_specific_weights:
                    weights = train_weight_tensor[paper_idx]
                    epsilon = 1e-8
                    logits = torch.logit(scores, eps=epsilon)
                    log_w = torch.log(torch.clamp(weights, min=epsilon))
                    scores = torch.sigmoid(logits + log_w.unsqueeze(0))

                # Single bulk reduction across dim=2 (the paper dimension).
                # Old code did .mean(dim=1), .max(dim=1)[0], or torch.quantile(..., dim=1)
                # once per reviewer inside the loop above.

                if self.percentile_select is not None:
                    q = max(0.0, min(1.0, self.percentile_select / 100.0))
                    nan_scores = scores.masked_fill(~mask.unsqueeze(0), float('nan'))
                    self.scores_matrix = torch.nanquantile(nan_scores, q, dim=2, interpolation='linear')
                elif self.average_score:
                    zero_scores = scores.masked_fill(~mask.unsqueeze(0), 0.0)
                    counts = mask.sum(dim=1).clamp(min=1).to(scores.dtype)
                    self.scores_matrix = zero_scores.sum(dim=2) / counts.unsqueeze(0)
                elif self.max_score:
                    neg_inf_scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
                    self.scores_matrix = neg_inf_scores.max(dim=2).values
            else:
                # No eligible reviewers (every train_note_id_list empty or all
                # publications dropped via bad_id_set). Produce a 0-column
                # matrix so downstream sparse generation / merging / CSV
                # emission all see a well-formed but empty result.
                self.scores_matrix = torch.empty((paper_num_test, 0), dtype=p2p_aff_norm.dtype)
            self.test_id_list = test_id_list
            self.reviewer_ids = reviewer_ids

        # Round once, vectorized — matches the previous per-row round(..., 4).
        self.scores_matrix = (self.scores_matrix * 10000).round() / 10000
        print(f"Computed preliminary scores for SciNCL.", flush=True)

        if matrix_path:
            print(f"Saving SciNCL scores matrix to {matrix_path}...", flush=True)
            torch.save({
                'scores': self.scores_matrix,
                'test_ids': self.test_id_list,
                'reviewer_ids': self.reviewer_ids,
            }, matrix_path)

        print("Done computing scincl scores.", flush=True)
        return self.scores_matrix

    def _remove_keys_from_cache(self, key):
        if self.redis:
            for key in self.redis.scan_iter(match=key+"*"):
                self.redis.delete(key)
