import time

from collections import defaultdict
import base64
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

KEEP_DIMS = [
    0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33,
    34, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 53, 54, 55, 57, 58, 59, 60, 61, 64, 65, 68, 71,
    72, 73, 74, 75, 78, 82, 86, 88, 89, 92, 95, 98, 99, 100, 102, 103, 104, 106, 108, 109, 111, 115, 116, 117,
    120, 122, 123, 124, 125, 126, 127, 128, 130, 131, 133, 134, 136, 137, 138, 139, 140, 142, 145, 146, 147, 149,
    150, 151, 152, 153, 156, 158, 159, 160, 162, 164, 167, 169, 170, 171, 175, 178, 179, 180, 181, 182, 183, 185,
    188, 190, 192, 194, 198, 199, 201, 203, 204, 205, 206, 207, 208, 211, 212, 214, 215, 217, 218, 219, 220, 221,
    222, 223, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 237, 238, 240, 244, 246, 247, 251, 253, 254, 256,
    257, 258, 259, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 282, 283, 284, 285,
    286, 287, 288, 289, 290, 291, 293, 294, 295, 297, 298, 301, 302, 304, 306, 307, 308, 309, 310, 311, 312, 316,
    317, 319, 321, 322, 324, 325, 326, 327, 329, 330, 331, 333, 334, 335, 337, 338, 339, 340, 341, 345, 347, 349,
    351, 354, 355, 356, 358, 359, 360, 363, 365, 368, 369, 372, 373, 376, 379, 382, 383, 384, 386, 388, 389, 390,
    391, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 408, 409, 410, 412, 413, 414, 416, 417, 418, 419,
    420, 421, 422, 423, 424, 426, 429, 430, 431, 433, 434, 436, 438, 442, 443, 448, 449, 450, 451, 453, 455, 456,
    457, 458, 459, 461, 462, 463, 464, 467, 468, 470, 471, 472, 474, 475, 476, 477, 478, 481, 482, 483, 484, 485,
    486, 487, 488, 489, 490, 491, 496, 497, 499, 500, 503, 505, 509, 511, 512, 513, 514, 516, 517, 518, 519, 520,
    522, 524, 525, 526, 527, 528, 529, 530, 532, 533, 534, 535, 537, 538, 543, 544, 545, 546, 547, 549, 550, 552,
    553, 554, 555, 556, 557, 559, 560, 561, 565, 567, 569, 570, 572, 573, 574, 575, 576, 578, 580, 581, 582, 583,
    584, 585, 587, 589, 591, 592, 594, 595, 596, 597, 598, 600, 601, 603, 605, 606, 607, 608, 609, 610, 611, 614,
    615, 617, 620, 623, 624, 626, 628, 629, 630, 633, 634, 636, 637, 638, 643, 644, 645, 646, 647, 648, 651, 653,
    654, 655, 656, 657, 658, 660, 661, 662, 663, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 676, 678, 679,
    681, 682, 683, 684, 685, 687, 688, 689, 690, 694, 695, 696, 697, 698, 699, 700, 701, 703, 704, 706, 707, 709,
    710, 714, 715, 717, 718, 719, 721, 723, 724, 725, 726, 727, 728, 733, 734, 735, 736, 737, 739, 740, 741, 743,
    745, 747, 750, 751, 752, 753, 754, 755, 756, 758, 759, 761, 764, 767
]
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
                 normalize_scores=True, scincl_hf_dir=None, embedding_compression=None):
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
        self.preliminary_scores = None
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
        if embedding_compression == 'int8':
            embedding_compression = 'int8_per_vector'
        if embedding_compression == 'int8_keep_dims':
            embedding_compression = 'int8_per_vector_keep_dims'
        valid_compressions = {None, 'float16', 'int8_per_vector', 'int8_per_vector_keep_dims'}
        if embedding_compression not in valid_compressions:
            raise ValueError(f"Unsupported SciNCL embedding_compression: {embedding_compression}")
        self.embedding_compression = embedding_compression

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

    def _build_embedding_jsonl(self, paper, embedding):
        embedding_array = embedding.detach().cpu().numpy().astype(np.float32)
        data = {
            'paper_id': paper['paper_id']
        }

        if self.embedding_compression is None:
            data['embedding'] = embedding_array.tolist()
        elif self.embedding_compression == 'float16':
            data['embedding_compression'] = 'float16'
            data['embedding_bytes'] = base64.b64encode(
                np.ascontiguousarray(embedding_array.astype(np.float16)).tobytes()
            ).decode('ascii')
        elif self.embedding_compression == 'int8_per_vector':
            max_abs = float(np.max(np.abs(embedding_array)))
            if max_abs == 0.0:
                scale = 1.0
                quantized = np.zeros_like(embedding_array, dtype=np.int8)
            else:
                scale = max_abs / 127.0
                quantized = np.clip(np.rint(embedding_array / scale), -127, 127).astype(np.int8)
            data['embedding_compression'] = 'int8_per_vector'
            data['embedding_scale'] = scale
            data['embedding_bytes'] = base64.b64encode(
                np.ascontiguousarray(quantized).tobytes()
            ).decode('ascii')
        elif self.embedding_compression == 'int8_per_vector_keep_dims':
            kept_embedding = embedding_array[KEEP_DIMS]
            max_abs = float(np.max(np.abs(kept_embedding)))
            if max_abs == 0.0:
                scale = 1.0
                quantized = np.zeros_like(kept_embedding, dtype=np.int8)
            else:
                scale = max_abs / 127.0
                quantized = np.clip(np.rint(kept_embedding / scale), -127, 127).astype(np.int8)
            data['embedding_compression'] = 'int8_per_vector_keep_dims'
            data['embedding_scale'] = scale
            data['embedding_bytes'] = base64.b64encode(
                np.ascontiguousarray(quantized).tobytes()
            ).decode('ascii')
        else:
            raise ValueError(f"Unsupported SciNCL embedding_compression: {self.embedding_compression}")

        if 'weight' in paper:
            data['weight'] = paper['weight']
        return json.dumps(data) + '\n'

    def _restore_embedding(self, paper_data, paper_emb_size_default=768):
        compression = paper_data.get('embedding_compression')
        if compression is None:
            paper_emb_size = len(paper_data['embedding'])
            assert paper_emb_size == 0 or paper_emb_size == paper_emb_size_default
            if paper_emb_size == 0:
                return [0] * paper_emb_size_default
            return paper_data['embedding']
        if compression == 'float16':
            restored = np.frombuffer(
                base64.b64decode(paper_data['embedding_bytes']),
                dtype=np.float16
            ).astype(np.float32)
            assert restored.shape[0] == paper_emb_size_default
            return restored.tolist()
        if compression == 'int8_per_vector':
            restored = np.frombuffer(
                base64.b64decode(paper_data['embedding_bytes']),
                dtype=np.int8
            ).astype(np.float32) * np.float32(paper_data['embedding_scale'])
            assert restored.shape[0] == paper_emb_size_default
            return restored.tolist()
        if compression == 'int8_per_vector_keep_dims':
            restored_keep_dims = np.frombuffer(
                base64.b64decode(paper_data['embedding_bytes']),
                dtype=np.int8
            ).astype(np.float32) * np.float32(paper_data['embedding_scale'])
            assert restored_keep_dims.shape[0] == len(KEEP_DIMS)
            restored = np.zeros(paper_emb_size_default, dtype=np.float32)
            restored[np.asarray(KEEP_DIMS, dtype=np.int64)] = restored_keep_dims
            return restored.tolist()
        raise ValueError(f"Unsupported SciNCL embedding_compression in cache: {compression}")

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

        pub_jsonl = []
        for batch_data in tqdm(self._fetch_batches(paper_data, self.batch_size), desc='Embedding Pubs', total=int(len(paper_data.keys())/self.batch_size), unit="batches"):
            pub_jsonl.extend(self._batch_predict(batch_data))

        with open(publications_path, 'w') as f:
            f.writelines(pub_jsonl)

    def all_scores(self, publications_path=None, submissions_path=None, scores_path=None, p2p_path=None):
        def load_emb_file(emb_file, load_weight=False):
            paper_emb_size_default = 768
            id_list = []
            emb_list = []
            weight_list = []
            bad_id_set = set()
            for line in emb_file:
                paper_data = json.loads(line.rstrip())
                paper_id = paper_data['paper_id']
                if paper_data.get('embedding_compression') is None and len(paper_data['embedding']) == 0:
                    bad_id_set.add(paper_id)
                paper_emb = self._restore_embedding(paper_data, paper_emb_size_default)
                id_list.append(paper_id)
                emb_list.append(paper_emb)
                if load_weight:
                    weight_list.append(paper_data['weight'])
            emb_tensor = torch.tensor(emb_list, device=torch.device('cpu'))
            emb_tensor = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 0.000000000001)
            weight_tensor = torch.tensor(weight_list, device=torch.device('cpu'), dtype=torch.float32)
            print(len(bad_id_set))
            return emb_tensor, id_list, bad_id_set, weight_tensor

        print('Loading cached publications...')
        with open(publications_path) as f_in:
            paper_emb_train, train_id_list, train_bad_id_set, train_weight_tensor = load_emb_file(f_in, load_weight=self.venue_specific_weights)
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

        # Note: Venue-specific weights are now applied per-reviewer in the scoring loop below

        if self.dump_p2p:
            p2p_dict = {}
            for i in range(paper_num_test):
                p2p_dict[test_id_list[i]] = {}
                for j in range(paper_num_train):
                    p2p_dict[test_id_list[i]][train_id_list[j]] = float(p2p_aff[i, j])
            with open(p2p_path, 'w') as f:
                json.dump(p2p_dict, f, indent=4)

        # Normalize all scores
        if self.normalize_scores:
            print("Normalizing scores...")
            min_val = p2p_aff.min()
            max_val = p2p_aff.max()
            if max_val - min_val == 0:
                p2p_aff_norm = torch.clamp(p2p_aff, 0.0, 1.0)
            else:
                p2p_aff_norm = (p2p_aff - min_val) / (max_val - min_val)
        else:
            print("Skipping normalization of scores...")
            p2p_aff_norm = p2p_aff

        csv_scores = []
        self.preliminary_scores = []

        if self.compute_paper_paper:
            for i in range(paper_num_train):
                for j in range(paper_num_test):
                    csv_line = '{match_id},{submission_id},{score}'.format(match_id=test_id_list[j], submission_id=train_id_list[i],
                                                                    score=round(p2p_aff_norm[j, i].item(), 4))
                    csv_scores.append(csv_line)
                    self.preliminary_scores.append((test_id_list[j], train_id_list[i], round(p2p_aff_norm[j, i].item(), 4)))
        else:
            for reviewer_id, train_note_id_list in self.pub_author_ids_to_note_id.items():
                if len(train_note_id_list) == 0:
                    continue
                train_paper_idx = []
                for paper_id in train_note_id_list:
                    if paper_id not in train_bad_id_set:
                        train_paper_idx.append(paper_id2train_idx[paper_id])
                train_paper_aff_j = p2p_aff_norm[:, train_paper_idx]

                # Apply venue-specific weights per reviewer
                if self.venue_specific_weights:
                    train_weight_j = train_weight_tensor[train_paper_idx]
                    # Logit-space transformation preserves bounds and probability mass
                    epsilon = 1e-8 ## Numerical stability
                    logits = torch.logit(train_paper_aff_j, eps=epsilon)
                    weighted_logits = logits + torch.log(torch.clamp(train_weight_j, min=epsilon)).unsqueeze(0)
                    train_paper_aff_j = torch.sigmoid(weighted_logits)

                if self.percentile_select is not None:
                    # Select score based on percentile
                    # q=1.0 (percentile_select=100) -> max score
                    # q=0.5 (percentile_select=50) -> median score
                    # q=0.0 (percentile_select=0) -> min score
                    q = self.percentile_select / 100.0
                    q = max(0.0, min(1.0, q)) 
                    all_paper_aff = torch.quantile(train_paper_aff_j, q, dim=1, interpolation='linear')
                elif self.average_score:
                    all_paper_aff = train_paper_aff_j.mean(dim=1)
                elif self.max_score:
                    all_paper_aff = train_paper_aff_j.max(dim=1)[0]
                for j in range(paper_num_test):
                    csv_line = '{note_id},{reviewer},{score}'.format(note_id=test_id_list[j], reviewer=reviewer_id,
                                                                    score=round(all_paper_aff[j].item(), 4))
                    csv_scores.append(csv_line)
                    self.preliminary_scores.append((test_id_list[j], reviewer_id, round(all_paper_aff[j].item(), 4)))

        if scores_path:
            with open(scores_path, 'w') as f:
                for csv_line in csv_scores:
                    f.write(csv_line + '\n')

        return self.preliminary_scores

    def _remove_keys_from_cache(self, key):
        if self.redis:
            for key in self.redis.scan_iter(match=key+"*"):
                self.redis.delete(key)
