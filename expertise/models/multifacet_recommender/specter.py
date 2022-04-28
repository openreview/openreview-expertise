from allennlp.commands.predict import _PredictManager
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, import_submodules
from allennlp.data import DatasetReader
from allennlp.models import Archive
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, DEFAULT_PREDICTORS

from collections import defaultdict
import json
import os
import torch
from tqdm import tqdm
from typing import Optional
import redisai
import numpy as np

from expertise.service.server import redis_conn_pool

import logging
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.common.from_params').disabled = True
logging.getLogger('allennlp.common.registrable').setLevel(logging.WARNING)
logging.getLogger('allennlp.nn.initializers').disabled = True

import_submodules('specter')
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


class _PredictManagerCustom(_PredictManager):
    """
    Source: https://github.com/allenai/specter/blob/master/scripts/embed.py

    Extends the following functions from allennlp's _PredictManager class
    `run` function to print predict progress
    """

    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool,
                 store_redis: bool = False,
                 redis_con=None) -> None:
        super(_PredictManagerCustom, self).__init__(predictor, input_file, output_file, batch_size, print_to_console,
                                                    has_dataset_reader)
        self.total_size = int(sum([1 for _ in open(self._input_file)]) / self._batch_size)
        self._store_redis = store_redis
        if store_redis:
            assert redis_con is not None, "Can't store in Redis, No redis connection provided"
        self._redis_con = redis_con

    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        if has_reader:
            for batch in tqdm(lazy_groups_of(self._get_instance_data(), self._batch_size), total=self.total_size,
                              unit="batches"):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(index, result, str(model_input_instance))
                    index = index + 1
        else:
            for batch_json in tqdm(lazy_groups_of(self._get_json_data(), self._batch_size), total=self.total_size,
                                   unit="batches"):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(index, result, json.dumps(model_input_json))
                    index = index + 1

        if self._output_file is not None:
            self._output_file.close()

    def _maybe_print_to_console_and_file(self,
                                         index: int,
                                         prediction: str,
                                         model_input: str = None) -> None:
        prediction_json = json.loads(prediction)
        if self._print_to_console:
            if model_input is not None:
                print(f"input {index}: ", model_input)
            print("prediction: ", prediction)

        if self._output_file is not None:
            self._output_file.write(prediction)
        if self._store_redis:
            self._redis_con.tensorset(key=prediction_json['paper_id'], tensor=np.array(prediction_json['embedding']))


def predictor_from_archive(archive: Archive, predictor_name: str = None,
                           paper_features_path: str = None) -> 'Predictor':
    """
    Source: https://github.com/allenai/specter/blob/master/scripts/embed.py

    Extends allennlp.predictors.predictor.from_archive to allow processing multiprocess reader
    paper_features_path is passed to replace the correct one if the dataset_reader is multiprocess
    """

    # Duplicate the config so that the config inside the archive doesn't get consumed
    config = archive.config.duplicate()

    if not predictor_name:
        model_type = config.get("model").get("type")
        if not model_type in DEFAULT_PREDICTORS:
            raise ConfigurationError(f"No default predictor for model type {model_type}.\n" \
                                     f"Please specify a predictor explicitly.")
        predictor_name = DEFAULT_PREDICTORS[model_type]

    dataset_config = config["dataset_reader"].as_dict()
    if dataset_config['type'] == 'multiprocess':
        dataset_config = dataset_config['base_reader']
        if paper_features_path:
            dataset_config['paper_features_path'] = paper_features_path
        dataset_reader_params = Params(dataset_config)

    else:
        dataset_reader_params = config["dataset_reader"]

    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    model = archive.model
    model.eval()

    return Predictor.by_name(predictor_name)(model, dataset_reader)


class SpecterPredictor:
    def __init__(self, specter_dir, work_dir, average_score=False, max_score=True, batch_size=16, use_cuda=True,
                 sparse_value=None, use_redis=False):
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
            self.cuda_device = 0
        else:
            self.cuda_device = -1
        self.preliminary_scores = None
        self.sparse_value = sparse_value
        if not os.path.exists(self.work_dir) and not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        self.use_redis = use_redis
        if use_redis:
            self.redis = redisai.Client(connection_pool=redis_conn_pool)
        else:
            self.redis = None

    def set_archives_dataset(self, archives_dataset):
        self.pub_note_id_to_author_ids = defaultdict(list)
        self.pub_author_ids_to_note_id = defaultdict(list)
        self.pub_note_id_to_abstract = {}
        self.pub_note_id_to_title = {}
        output_dict = {}
        paper_ids_list = []
        for profile_id, publications in archives_dataset.items():
            for publication in publications:
                self.pub_note_id_to_author_ids[publication['id']].append(profile_id)
                self.pub_author_ids_to_note_id[profile_id].append(publication['id'])
                self.pub_note_id_to_title[publication['id']] = publication['content'].get('title', "")
                self.pub_note_id_to_abstract[publication['id']] = publication['content'].get('abstract', "")
                if self.redis is None or not self.redis.exists(publication['id']):
                    if publication['id'] in output_dict:
                        output_dict[publication['id']]["authors"].append(profile_id)
                    else:
                        paper_ids_list.append(publication['id'])
                        output_dict[publication['id']] = {"title": self.pub_note_id_to_title[publication['id']],
                                                          "abstract": self.pub_note_id_to_abstract[publication['id']],
                                                          "paper_id": publication["id"],
                                                          "authors": [profile_id]}
        with open(os.path.join(self.work_dir, "specter_reviewer_paper_data.json"), 'w') as f_out:
            json.dump(output_dict, f_out, indent=1)
        with open(os.path.join(self.work_dir, "specter_reviewer_paper_ids.txt"), 'w') as f_out:
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
        with open(os.path.join(self.work_dir, "specter_submission_paper_data.json"), 'w') as f_out:
            json.dump(output_dict, f_out, indent=1)
        with open(os.path.join(self.work_dir, "specter_submission_paper_ids.txt"), 'w') as f_out:
            f_out.write('\n'.join(paper_ids_list)+'\n')

    def embed_submissions(self, submissions_path=None):
        print('Embedding submissions...')
        metadata_file = os.path.join(self.work_dir, "specter_submission_paper_data.json")
        ids_file = os.path.join(self.work_dir, "specter_submission_paper_ids.txt")

        # Overrides default config in the saved specter archive
        overrides = json.dumps({'model': {'predict_mode': 'true', 'include_venue': 'false',
                                          'text_field_embedder': {
                                              'token_embedders': {
                                                  'bert': {
                                                      'pretrained_model': os.path.join(self.specter_dir,
                                                                                       "data/scibert_scivocab_uncased/scibert.tar.gz")
                                                  }
                                              }
                                          }
                                          },
                                "train_data_path": os.path.join(self.specter_dir, "data/train.csv"),
                                "validation_data_path": os.path.join(self.specter_dir, "data/val.csv"),
                                "test_data_path": os.path.join(self.specter_dir, "data/test.csv"),
                                'dataset_reader': {'type': 'specter_data_reader', 'predict_mode': 'true',
                                                   'paper_features_path': metadata_file,
                                                   'included_text_fields': 'abstract title',
                                                   'cache_path': os.path.join(self.specter_dir,
                                                                              'data/dataset-instance-cache/'),
                                                   'data_file': os.path.join(self.specter_dir, 'data/train.json'),
                                                   'token_indexers': {
                                                       'bert': {
                                                           "pretrained_model": os.path.join(self.specter_dir,
                                                                                            "data/scibert_scivocab_uncased/vocab.txt")
                                                       }
                                                   }
                                                   },
                                'vocabulary': {'directory_path': self.vocab_dir}
                                })

        archive = load_archive(self.model_archive_file,
                               weights_file=None,
                               cuda_device=self.cuda_device,
                               overrides=overrides)
        predictor = predictor_from_archive(archive, self.predictor_name, metadata_file)

        manager = _PredictManagerCustom(predictor,
                                        ids_file,
                                        submissions_path,
                                        self.batch_size,
                                        False,
                                        False)
        manager.run()

    def embed_publications(self, publications_path=None):
        assert publications_path or self.redis, "Either publications_path must be given or use_redis must be set to true"
        print('Embedding publications...')
        metadata_file = os.path.join(self.work_dir, "specter_reviewer_paper_data.json")
        ids_file = os.path.join(self.work_dir, "specter_reviewer_paper_ids.txt")

        # Overrides default config in the saved specter archive
        overrides = json.dumps({'model': {'predict_mode': 'true', 'include_venue': 'false',
                                          'text_field_embedder': {
                                              'token_embedders': {
                                                  'bert': {
                                                      'pretrained_model': os.path.join(self.specter_dir, "data/scibert_scivocab_uncased/scibert.tar.gz")
                                                  }
                                              }
                                          }
                                          },
                                "train_data_path": os.path.join(self.specter_dir, "data/train.csv"),
                                "validation_data_path": os.path.join(self.specter_dir, "data/val.csv"),
                                "test_data_path": os.path.join(self.specter_dir, "data/test.csv"),
                                'dataset_reader': {'type': 'specter_data_reader', 'predict_mode': 'true',
                                                   'paper_features_path': metadata_file,
                                                   'included_text_fields': 'abstract title',
                                                   'cache_path': os.path.join(self.specter_dir,
                                                                              'data/dataset-instance-cache/'),
                                                   'data_file': os.path.join(self.specter_dir, 'data/train.json'),
                                                   'token_indexers': {
                                                       'bert': {
                                                           "pretrained_model": os.path.join(self.specter_dir,
                                                                                            "data/scibert_scivocab_uncased/vocab.txt")
                                                       }
                                                   }
                                                   },
                                'vocabulary': {'directory_path': self.vocab_dir}
                                })
        archive = load_archive(self.model_archive_file,
                               weights_file=None,
                               cuda_device=self.cuda_device,
                               overrides=overrides)
        predictor = predictor_from_archive(archive, self.predictor_name, metadata_file)
        redis_client = self.redis.client() if self.use_redis else None
        manager = _PredictManagerCustom(predictor,
                                        ids_file,
                                        publications_path,
                                        self.batch_size,
                                        False,
                                        False,
                                        store_redis=self.use_redis,
                                        redis_con=redis_client)
        manager.run()

    def all_scores(self, publications_path=None, submissions_path=None, scores_path=None):
        def load_emb_file(emb_file):
            paper_emb_size_default = 768
            id_list = []
            emb_list = []
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
            emb_tensor = torch.tensor(emb_list, device=torch.device('cpu'))
            emb_tensor = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 0.000000000001)
            print(len(bad_id_set))
            return emb_tensor, id_list, bad_id_set

        def load_from_redis():
            paper_emb_size_default = 768
            id_list = self.pub_note_id_to_title.keys()
            emb_list = []
            bad_id_set = set()
            for paper_id in id_list:
                try:
                    paper_emb = self.redis.tensorget(key=paper_id, as_numpy_mutable=True)
                    assert len(paper_emb) == paper_emb_size_default
                    emb_list.append(paper_emb)
                except Exception as e:
                    bad_id_set.add(paper_id)

            emb_tensor = torch.tensor(emb_list, device=torch.device('cpu'))
            emb_tensor = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 0.000000000001)
            if bad_id_set:
                print(f"No Embedding found for {len(bad_id_set)} Papers: ")
                print(bad_id_set)
            return emb_tensor, id_list, bad_id_set

        print('Loading cached publications...')
        if self.use_redis:
            paper_emb_train, train_id_list, train_bad_id_set = load_from_redis()
        else:
            with open(publications_path) as f_in:
                paper_emb_train, train_id_list, train_bad_id_set = load_emb_file(f_in)
        paper_num_train = len(train_id_list)

        paper_id2train_idx = {}
        for idx, paper_id in enumerate(train_id_list):
            paper_id2train_idx[paper_id] = idx

        with open(submissions_path) as f_in:
            print('Loading cached submissions...')
            paper_emb_test, test_id_list, test_bad_id_set = load_emb_file(f_in)
            paper_num_test = len(test_id_list)

        print('Computing all scores...')
        p2p_aff = torch.empty((paper_num_test, paper_num_train), device=torch.device('cpu'))
        for i in range(paper_num_test):
            p2p_aff[i, :] = torch.sum(paper_emb_test[i, :].unsqueeze(dim=0) * paper_emb_train, dim=1)

        csv_scores = []
        self.preliminary_scores = []
        for reviewer_id, train_note_id_list in self.pub_author_ids_to_note_id.items():
            if len(train_note_id_list) == 0:
                continue
            train_paper_idx = []
            for paper_id in train_note_id_list:
                if paper_id not in train_bad_id_set:
                    train_paper_idx.append(paper_id2train_idx[paper_id])
            train_paper_aff_j = p2p_aff[:, train_paper_idx]

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
