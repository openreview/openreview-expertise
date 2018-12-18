import json
import random
import os
import pickle

from . import utils

class Config(object):
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.path_by_filename = {}
        self.experiment_dir = os.path.dirname(config_file_path)
        self.setup_dir = os.path.join(self.experiment_dir, 'setup')
        self.train_dir = os.path.join(self.experiment_dir, 'train')
        self.test_dir = os.path.join(self.experiment_dir, 'test')
        self.infer_dir = os.path.join(self.experiment_dir, 'infer')

        # self.learning_rate = 0.0001
        # self.l2penalty = 10.0
        # self.vocab_file = None
        # self.train_file = None
        # self.dev_file = None
        # self.test_file = None
        # self.num_minibatches = 100000
        # self.batch_size = 100
        # self.dev_batch_size = 101
        # self.max_string_len = 20
        # self.embedding_dim = 100
        # self.rnn_hidden_size = 100
        # self.random_seed = 2524
        # self.bidirectional = True
        # self.dropout_rate = 0.2
        # self.eval_every = 5000
        # self.clip = 0.25
        # self.num_layers = 3
        # self.filter_count = 25
        # self.filter_count2 = 25
        # self.filter_count3 = 25
        # self.codec = 'UTF-8'
        # self.experiment_out_dir = "experiments"
        # self.dataset_name = "dataset"
        # self.model_name = "model"
        # self.increasing = False
        # self.use_cuda = False
        # self.fasttext = None
        # self.zip_file_name = ""
        # self.random = random.Random(self.random_seed)

        if config_file_path:
            with open(config_file_path) as f:
                loaded_dict = json.load(f)
                self.__dict__.update(loaded_dict)

    def to_json(self):
        res = {}
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is str or type(self.__dict__[k]) is float or type(self.__dict__[k]) is int:
                res[k] = self.__dict__[k]
        return json.dumps(res)

    def save_config(self):
        print('saving config')
        with open(self.config_file_path, 'w') as f:
            json.dump(self.__dict__, f, sort_keys=True,
                indent=4, separators=(',', ': '))

    def setup_load(self, filename):
        filepath = os.path.join(self.setup_dir, filename)

        if filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filename.endswith('.jsonl'):
            data_list = []
            for data in utils.jsonl_reader(filepath):
                data_list.append(data)
            return data_list

    def __save(self, subdir, data, filename, delimiter):
        '''
        Given a piece of data and a filename, writes the data to this config's setup directory.

        if filename ends in .pkl, the data object is saved as a pickle.

        if filename ends in .tsv or .csv, the data is written as a csv/tsv.
            (in this case, data must be an iterable over lists, where each
            list contains the line items)
        '''


        filepath = os.path.join(subdir, filename)
        # todo: automatically make these directories

        if filename.endswith('.pkl'):
            dump_file = utils.dump_pkl
        elif filename.endswith('.jsonl'):
            dump_file = utils.dump_jsonl
        elif filename.endswith('.json'):
            dump_file = utils.dump_json
        elif filename.endswith('.tsv'):
            dump_file = utils.dump_csv
        else:
            raise AssertionError('filename must end with one of the following: .pkl, .json, .jsonl, .tsv')

        dump_file(filepath, data)
        self.path_by_filename[filename] = filepath

        # self.save_config()

        return self.path_by_filename[filename]

    def setup_save(self, data, filename, delimiter='\t'):
        return self.__save(self.setup_dir, data, filename, delimiter)

    def train_save(self, data, filename, delimiter='\t'):
        return self.__save(self.train_dir, data, filename, delimiter)

    def test_save(self, data, filename, delimiter='\t'):
        return self.__save(self.test_dir, data, filename, delimiter)

    def infer_save(self, data, filename, delimiter='\t'):
        return self.__save(self.infer_dir, data, filename, delimiter)

    def setup_path(self, filename):
        return os.path.join(self.setup_dir, filename)

    def test_path(self, filename):
        return os.path.join(self.test_dir, filename)
