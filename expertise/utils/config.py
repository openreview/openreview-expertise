import json
import random
import os

class Config(object):
    def __init__(self,filename=None):
        self.learning_rate = 0.0001
        self.l2penalty = 10.0
        self.vocab_file = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None
        self.num_minibatches = 100000
        self.batch_size = 100
        self.dev_batch_size = 101
        self.max_string_len = 20
        self.embedding_dim = 100
        self.rnn_hidden_size = 100
        self.random_seed = 2524
        self.bidirectional = True
        self.dropout_rate = 0.2
        self.eval_every = 5000
        self.clip = 0.25
        self.num_layers = 3
        self.filter_count = 25
        self.filter_count2 = 25
        self.filter_count3 = 25
        self.codec = 'UTF-8'
        self.experiment_out_dir = "experiments"
        self.dataset_name = "dataset"
        self.model_name = "model"
        self.increasing = False
        self.use_cuda = False
        self.fasttext = None
        self.zip_file_name = ""
        self.random = random.Random(self.random_seed)

        if filename:
            with open(filename) as f:
                loaded_dict = json.load(f)
                self.__dict__.update(loaded_dict)

    def to_json(self):
        res = {}
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is str or type(self.__dict__[k]) is float or type(self.__dict__[k]) is int:
                res[k] = self.__dict__[k]
        return json.dumps(res)

    def save_config(self,exp_dir):
        with open(os.path.join(exp_dir,"config.json"), 'w') as fout:
            fout.write(self.to_json())
            fout.write("\n")

