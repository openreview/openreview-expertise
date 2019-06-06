import json
import random
import os
import pickle

class Config(object):
    def __init__(self, config):
        self._config = config # set it to conf
        for k, v in self._config.items():
            setattr(self, k, v)

    def save_config(self, outfile):
        with open(outfile, 'w') as f:
            json.dump(self._config, f, indent=4, separators=(',', ': '))
