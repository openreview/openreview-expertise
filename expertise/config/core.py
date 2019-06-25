from collections import OrderedDict
import json
import random
import os
import pickle
import pkgutil
import expertise

class ModelConfig(object):
    def __init__(self, **kwargs):
        self._config = {}

        # valid_model_names = expertise.available_models()
        # if not 'model' in kwargs:
        #     raise AttributeError(
        #         f'ModelConfig requires a model. Select from {valid_model_names}')

        # model = kwargs['model']

        # if model not in valid_model_names:
        #     raise ValueError(
        #         f'"model" attribute must be one of {valid_model_names}')

        # model_default_file = os.path.join(
        #     expertise.model_importers()[model].path, model, f'{model}_default.json')

        # with open(model_default_file) as f:
        #     model_default_config = json.load(f, object_pairs_hook=OrderedDict)

        # self._config = model_default_config

        self.update(**kwargs)

    def __repr__(self):
        return json.dumps(self._config, indent=4)

    def update(self, **kwargs):
        self._config = OrderedDict({**self._config, **kwargs})

        for k, v in self._config.items():
            setattr(self, k, v)

    def save(self, outfile):
        with open(outfile, 'w') as f:
            json.dump(self._config, f, indent=4, separators=(',', ': '))

    def update_from_file(self, file):
        config_path = os.path.abspath(file)

        with open(config_path) as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        self.update(**data)
