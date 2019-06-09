from collections import OrderedDict
import json
import random
import os
import pickle
import pkgutil
from expertise import models

def model_importers():
    return {m: i for i, m, _ in pkgutil.iter_modules(
        models.__path__)}

def available_models():
    return [k for k in model_importers().keys()]

class ModelConfig(object):
    def __init__(self, **kwargs):
        self._config = kwargs

        valid_model_names = available_models()
        if not 'model' in self._config:
            raise AttributeError(
                f'ModelConfig requires a model. Select from {valid_model_names}')

        model = self._config['model']

        if model not in valid_model_names:
            raise ValueError(
                f'"model" attribute must be one of {valid_model_names}')

        model_default_file = os.path.join(
            model_importers()[model].path, model, f'{model}_default.json')

        with open(model_default_file) as f:
            model_default_config = json.load(f, object_pairs_hook=OrderedDict)

        self._config = OrderedDict({**model_default_config, **self._config})

        for k, v in self._config.items():
            setattr(self, k, v)

    def __repr__(self):
        return json.dumps(self._config, indent=4)

    def save(self, outfile):
        with open(outfile, 'w') as f:
            json.dump(self._config, f, indent=4, separators=(',', ': '))


