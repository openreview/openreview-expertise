from collections import OrderedDict
import json
import random
from pathlib import Path
import pickle
import pkgutil
import expertise

class ModelConfig(object):
    def __init__(self, **kwargs):
        if kwargs.get('config_file_path'):
            config_file_path = Path(kwargs['config_file_path'])
            with open(config_file_path) as file_handle:
                self.config = json.load(file_handle)
        elif kwargs.get('config_dict'):
            self.config = kwargs['config_dict']

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

        # self.update(**kwargs)

    def __repr__(self):
        return json.dumps(self.config, indent=4)

    def get(self):
        return self.config

    def update(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def save(self, outfile):
        with open(outfile, 'w') as f:
            json.dump(self.config, f, indent=4, separators=(',', ': '))

    def update_from_file(self, file):
        config_path = Path(file).resolve()

        with open(config_path) as file_handle:
            data = json.load(file_handle)

        self.update(**data)
