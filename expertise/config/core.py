from collections import UserDict
import json
import random
from pathlib import Path
import pickle
import pkgutil
import expertise

class ModelConfig(UserDict):
    def __init__(self, **kwargs):
        super(UserDict, self).__init__()
        if kwargs.get('config_file_path'):
            config_file_path = Path(kwargs['config_file_path'])
            with open(config_file_path) as file_handle:
                self.data = json.load(file_handle)
        elif kwargs.get('config_dict'):
            self.data = kwargs['config_dict']

        ModelConfig.validate_weight_specification(self.data)

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def update(self, **kwargs):
        self.data = {**self.data, **kwargs}

    def save(self, outfile):
        with open(outfile, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(',', ': '))

    def update_from_file(self, file):
        config_path = Path(file).resolve()

        with open(config_path) as file_handle:
            data = json.load(file_handle)

        self.update(**data)

    def validate_weight_specification(config):
        # Validate weight specification
        dataset_params = config.get('dataset', {})
        weight_specification = dataset_params.get('weight_specification', None)
        if weight_specification:
            if not isinstance(weight_specification, list):
                raise ValueError('weight_specification must be a list')
            for venue_spec in weight_specification:
                if not isinstance(venue_spec, dict):
                    raise ValueError('Objects in weight_specification must be dictionaries')

                # Count how many matching keys are present
                matching_keys = ['prefix', 'value', 'articleSubmittedToOpenReview']
                present_keys = [key for key in matching_keys if key in venue_spec]
                if len(present_keys) > 1:
                    raise KeyError(f'Objects in weight_specification must have exactly one of [prefix, value, articleSubmittedToOpenReview]. Found: {present_keys}')

                if 'prefix' not in venue_spec and 'value' not in venue_spec and 'articleSubmittedToOpenReview' not in venue_spec:
                    raise KeyError('Objects in weight_specification must have a prefix, value, or articleSubmittedToOpenReview key')
                if 'weight' not in venue_spec:
                    raise KeyError('Objects in weight_specification must have a weight key')
                
                if 'articleSubmittedToOpenReview' in venue_spec and not isinstance(venue_spec['articleSubmittedToOpenReview'], bool):
                    raise KeyError('The articleSubmittedToOpenReview key can only have a boolean value')

                # weight must be an integer or float
                if not isinstance(venue_spec['weight'], int) and not isinstance(venue_spec['weight'], float):
                    raise ValueError('weight must be an integer or float greater than 0')
                else:
                    if venue_spec['weight'] <= 0:
                        raise ValueError('weight must be an integer or float greater than 0')
