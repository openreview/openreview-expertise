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
        if kwargs.get("config_file_path"):
            config_file_path = Path(kwargs["config_file_path"])
            with open(config_file_path) as file_handle:
                self.data = json.load(file_handle)
        elif kwargs.get("config_dict"):
            self.data = kwargs["config_dict"]

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def update(self, **kwargs):
        self.data = {**self.data, **kwargs}

    def save(self, outfile):
        with open(outfile, "w") as f:
            json.dump(self.data, f, indent=4, separators=(",", ": "))

    def update_from_file(self, file):
        config_path = Path(file).resolve()

        with open(config_path) as file_handle:
            data = json.load(file_handle)

        self.update(**data)
