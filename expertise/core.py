import pkgutil
from . import models


def model_importers():
    return {m: i for i, m, _ in pkgutil.iter_modules(models.__path__)}


def available_models():
    return [k for k in model_importers().keys()]


def load_model(module_name):
    return model_importers()[module_name].find_module(module_name).load_module()
