# Shree KRISHNAya Namaha
# A Factory method that returns a Model
# Author: Nagabhushan S N
# Last Modified: 15/09/2023

import importlib.util
import inspect


def get_model(configs: dict, **kwargs):
    filename = configs['model']['name']
    classname = f'{filename[:-2]}'
    model = None
    module = importlib.import_module(f'models.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs, **kwargs)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
