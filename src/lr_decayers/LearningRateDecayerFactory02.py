# Shree KRISHNAya Namaha
# A Factory method that returns a Learning Rate Decayer
# Author: Nagabhushan S N
# Last Modified: 20/07/2022

import importlib.util
import inspect

from lr_decayers.LearningRateDecayerParent01 import LearningRateDecayerParent


def get_lr_decayer(configs: dict, optimizer_name: str) -> LearningRateDecayerParent:
    lr_decayer = None
    for optimizer_configs in configs['optimizers']:
        if optimizer_configs['name'] != optimizer_name:
            continue
        filename = optimizer_configs['lr_decayer_name']
        classname = filename[:-2]
        module = importlib.import_module(f'lr_decayers.{filename}')
        candidate_classes = inspect.getmembers(module, inspect.isclass)
        for candidate_class in candidate_classes:
            if candidate_class[0] == classname:
                lr_decayer = candidate_class[1](configs, optimizer_configs)
                break
    # if lr_decayer is None:
    #     raise RuntimeError(f'Unknown lr decayer: {optimizer_name}')
    return lr_decayer
