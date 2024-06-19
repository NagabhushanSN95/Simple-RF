# Shree KRISHNAya Namaha
# A Factory method that returns an Optimizer
# Extended from OptimizerFactory01.py. If lr_initial is not available, it is not provided
# Author: Nagabhushan S N
# Last Modified: 22/06/2023

import torch


def get_optimizer(configs: dict, optimizer_name: str, model_params: list):
    optimizer = None
    for optimizer_configs in configs['optimizers']:
        if optimizer_configs['name'] != optimizer_name:
            continue
        if 'lr_initial' in optimizer_configs:
            optimizer = torch.optim.Adam(model_params,
                                         lr=optimizer_configs['lr_initial'],
                                         betas=(optimizer_configs['beta1'], optimizer_configs['beta2']))
        else:
            optimizer = torch.optim.Adam(model_params,
                                         betas=(optimizer_configs['beta1'], optimizer_configs['beta2']))
    return optimizer
