# Shree KRISHNAya Namaha
# A Factory method that returns an Optimizer
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

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
