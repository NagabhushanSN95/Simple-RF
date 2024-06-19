# Shree KRISHNAya Namaha
# Learning Rate Decayer
# Extended from NeRFLearningRateDecayer02.py. Learing Rate Decayer used in TensoRF.
# Author: Nagabhushan S N
# Last Modified: 12/07/2023

from pathlib import Path

from lr_decayers.LearningRateDecayerParent02 import LearningRateDecayerParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class TensoRFLearningRateDecayer(LearningRateDecayerParent):
    def __init__(self, configs: dict, optimizer_configs: dict):
        self.configs = configs
        self.optimizer_configs = optimizer_configs
        self.lr_decay_ratio = self.optimizer_configs['lr_decay_ratio']
        self.curr_lr_to_init_lr_ratio = 1
        if self.optimizer_configs['lr_decay_iters'] is not None:
            self.lr_decay_iters = self.optimizer_configs['lr_decay_iters']
        else:
            self.lr_decay_iters = self.configs['num_iterations']
        return

    def get_updated_learning_rate(self, iter_num, lr_initial):
        self.curr_lr_to_init_lr_ratio *= self.get_learning_rate_scale(iter_num)
        new_lr = lr_initial * self.curr_lr_to_init_lr_ratio
        new_lr1 = lr_initial * (self.lr_decay_ratio ** (iter_num / self.lr_decay_iters))
        return new_lr

    def get_learning_rate_scale(self, iter_num):
        lr_scale = self.lr_decay_ratio ** (1 / self.lr_decay_iters)
        return lr_scale
