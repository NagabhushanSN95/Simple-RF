# Shree KRISHNAya Namaha
# Learning Rate Decayer
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

from pathlib import Path

from lr_decayers.LearningRateDecayerParent02 import LearningRateDecayerParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class NeRFLearningRateDecayer(LearningRateDecayerParent):
    def __init__(self, configs: dict, optimizer_configs: dict):
        self.configs = configs
        self.optimizer_configs = optimizer_configs
        self.decay_rate = 0.1
        self.decay_steps = self.optimizer_configs['lr_decay'] * 1000
        return

    def get_updated_learning_rate(self, iter_num, lr_initial):
        new_lr = lr_initial * (self.decay_rate ** (iter_num / self.decay_steps))
        return new_lr

    def get_learning_rate_scale(self, iter_num):
        lr_scale = self.decay_rate ** (1 / self.decay_steps)
        return lr_scale
