# Shree KRISHNAya Namaha
# Abstract class
# Extended from LearningRateDecayerParent01.py. Instead of returning updated learning rate, it returns the decay factor,
# which has to then be multiplied with the existing learning rate. This allows using a single decayers when there are
# different initial learning rates for different parameters, but the same decay.
# Author: Nagabhushan S N
# Last Modified: 12/07/2023

import abc

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class LearningRateDecayerParent:
    @abc.abstractmethod
    def get_updated_learning_rate(self, iter_num, lr_initial):
        pass

    @abc.abstractmethod
    def get_learning_rate_scale(self, iter_num):
        pass
