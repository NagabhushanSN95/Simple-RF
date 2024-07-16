# Shree KRISHNAya Namaha
# Abstract class for learning rate decayers
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

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
