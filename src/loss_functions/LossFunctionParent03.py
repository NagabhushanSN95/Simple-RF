# Shree KRISHNAya Namaha
# Abstract parent class
# Extended from LossFunctionParent03.py. Model passed to compute_loss function
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import abc


class LossFunctionParent:
    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = False):
        pass
