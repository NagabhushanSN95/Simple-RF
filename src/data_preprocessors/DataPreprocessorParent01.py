# Shree KRISHNAya Namaha
# A parent class for all dataloaders
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import abc
from pathlib import Path
from typing import Optional

import numpy

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DataPreprocessorParent:
    mode: str
    raw_data_dict: dict
    preprocessed_data_dict: Optional[dict]

    @abc.abstractmethod
    def get_model_configs(self):
        pass

    @abc.abstractmethod
    def get_next_batch(self, iter_num: int, image_num: int = None):
        pass

    @abc.abstractmethod
    def create_test_data(self, pose: numpy.ndarray, view_pose: Optional[numpy.ndarray], preprocess_pose: bool = True):
        pass

    @abc.abstractmethod
    def retrieve_inference_outputs(self, network_outputs: dict):
        pass
