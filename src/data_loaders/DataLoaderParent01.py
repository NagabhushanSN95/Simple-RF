# Shree KRISHNAya Namaha
# A parent class for all dataloaders
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024
import abc

from pathlib import Path

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DataLoaderParent:
    @abc.abstractmethod
    def load_data(self):
        pass
