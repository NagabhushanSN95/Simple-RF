# Shree KRISHNAya Namaha
# Utility functions for losses
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024


def update_loss_map_dict(old_dict: dict, new_dict: dict, suffix: str):
    for key in new_dict.keys():
        old_dict[f'{key}_{suffix}'] = new_dict[key]
    return old_dict
