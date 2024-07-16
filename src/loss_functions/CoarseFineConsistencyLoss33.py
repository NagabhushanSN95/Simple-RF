# Shree KRISHNAya Namaha
# MSE loss between coarse and fine depths for main model.
# Last Modified: 20/06/2024

from pathlib import Path

import torch

from loss_functions.LossFunctionParent03 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class CoarseFineConsistencyLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_model_needed = 'coarse_model' in self.configs['model']
        self.fine_model_needed = 'fine_model' in self.configs['model']
        self.ndc = self.configs['data_loader']['ndc']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = True):
        total_loss = torch.tensor(0).to(input_dict['target_rgb'])
        loss_maps = {}

        # Requires both coarse and fine models to compute the loss
        if not (self.coarse_model_needed and self.fine_model_needed):
            loss_dict = {
                'loss_value': total_loss,
            }
            return loss_dict

        if not self.ndc:
            depth_coarse = output_dict['depth_coarse']
            depth_fine = output_dict['depth_fine']
        else:
            depth_coarse = output_dict['depth_ndc_coarse']
            depth_fine = output_dict['depth_ndc_fine']

        loss_map = torch.square(depth_coarse - depth_fine)
        total_loss = torch.mean(loss_map)

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: loss_map,
            }
        return loss_dict
