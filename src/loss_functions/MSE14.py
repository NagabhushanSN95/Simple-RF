# Shree KRISHNAya Namaha
# MSE loss function for main and augmented models. Supports regularizing specified models only. Supports NeRF16 and
# TensoRF07.
# Combining MSE12.py and MSE13.py. A config added that specifies which models to regularize.
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent03 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_model_needed = 'coarse_model' in self.configs['model']
        self.fine_model_needed = 'fine_model' in self.configs['model']
        self.augmentations_needed = 'augmentations' in self.configs['model']
        self.models_to_regularize = self.loss_configs['models_to_regularize']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = False):
        total_loss = torch.tensor(0, dtype=torch.float32).to(input_dict['target_rgb'].device)
        loss_maps = {}

        indices_mask = input_dict['indices_mask_nerf']
        target_rgb = input_dict['target_rgb']

        if self.coarse_model_needed and ('main_coarse' in self.models_to_regularize):
            pred_rgb_coarse = output_dict['rgb_coarse']
            loss_coarse = self.compute_mse(pred_rgb_coarse, target_rgb, indices_mask, return_loss_maps)
            total_loss += (loss_coarse['loss_value'] * self.models_to_regularize['main_coarse'])
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')

        if self.fine_model_needed and ('main_fine' in self.models_to_regularize):
            pred_rgb_fine = output_dict['rgb_fine']
            loss_fine = self.compute_mse(pred_rgb_fine, target_rgb, indices_mask, return_loss_maps)
            total_loss += (loss_fine['loss_value'] * self.models_to_regularize['main_fine'])
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix='fine')

        if self.augmentations_needed:
            for augmentation_configs in self.configs['model']['augmentations']:
                aug_name = augmentation_configs['name']

                if ('coarse_model' in augmentation_configs) and (f'{aug_name}_coarse' in self.models_to_regularize):
                    aug_pred_rgb_coarse = output_dict[f'{aug_name}_rgb_coarse']
                    loss_aug_coarse = self.compute_mse(aug_pred_rgb_coarse, target_rgb, indices_mask, return_loss_maps)
                    total_loss += (loss_aug_coarse['loss_value'] * self.models_to_regularize[f'{aug_name}_coarse'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_aug_coarse['loss_maps'],
                                                                     suffix=f'{aug_name}_coarse')

                if ('fine_model' in augmentation_configs) and (f'{aug_name}_fine' in self.models_to_regularize):
                    aug_pred_rgb_fine = output_dict[f'{aug_name}_rgb_fine']
                    loss_aug_fine = self.compute_mse(aug_pred_rgb_fine, target_rgb, indices_mask, return_loss_maps)
                    total_loss += (loss_aug_fine['loss_value'] * self.models_to_regularize[f'{aug_name}_fine'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_aug_fine['loss_maps'],
                                                                     suffix=f'{aug_name}_fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def compute_mse(pred_value, true_value, indices_mask, return_loss_maps: bool):
        pred_value = pred_value[indices_mask]
        true_value = true_value[indices_mask]
        error = pred_value - true_value
        mse = torch.mean(torch.square(error), dim=1)
        mean_mse = torch.mean(mse) if pred_value.numel() > 0 else 0
        loss_dict = {
            'loss_value': mean_mse,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: mse
            }
        return loss_dict
