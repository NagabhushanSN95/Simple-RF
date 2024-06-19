# Shree KRISHNAya Namaha
# Sparse Depth MSE loss function for main and augmented models. Supports regularizing specified models only. Supports
# NeRF16 and TensoRF07.
# Combining SparseDepthMSE12.py and SparseDepthMSE13.py. A config added that specifies which models to regularize.
# Author: Nagabhushan S N
# Last Modified: 11/11/2023

from pathlib import Path

import torch
from loss_functions import LossUtils01
from loss_functions.LossFunctionParent03 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class SparseDepthMSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_model_needed = 'coarse_model' in self.configs['model']
        self.fine_model_needed = 'fine_model' in self.configs['model']
        self.augmentations_needed = 'augmentations' in self.configs['model']
        self.models_to_regularize = self.loss_configs['models_to_regularize']
        self.ndc = self.configs['data_loader']['ndc']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = False):
        total_loss = torch.tensor(0, dtype=torch.float32).to(input_dict['target_rgb'].device)
        loss_maps = {}

        # Sparse depth loss is computed only for batches - not for full images
        if 'indices_mask_sparse_depth' not in input_dict:
            return {'loss_value': total_loss}

        gt_depth = input_dict['sparse_depth_values'][:, 0]
        indices_mask = input_dict['indices_mask_sparse_depth']

        if self.coarse_model_needed and ('main_coarse' in self.models_to_regularize):
            pred_depth_coarse = output_dict['depth_coarse']
            loss_coarse = self.compute_depth_loss(pred_depth_coarse, gt_depth, indices_mask, return_loss_maps)
            total_loss += (loss_coarse['loss_value'] * self.models_to_regularize['main_coarse'])
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')

        if self.fine_model_needed and ('main_fine' in self.models_to_regularize):
            pred_depth_fine = output_dict['depth_fine']
            loss_fine = self.compute_depth_loss(pred_depth_fine, gt_depth, indices_mask, return_loss_maps)
            total_loss += (loss_fine['loss_value'] * self.models_to_regularize['main_fine'])
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix='fine')

        if self.augmentations_needed:
            for augmentation_configs in self.configs['model']['augmentations']:
                aug_name = augmentation_configs['name']

                if ('coarse_model' in augmentation_configs) and (f'{aug_name}_coarse' in self.models_to_regularize):
                    aug_pred_depth_coarse = output_dict[f'{aug_name}_depth_coarse']
                    loss_aug_coarse = self.compute_depth_loss(aug_pred_depth_coarse, gt_depth, indices_mask,
                                                              return_loss_maps)
                    total_loss += (loss_aug_coarse['loss_value'] * self.models_to_regularize[f'{aug_name}_coarse'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_aug_coarse['loss_maps'],
                                                                     suffix=f'{aug_name}_coarse')

                if ('fine_model' in augmentation_configs) and (f'{aug_name}_fine' in self.models_to_regularize):
                    aug_pred_depth_fine = output_dict[f'{aug_name}_depth_fine']
                    loss_aug_fine = self.compute_depth_loss(aug_pred_depth_fine, gt_depth, indices_mask,
                                                            return_loss_maps)
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
    def compute_depth_loss(pred_depth, true_depth, indices_mask, return_loss_maps: bool):
        pred_depth = pred_depth[indices_mask]
        true_depth = true_depth[indices_mask]
        error = pred_depth - true_depth
        squared_error = torch.square(error)
        depth_loss = torch.mean(squared_error) if pred_depth.numel() > 0 else 0
        loss_dict = {
            'loss_value': depth_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                # No loss maps
            }
        return loss_dict
