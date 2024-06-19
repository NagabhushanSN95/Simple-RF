# Shree KRISHNAya Namaha
# Total Variation loss function
# Extended from TotalVariationLoss02.py for augmented models.
# Author: Nagabhushan S N
# Last Modified: 19/10/2023

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent03 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class TotalVariationLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.augmentations_needed = 'augmentations' in self.configs['model']
        self.lr_decay_ratio, self.lr_decay_iters = self.get_lr_decay_ratio_and_iters(configs)
        return
    
    def get_lr_decay_ratio_and_iters(self, configs):
        lr_decay_ratio, lr_decay_iters = None, None
        for optimizer_configs in configs['optimizers']:
            if optimizer_configs['name'] == 'optimizer_main':
                lr_decay_ratio = optimizer_configs['lr_decay_ratio']
                if optimizer_configs['lr_decay_iters'] is not None:
                    lr_decay_iters = optimizer_configs['lr_decay_iters']
                else:
                    lr_decay_iters = self.configs['num_iterations']
        return lr_decay_ratio, lr_decay_iters

    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = False):
        total_loss = torch.tensor(0, dtype=torch.float32).to(input_dict['target_rgb'].device)
        loss_maps = {}

        iter_weight = self.get_iter_weight(input_dict['iter_num'])

        if self.augmentations_needed:
            for augmentation_configs in self.configs['model']['augmentations']:
                aug_name = augmentation_configs['name']
                aug_model_list = list(filter(lambda x: x['name'] == aug_name, model.module.augmented_models))
                if len(aug_model_list) != 1:
                    raise RuntimeError

                if 'coarse_model' in augmentation_configs:
                    components_density, components_color = self.get_components(aug_model_list[0]['coarse_model'])
        
                    tv_density_coarse = self.compute_tv_loss(components_density, iter_weight, return_loss_maps)
                    total_loss += (tv_density_coarse['loss_value'] * self.loss_configs['weight_density'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, tv_density_coarse['loss_maps'], suffix=f'{aug_name}_coarse_density')
        
                    tv_color_coarse = self.compute_tv_loss(components_color, iter_weight, return_loss_maps)
                    total_loss += (tv_color_coarse['loss_value'] * self.loss_configs['weight_color'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, tv_color_coarse['loss_maps'], suffix=f'{aug_name}_coarse_color')

                if 'fine_model' in augmentation_configs:
                    components_density, components_color = self.get_components(aug_model_list[0]['fine_model'])
        
                    tv_density_fine = self.compute_tv_loss(components_density, iter_weight, return_loss_maps)
                    total_loss += (tv_density_fine['loss_value'] * self.loss_configs['weight_density'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, tv_density_fine['loss_maps'], suffix=f'{aug_name}_fine_density')
        
                    tv_color_fine = self.compute_tv_loss(components_color, iter_weight, return_loss_maps)
                    total_loss += (tv_color_fine['loss_value'] * self.loss_configs['weight_color'])
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, tv_color_fine['loss_maps'], suffix=f'{aug_name}_fine_color')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    @staticmethod
    def get_components(model):
        if model.__class__.__name__ == 'CpDecomposedTensor':
            components_density = model.vectors_density
            components_color = model.vectors_color
        elif model.__class__.__name__ == 'VmDecomposedTensor':
            components_density = model.matrices_density
            components_color = model.matrices_color
        else:
            raise RuntimeError
        return components_density, components_color

    @staticmethod
    def compute_tv_loss(components, iter_weight, return_loss_maps: bool):
        tv_loss = 0
        for component in components:
            squared_diff_h = torch.pow((component[:, :, 1:, :] - component[:, :, :-1, :]), 2)
            squared_diff_w = torch.pow((component[:, :, :, 1:] - component[:, :, :, :-1]), 2)
            numel_h = max(torch.numel(squared_diff_h), 1)  # To avoid division by 0 for vectors
            numel_w = max(torch.numel(squared_diff_w), 1)
            tv_h = squared_diff_h.sum() / numel_h
            tv_w = squared_diff_w.sum() / numel_w
            matrix_tv_loss = 2*(tv_h + tv_w)
            tv_loss += (matrix_tv_loss * iter_weight)
        loss_dict = {
            'loss_value': tv_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                
            }
        return loss_dict
    
    def get_iter_weight(self, iter_num):
        weight = (self.lr_decay_ratio ** ((iter_num + 1) / self.lr_decay_iters))
        return weight
