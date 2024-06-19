# Shree KRISHNAya Namaha
# Loss on sigma to consolidate weights/mass in augmented models. This is based on the entropy regularization in InfoNeRF paper.
# Modified MassConcentrationLoss04.py by binning weights before computing entropy.
# Supports NeRF16 and TensoRF07
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil, Harsha Mupparaju
# Last Modified: 20/06/2024

from pathlib import Path

from matplotlib import pyplot

import torch
import torch.nn.functional as F

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent03 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MassConcentrationLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.augmentations_needed = 'augmentations' in self.configs['model']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = False):
        total_loss = torch.tensor(0, dtype=torch.float32).to(input_dict['target_rgb'].device)
        loss_maps = {}

        indices_mask = input_dict['indices_mask_nerf']

        if self.augmentations_needed:
            for augmentation_configs in self.configs['model']['augmentations']:
                aug_name = augmentation_configs['name']
                aug_model_list = list(filter(lambda x: x['name'] == aug_name, model.module.augmented_models))
                if len(aug_model_list) != 1:
                    raise RuntimeError

                if 'coarse_model' in augmentation_configs:
                    weights_coarse = output_dict[f'{aug_name}_weights_coarse']
                    loss_coarse = self.compute_entropy_loss(weights_coarse, indices_mask, return_loss_maps)
                    total_loss += loss_coarse['loss_value']
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix=f'{aug_name}_coarse')

                if 'fine_model' in augmentation_configs:
                    weights_fine = output_dict[f'{aug_name}_weights_fine']
                    loss_fine = self.compute_entropy_loss(weights_fine, indices_mask, return_loss_maps)
                    total_loss += loss_fine['loss_value']
                    if return_loss_maps:
                        loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix=f'{aug_name}_fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    def compute_entropy_loss(self, weights: torch.Tensor, indices_mask: torch.Tensor, return_loss_maps: bool):
        """
        @param: weights (num_rays, num_samples)
        """
        weights = weights[indices_mask]
        elements_per_bin = weights.shape[1] // self.loss_configs['num_bins']
        remaining_elements = weights.shape[1] % self.loss_configs['num_bins']
        bigger_bin_idx = torch.linspace(0, self.loss_configs['num_bins']-1, remaining_elements, device=weights.device).long()
        bin_sizes = [(elements_per_bin + 1) if i in bigger_bin_idx else elements_per_bin for i in range(self.loss_configs['num_bins'])]
        reshaped_weights = torch.split(weights, bin_sizes, dim=1)  # [(num_rays, num_samples_per_bin)]*num_bins
        binned_weights = torch.stack([torch.sum(bin_weights, dim=1) for bin_weights in reshaped_weights], dim=1)  # (num_rays, num_bins)
        binned_ray_loss = torch.sum(torch.special.entr(binned_weights + 1e-3), dim=1)
        mean_loss = torch.mean(binned_ray_loss) if binned_ray_loss.numel() > 0 else 0
        loss_dict = {
            'loss_value': mean_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: binned_ray_loss
            }
        return loss_dict
