# Shree KRISHNAya Namaha
# Depth MSE loss function between Main Coarse NeRF and Points Augmentation. Reprojection error (patch-wise) is employed
# to determine the more accurate depth estimate. For patches mapping outside the
# frame, larger depth is considered more accurate.
# the frame.
# Last Modified: 20/06/2024

import torch
from torch import Tensor
from pathlib import Path
import torch.nn.functional as F

from loss_functions.LossFunctionParent03 import LossFunctionParent
from utils import CommonUtils04 as CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class AugmentationsDepthLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_model_needed = 'coarse_model' in self.configs['model']
        self.fine_model_needed = 'fine_model' in self.configs['model']
        self.augmentations_needed = 'augmentations' in self.configs['model']
        self.px, self.py = self.loss_configs['patch_size']
        self.hpx, self.hpy = [x // 2 for x in self.loss_configs['patch_size']]
        self.rmse_threshold = self.loss_configs['rmse_threshold']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, model, return_loss_maps: bool = False):
        total_loss = torch.tensor(0).to(input_dict['target_rgb'])
        loss_maps = {}

        rays_o = output_dict['rays_o']
        rays_d = output_dict['rays_d']
        gt_images = input_dict['common_data']['images']
        intrinsics = output_dict['intrinsics'].detach()
        extrinsics = output_dict['extrinsics_all'].detach()  # (num_views, 4, 4)
        resolution = input_dict['common_data']['resolution']
        indices_mask_nerf = input_dict['indices_mask_nerf']
        pixel_ids = input_dict['pixel_id'].long()

        if self.coarse_model_needed and self.augmentations_needed:
            depth_coarse_main = output_dict['depth_coarse']

            for augmentation_configs in self.configs['model']['augmentations']:
                if 'coarse_model' in augmentation_configs:
                    aug_name = augmentation_configs['name']
                    depth_coarse_aug = output_dict[f'{aug_name}_depth_coarse']
                    loss_coarse = self.compute_depth_loss(depth_coarse_main, depth_coarse_aug, indices_mask_nerf,
                                                          rays_o, rays_d, extrinsics, gt_images, pixel_ids, intrinsics,
                                                          resolution, return_loss_maps)
                    total_loss += loss_coarse['loss_value']
                    if return_loss_maps:
                        coarse_loss_maps = loss_coarse['loss_maps']
                        loss_maps[f'{this_filename}_{aug_name}_coarse_main'] = coarse_loss_maps[f'{this_filename}_1']
                        loss_maps[f'{this_filename}_{aug_name}_coarse_augmented'] = coarse_loss_maps[f'{this_filename}_2']

        if self.fine_model_needed and self.augmentations_needed:
            depth_fine_main = output_dict['depth_fine']

            for augmentation_configs in self.configs['model']['augmentations']:
                if 'fine_model' in augmentation_configs:
                    aug_name = augmentation_configs['name']
                    depth_fine_aug = output_dict[f'{aug_name}_depth_fine']
                    loss_fine = self.compute_depth_loss(depth_fine_main, depth_fine_aug, indices_mask_nerf,
                                                          rays_o, rays_d, extrinsics, gt_images, pixel_ids, intrinsics,
                                                          resolution, return_loss_maps)
                    total_loss += loss_fine['loss_value']
                    if return_loss_maps:
                        fine_loss_maps = loss_fine['loss_maps']
                        loss_maps[f'{this_filename}_{aug_name}_fine_main'] = fine_loss_maps[f'{this_filename}_1']
                        loss_maps[f'{this_filename}_{aug_name}_fine_augmented'] = fine_loss_maps[f'{this_filename}_2']

        loss_dict = {
            'loss_value': total_loss,
        }

        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict

    def compute_depth_loss(self, depth1, depth2, indices_mask_nerf,
                           rays_o, rays_d, gt_poses, gt_images, pixel_ids, intrinsics, resolution,
                           return_loss_maps: bool) -> dict:
        total_loss = 0

        loss_nerf, nerf_mse_map1, nerf_mse_map2 = self.compute_loss_nerf(depth1, depth2, indices_mask_nerf,
                                                                         rays_o, rays_d, gt_poses, gt_images, pixel_ids,
                                                                         intrinsics, resolution)
        total_loss += loss_nerf

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                f'{this_filename}_1': nerf_mse_map1,
                f'{this_filename}_2': nerf_mse_map2
            }
        return loss_dict

    def compute_loss_nerf(self, depth1, depth2, indices_mask_nerf, rays_o, rays_d, gt_poses, gt_images,
                          pixel_ids, intrinsics, resolution) -> tuple[Tensor, Tensor, Tensor]:
        """
        Computes the loss for nerf samples (and not for sparse_depth or any other samples)

        Naming convention
        1, 2 -> refers to two different models
        a, b -> refers to source view and the other reprojection view

        :param depth1:
        :param depth2:
        :param indices_mask_nerf:
        :param rays_o:
        :param rays_d:
        :param gt_poses:
        :param gt_images:
        :param pixel_ids:
        :param intrinsics:
        :param resolution:
        :return:
        """
        h, w = resolution
        image_ids = pixel_ids[:, 0]

        rays_o = rays_o[indices_mask_nerf]
        rays_d = rays_d[indices_mask_nerf]
        depth1 = depth1[indices_mask_nerf]
        depth2 = depth2[indices_mask_nerf]

        gt_origins = gt_poses[:, :3, 3]
        distances = torch.sqrt(torch.sum(torch.square(gt_origins[image_ids].unsqueeze(1).repeat([1, gt_origins.shape[0], 1]) - gt_origins), dim=2))
        # Taking second smallest value as the smallest distance will always be with the same view at 0.0. Kth value
        # by default randomly returns one index if two distances are the same. Which works for our use-case.
        closest_image_ids = torch.kthvalue(distances, 2, dim=1)[1]

        image_ids_a = image_ids[indices_mask_nerf]
        pixel_ids_a = pixel_ids[indices_mask_nerf]
        image_ids_b = closest_image_ids[indices_mask_nerf]

        poses_b = gt_poses[image_ids_b]
        points1a = rays_o + rays_d * depth1.unsqueeze(-1)
        points2a = rays_o + rays_d * depth2.unsqueeze(-1)

        pos1b = CommonUtils.reproject(points1a.detach(), poses_b, intrinsics).round().long()
        pos2b = CommonUtils.reproject(points2a.detach(), poses_b, intrinsics).round().long()

        x_a, y_a = pixel_ids_a[:, 1], pixel_ids_a[:, 2]
        x1b, y1b = pos1b[:, 0], pos1b[:, 1]
        x2b, y2b = pos2b[:, 0], pos2b[:, 1]

        # Ignore reprojections that were set outside the image
        valid_mask_a = (x_a >= self.hpx) & (x_a < w - self.hpx) & (y_a >= self.hpy) & (y_a < h - self.hpy)
        valid_mask_1b = (x1b >= self.hpx) & (x1b < w - self.hpx) & (y1b >= self.hpy) & (y1b < h - self.hpy)
        valid_mask_2b = (x2b >= self.hpx) & (x2b < w - self.hpx) & (y2b >= self.hpy) & (y2b < h - self.hpy)

        x1b1, y1b1 = torch.clip(x1b, 0, w - 1).long(), torch.clip(y1b, 0, h - 1).long()
        x2b1, y2b1 = torch.clip(x2b, 0, w - 1).long(), torch.clip(y2b, 0, h - 1).long()
        patches_a = torch.zeros(image_ids_a.shape[0], self.py, self.px, gt_images.shape[3]).to(image_ids_a.device)  # (nr, py, px, 3)
        patches1b = torch.zeros(image_ids_b.shape[0], self.py, self.px, gt_images.shape[3]).to(image_ids_b.device)  # (nr, py, px, 3)
        patches2b = torch.zeros(image_ids_b.shape[0], self.py, self.px, gt_images.shape[3]).to(image_ids_b.device)  # (nr, py, px, 3)
        gt_images_padded = F.pad(gt_images, (0, 0, 0, self.hpy, 0, self.hpx), mode='constant', value=0)
        for i, y_offset in enumerate(range(-self.hpy, self.hpy + 1)):  # y_offset: [-2, -1, 0, 1, 2]
            for j, x_offset in enumerate(range(-self.hpx, self.hpx + 1)):
                patches_a[:, i, j, :] = gt_images_padded[image_ids_a, y_a + y_offset, x_a + x_offset]
                patches1b[:, i, j, :] = gt_images_padded[image_ids_b, y1b1 + y_offset, x1b1 + x_offset]
                patches2b[:, i, j, :] = gt_images_padded[image_ids_b, y2b1 + y_offset, x2b1 + x_offset]

        rmse1 = self.compute_patch_rmse(patches_a, patches1b)
        rmse2 = self.compute_patch_rmse(patches_a, patches2b)

        # mask1 is true wherever model1 is more accurate
        mask1 = ((rmse1 < rmse2) | (~valid_mask_2b)) & (rmse1 < self.rmse_threshold) & valid_mask_1b & valid_mask_a
        # mask2 is true wherever model2 is more accurate
        mask2 = ((rmse2 < rmse1) | (~valid_mask_1b)) & (rmse2 < self.rmse_threshold) & valid_mask_2b & valid_mask_a

        # Find the pixels where all depths are invalid
        both_invalid_mask = ~(torch.stack([valid_mask_1b, valid_mask_2b], dim=0).any(dim=0))  # (nr, )
        # For the pixels where all depths are invalid, set mask1=True if depth1 > depth2
        mask1 = mask1 | (both_invalid_mask & (depth1 > depth2))
        mask2 = mask2 | (both_invalid_mask & (depth2 > depth1))

        # depth_mse1 is loss on depth1; depth_mse2 is loss on depth2
        depth_mse1, depth_mse_map1 = self.compute_depth_mse(depth1, depth2.detach(), mask2)  # (nr, )
        depth_mse2, depth_mse_map2 = self.compute_depth_mse(depth2, depth1.detach(), mask1)
        loss = depth_mse1 + depth_mse2
        return loss, depth_mse_map1, depth_mse_map2

    @classmethod
    def compute_patch_rmse(cls, patch1: Tensor, patch2: Tensor) -> Tensor:
        """

        Args:
            patch1: (num_rays, patch_size, patch_size, 3)
            patch2: (num_rays, patch_size, patch_size, 3)

        Returns:
            rmse: (num_rays, )

        """
        rmse = torch.sqrt(torch.mean(torch.square(patch1 - patch2), dim=(1, 2, 3)))
        return rmse

    @classmethod
    def compute_depth_mse(cls, pred_depth: Tensor, gt_depth: Tensor, mask: Tensor = None) -> tuple[Tensor, Tensor]:
        """

        Args:
            pred_depth: (num_rays, )
            gt_depth: (num_rays, )
            mask: (num_rays, ); Loss is computed only where mask is True

        Returns:

        """
        zero_tensor = torch.tensor(0).to(pred_depth)
        if mask is not None:
            pred_depth[~mask] = 0
            gt_depth[~mask] = 0
        loss_map = torch.square(pred_depth - gt_depth)
        mse = torch.mean(loss_map) if pred_depth.numel() > 0 else zero_tensor
        return mse, loss_map
