# Shree KRISHNAYa Namaha
# Preprocesses data for NeRF, MipNeRF, Colmap sparse depth, dense depth, visibility prior. Supports TensoRF and
# camera params optimization.
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

from typing import Optional, Union, List

import numpy
import skimage.io
import skimage.transform
import torch
from tqdm import tqdm

from data_preprocessors.DataPreprocessorParent01 import DataPreprocessorParent
from utils import CommonUtils04 as CommonUtils


class DataPreprocessor(DataPreprocessorParent):
    def __init__(self, configs: dict, mode: str, raw_data_dict: Optional[dict] = None, model_configs: Optional[dict] = None):
        self.configs = configs
        self.mode = mode.lower()
        self.bd_factor = self.configs['data_loader']['bd_factor']
        self.use_batching = self.configs['data_loader']['batching']
        self.ndc = self.configs['data_loader']['ndc']
        self.downsampling_factor = self.configs['data_loader']['downsampling_factor']
        self.nerf_synthetic = self.configs['database'] == 'NeRF_Synthetic'  # TODO: Remove this
        self.device = CommonUtils.get_device(self.configs['device'])

        self.raw_data_dict = raw_data_dict
        self.preprocessed_data_dict = None
        self.model_configs = model_configs
        self.i_batch = 0
        self.num_rays = self.configs['data_loader']['num_rays']

        self.mip_nerf_model = 'MipNeRF' in self.configs['model']['name']
        self.tensorf_model = 'TensoRF' in self.configs['model']['name']

        self.sparse_depth_needed = 'sparse_depth' in self.configs['data_loader']
        self.dense_depth_needed = 'dense_depth' in self.configs['data_loader']
        self.visibility_prior_needed = 'visibility_prior' in self.configs['data_loader']
        self.pts_augmentation_needed = 'points_augmentation' in self.configs['model']
        self.views_augmentation_needed = 'views_augmentation' in self.configs['model']

        if self.sparse_depth_needed:
            self.i_batch_sparse_depth = 0
            self.num_rays_sparse_depth = self.configs['data_loader']['sparse_depth']['num_rays']

        if self.mode in ['train', 'validation', 'test-optimization']:
            self.preprocessed_data_dict = self.preprocess_data(self.raw_data_dict)
            if self.mode == 'train':
                self.model_configs = self.create_model_configs(self.preprocessed_data_dict)
        return

    def preprocess_data(self, raw_data_dict):
        preprocessed_data_dict = {}

        preprocessed_raw_data_dict = self.preprocess_raw_data(raw_data_dict)
        self.update_dict_recursively(preprocessed_data_dict, preprocessed_raw_data_dict)
        self.add_residual_dict_items(preprocessed_data_dict, preprocessed_raw_data_dict)

        if self.use_batching:
            cache_data_dict = self.create_cache(preprocessed_data_dict)
            self.update_dict_recursively(preprocessed_data_dict, cache_data_dict)
            self.add_residual_dict_items(preprocessed_data_dict, cache_data_dict)

        preprocessed_data_dict = CommonUtils.move_to_device(preprocessed_data_dict, self.device)
        return preprocessed_data_dict

    def create_model_configs(self, preprocessed_data_dict):
        # Create model configs to save
        model_configs = {
            'resolution': preprocessed_data_dict['nerf_data']['resolution'],
            'bounds': preprocessed_data_dict['nerf_data']['bounds'].tolist(),
            'translation_scale': preprocessed_data_dict['nerf_data'].get('sc', 1),
            f'{self.mode}_frame_nums': preprocessed_data_dict['frame_nums'].tolist(),
            'intrinsics': preprocessed_data_dict['nerf_data']['intrinsics'].tolist(),
            'extrinsics': preprocessed_data_dict['nerf_data']['poses'].tolist(),
            'average_pose': preprocessed_data_dict['nerf_data']['average_pose'].tolist(),
            'near': preprocessed_data_dict['nerf_data']['near'],
            'far': preprocessed_data_dict['nerf_data']['far'],
        }
        if self.ndc:
            model_configs['near_ndc'] = preprocessed_data_dict['nerf_data']['near_ndc']
            model_configs['far_ndc'] = preprocessed_data_dict['nerf_data']['far_ndc']
        if self.tensorf_model:
            model_configs['bounding_box'] = preprocessed_data_dict['tensorf_data']['bounding_box']
        return model_configs

    def get_model_configs(self):
        return self.model_configs

    # --------------------- Methods that preprocess raw data ----------------------------------- #

    def preprocess_raw_data(self, raw_data_dict: dict):
        return_dict = {}

        return_dict['nerf_data'] = self.preprocess_raw_nerf_data(raw_data_dict, return_dict)

        if self.tensorf_model and self.mode == 'train':
            return_dict['tensorf_data'] = self.preprocess_tensorf_data(raw_data_dict, return_dict)

        if self.sparse_depth_needed and self.mode == 'train':
            return_dict['sparse_depth_data'] = self.preprocess_raw_sparse_depth_data(raw_data_dict, return_dict)

        if self.dense_depth_needed and self.mode == 'train':
            return_dict['dense_depth_data'] = self.preprocess_raw_dense_depth_data(raw_data_dict, return_dict)

        if self.visibility_prior_needed and self.mode == 'train':
            return_dict['visibility_prior_data'] = self.preprocess_raw_visibility_prior_data(raw_data_dict, return_dict)

        self.add_residual_dict_items(return_dict, raw_data_dict)
        return return_dict

    def preprocess_raw_nerf_data(self, raw_data_dict: dict, processed_data_dict: dict):
        images = raw_data_dict['nerf_data']['images']
        poses = raw_data_dict['nerf_data']['extrinsics']
        intrinsics = raw_data_dict['nerf_data']['intrinsics']
        bds = raw_data_dict['nerf_data']['bounds']
        resolution = raw_data_dict['nerf_data']['resolution']

        images = self.preprocess_images(images)
        resolution = [int(x) for x in resolution]

        if self.downsampling_factor > 1:
            images = numpy.stack([skimage.transform.rescale(image, 1/self.downsampling_factor, anti_aliasing=True, multichannel=True) for image in images])
            resolution = [x // self.downsampling_factor for x in resolution]
            intrinsics[:, :2] /= self.downsampling_factor

        return_dict = {
            'images': images,
            'resolution': resolution,
            'intrinsics': intrinsics.astype('float32'),
        }

        if self.mode == 'train':
            preprocessed_poses_dict = self.preprocess_poses({
                'poses': poses.copy(),
                'bounds': bds,
                'bd_factor': self.configs['data_loader']['bd_factor']
            },
                train_mode=True)
        else:
            preprocessed_poses_dict = self.preprocess_poses({
                'poses': poses.copy(),
                'bounds': bds,
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)

        return_dict.update(preprocessed_poses_dict)
        return_dict['raw_poses'] = poses

        bds = preprocessed_poses_dict['bounds']
        if not self.ndc:
            near = bds[0] * .9
            far = bds[1] * 1.
            if self.nerf_synthetic:  # TODO: remove this
                near = bds[0]
                far = bds[1]
        else:
            bd_factor = self.bd_factor if self.bd_factor is not None else 1
            near = float(bds[0] * bd_factor)
            far = float(bds[1])
            near_ndc = 0.
            far_ndc = 1.

        return_dict['near'] = near
        return_dict['far'] = far
        if self.ndc:
            return_dict['near_ndc'] = near_ndc
            return_dict['far_ndc'] = far_ndc

        return return_dict

    def preprocess_tensorf_data(self, raw_data_dict: dict, processed_data_dict: dict):
        tensorf_data_dict = {
            'bounding_box': raw_data_dict['tensorf_data']['bounding_box']
        }
        return tensorf_data_dict

    def preprocess_raw_sparse_depth_data(self, raw_data_dict: dict, processed_data_dict: dict):
        return_dict = {}

        sc = processed_data_dict['nerf_data']['sc']
        h, w = processed_data_dict['nerf_data']['resolution']
        depths, reproj_errors = [], []
        for frame_num in raw_data_dict['frame_nums']:
            depth = -1 * numpy.ones((h, w))
            reproj_error = -1 * numpy.ones((h, w))
            if frame_num in raw_data_dict['sparse_depth_data']:
                frame_depth_data = raw_data_dict['sparse_depth_data'][frame_num]
                x, y = frame_depth_data['x'].to_numpy(), frame_depth_data['y'].to_numpy()
                if self.downsampling_factor > 1:
                    x = x / self.downsampling_factor
                    y = y / self.downsampling_factor
                x, y = numpy.clip(numpy.round(x), 0, w-1).astype('int'), numpy.clip(numpy.round(y), 0, h-1).astype('int')
                depth[y, x] = frame_depth_data['depth'].to_numpy() * sc
                reproj_error[y, x] = frame_depth_data['reprojection_error'].to_numpy()
            depths.append(depth)
            reproj_errors.append(reproj_error)
        return_dict['depths'] = numpy.stack(depths)
        return_dict['reprojection_errors'] = numpy.stack(reproj_errors)
        return return_dict

    def preprocess_raw_dense_depth_data(self, raw_data_dict: dict, processed_data_dict: dict):
        return_dict = {}

        sc = processed_data_dict['nerf_data']['sc']
        depths = raw_data_dict['dense_depth_data']['depth_values'] * sc
        depth_weights = raw_data_dict['dense_depth_data']['depth_weights']

        if self.downsampling_factor > 1:
            depths = numpy.stack([skimage.transform.rescale(depth, 1/self.downsampling_factor) for depth in depths])
            depth_weights = numpy.stack([skimage.transform.rescale(weights, 1/self.downsampling_factor) for weights in depth_weights])
        return_dict['depth_values'] = depths
        return_dict['depth_weights'] = depth_weights
        self.add_residual_dict_items(return_dict, raw_data_dict['dense_depth_data'])
        return return_dict

    def preprocess_raw_visibility_prior_data(self, raw_data_dict: dict, processed_data_dict: dict):
        return_dict = {}

        if self.configs['data_loader']['visibility_prior']['load_masks']:
            masks = raw_data_dict['visibility_prior_data']['masks']
            n, _, h, w = masks.shape
            masks = numpy.reshape(masks, (n * (n-1), h, w))
            if self.downsampling_factor > 1:
                masks = numpy.stack([skimage.transform.rescale(mask, 1/self.downsampling_factor) for mask in masks]).astype(bool)
            masks = numpy.reshape(masks, (n, n-1, h, w))
            return_dict['masks'] = masks

        if self.configs['data_loader']['visibility_prior']['load_weights']:
            weights = raw_data_dict['visibility_prior_data']['weights']
            n, _, h, w = weights.shape
            weights = numpy.reshape(weights, (n * (n-1), h, w))
            if self.downsampling_factor > 1:
                weights = numpy.stack([skimage.transform.rescale(weight, 1/self.downsampling_factor, anti_aliasing=True) for weight in weights])
            weights = numpy.reshape(weights, (n, n - 1, h, w))
            return_dict['weights'] = weights

        self.add_residual_dict_items(return_dict, raw_data_dict['visibility_prior_data'])
        return return_dict

    # ----------------------- Methods that create cache, if batching is enabled ----------------------- #

    def create_cache(self, preprocessed_data_dict: dict):
        cache_data_dict = {}

        indices = self.generate_indices(preprocessed_data_dict, cache_data_dict)
        cache_data_dict['indices'] = indices

        cache_data_dict['nerf_data'] = self.preprocess_nerf_data(preprocessed_data_dict, cache_data_dict)

        # if self.mip_nerf_model:
        #     cache_data_dict['mip_nerf_data'] = self.preprocess_mip_nerf_data(preprocessed_data_dict, cache_data_dict)

        if self.tensorf_model:
            self.generate_indices(preprocessed_data_dict, cache_data_dict)

        if self.sparse_depth_needed and self.mode == 'train':
            cache_data_dict['sparse_depth_data'] = self.preprocess_sparse_depth_data(preprocessed_data_dict, cache_data_dict)

        if self.dense_depth_needed and self.mode == 'train':
            cache_data_dict['dense_depth_data'] = self.preprocess_dense_depth_data(preprocessed_data_dict, cache_data_dict)

        if self.visibility_prior_needed and self.mode == 'train':
            cache_data_dict['visibility_prior_data'] = self.preprocess_visibility_prior_data(preprocessed_data_dict, cache_data_dict)

        cache_data_dict['common_data'] = self.preprocess_common_data(preprocessed_data_dict, cache_data_dict)

        return cache_data_dict

    def generate_indices(self, data_dict: dict, cache_data_dict: Optional[dict], iter_num: int = 0):
        n = len(data_dict['nerf_data']['images'])
        h, w = data_dict['nerf_data']['resolution']
        num_rays = n * h * w
        indices = numpy.arange(num_rays)

        if ('precrop_fraction' in self.configs['data_loader']) and \
                (self.configs['data_loader']['precrop_fraction'] < 1) and \
                (iter_num < self.configs['data_loader']['precrop_iterations']):
            h1 = int(round(h/2 * (1 - self.configs['data_loader']['precrop_fraction'])))
            h2 = int(round(h/2 * (1 + self.configs['data_loader']['precrop_fraction'])))
            w1 = int(round(w/2 * (1 - self.configs['data_loader']['precrop_fraction'])))
            w2 = int(round(w/2 * (1 + self.configs['data_loader']['precrop_fraction'])))
            indices_reshaped = numpy.reshape(indices, (n, h, w))
            cropped_indices = indices_reshaped[:, h1:h2, w1:w2]
            indices = cropped_indices.ravel()

        # For TensoRF
        # TODO: Disabled currently since it takes a lot of time
        if (iter_num == 0) and self.tensorf_model and (not self.ndc) and ('nerf_data' in cache_data_dict):
            # TODO: Do this on torch cuda
            chunk = self.configs['model']['chunk']
            chunk = 10240*5 # TODO: do not hard-code
            idx_chunks = numpy.split(indices, indices.size // chunk)
            valid_index_masks = []
            for i, idx_chunk in enumerate(tqdm(idx_chunks)):
                pixel_id = cache_data_dict['nerf_data']['pixel_id'][idx_chunk]
                # TODO: Check if the render() function can be reused to do this
                image_id = pixel_id[:, 0].cpu().numpy()
                intrinsics = data_dict['nerf_data']['intrinsics'][image_id]
                extrinsics = data_dict['nerf_data']['poses'][image_id]
                rays_o, rays_d = CommonUtils.get_rays_np(pixel_id, intrinsics, extrinsics, mip_nerf_used=False, nerf_synthetic=self.nerf_synthetic)
                rays_o[:, :1] *= -1  # TODO: check if moving this to get_rays_tr affects NeRF performance
                rays_d[:, :1] *= -1
                if self.nerf_synthetic:
                    rays_o[:, :1] *= -1
                    rays_d[:, :1] *= -1
                    rays_d = rays_d / numpy.linalg.norm(rays_d, axis=1, keepdims=True)

                # Sample z_vals
                vec = numpy.where(rays_d == 0, numpy.full_like(rays_d, 1e-6), rays_d)
                bounding_box_current = numpy.array([[-1.5, -1.67, -1], [1.5, 1.67, 1]])
                if self.nerf_synthetic:
                    bounding_box_current = numpy.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
                rate_a = (bounding_box_current[1] - rays_o) / vec
                rate_b = (bounding_box_current[0] - rays_o) / vec
                t_min = numpy.max(numpy.minimum(rate_a, rate_b), axis=-1)
                t_max = numpy.min(numpy.maximum(rate_a, rate_b), axis=-1)
                valid_index_mask = t_max > t_min
                valid_index_masks.append(valid_index_mask)
            valid_index_masks = numpy.concatenate(valid_index_masks)
            indices = indices[valid_index_masks]

        numpy.random.shuffle(indices)
        return indices

    @staticmethod
    def preprocess_common_data(data_dict: dict, cache_data_dict: dict) -> dict:
        images = data_dict['nerf_data']['images']
        poses = data_dict['nerf_data']['poses']
        intrinsics = data_dict['nerf_data']['intrinsics']
        all_data_dict = {
            'images': torch.from_numpy(images),
            'poses': torch.from_numpy(poses),
            'intrinsics': torch.from_numpy(intrinsics),
        }
        return all_data_dict
    
    def preprocess_nerf_data(self, data_dict: dict, cache_data_dict: dict):
        images = data_dict['nerf_data']['images']
        poses = data_dict['nerf_data']['poses']
        resolution = data_dict['nerf_data']['resolution']
        h, w = resolution

        # For random ray batching
        pixel_id_list = []
        for i in range(poses.shape[0]):
            grid_x, grid_y = numpy.meshgrid(
                numpy.arange(w, dtype=numpy.float32),
                numpy.arange(h, dtype=numpy.float32),
                indexing='xy')  # i: H x W, j: H x W
            image_id = i * numpy.ones_like(grid_x)
            pixel_id = numpy.stack([image_id, grid_x, grid_y], axis=2)
            pixel_id_list.append(pixel_id)
        pixel_id = numpy.stack(pixel_id_list, 0)  # (n, h, w, 3)
        pixel_id = numpy.reshape(pixel_id, (-1, 3)).astype(numpy.int32)  # (n*h*w, 3)

        all_data_dict = {
            'pixel_id': torch.from_numpy(pixel_id),
        }

        target_rgb = numpy.reshape(images, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
        all_data_dict['target_rgb'] = torch.from_numpy(target_rgb)
        return all_data_dict

    # def preprocess_mip_nerf_data(self, data_dict: dict, cache_data_dict: dict):
    #     n = data_dict['num_frames']
    #     h, w = data_dict['nerf_data']['resolution']
    #     rays_d = cache_data_dict['nerf_data']['rays_d'].numpy()  # (n*h*w, 3)
    #     rays_d = numpy.reshape(rays_d, (n, h, w, 3))  # (n, h, w, 3)
    #     radii = self.get_radii(rays_d)  # [n, h, w, 1]
    #     radii = numpy.reshape(radii, (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
    #     all_data_dict = {
    #         'radii': torch.from_numpy(radii),
    #     }
    #     if self.ndc:
    #         rays_o_ndc = cache_data_dict['nerf_data']['rays_o_ndc'].numpy()  # (n*h*w, 3)
    #         rays_o_ndc = numpy.reshape(rays_o_ndc, (n, h, w, 3))  # (n, h, w, 3)
    #         radii_ndc = self.get_radii_ndc(rays_o_ndc)
    #         radii_ndc = numpy.reshape(radii_ndc, (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
    #         all_data_dict['radii_ndc'] = torch.from_numpy(radii_ndc)
    #     return all_data_dict

    def preprocess_sparse_depth_data(self, data_dict: dict, cache_data_dict: dict):
        sparse_depth_data = data_dict['sparse_depth_data']

        sparse_depths = numpy.reshape(sparse_depth_data['depths'], (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
        reproj_errors = numpy.reshape(sparse_depth_data['reprojection_errors'], (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
        indices_sd = numpy.where(sparse_depths > 0)[0]
        numpy.random.shuffle(indices_sd)
        all_data_dict = {
            'indices': indices_sd,
            'depths': torch.from_numpy(sparse_depths),
            'reprojection_errors': torch.from_numpy(reproj_errors)
        }

        # If camera optimization enabled
        points_3d = -1 * numpy.ones((sparse_depths.shape[0], 3), dtype=numpy.float32)  # (n*h*w, 3)
        pixel_ids_sd = cache_data_dict['nerf_data']['pixel_id'][indices_sd]  # (n_sd, 3)
        image_id_sd = pixel_ids_sd[:, 0]  # (n_sd, )
        intrinsics_sd = data_dict['nerf_data']['intrinsics'][image_id_sd]
        extrinsics_sd = data_dict['nerf_data']['poses'][image_id_sd]
        rays_o_sd, rays_d_sd = CommonUtils.get_rays_np(pixel_ids_sd, intrinsics_sd, extrinsics_sd, mip_nerf_used=False)
        points_3d_sd = rays_o_sd + sparse_depths[indices_sd] * rays_d_sd
        points_3d[indices_sd] = points_3d_sd
        all_data_dict['points_3d'] = torch.from_numpy(points_3d)
        return all_data_dict

    def preprocess_dense_depth_data(self, data_dict: dict, cache_data_dict: dict):
        dense_depth_data = data_dict['dense_depth_data']

        dense_depths = numpy.reshape(dense_depth_data['depth_values'], (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
        dense_depth_weights = numpy.reshape(dense_depth_data['depth_weights'], (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
        all_data_dict = {
            'depth_values': torch.from_numpy(dense_depths),
            'depth_weights': torch.from_numpy(dense_depth_weights)
        }
        return all_data_dict

    def preprocess_visibility_prior_data(self, data_dict: dict, cache_data_dict: dict):
        all_data_dict = {}

        # Ray batching for other views
        num_frames = data_dict['frame_nums'].size
        if num_frames < 2:
            return all_data_dict

        if self.configs['data_loader']['visibility_prior']['load_masks']:
            visibility_prior_mask_images = data_dict['visibility_prior_data']['masks'].astype(numpy.float32)  # (n, n-1, h, w)
            visibility_prior_mask_images = numpy.transpose(visibility_prior_mask_images, [0, 2, 3, 1])  # (n, h, w, n-1)
            visibility_prior_masks = numpy.reshape(visibility_prior_mask_images, (-1, num_frames-1))  # (n*h*w, n-1)
            all_data_dict['masks'] = torch.from_numpy(visibility_prior_masks)
            all_data_dict['mask_images'] = torch.from_numpy(visibility_prior_mask_images)

        if self.configs['data_loader']['visibility_prior']['load_weights']:
            visibility_prior_weight_images = data_dict['visibility_prior_data']['weights'].astype(numpy.float32)  # (n, n-1, h, w)
            visibility_prior_weight_images = numpy.transpose(visibility_prior_weight_images, [0, 2, 3, 1])  # (n, h, w, n-1)
            visibility_prior_weights = numpy.reshape(visibility_prior_weight_images, (-1, num_frames-1))  # (n*h*w, n-1)
            all_data_dict['weights'] = torch.from_numpy(visibility_prior_weights)
            all_data_dict['weight_images'] = torch.from_numpy(visibility_prior_weight_images)

        return all_data_dict

    # ---------------- Methods to return next batch during training/validation ------------------ #

    def get_next_batch(self, iter_num: int, image_num: int = None):
        if self.use_batching:
            return_dict = self.load_cached_next_batch(iter_num, image_num)
        else:
            return_dict = self.load_uncached_next_batch(iter_num, image_num)
        return return_dict
    
    def load_cached_next_batch(self, iter_num, image_num):
        return_dict = {}
        return_dict['common_data'] = {}

        indices_dict = self.select_batch_indices(iter_num, image_num)
        return_dict.update(indices_dict)

        nerf_data_dict = self.load_nerf_cached_batch(iter_num, indices_dict)
        return_dict.update(nerf_data_dict)

        # if self.mip_nerf_model:
        #     mip_nerf_depth_data_dict = self.load_mip_nerf_cached_batch(indices_dict, return_dict)
        #     return_dict.update(mip_nerf_depth_data_dict)

        if self.sparse_depth_needed and (self.mode == 'train'):
            sparse_depth_data_dict = self.load_sparse_depth_cached_batch(indices_dict, return_dict)
            return_dict.update(sparse_depth_data_dict)

        if self.dense_depth_needed and (self.mode == 'train'):
            dense_depth_data_dict = self.load_dense_depth_cached_batch(indices_dict, return_dict)
            return_dict.update(dense_depth_data_dict)

        if self.visibility_prior_needed and (self.mode == 'train'):
            visibility_prior_data_dict = self.load_visibility_prior_cached_batch(indices_dict, return_dict)
            return_dict.update(visibility_prior_data_dict)

        if self.mode == 'train':
            # Loaded during training and for validation with train-data-loader only
            common_data_dict = self.load_common_cached_batch(indices_dict, return_dict)
            return_dict.update(common_data_dict)

        # Copy common data num_gpus times so that it will be available to all GPUs
        num_gpus = len(self.configs['device'])
        for key in return_dict['common_data'].keys():
            if isinstance(return_dict['common_data'][key], torch.Tensor):
                ones_shape = [1] * return_dict['common_data'][key].ndim
                return_dict['common_data'][key] = return_dict['common_data'][key][None].repeat([num_gpus, *ones_shape])
        return return_dict

    def select_batch_indices(self, iter_num, image_num):
        return_dict = {}

        if image_num is None:
            if iter_num == self.configs['data_loader']['precrop_iterations']:
                self.generate_indices(self.preprocessed_data_dict, None, iter_num)
            indices = self.preprocessed_data_dict['indices'][self.i_batch: self.i_batch + self.num_rays]
            self.i_batch += self.num_rays
            if self.i_batch >= self.preprocessed_data_dict['indices'].size:
                numpy.random.shuffle(self.preprocessed_data_dict['indices'])
                self.i_batch = 0
        else:
            h, w = self.preprocessed_data_dict['nerf_data']['resolution']
            image_index = numpy.where(self.preprocessed_data_dict['frame_nums'] == image_num)[0].item()
            indices = numpy.arange(h * w) + (image_index * h * w)
        indices_id = numpy.ones_like(indices)  # To identify which part added the indices

        if self.sparse_depth_needed and (self.mode == 'train') and (image_num is None):
            indices_sd = self.preprocessed_data_dict['sparse_depth_data']['indices'][self.i_batch_sparse_depth: self.i_batch_sparse_depth + self.num_rays_sparse_depth]
            indices_id_sd = 2 * numpy.ones_like(indices_sd)
            self.i_batch_sparse_depth += self.num_rays_sparse_depth
            if self.i_batch_sparse_depth >= self.preprocessed_data_dict['sparse_depth_data']['indices'].size:
                numpy.random.shuffle(self.preprocessed_data_dict['sparse_depth_data']['indices'])
                self.i_batch_sparse_depth = 0
            indices = numpy.concatenate([indices, indices_sd])
            indices_id = numpy.concatenate([indices_id, indices_id_sd])

        # indices are not moved to gpu in cached data since numpy shuffle is applied on the indices after every epoch
        return_dict['indices'] = torch.from_numpy(indices).to(self.device)
        return_dict['indices_mask_nerf'] = torch.from_numpy(indices_id == 1).to(self.device)
        if self.sparse_depth_needed and (self.mode == 'train') and (image_num is None):
            return_dict['indices_mask_sparse_depth'] = torch.from_numpy(indices_id == 2).to(self.device)
        return return_dict

    def load_nerf_cached_batch(self, iter_num, indices_dict):
        num_frames = self.preprocessed_data_dict['frame_nums'].size
        indices = indices_dict['indices']
        indices_mask_nerf = indices_dict['indices_mask_nerf']
        indices_nerf = indices[indices_mask_nerf]

        # Initialize values to -1. Will be filled later in respective functions
        pixel_id = -1 * torch.ones((indices.shape[0], 3), dtype=torch.int32).to(self.device)
        target_rgb = -1 * torch.ones((indices.shape[0], 3)).to(self.device)

        pixel_id[indices_mask_nerf] = self.preprocessed_data_dict['nerf_data']['pixel_id'][indices_nerf]
        target_rgb[indices_mask_nerf] = self.preprocessed_data_dict['nerf_data']['target_rgb'][indices_nerf]

        return_dict = {
            'iter_num': iter_num,
            'num_frames': num_frames,
            'pixel_id': pixel_id,
            'target_rgb': target_rgb,
        }
        return return_dict

    # def load_mip_nerf_cached_batch(self, indices_dict, batch_dict):
    #     return_dict = {}
    #
    #     indices = indices_dict['indices']
    #     indices_mask_nerf = indices_dict['indices_mask_nerf']
    #     indices_nerf = indices[indices_mask_nerf]
    #
    #     # Initialize values to -1. Will be filled later in respective functions
    #     radii = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
    #     radii[indices_mask_nerf] = self.preprocessed_data_dict['nerf_data']['radii'][indices_nerf]
    #     return_dict['radii'] = radii
    #
    #     if self.ndc:
    #         radii_ndc = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
    #         radii_ndc[indices_mask_nerf] = self.preprocessed_data_dict['nerf_data']['radii_ndc'][indices_nerf]
    #         return_dict['radii_ndc'] = radii_ndc
    #     return return_dict

    def load_sparse_depth_cached_batch(self, indices_dict, batch_dict):
        return_dict = {}

        if 'indices_mask_sparse_depth' not in indices_dict:
            return return_dict

        indices = indices_dict['indices']
        indices_mask_sd = indices_dict['indices_mask_sparse_depth']
        indices_sd = indices[indices_mask_sd]

        # fill in values that were earlier initialized to -1
        return_dict['pixel_id'] = batch_dict['pixel_id']
        return_dict['pixel_id'][indices_mask_sd] = self.preprocessed_data_dict['nerf_data']['pixel_id'][indices_sd]

        # Initialize values to -1. Will be filled later in respective functions
        sparse_depths = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
        reproj_errors = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
        points_3d = -1 * torch.ones((indices.shape[0], 3)).to(self.device)

        sparse_depths[indices_mask_sd] = self.preprocessed_data_dict['sparse_depth_data']['depths'][indices_sd]
        reproj_errors[indices_mask_sd] = self.preprocessed_data_dict['sparse_depth_data']['reprojection_errors'][indices_sd]
        points_3d[indices_mask_sd] = self.preprocessed_data_dict['sparse_depth_data']['points_3d'][indices_sd]

        return_dict['sparse_depth_values'] = sparse_depths
        return_dict['sparse_depth_errors'] = reproj_errors
        return_dict['sparse_depth_points3d'] = points_3d
        return return_dict

    def load_dense_depth_cached_batch(self, indices_dict, batch_dict):
        return_dict = {}

        indices = indices_dict['indices']
        indices_mask_nerf = indices_dict['indices_mask_nerf']
        indices_nerf = indices[indices_mask_nerf]

        # Initialize values to -1. Will be filled later in respective functions
        dense_depths = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
        dense_depth_weights = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
        dense_depths[indices_mask_nerf] = self.preprocessed_data_dict['dense_depth_data']['depth_values'][indices_nerf]
        dense_depth_weights[indices_mask_nerf] = self.preprocessed_data_dict['dense_depth_data']['depth_weights'][indices_nerf]
        return_dict['dense_depth_values'] = dense_depths
        return_dict['dense_depth_weights'] = dense_depth_weights
        return return_dict

    def load_visibility_prior_cached_batch(self, indices_dict, batch_dict):
        return_dict = {}
        return_dict['common_data'] = batch_dict['common_data']

        indices = indices_dict['indices']
        indices_mask_nerf = indices_dict['indices_mask_nerf']
        indices_nerf = indices[indices_mask_nerf]

        poses = torch.from_numpy(self.preprocessed_data_dict['nerf_data']['poses']).to(self.device)
        return_dict['common_data']['poses'] = poses

        if self.configs['data_loader']['visibility_prior']['load_masks']:
            num_frames = self.preprocessed_data_dict['frame_nums'].size
            vc_masks = -1 * torch.ones((indices.shape[0], num_frames-1)).to(self.device)
            vc_masks[indices_mask_nerf] = self.preprocessed_data_dict['visibility_prior_data']['masks'][indices_nerf]
            return_dict['visibility_prior_masks'] = vc_masks

        if self.configs['data_loader']['visibility_prior']['load_weights']:
            vc_weights = -1 * torch.ones((indices.shape[0], 1)).to(self.device)
            vc_weights[indices_mask_nerf] = self.preprocessed_data_dict['visibility_prior_data']['weights'][indices_nerf]
            return_dict['visibility_prior_weights'] = vc_weights

        return return_dict

    def load_common_cached_batch(self, indices_dict, batch_dict):
        return_dict = {}
        return_dict['common_data'] = batch_dict['common_data']
        return_dict['common_data']['frame_nums'] = self.preprocessed_data_dict['frame_nums']
        return_dict['common_data']['poses'] = self.preprocessed_data_dict['common_data']['poses']
        return_dict['common_data']['intrinsics'] = self.preprocessed_data_dict['common_data']['intrinsics']
        return_dict['common_data']['images'] = self.preprocessed_data_dict['common_data']['images']
        return_dict['common_data']['resolution'] = self.preprocessed_data_dict['nerf_data']['resolution']
        return return_dict

    def load_uncached_next_batch(self, iter_num, image_num):
        """
        Random from one image.
        This function currently supports loading nerf data only and nothing else. It has been tested for camera-params optimization only.
        """
        images = self.preprocessed_data_dict['nerf_data']['images']
        # extrinsics = self.preprocessed_data_dict['nerf_data']['poses']
        resolution = self.preprocessed_data_dict['nerf_data']['resolution']
        # intrinsics = self.preprocessed_data_dict['nerf_data']['intrinsics']
        # near = self.preprocessed_data_dict['nerf_data']['near']
        h, w = resolution

        if image_num is None:
            image_num = numpy.random.randint(0, images.shape[0])
        target_image = torch.from_numpy(images[image_num])

        if ('precrop_fraction' in self.configs['data_loader']) and \
                (self.configs['data_loader']['precrop_fraction'] < 1) and \
                (iter_num < self.configs['data_loader']['precrop_iterations']):
            raise NotImplementedError

        coords = torch.stack(torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = numpy.random.choice(coords.shape[0], size=[self.num_rays], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        image_id = image_num * torch.ones_like(select_coords[:, 0])
        pixel_id = torch.stack([image_id, select_coords[:, 1], select_coords[:, 0]], dim=1)
        target_rgb = target_image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        indices_id = numpy.ones_like(select_inds)
        indices_mask_nerf = torch.from_numpy(indices_id == 1)

        return_dict = {
            'iter_num': iter_num,
            'num_frames': images.shape[0],
            'indices_mask_nerf': indices_mask_nerf,
            'pixel_id': pixel_id,
            'target_rgb': target_rgb,
        }
        return_dict = CommonUtils.move_to_device(return_dict, self.device)
        return return_dict

    # --------------------------------- Inference methods ----------------------------------- #

    def create_test_data(self, *, pose: numpy.ndarray, view_pose: Optional[numpy.ndarray] = None,
                         secondary_poses: Optional[List[numpy.ndarray]] = None, preprocess_pose: bool = True,
                         intrinsic: Optional[numpy.ndarray] = None, view_intrinsic: Optional[numpy.ndarray] = None,
                         secondary_intrinsics: Optional[List[numpy.ndarray]] = None):
        # Create a copy of poses so that any preprocessing does not affect the poses in parent files.
        pose = pose.copy()
        if view_pose is not None:
            view_pose = view_pose.copy()
        if secondary_poses is not None:
            secondary_poses = [secondary_pose.copy() for secondary_pose in secondary_poses]

        if preprocess_pose:
            processed_pose = self.preprocess_poses({
                'poses': pose[None],
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)['poses'][0]
        else:
            processed_pose = pose.astype('float32')
        resolution = self.model_configs['resolution']
        if intrinsic is None:
            intrinsics = numpy.array(self.model_configs['intrinsics'])
            intrinsic = numpy.mean(intrinsics, axis=0)
        intrinsic = intrinsic.astype('float32')
        if (view_pose is not None) and preprocess_pose:
            processed_view_pose = self.preprocess_poses({
                'poses': view_pose[None],
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)['poses'][0]
        else:
            processed_view_pose = view_pose

        if (secondary_poses is not None) and preprocess_pose:
            processed_sec_poses = self.preprocess_poses({
                'poses': numpy.array(secondary_poses),
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)['poses']
        else:
            processed_sec_poses = secondary_poses

        h, w = resolution
        grid_x, grid_y = numpy.meshgrid(
            numpy.arange(w, dtype=numpy.float32),
            numpy.arange(h, dtype=numpy.float32),
            indexing='xy')  # i: H x W, j: H x W
        image_id = 0 * numpy.ones_like(grid_x)
        pixel_id = numpy.stack([image_id, grid_x, grid_y], axis=2)
        pixel_id = pixel_id.reshape((-1, 3)).astype('int32')
        pixel_id = torch.from_numpy(pixel_id)

        processed_pose = torch.from_numpy(processed_pose)
        processed_view_pose = torch.from_numpy(processed_view_pose) if processed_view_pose is not None else None
        processed_sec_poses = torch.from_numpy(processed_sec_poses) if processed_sec_poses is not None else None
        intrinsic = torch.from_numpy(intrinsic)
        view_intrinsic = torch.from_numpy(view_intrinsic) if view_intrinsic is not None else None
        secondary_intrinsics = torch.from_numpy(secondary_intrinsics) if secondary_intrinsics is not None else None

        input_batch = {
            'num_frames': 1,
            'pixel_id': pixel_id,
            'common_data': {
                'processed_pose': processed_pose,
                'processed_view_pose': processed_view_pose,
                'processed_secondary_poses': processed_sec_poses,
                'intrinsic': intrinsic,
                'view_intrinsic': view_intrinsic,
                'secondary_intrinsics': secondary_intrinsics,
            }
        }

        # Copy common data num_gpus times so that it will be available to all GPUs
        num_gpus = len(self.configs['device'])
        for key in input_batch['common_data'].keys():
            if isinstance(input_batch['common_data'][key], torch.Tensor):
                ones_shape = [1] * input_batch['common_data'][key].ndim
                input_batch['common_data'][key] = input_batch['common_data'][key][None].repeat([num_gpus, *ones_shape])

        input_batch = CommonUtils.move_to_device(input_batch, self.device)
        return input_batch

    def retrieve_inference_outputs(self, network_outputs: dict):
        h, w = self.model_configs['resolution']
        processed_outputs = self.post_process_output(network_outputs)
        if 'fine_model' in self.configs['model']:
            suffix = '_fine'
        elif 'coarse_model' in self.configs['model']:
            suffix = '_coarse'
        else:
            raise RuntimeError
        image = self.post_process_image(processed_outputs[f'rgb{suffix}'].reshape(h, w, 3))
        depth = self.post_process_depth(processed_outputs[f'depth{suffix}'].reshape(h, w))
        depth_var = self.post_process_depth(processed_outputs[f'depth_var{suffix}'].reshape(h, w))
        return_dict = {
            'image': image,
            'depth': depth,
            'depth_var': depth_var,
        }
        if self.ndc:
            depth_ndc = self.post_process_depth(processed_outputs[f'depth_ndc{suffix}'].reshape(h, w))
            depth_var_ndc = self.post_process_depth(processed_outputs[f'depth_var_ndc{suffix}'].reshape(h, w))
            return_dict['depth_ndc'] = depth_ndc
            return_dict['depth_var_ndc'] = depth_var_ndc
        if f'visibility2{suffix}' in processed_outputs:
            visibility2_flat = processed_outputs[f'visibility2{suffix}']  # (h * w, nf-1)
            visibility2 = visibility2_flat.reshape((h, w, -1))  # (h, w, nf-1)
            visibility2 = visibility2.transpose([2, 0, 1])  # (nf-1, h, w)
            visibility2 = self.post_process_visibility(visibility2)
            return_dict['visibility2'] = visibility2
        return return_dict

    # ------------------------------- Data pre/post process methods -------------------------------- #

    def preprocess_images(self, images: numpy.ndarray):
        images = images.astype('float32') / 255
        if self.configs['model']['white_bkgd']:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
        return images

    def preprocess_poses(self, input_dict: dict, train_mode: bool):
        return_dict = {}
        poses = input_dict['poses']

        if train_mode:
            if 'bounds' in input_dict:
                bds = input_dict['bounds']
                # Rescale if bd_factor is provided
                if ('bd_factor' in input_dict) and (input_dict['bd_factor'] is not None):
                    bd_factor = input_dict['bd_factor']
                    sc = 1./(float(bds[0]) * bd_factor)
                else:
                    sc = 1
                poses[:, :3, 3] *= sc
                bds *= sc
                return_dict['sc'] = sc
                return_dict['bounds'] = bds
            if self.configs['data_loader']['recenter_camera_poses']:
                avg_pose = self.compute_average_pose(poses)
            else:
                avg_pose = numpy.eye(4)
            return_dict['average_pose'] = avg_pose
        else:
            sc = input_dict['translation_scale']
            poses[:, :3, 3] *= sc
            if 'bounds' in input_dict:
                bds = input_dict['bounds']
                bds *= sc
                return_dict['bounds'] = bds
            avg_pose = input_dict['average_pose']

        poses = self.recenter_poses(poses, avg_pose)
        poses = self.convert_pose_to_standard_coordinates(poses)

        if self.configs['data_loader']['spherify']:
            poses, render_poses, bds = self.spherify_poses(poses, return_dict['bounds'])
            return_dict['bounds'] = bds

        return_dict['poses'] = poses.astype(numpy.float32)
        return return_dict

    @staticmethod
    def recenter_poses(poses, avg_pose):
        centered_poses = avg_pose[None] @ numpy.linalg.inv(poses)
        return centered_poses

    def convert_pose_to_standard_coordinates(self, poses):
        # Convert from Colmap/RE10K convention to NeRF convention: (x,-y,-z) to (x,y,z)
        perm_matrix = numpy.eye(3)
        perm_matrix[1, 1] = -1
        perm_matrix[2, 2] = -1
        std_poses = self.change_coordinate_system(poses, perm_matrix)
        return std_poses

    @staticmethod
    def compute_average_pose(poses: numpy.ndarray):
        def normalize(x):
            return x / numpy.linalg.norm(x)

        def viewmatrix(z, up, pos):
            vec2 = normalize(z)
            vec1_avg = up
            vec0 = normalize(numpy.cross(vec1_avg, vec2))
            vec1 = normalize(numpy.cross(vec2, vec0))
            m = numpy.stack([vec0, vec1, vec2, pos], 1)
            bottom = numpy.array([0, 0, 0, 1])[None]
            matrix = numpy.concatenate([m, bottom], axis=0)
            return matrix

        # compute average pose in camera2world system
        rot_mats = poses[:, :3, :3]
        rot_inverted = numpy.transpose(rot_mats, axes=[0, 2, 1])
        translations = poses[:, :3, 3:]
        rotated_translations = -rot_inverted @ translations
        avg_translation = numpy.mean(rotated_translations, axis=0)[:, 0]

        vec2 = normalize(rot_inverted[:, :3, 2].sum(0))
        up = rot_inverted[:, :3, 1].sum(0)
        avg_pose_c2w = viewmatrix(vec2, up, avg_translation)
        avg_pose = numpy.linalg.inv(avg_pose_c2w)  # convert avg_pose to world2camera system
        return avg_pose

    @staticmethod
    def change_coordinate_system(poses: numpy.ndarray, p: numpy.ndarray):
        changed_poses = []
        for pose in poses:
            r = pose[:3, :3]
            t = pose[:3, 3:]
            rc = p.T @ r @ p
            tc = p @ t
            changed_pose = numpy.concatenate([numpy.concatenate([rc, tc], axis=1), pose[3:]], axis=0)
            changed_poses.append(changed_pose)
        changed_poses = numpy.stack(changed_poses)
        return changed_poses

    def spherify_poses(self, poses, bds):
        # TODO SNB: Understand where spherify is required/used
        p34_to_44 = lambda p : numpy.concatenate([p, numpy.tile(numpy.reshape(numpy.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

        rays_d = poses[:,:3,2:3]
        rays_o = poses[:,:3,3:4]

        def min_line_dist(rays_o, rays_d):
            A_i = numpy.eye(3) - rays_d * numpy.transpose(rays_d, [0,2,1])
            b_i = -A_i @ rays_o
            pt_mindist = numpy.squeeze(-numpy.linalg.inv((numpy.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)

        center = pt_mindist
        up = (poses[:,:3,3] - center).mean(0)

        vec0 = self.normalize(up)
        vec1 = self.normalize(numpy.cross([.1,.2,.3], vec0))
        vec2 = self.normalize(numpy.cross(vec0, vec1))
        pos = center
        c2w = numpy.stack([vec1, vec2, vec0, pos], 1)

        poses_reset = numpy.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

        rad = numpy.sqrt(numpy.mean(numpy.sum(numpy.square(poses_reset[:,:3,3]), -1)))

        sc = 1./rad
        poses_reset[:,:3,3] *= sc
        bds *= sc
        rad *= sc

        centroid = numpy.mean(poses_reset[:,:3,3], 0)
        zh = centroid[2]
        radcircle = numpy.sqrt(rad**2-zh**2)
        new_poses = []

        for th in numpy.linspace(0.,2.*numpy.pi, 120):

            camorigin = numpy.array([radcircle * numpy.cos(th), radcircle * numpy.sin(th), zh])
            up = numpy.array([0,0,-1.])

            vec2 = self.normalize(camorigin)
            vec0 = self.normalize(numpy.cross(vec2, up))
            vec1 = self.normalize(numpy.cross(vec2, vec0))
            pos = camorigin
            p = numpy.stack([vec0, vec1, vec2, pos], 1)

            new_poses.append(p)

        new_poses = numpy.stack(new_poses, 0)

        new_poses = numpy.concatenate([new_poses, numpy.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
        poses_reset = numpy.concatenate([poses_reset[:,:3,:4], numpy.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

        return poses_reset, new_poses, bds

    def post_process_output(self, output_batch: Union[torch.Tensor, list, tuple, dict]) -> Union[numpy.ndarray, list, tuple, dict]:
        if isinstance(output_batch, torch.Tensor):
            processed_batch = output_batch.detach().cpu().numpy()
        elif isinstance(output_batch, list) or isinstance(output_batch, tuple):
            processed_batch = []
            for list_element in output_batch:
                processed_batch.append(self.post_process_output(list_element))
        elif isinstance(output_batch, dict):
            processed_batch = {}
            for key in output_batch.keys():
                processed_batch[key] = self.post_process_output(output_batch[key])
        else:
            raise RuntimeError(f'How do I post process an object of type: {type(output_batch)}?')
        return processed_batch

    @staticmethod
    def post_process_image(rgb: numpy.ndarray):
        clipped_image = numpy.clip(rgb, a_min=0, a_max=1)
        uint8_image = numpy.round(clipped_image * 255).astype('uint8')
        return uint8_image

    @staticmethod
    def post_process_depth(depth: numpy.ndarray):
        clipped_depth = numpy.clip(depth, a_min=0, a_max=numpy.inf).astype('float32')
        return clipped_depth

    @staticmethod
    def post_process_visibility(visibility: numpy.ndarray):
        proc_visibility = visibility.astype('float32')
        return proc_visibility

    # ------------------------------------ Utilities ------------------------------------------- #
    @staticmethod
    def update_dict_recursively(tgt_dict, src_dict):
        for key in src_dict:
            if not isinstance(src_dict[key], dict):
                continue
            if key not in tgt_dict:
                tgt_dict[key] = {}
            tgt_dict[key].update(src_dict[key])
        return

    @staticmethod
    def add_residual_dict_items(tgt_dict, src_dict):
        for key in src_dict:
            if key not in tgt_dict:
                tgt_dict[key] = src_dict[key]
        return
