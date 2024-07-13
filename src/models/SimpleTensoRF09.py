# Shree KRISHNAya Namaha
# Extended from TensoRF07.py. Bug fixes w.r.t. optimizing camera parameters
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import abc

import numpy
import torch
import torch.nn.functional as F
from torch.nn import ModuleDict
from torch.nn import ModuleList

from utils import CommonUtils04 as CommonUtils


class SimpleTensoRF(torch.nn.Module):
    def __init__(self, configs: dict, model_configs: dict):
        super().__init__()
        self.configs = configs
        self.model_configs = model_configs
        self.ndc = self.configs['data_loader']['ndc']
        self.coarse_model_needed = 'coarse_model' in self.configs['model']
        self.fine_model_needed = 'fine_model' in self.configs['model']
        self.predict_visibility = (self.coarse_model_needed and self.configs['model']['coarse_model']['predict_visibility']) or \
                                  (self.fine_model_needed and self.configs['model']['fine_model']['predict_visibility'])
        self.nerf_synthetic = self.configs['database'] == 'NeRF_Synthetic'  # TODO: Remove this

        self.coarse_model = None
        self.fine_model = None
        self.intrinsics_learner = None
        self.extrinsics_learner = None
        self.augmentations_needed = 'augmentations' in configs['model']
        if self.augmentations_needed:
            self.augmented_models = []
            self.augmented_models_nn = []
        self.build_nerf()

        self.optimizers = None
        self.train_data_preprocessor = None
        return

    def build_nerf(self):
        if self.coarse_model_needed:
            self.coarse_model = self.get_tensor_model('coarse_model', self.configs, self.configs['model']['coarse_model'], self.model_configs)

        if self.fine_model_needed:
            self.fine_model = self.get_tensor_model('fine_model', self.configs, self.configs['model']['fine_model'], self.model_configs)

        self.intrinsics_learner = IntrinsicsLearner(numpy.array(self.model_configs['intrinsics']),
                                                    learn_focal=self.configs['model']['learn_camera_focal_length'])
        self.extrinsics_learner = ExtrinsicsLearner(numpy.array(self.model_configs['extrinsics']),
                                                    learn_rotation=self.configs['model']['learn_camera_rotation'],
                                                    learn_translation=self.configs['model']['learn_camera_translation'])

        if self.augmentations_needed:
            for augmentation_configs in self.configs['model']['augmentations']:
                aug_name = augmentation_configs['name']
                coarse_model = None
                fine_model = None
                aug_model_nn = ModuleDict()
                if 'coarse_model' in augmentation_configs:
                    name = f'{aug_name}_coarse_model'
                    coarse_model = self.get_tensor_model(name, self.configs, augmentation_configs['coarse_model'], self.model_configs)
                    aug_model_nn['coarse_model'] = coarse_model
                if 'fine_model' in augmentation_configs:
                    name = f'{aug_name}_fine_model'
                    fine_model = self.get_tensor_model(name, self.configs, augmentation_configs['fine_model'], self.model_configs)
                    aug_model_nn['fine_model'] = fine_model
                # self.augmented_models.append(ModuleDict({
                #     'name': aug_name,
                #     'coarse_model': coarse_model,
                #     'fine_model': fine_model,
                # }))
                self.augmented_models.append({
                    'name': aug_name,
                    'coarse_model': coarse_model,
                    'fine_model': fine_model,
                })
                self.augmented_models_nn.append(aug_model_nn)
            self.augmented_models_nn = ModuleList(self.augmented_models_nn)
        return

    def get_trainable_parameters(self, optimizer_configs):
        trainable_params = []
        if self.coarse_model_needed:
            trainable_params.extend(self.coarse_model.get_trainable_parameters(optimizer_configs))
        if self.fine_model_needed:
            trainable_params.extend(self.fine_model.get_trainable_parameters(optimizer_configs))
        if self.augmentations_needed:
            for augmented_model in self.augmented_models:
                if augmented_model['coarse_model'] is not None:
                    trainable_params.extend(augmented_model['coarse_model'].get_trainable_parameters(optimizer_configs))
                if augmented_model['fine_model'] is not None:
                    trainable_params.extend(augmented_model['fine_model'].get_trainable_parameters(optimizer_configs))
        # TODO: add trainable parameters of Intrinsics and Extrinsics learners
        return trainable_params

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (name == 'optimizers') and (value is not None):
            self.set_models_attribute(name, value)
        return

    def set_models_attribute(self, name: str, value):
        self.set_model_attribute(self.coarse_model, name, value)
        self.set_model_attribute(self.fine_model, name, value)
        if self.augmentations_needed:
            for augmented_model in self.augmented_models:
                if augmented_model['coarse_model'] is not None:
                    self.set_model_attribute(augmented_model['coarse_model'], name, value)
                if augmented_model['fine_model'] is not None:
                    self.set_model_attribute(augmented_model['fine_model'], name, value)
        return

    @staticmethod
    def set_model_attribute(model, name: str, value):
        if model is not None and hasattr(model, name):
            setattr(model, name, value)
        return

    def rebuild_camera_params_learners(self, *, intrinsics: numpy.ndarray = None, extrinsics = None, device):
        if intrinsics is not None:
            self.intrinsics_learner = IntrinsicsLearner(initial_intrinsics=intrinsics,
                                                        learn_focal=self.configs['model']['learn_camera_focal_length'])
            self.intrinsics_learner = self.intrinsics_learner.to(device)
        if extrinsics is not None:
            self.extrinsics_learner = ExtrinsicsLearner(initial_extrinsics=extrinsics,
                                                        learn_rotation=self.configs['model']['learn_camera_rotation'],
                                                        learn_translation=self.configs['model']['learn_camera_translation'])
            self.extrinsics_learner = self.extrinsics_learner.to(device)
        return

    def forward(self, input_batch: dict, *, retraw: bool = False, sec_views_vis: bool = False, mode: str = None):
        input_batch = self.deep_dict_copy(input_batch)
        if 'common_data' in input_batch.keys():
            # unpack common data
            for key in input_batch['common_data'].keys():
                if isinstance(input_batch['common_data'][key], torch.Tensor):
                    input_batch['common_data'][key] = input_batch['common_data'][key][0]

        if self.training and (mode != 'test_camera_params_optimization') and (input_batch['sub_batch_index'] == 0):
            # Run model modifications: Has to be done before the forward prop, so that loss can be back-propped.
            self.run_model_modifications(input_batch['iter_num'])
            # Update pixels to train
            self.filter_pixel_ids(input_batch['iter_num'])

        image_id = input_batch['pixel_id'][:, 0]
        intrinsics = self.intrinsics_learner(image_id)
        extrinsics = self.extrinsics_learner(image_id)
        all_extrinsics = self.extrinsics_learner(torch.arange(input_batch['num_frames']).to(image_id))
        input_batch['intrinsics'] = intrinsics
        input_batch['extrinsics'] = extrinsics

        if mode == 'camera_params_only':
            render_output_dict = {}
        else:
            render_output_dict = self.render(input_batch, retraw=retraw or self.training, sec_views_vis=sec_views_vis or self.training, mode=mode)

        render_output_dict['intrinsics'] = intrinsics
        render_output_dict['extrinsics'] = extrinsics
        render_output_dict['extrinsics_all'] = all_extrinsics
        return render_output_dict

    def render(self, input_dict: dict, *, retraw: bool, sec_views_vis: bool, mode: str):
        all_ret = self.batchify_rays(input_dict, retraw=retraw, sec_views_vis=sec_views_vis, mode=mode)
        return all_ret

    def batchify_rays(self, input_dict: dict, *, retraw, sec_views_vis, mode):
        """
        Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        num_rays = input_dict['pixel_id'].shape[0]
        chunk = self.configs['model']['chunk']
        for i in range(0, num_rays, chunk):
            render_rays_dict = {}
            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor) and (input_dict[key].shape[0] == num_rays):
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                elif isinstance(input_dict[key], numpy.ndarray) and (input_dict[key].shape[0] == num_rays):  # indices
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                else:
                    render_rays_dict[key] = input_dict[key]

            ret = self.render_rays(render_rays_dict, retraw=retraw, sec_views_vis=sec_views_vis, mode=mode)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = self.merge_mini_batch_data(all_ret)
        return all_ret

    def render_rays(self, input_dict: dict, *, retraw, sec_views_vis, mode):
        return_dict = {}
        pixel_id = input_dict['pixel_id']  # (image_id, x, y)
        image_id = pixel_id[:, 0]
        resolution = self.model_configs['resolution']
        intrinsics = input_dict['intrinsics']
        extrinsics = input_dict['extrinsics']
        num_rays = pixel_id.shape[0]

        # rays_o = input_dict['rays_o']
        # rays_d = input_dict['rays_d']
        rays_o, rays_d = CommonUtils.get_rays_tr(pixel_id, intrinsics, extrinsics, mip_nerf_used=True, nerf_synthetic=self.nerf_synthetic)
        rays_o[:, :1] *= -1  # TODO: check if moving this to get_rays_tr affects NeRF performance
        rays_d[:, :1] *= -1
        if self.nerf_synthetic:
            rays_o[:, :1] *= -1
            rays_d[:, :1] *= -1
            rays_d = rays_d / torch.linalg.norm(rays_d, dim=1, keepdim=True)
        return_dict['rays_o'] = rays_o
        return_dict['rays_d'] = rays_d
        if self.ndc:
            rays_o_ndc, rays_d_ndc = CommonUtils.get_ndc_rays_tr(rays_o, rays_d, resolution, intrinsics,
                                                                 near=self.model_configs['near'])
            return_dict['rays_o_ndc'] = rays_o_ndc
            return_dict['rays_d_ndc'] = rays_d_ndc
        if (self.coarse_model_needed and self.configs['model']['coarse_model']['use_view_dirs']) or \
                (self.fine_model_needed and self.configs['model']['fine_model']['use_view_dirs']):
            # provide ray directions as input
            if mode == 'static_camera':
                if input_dict['common_data']['view_intrinsic'] is not None:
                    view_intrinsics = input_dict['common_data']['view_intrinsic'][None].repeat((num_rays, 1, 1))
                else:
                    view_intrinsics = intrinsics
                view_extrinsics = input_dict['common_data']['processed_view_pose'][None].repeat((num_rays, 1, 1))
                view_rays_o, view_rays_d = CommonUtils.get_rays_tr(pixel_id, view_intrinsics, view_extrinsics, mip_nerf_used=False)
                if not self.ndc:
                    view_dirs = CommonUtils.get_view_dirs_tr(view_rays_d)
                else:
                    view_rays_o_ndc, view_rays_d_ndc = CommonUtils.get_ndc_rays_tr(view_rays_o, view_rays_d, resolution, view_intrinsics, near=self.model_configs['near'])
                    view_dirs = CommonUtils.get_view_dirs_tr(view_rays_d_ndc)
            else:
                if not self.ndc:
                    view_dirs = CommonUtils.get_view_dirs_tr(rays_d)
                else:
                    view_dirs = CommonUtils.get_view_dirs_tr(rays_d_ndc)
            return_dict['view_dirs'] = view_dirs

        if self.predict_visibility and sec_views_vis:
            if self.training:
                poses = input_dict['common_data']['poses']
                num_frames = input_dict['num_frames']
                rays_o2 = []
                for i in range(num_frames-1):
                    other_image_id = i + (i >= image_id).long()
                    poses2i = poses[other_image_id]  # (n, 3, 4)
                    rays_o2i = poses2i[:, :3, 3]  # (n, 3)
                    rays_o2.append(rays_o2i)
                rays_o2 = torch.stack(rays_o2, dim=1)  # (nr, nf-1, 3)
            else:
                sec_poses = input_dict['common_data']['processed_secondary_poses']  # (nf-1, 4, 4)
                rays_o2 = sec_poses[:, :3, 3][None].repeat((num_rays, 1, 1))  # (nr, nf-1, 3)
            return_dict['rays_o2'] = rays_o2

        if self.coarse_model_needed:
            z_vals_coarse = self.get_z_vals_coarse(input_dict, return_dict, self.coarse_model)

            if not self.ndc:
                pts_coarse = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse[...,:,None] # [num_rays, num_samples, 3]
            else:
                pts_coarse = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_coarse[..., :, None]  # [num_rays, num_samples, 3]

            model_input_coarse = {
                'iter_num': input_dict.get('iter_num'),
                'pts': pts_coarse,
                'z_vals': z_vals_coarse,
                'rays_o': rays_o,
                'rays_d': rays_d,
            }
            if not self.ndc:
                model_input_coarse['z_vals'] = z_vals_coarse
            else:
                model_input_coarse['z_vals_ndc'] = z_vals_coarse
                model_input_coarse['rays_d_ndc'] = rays_d_ndc
            if self.configs['model']['coarse_model']['use_view_dirs']:
                model_input_coarse['view_dirs'] = view_dirs
            if self.coarse_model.predict_visibility and sec_views_vis:
                view_dirs2 = self.compute_other_view_dirs(z_vals_coarse, rays_o, rays_d, rays_o2)
                model_input_coarse['view_dirs2'] = view_dirs2

            model_outputs_coarse = self.coarse_model(model_input_coarse, retraw)
            weights_coarse = model_outputs_coarse['weights']

            return_dict['z_vals_coarse'] = z_vals_coarse
            for key in model_outputs_coarse:
                return_dict[f'{key}_coarse'] = model_outputs_coarse[key]

            if self.augmentations_needed and self.training and (mode != 'test_camera_params_optimization'):
                for augmented_model in self.augmented_models:
                    aug_name = augmented_model['name']
                    if augmented_model['coarse_model'] is not None:
                        aug_outputs_coarse = augmented_model['coarse_model'](model_input_coarse, retraw)
                        for key in aug_outputs_coarse:
                            return_dict[f'{aug_name}_{key}_coarse'] = aug_outputs_coarse[key]

        if self.fine_model_needed:
            z_vals_fine = self.get_z_vals_fine(z_vals_coarse, weights_coarse)
            if not self.ndc:
                pts_fine = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_fine[...,:,None]  # [num_rays, num_samples, 3]
            else:
                pts_fine = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_fine[..., :, None]  # [num_rays, num_samples, 3]

            model_input_fine = {
                'iter_num': input_dict.get('iter_num'),
                'pts': pts_fine,
                'z_vals': z_vals_fine,
                'rays_o': rays_o,
                'rays_d': rays_d,
            }
            if not self.ndc:
                model_input_fine['z_vals'] = z_vals_fine
            else:
                model_input_fine['z_vals_ndc'] = z_vals_fine
                model_input_fine['rays_d_ndc'] = rays_d_ndc
            if self.configs['model']['fine_model']['use_view_dirs']:
                model_input_fine['view_dirs'] = view_dirs
            if self.fine_model.predict_visibility and sec_views_vis:
                view_dirs2 = self.compute_other_view_dirs(z_vals_fine, rays_o, rays_d, rays_o2)
                model_input_fine['view_dirs2'] = view_dirs2

            model_outputs_fine = self.fine_model(model_input_fine, retraw)
            # weights_fine = model_outputs_fine['weights']

            return_dict['z_vals_fine'] = z_vals_fine
            for key in model_outputs_fine:
                return_dict[f'{key}_fine'] = model_outputs_fine[key]

            if self.augmentations_needed and self.training and (mode != 'test_camera_params_optimization'):
                for augmented_model in self.augmented_models:
                    aug_name = augmented_model['name']
                    if augmented_model['fine_model'] is not None:
                        aug_outputs_fine = augmented_model['fine_model'](model_input_fine, retraw)
                        for key in aug_outputs_fine:
                            return_dict[f'{aug_name}_{key}_fine'] = aug_outputs_fine[key]

        if not retraw:
            if self.coarse_model_needed:
                del return_dict['z_vals_coarse'], return_dict['alpha_coarse'], return_dict['visibility_coarse'], return_dict['weights_coarse']
            if self.fine_model_needed:
                del return_dict['z_vals_fine'], return_dict['alpha_fine'], return_dict['visibility_fine'], return_dict['weights_fine']
            if self.augmentations_needed and self.training and (mode != 'test_camera_params_optimization'):
                for augmented_model in self.augmented_models:
                    aug_name = augmented_model['name']
                    if augmented_model['coarse_model'] is not None:
                        del return_dict[f'{aug_name}_alpha_coarse'], return_dict[f'{aug_name}_visibility_coarse'], return_dict[f'{aug_name}_weights_coarse']
                    if augmented_model['fine_model'] is not None:
                        del return_dict[f'{aug_name}_alpha_fine'], return_dict[f'{aug_name}_visibility_fine'], return_dict[f'{aug_name}_weights_fine']

        return return_dict

    def get_z_vals_coarse(self, input_dict: dict, output_dict: dict, tensor_model = None):
        num_rays = input_dict['pixel_id'].shape[0]
        if not self.ndc:
            near, far = self.model_configs['near'], self.model_configs['far']
        else:
            near, far = self.model_configs['near_ndc'], self.model_configs['far_ndc']

        perturb = self.training and self.configs['model']['perturb']
        lindisp = self.configs['model']['lindisp']

        num_samples_coarse = tensor_model.num_samples.item()
        if self.ndc:
            t_vals = torch.linspace(0., 1., steps=num_samples_coarse).to(input_dict['pixel_id'].device)
            if not lindisp:
                z_vals_coarse = near * (1.-t_vals) + far * t_vals
            else:
                z_vals_coarse = 1./(1. / near * (1.-t_vals) + 1. / far * t_vals)

            z_vals_coarse = z_vals_coarse.expand([num_rays, num_samples_coarse])

            if perturb:
                # get intervals between samples
                mids = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
                upper = torch.cat([mids, z_vals_coarse[..., -1:]], -1)
                lower = torch.cat([z_vals_coarse[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals_coarse.shape).to(input_dict['pixel_id'].device)

                z_vals_coarse = lower + (upper - lower) * t_rand
        else:
            # TODO: understand this
            # Different mechanism is used to determine points outside the bounding box. Check the below
            # TensorBase > sample_ray(): https://github.com/apchenstu/TensoRF/blob/main/models/tensorBase.py#L277-L296
            # https://github.com/apchenstu/TensoRF/issues/24#issuecomment-1213256921
            # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html
            rays_o = output_dict['rays_o']
            rays_d = output_dict['rays_d']
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (tensor_model.bounding_box[1] - rays_o) / vec
            rate_b = (tensor_model.bounding_box[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

            rng = torch.arange(num_samples_coarse)[None].float()
            if perturb:
                rng = rng.repeat(rays_d.shape[-2], 1)
                rng += torch.rand_like(rng[:, [0]])
            step = tensor_model.step_size * rng.to(rays_o.device)
            z_vals_coarse = (t_min[..., None] + step)
        return z_vals_coarse

    def get_z_vals_fine(self, z_vals_coarse, weights_coarse):
        num_samples_fine = self.configs['model']['fine_model']['num_samples']
        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False

        z_vals_mid = .5 * (z_vals_coarse[...,1:] + z_vals_coarse[...,:-1])
        z_samples = self.sample_pdf(z_vals_mid, weights_coarse[...,1:-1], num_samples_fine, det=(not perturb))
        z_samples = z_samples.detach()

        z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
        return z_vals_fine

    def compute_other_view_dirs(self, z_vals, rays_o, rays_d, rays_o2):
        if self.ndc:
            near = self.model_configs['near']
            tn = -(near + rays_o[..., 2]) / rays_d[..., 2]
            z_vals = (((rays_o[..., None, 2] + tn[..., None] * rays_d[..., None, 2]) / (1 - z_vals + 1e-6)) - rays_o[..., None, 2]) / rays_d[..., None, 2]
        pts = rays_o[..., None, :] + z_vals[..., None] * rays_d[..., None, :]
        view_dirs_other = (pts[:, :, None] - rays_o2[..., None, :, :])  # (nr, ns, nf-1, 3)
        view_dirs_other = view_dirs_other / torch.norm(view_dirs_other, dim=-1, keepdim=True)
        return view_dirs_other

    # Hierarchical sampling (section 5.2)
    @staticmethod
    def sample_pdf(bins, weights, N_samples, det=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples).to(weights.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights.device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def run_model_modifications(self, iter_num):
        if self.coarse_model_needed:
            self.coarse_model.run_model_modifications(iter_num)
        if self.fine_model_needed:
            self.fine_model.run_model_modifications(iter_num)
        if self.augmentations_needed:
            for augmented_model in self.augmented_models:
                if augmented_model['coarse_model'] is not None:
                    augmented_model['coarse_model'].run_model_modifications(iter_num)
                if augmented_model['fine_model'] is not None:
                    augmented_model['fine_model'].run_model_modifications(iter_num)
        return

    def filter_pixel_ids(self, iter_num):
        if self.ndc or (iter_num != self.coarse_model.tensor_configs['alpha_mask_update_iters'][1]):
            return
        all_indices = torch.from_numpy(numpy.sort(self.train_data_preprocessor.preprocessed_data_dict['indices']))
        chunk = self.configs['model']['chunk']
        idx_chunks = torch.split(all_indices, chunk)
        tensor_model = self.coarse_model if self.coarse_model_needed else self.fine_model
        num_samples_coarse = tensor_model.num_samples.item()
        valid_index_masks = []
        for idx_chunk in idx_chunks:
            pixel_id = self.train_data_preprocessor.preprocessed_data_dict['nerf_data']['pixel_id'][idx_chunk]
            image_id = pixel_id[:, 0]
            intrinsics = self.intrinsics_learner(image_id)
            extrinsics = self.extrinsics_learner(image_id)
            rays_o, rays_d = CommonUtils.get_rays_tr(pixel_id, intrinsics, extrinsics, mip_nerf_used=False)
            rays_o[:, :1] *= -1  # TODO: check if moving this to get_rays_tr affects NeRF performance
            rays_d[:, :1] *= -1

            # Sample z_vals
            near, far = self.model_configs['near'], self.model_configs['far']
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (tensor_model.bounding_box[1] - rays_o) / vec
            rate_b = (tensor_model.bounding_box[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
            rng = torch.arange(num_samples_coarse)[None].float()
            step = tensor_model.step_size * rng.to(rays_o.device)
            z_vals_coarse = (t_min[..., None] + step)

            pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_coarse[..., :, None]  # [num_rays, num_samples, 3]
            pts_validity_mask_coarse = ((tensor_model.bounding_box[0] <= pts_coarse) & (pts_coarse <= tensor_model.bounding_box[1])).all(dim=-1)
            alphas = tensor_model.alpha_mask.sample_alpha(pts_coarse[pts_validity_mask_coarse])
            alpha_mask = alphas > 0
            valid_index_masks.append(alpha_mask)
        valid_index_masks = torch.cat(valid_index_masks)
        valid_indices = all_indices[valid_index_masks].cpu().numpy()
        numpy.random.shuffle(valid_indices)
        self.train_data_preprocessor.preprocessed_data_dict['indices'] = valid_indices
        return

    @staticmethod
    def merge_mini_batch_data(data_chunks: dict):
        merged_data = {}
        for key in data_chunks:
            if isinstance(data_chunks[key][0], torch.Tensor):
                merged_data[key] = torch.cat(data_chunks[key], dim=0)
            else:
                raise NotImplementedError
        return merged_data

    @classmethod
    def deep_dict_copy(cls, input_dict: dict):
        output_dict = {}
        for key in input_dict.keys():
            if isinstance(input_dict[key], dict):
                output_dict[key] = cls.deep_dict_copy(input_dict[key])
            else:
                output_dict[key] = input_dict[key]
        return output_dict

    @staticmethod
    def get_tensor_model(name, configs, tensor_configs, model_configs):
        decomposition_type = tensor_configs['decomposition_type']
        match decomposition_type:
            case "CandecompParafac":
                tensor_model = CpDecomposedTensor(name, configs, tensor_configs, model_configs)
            case "VectorMatrix":
                tensor_model = VmDecomposedTensor(name, configs, tensor_configs, model_configs)
            case _:
                raise RuntimeError(f'Unknown tensor decomposition: {decomposition_type}')
        return tensor_model


class PositionalEncoder:
    def __init__(self, configs):
        self.configs = configs
        self.out_dim = None
        self.pos_enc_fns = []
        self.create_pos_enc_fns()
        return

    def create_pos_enc_fns(self):
        d = self.configs['input_dims']
        out_dim = 0
        if self.configs['include_input']:
            self.pos_enc_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.configs['max_freq_log2']
        N_freqs = self.configs['num_freqs']

        if self.configs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.configs['periodic_fns']:
                self.pos_enc_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq.to(x.device)))
                out_dim += d

        self.out_dim = out_dim
        return

    def encode(self, inputs):
        return torch.cat([fn(inputs) for fn in self.pos_enc_fns], -1)


class LowRankTensor(torch.nn.Module):
    def __init__(self, name, configs, tensor_configs, model_configs):
        super().__init__()
        self.name = name
        self.configs = configs  # entire train configs
        self.tensor_configs = tensor_configs  # configs specific to this tensor grid instantiation
        self.model_configs = model_configs  # configs for the entire TensoRF model

        self.ndc = self.configs['data_loader']['ndc']
        self.predict_visibility = self.tensor_configs['predict_visibility']

        # Register variables that need to be saved as buffer
        self.register_buffer('bounding_box', torch.zeros(size=(2, 3)))
        self.register_buffer('resolution', torch.zeros(size=(3,)))
        self.register_buffer('num_samples', torch.tensor(0))
        self.register_buffer('bounding_box_size', torch.zeros(size=(3,)))
        self.register_buffer('voxel_length', torch.zeros(size=(3,)))
        self.register_buffer('step_size', torch.tensor(0))

        bounding_box = self.tensor_configs.get('bounding_box', self.model_configs['bounding_box'])
        self.bounding_box = torch.tensor(bounding_box)  # co-ordinates of the vertices of the tensor grid in world co-ordinates
        self.resolution = self.compute_resolution_in_voxels(tensor_configs['num_voxels_initial'], self.bounding_box)  # num voxels in each of the dimensions
        self.num_samples = None  # num samples along a ray
        self.bounding_box_size = None  # length of bounding box in each of the dimensions in world coordinates
        self.voxel_length = None  # length of the voxels in each of the dimension, in world coordinates
        self.step_size = None  # distance between samples in world coordinates

        self.alpha_mask = None
        self.density_predictor = None
        self.color_predictor = None
        self.optimizers = None

        self.update_tensor_params(self.resolution, self.bounding_box)

        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)
        return

    def build_model(self):
        self.build_tensor()
        self.density_predictor = self.get_density_predictor()
        self.color_predictor = self.get_color_predictor()
        return

    @staticmethod
    def compute_resolution_in_voxels(num_voxels: int, bounding_box: torch.Tensor) -> torch.Tensor:
        """
        Computes the resolution of the tensor grid in terms of number of voxels in each of the dimension.
        :param num_voxels: total number of voxels in the tensor grid
        :param bounding_box: co-ordinates of the vertices of the tensor grid in world co-ordinates
        :return:
        """
        xyz_min, xyz_max = bounding_box
        num_dims = len(xyz_min)
        # Voxels are assumed to be of cubical shape
        voxel_size = ((xyz_max - xyz_min).prod() / num_voxels).pow(1 / num_dims)  # length of each side of the voxels
        voxel_resolution = ((xyz_max - xyz_min) / voxel_size).long()  # number of voxels along each dimension
        return voxel_resolution

    def compute_num_samples(self, resolution: torch.Tensor, num_voxels_per_sample: float) -> torch.Tensor:
        """
        Computes the total number of samples required for the ray that passes diagonally through the tensor grid.
        :param resolution: num voxels in each of the dimensions
        :param num_voxels_per_sample: Inverse of the number of samples required per voxel, when the ray is traversing
                                      along an edge of the voxel. Recall that voxels are assumed to be cubic.
        :return:
        """
        num_samples = (torch.linalg.norm(resolution.float()) / num_voxels_per_sample).round().long()
        num_samples = min(self.tensor_configs['num_samples_max'], num_samples)
        return num_samples

    def update_tensor_params(self, new_resolution, new_bounding_box):
        """
        Updates resolution,
        :param new_resolution: num voxels in each of the dimensions
        :return:
        """
        self.bounding_box = new_bounding_box.float()
        self.bounding_box_size = self.bounding_box[1] - self.bounding_box[0]

        self.resolution = new_resolution.long()
        self.voxel_length = self.bounding_box_size / (self.resolution - 1)
        self.step_size = torch.mean(self.voxel_length) * self.tensor_configs['num_voxels_per_sample']
        self.num_samples = self.compute_num_samples(self.resolution, self.tensor_configs['num_voxels_per_sample'])
        return

    @abc.abstractmethod
    def build_tensor(self):
        pass

    def get_density_predictor(self):
        match self.tensor_configs['density_predictor']:
            case 'ReLU':
                density_predictor = F.relu
            case 'SoftPlus':
                density_predictor = lambda x: F.softplus(x + self.tensor_configs['density_offset'])
            case _:
                raise NotImplementedError
        return density_predictor

    def get_color_predictor(self):
        match self.tensor_configs['color_predictor']:
            case 'MLP_PositionalEncoding':
                raise NotImplementedError
            case 'MLP_Features':
                color_predictor = MlpFeaturesColorPredictor(self.configs, self.tensor_configs, self.model_configs,
                                                            self.tensor_configs['features_dimension_color'],
                                                            self.tensor_configs['features_positional_encoding_degree'],
                                                            self.tensor_configs['views_positional_encoding_degree'],
                                                            self.tensor_configs['num_units_color_predictor'])
            case 'MLP':
                raise NotImplementedError
            case 'SphericalHarmonics':
                raise NotImplementedError
            case 'RGB':
                raise NotImplementedError
            case _:
                raise NotImplementedError
        return color_predictor

    def forward(self, input_dict: dict, retraw: bool):
        input_dict = input_dict.copy()

        pts = input_dict['pts']
        pts_validity_mask = ((self.bounding_box[0] <= pts) & (pts <= self.bounding_box[1])).all(dim=-1)

        if self.alpha_mask is not None:
            alphas = self.alpha_mask.sample_alpha(pts[pts_validity_mask])
            alpha_mask = alphas > 0
            pts_validity_mask[pts_validity_mask.clone()] &= alpha_mask

        # Pre-process the inputs
        raw_output_dict, output_dict = {}, {}
        pts_normalized = self.normalize_points(pts)
        density_input_dict = {
            'pts': pts_normalized,
            'pts_validity_mask': pts_validity_mask,
        }

        # Read Volume Density
        sigma = self.get_volume_density(density_input_dict)
        raw_output_dict['sigma'] = sigma
        vr_output_dict = self.get_volume_rendering_weights(input_dict, raw_output_dict)
        output_dict.update(vr_output_dict)

        surface_pts_mask = output_dict['weights'] > self.tensor_configs['ray_marching_weight_threshold']
        color_input_dict = {
            'pts': pts_normalized,
            'pts_validity_mask': surface_pts_mask,
        }

        if self.tensor_configs['use_view_dirs']:
            view_dirs = input_dict['view_dirs']
            if view_dirs.ndim == 2:
                view_dirs = view_dirs[:, None].expand(input_dict['pts'].shape)
            color_input_dict['view_dirs'] = view_dirs

            if self.tensor_configs['predict_visibility'] and ('view_dirs2' in input_dict):
                view_dirs2 = input_dict['view_dirs2']
                color_input_dict['view_dirs2'] = view_dirs2

        # Read Color
        rgb = self.get_color(color_input_dict)
        raw_output_dict['rgb'] = rgb
        rgb_map = self.volume_render(rgb, output_dict['weights'])
        if self.configs['model']['white_bkgd'] or (self.training and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1 - output_dict['acc'][..., None])
        output_dict['rgb'] = rgb_map

        # Read visibility if required
        if self.predict_visibility:
            visibility, visibility2 = self.get_visibility(input_dict, raw_output_dict, output_dict)
            raw_output_dict['visibility'] = visibility
            raw_output_dict['visibility2'] = visibility2
            visibility2_map = self.volume_render(visibility2, output_dict['weights'])
            output_dict['visibility2'] = visibility2_map

        if retraw:
            for key in raw_output_dict.keys():
                output_dict[f'raw_{key}'] = raw_output_dict[key]
        return output_dict

    def normalize_points(self, pts: torch.Tensor):
        norm_pts = ((pts - self.bounding_box[0]) / self.bounding_box_size) * 2 - 1
        return norm_pts

    def get_volume_rendering_weights(self, input_dict: dict, raw_output_dict):
        z_vals = input_dict.get('z_vals')
        z_vals_ndc = input_dict.get('z_vals_ndc')
        rays_o = input_dict['rays_o']
        rays_d = input_dict['rays_d']
        rays_d_ndc = input_dict.get('rays_d_ndc')
        sigma = raw_output_dict['sigma'][..., 0]  # (num_rays, num_samples)

        if not self.ndc:
            inf_depth = 1e10
            z_vals1 = torch.cat([z_vals, torch.Tensor([inf_depth]).to(rays_d.device).expand(z_vals[...,:1].shape)], -1)
            z_dists = z_vals1[...,1:] - z_vals1[..., :-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d[..., None, :], dim=-1)
        else:
            inf_depth = 1
            z_vals1 = torch.cat([z_vals_ndc, torch.Tensor([inf_depth]).to(rays_d.device).expand(z_vals_ndc[...,:1].shape)], -1)
            z_dists = z_vals1[...,1:] - z_vals1[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d_ndc[...,None,:], dim=-1)

        # Why distance_scale is required https://github.com/apchenstu/TensoRF/issues/14
        alpha = 1. - torch.exp(-sigma * delta * self.tensor_configs['distance_scale'])  # [N_rays, N_samples]
        visibility = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(rays_d.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * visibility
        acc_map = torch.sum(weights, dim=-1)

        if not self.ndc:
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)
        else:
            depth_map_ndc = torch.sum(weights * z_vals_ndc, dim=-1) / (acc_map + 1e-6)
            depth_var_map_ndc = torch.sum(weights * torch.square(z_vals_ndc - depth_map_ndc[..., None]), dim=-1)
            z_vals = CommonUtils.convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d)
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)

        return_dict = {
            'acc': acc_map,
            'alpha': alpha,
            'visibility': visibility,
            'weights': weights,
            'depth': depth_map,
            'depth_var': depth_var_map,
        }

        if self.ndc:
            return_dict['depth_ndc'] = depth_map_ndc
            return_dict['depth_var_ndc'] = depth_var_map_ndc
        return return_dict

    @staticmethod
    def volume_render(raw_value, weights):
        rendered_value = torch.sum(weights[..., None] * raw_value, dim=-2)
        return rendered_value

    def run_model_modifications(self, iter_num):
        if self.training and (iter_num in self.tensor_configs['alpha_mask_update_iters']):
            bounding_box_new = self.update_alpha_mask(iter_num)
            if iter_num == self.tensor_configs['alpha_mask_update_iters'][0]:
                self.shrink_tensor(bounding_box_new)

        if self.training and (iter_num in self.tensor_configs['tensor_upsampling_iters']):
            self.upsample_model_resolution(iter_num)
            self.reconfigure_optimizer()
        return

    def upsample_model_resolution(self, iter_num):
        num_voxels_new = self.get_new_num_voxels(iter_num)
        self.resolution = self.compute_resolution_in_voxels(num_voxels_new, self.bounding_box)
        self.update_tensor_params(self.resolution, self.bounding_box)
        return

    def get_new_num_voxels(self, iter_num):
        if iter_num not in self.tensor_configs['tensor_upsampling_iters']:
            raise RuntimeError('LowRankTensor.get_new_num_voxels() called at invalid iteration number')
        upsample_num = self.tensor_configs['tensor_upsampling_iters'].index(iter_num) + 1
        num_upsamples = len(self.tensor_configs['tensor_upsampling_iters'])
        log_num_voxels_initial = numpy.log(self.tensor_configs['num_voxels_initial'])
        log_num_voxels_final = numpy.log(self.tensor_configs['num_voxels_final'])
        log_num_voxels = log_num_voxels_initial + (log_num_voxels_final - log_num_voxels_initial) * upsample_num / num_upsamples
        num_voxels = int(numpy.round(numpy.exp(log_num_voxels)))
        return num_voxels

    def update_alpha_mask(self, iter_num):
        grid_size = tuple(self.resolution.detach().cpu().numpy())
        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0]),
            torch.linspace(0, 1, grid_size[1]),
            torch.linspace(0, 1, grid_size[2]),
        ), -1).to(self.resolution.device)
        dense_xyz = self.bounding_box[0] * (1 - samples) + self.bounding_box[1] * samples

        alpha = self.compute_alpha(dense_xyz.view(-1, 3), self.step_size).view(grid_size)

        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(grid_size[::-1])
        alpha[alpha >= self.tensor_configs['alpha_mask_threshold']] = 1
        alpha[alpha < self.tensor_configs['alpha_mask_threshold']] = 0

        self.alpha_mask = AlphaGridMask(alpha, self.bounding_box)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        bounding_box_new = torch.stack((xyz_min, xyz_max))
        return bounding_box_new

    def compute_alpha(self, xyz_locs, length=1):
        if self.alpha_mask is not None:
            alphas = self.alpha_mask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_normalized = self.normalize_points(xyz_locs)
            sigma_input_dict = {
                'pts': xyz_normalized,
                'pts_validity_mask': alpha_mask,
            }
            sigma = self.get_volume_density(sigma_input_dict)

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    def shrink_tensor(self, bounding_box_new):
        xyz_min, xyz_max = bounding_box_new
        t_l, b_r = (xyz_min - self.bounding_box[0]) / self.voxel_length, (xyz_max - self.bounding_box[0]) / self.voxel_length
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.resolution]).amin(0)

        if not torch.equal(self.alpha_mask.resolution, self.resolution):
            t_l_r, b_r_r = t_l / (self.resolution - 1), (b_r - 1) / (self.resolution - 1)
            correct_bounding_box = torch.zeros_like(bounding_box_new)
            correct_bounding_box[0] = (1 - t_l_r) * self.bounding_box[0] + t_l_r * self.bounding_box[1]
            correct_bounding_box[1] = (1 - b_r_r) * self.bounding_box[0] + b_r_r * self.bounding_box[1]
            bounding_box_new = correct_bounding_box

        resolution_new = b_r - t_l
        self.update_tensor_params(resolution_new, bounding_box_new)
        return t_l, b_r

    def reconfigure_optimizer(self):
        optimizer = self.optimizers['optimizer_nerf']  # TODO: rename this to just main
        # TODO: Change configs to accept a dict of optimizers instead of a list of optimizers
        optimizer_configs = next(filter(lambda opt_configs: opt_configs['name'] == 'optimizer_main', self.configs['optimizers']))

        # Clear the optimizer params corresponding to this model
        # https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814/7
        # optimizer.param_groups.clear()
        # optimizer.state.clear()
        param_groups = self.get_trainable_parameters(optimizer_configs)
        for param_group in param_groups:
            param_index = 0
            for i, param_group1 in enumerate(optimizer.param_groups):
                num_params = len(param_group1['params'])
                if param_group1['name'] == param_group['name']:
                    del optimizer.param_groups[i]
                    for j in range(num_params):
                        if param_index < len(list(optimizer.state.keys())):
                            del optimizer.state[list(optimizer.state.keys())[param_index]]
                        else:
                            # TODO: remove this if it never prints
                            print(f'Unable to delete item at {param_index} since list has only {len(list(optimizer.state.keys()))} elements')
                else:
                    param_index += num_params

        # Add new params
        for param_group in param_groups:
            optimizer.add_param_group(param_group)
        return

    def load_state_dict_pre_hook(self, model_state_dict, prefix, *args, **kwargs):
        """
        When loading pre-trained model, AlphaGridMask will be initialized to None. Creating this is necessary for 
        weights loading to work properly. This is actually a repetition, but I don't see a better way.
        However, this makes uses of pre-hook which can be changed at any point of time. If the code breaks due to this
        change in PyTorch, update the code accordingly. I can see three cases in which the current code can break.
          1. the hook name is changed - update the hook name accordingly.
          2. Customizing load_state_dict() is made available in which case do the following in the customized method
            (https://discuss.pytorch.org/t/customize-state-dict-and-load-state-dict-pytorch/188648?u=nagabhushansn95)
          3. PyTorch allows loading shape mismatched weights - in which case AlphaGridMask can be initialized to zero tensor 
            (https://github.com/pytorch/pytorch/issues/40859)
        """
        alpha_volume = model_state_dict[f'{prefix}alpha_mask.alpha_volume'].float()
        bounding_box = model_state_dict[f'{prefix}alpha_mask.bounding_box']
        self.alpha_mask = AlphaGridMask(alpha_volume, bounding_box)
        return


class CpDecomposedTensor(LowRankTensor):
    def __init__(self, name, configs, tensor_configs, model_configs):
        super().__init__(name, configs, tensor_configs, model_configs)

        # Define the axes/dimensions for the Candecomp-Parafac decomposition
        self.vector_axes = [2, 1, 0]

        # Create the vectors and matrices for the vector-matrix decomposed tensor
        self.vectors_density = None
        self.vectors_color = None
        self.basis_matrix_color = None  # basis vectors for the color channels

        self.build_model()
        return

    def build_tensor(self):
        """
        Creates the decomposed components corresponding to the tensor representing the scene
        :return:
        """
        self.vectors_density = self.create_decomposed_tensor(self.tensor_configs['num_components_density'], self.resolution, init_params_scale=0.1)
        self.vectors_color = self.create_decomposed_tensor(self.tensor_configs['num_components_color'], self.resolution, init_params_scale=0.1)
        self.basis_matrix_color = torch.nn.Linear(sum(self.tensor_configs['num_components_color']),self.tensor_configs['features_dimension_color'], bias=False)
        return

    def create_decomposed_tensor(self, num_components, resolution, init_params_scale):
        vectors = []
        for i in range(len(self.vector_axes)):  # i in [0, 1, 2]
            vec_axis = self.vector_axes[i]
            vector = torch.nn.Parameter(init_params_scale * torch.randn((1, num_components[0], resolution[vec_axis], 1)))  # (1, r_z, z, 1)
            vectors.append(vector)
        vectors_params = torch.nn.ParameterList(vectors)
        return vectors_params

    def get_trainable_parameters(self, optimizer_configs):
        tensor_trainable_params = torch.nn.ParameterList()
        network_trainable_params = torch.nn.ParameterList()

        tensor_trainable_params.extend(self.vectors_density)
        tensor_trainable_params.extend(self.vectors_color)
        network_trainable_params.extend(self.basis_matrix_color.parameters())

        if isinstance(self.density_predictor, torch.nn.Module):
            network_trainable_params.extend(self.density_predictor.parameters())
        if isinstance(self.color_predictor, torch.nn.Module):
            network_trainable_params.extend(self.color_predictor.parameters())

        trainable_params = [
            {
                'name': f'{self.name}_tensor_params',
                'params': tensor_trainable_params,
                'lr': optimizer_configs['lr_initial_tensor']
            },
            {
                'name': f'{self.name}_network_params',
                'params': network_trainable_params,
                'lr': optimizer_configs['lr_initial_network']
            },
        ]
        return trainable_params

    def load_state_dict_pre_hook(self, model_state_dict, prefix, *args, **kwargs):
        """
        When loading pre-trained model, the resolution of the vectors may be different than those after initialization.
        So, these have to be corrected before loading the weights.
        However, this makes uses of pre-hook which can be changed at any point of time. If the code breaks due to this
        change in PyTorch, update the code accordingly. I can see three cases in which the current code can break.
          1. the hook name is changed - update the hook name accordingly.
          2. Customizing load_state_dict() is made available in which case do the following in the customized method
            (https://discuss.pytorch.org/t/customize-state-dict-and-load-state-dict-pytorch/188648?u=nagabhushansn95)
          3. PyTorch allows loading shape mismatched weights - in which case there is no need for the following code. It can
            be completely omitted (https://github.com/pytorch/pytorch/issues/40859)
        """
        super().load_state_dict_pre_hook(model_state_dict, prefix, *args, **kwargs)
        new_resolution = model_state_dict[f'{prefix}resolution']
        self.vectors_density = self.upsample_vectors(self.vectors_density, new_resolution)
        self.vectors_color = self.upsample_vectors(self.vectors_color, new_resolution)
        return

    def get_volume_density(self, input_dict: dict):
        # Read Sigma for valid points only
        pts = input_dict['pts']
        # validity_mask_default is not used currently. Added as a fail-safe for future usage.
        validity_mask_default = torch.ones(size=pts.shape[:2], dtype=bool).to(pts.device)
        validity_mask = input_dict.get('pts_validity_mask', validity_mask_default)
        sigma = torch.zeros([*pts.shape[:-1], 1]).to(pts)
        valid_pts = pts[validity_mask]

        if valid_pts.any():
            coordinate_line = torch.stack((valid_pts[..., self.vector_axes[0]], valid_pts[..., self.vector_axes[1]], valid_pts[..., self.vector_axes[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

            line_coef_point = F.grid_sample(self.vectors_density[0], coordinate_line[[0]], align_corners=True).view(-1, *valid_pts.shape[:1])
            line_coef_point = line_coef_point * F.grid_sample(self.vectors_density[1], coordinate_line[[1]], align_corners=True).view(-1, *valid_pts.shape[:1])
            line_coef_point = line_coef_point * F.grid_sample(self.vectors_density[2], coordinate_line[[2]], align_corners=True).view(-1, *valid_pts.shape[:1])
            sigma_features = torch.sum(line_coef_point, dim=0)
            valid_sigma = self.density_predictor(sigma_features)[..., None]
            sigma[validity_mask] = valid_sigma
        return sigma

    def get_color(self, input_dict: dict):
        # Read rgb for surface points only
        pts = input_dict['pts']
        # validity_mask_default is not used currently. Added as a fail-safe for future usage.
        validity_mask_default = torch.ones(size=pts.shape[:2], dtype=bool).to(pts.device)
        validity_mask = input_dict.get('pts_validity_mask', validity_mask_default)
        rgb = torch.zeros([*pts.shape[:-1], 3]).to(pts)
        valid_pts = pts[validity_mask]

        if validity_mask.any():
            coordinate_line = torch.stack((valid_pts[..., self.vector_axes[0]], valid_pts[..., self.vector_axes[1]], valid_pts[..., self.vector_axes[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

            line_coef_point = F.grid_sample(self.vectors_color[0], coordinate_line[[0]], align_corners=True).view(-1, *valid_pts.shape[:1])
            line_coef_point = line_coef_point * F.grid_sample(self.vectors_color[1], coordinate_line[[1]], align_corners=True).view(-1, *valid_pts.shape[:1])
            line_coef_point = line_coef_point * F.grid_sample(self.vectors_color[2], coordinate_line[[2]], align_corners=True).view(-1, *valid_pts.shape[:1])
            color_features = self.basis_matrix_color(line_coef_point.T)
            color_pred_input_dict = {
                'features': color_features,
            }
            if self.tensor_configs['view_dependent_color']:
                color_pred_input_dict['view_dirs'] = input_dict['view_dirs'][validity_mask]
            color_pred_output_dict = self.color_predictor(color_pred_input_dict)
            surface_rgb = color_pred_output_dict['color']
            rgb[validity_mask] = surface_rgb
        return rgb

    def get_visibility(self, input_batch: dict, raw_output_dict: dict, output_dict: dict):
        raise NotImplementedError

    def upsample_model_resolution(self, iter_num):
        super().upsample_model_resolution(iter_num)
        # upsample the tensor elements
        self.vectors_density = self.upsample_vectors(self.vectors_density, self.resolution)
        self.vectors_color = self.upsample_vectors(self.vectors_color, self.resolution)
        return

    def upsample_vectors(self, vectors, resolution_new):
        upsampled_vectors = []
        for i in range(len(self.vector_axes)):
            vector_axis = self.vector_axes[i]
            upsampled_vector = torch.nn.Parameter(F.interpolate(vectors[i].data,
                                                                size=(resolution_new[vector_axis], 1),
                                                                mode='bilinear', align_corners=True))
            upsampled_vectors.append(upsampled_vector)

        vectors_params = torch.nn.ParameterList(upsampled_vectors)
        return vectors_params

    def shrink_tensor(self, bounding_box_new):
        t_l, b_r = super().shrink_tensor(bounding_box_new)
        for i in range(len(self.vector_axes)):
            mode0 = self.vector_axes[i]
            self.vectors_density[i] = torch.nn.Parameter(
                self.vectors_density[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            self.vectors_color[i] = torch.nn.Parameter(
                self.vectors_color[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
        return


class VmDecomposedTensor(LowRankTensor):
    def __init__(self, name, configs, tensor_configs, model_configs):
        super().__init__(name, configs, tensor_configs, model_configs)

        # Define the axes/dimensions for the vector-matrix decomposition
        self.matrix_axes = [[0, 1], [0, 2], [1, 2]]
        self.vector_axes = [2, 1, 0]

        # Create the vectors and matrices for the vector-matrix decomposed tensor
        self.matrices_density = None
        self.vectors_density = None
        self.matrices_color = None
        self.vectors_color = None
        self.basis_matrix_color = None  # basis vectors for the color channels

        self.build_model()
        return

    def build_tensor(self):
        """
        Creates the decomposed components corresponding to the tensor representing the scene
        :return:
        """
        self.matrices_density, self.vectors_density = self.create_decomposed_tensor(self.tensor_configs['num_components_density'], self.resolution, init_params_scale = 0.1)
        self.matrices_color, self.vectors_color = self.create_decomposed_tensor(self.tensor_configs['num_components_color'], self.resolution, init_params_scale = 0.1)
        self.basis_matrix_color = torch.nn.Linear(sum(self.tensor_configs['num_components_color']), self.tensor_configs['features_dimension_color'], bias=False)
        return

    def create_decomposed_tensor(self, num_components, resolution, init_params_scale):
        matrices, vectors = [], []
        for i in range(len(self.vector_axes)):  # i in [0, 1, 2]
            mat_axis0, mat_axis1 = self.matrix_axes[i]
            vec_axis = self.vector_axes[i]
            matrix = torch.nn.Parameter(init_params_scale * torch.randn((1, num_components[i], resolution[mat_axis1], resolution[mat_axis0])))  # (1, r_xy, y, x)
            vector = torch.nn.Parameter(init_params_scale * torch.randn((1, num_components[i], resolution[vec_axis], 1)))  # (1, r_z, z, 1)
            matrices.append(matrix)
            vectors.append(vector)
        matrices_params = torch.nn.ParameterList(matrices)
        vectors_params = torch.nn.ParameterList(vectors)
        return matrices_params, vectors_params

    def get_trainable_parameters(self, optimizer_configs):
        tensor_trainable_params = torch.nn.ParameterList()
        network_trainable_params = torch.nn.ParameterList()

        tensor_trainable_params.extend(self.vectors_density)
        tensor_trainable_params.extend(self.matrices_density)
        tensor_trainable_params.extend(self.vectors_color)
        tensor_trainable_params.extend(self.matrices_color)
        network_trainable_params.extend(self.basis_matrix_color.parameters())

        if isinstance(self.density_predictor, torch.nn.Module):
            network_trainable_params.extend(self.density_predictor.parameters())
        if isinstance(self.color_predictor, torch.nn.Module):
            network_trainable_params.extend(self.color_predictor.parameters())

        trainable_params = [
            {
                'name': f'{self.name}_tensor_params',
                'params': tensor_trainable_params,
                'lr': optimizer_configs['lr_initial_tensor']
            },
            {
                'name': f'{self.name}_network_params',
                'params': network_trainable_params,
                'lr': optimizer_configs['lr_initial_network']
            },
        ]
        return trainable_params

    def load_state_dict_pre_hook(self, model_state_dict, prefix, *args, **kwargs):
        """
        When loading pre-trained model, the resolution of the vectors and matrices may be different than those after
        initialization. So, these have to be corrected before loading the weights.
        However, this makes uses of pre-hook which can be changed at any point of time. If the code breaks due to this
        change in PyTorch, update the code accordingly. I can see three cases in which the current code can break.
          1. the hook name is changed - update the hook name accordingly.
          2. Customizing load_state_dict() is made available in which case do the following in the customized method
            (https://discuss.pytorch.org/t/customize-state-dict-and-load-state-dict-pytorch/188648?u=nagabhushansn95)
          3. PyTorch allows loading shape mismatched weights - in which case there is no need for the following code. It can
            be completely omitted (https://github.com/pytorch/pytorch/issues/40859)
        """
        super().load_state_dict_pre_hook(model_state_dict, prefix, *args, **kwargs)
        new_resolution = model_state_dict[f'{prefix}resolution']
        self.matrices_density, self.vectors_density = self.upsample_vectors_and_matrices(self.matrices_density, self.vectors_density, new_resolution)
        self.matrices_color, self.vectors_color = self.upsample_vectors_and_matrices(self.matrices_color, self.vectors_color, new_resolution)
        return

    def get_volume_density(self, input_dict: dict):
        # Read Sigma for valid points only
        pts = input_dict['pts']
        # validity_mask_default is not used currently. Added as a fail-safe for future usage.
        validity_mask_default = torch.ones(size=pts.shape[:2], dtype=bool).to(pts.device)
        validity_mask = input_dict.get('pts_validity_mask', validity_mask_default)
        sigma = torch.zeros([*pts.shape[:-1], 1]).to(pts)
        valid_pts = pts[validity_mask]

        if valid_pts.any():
            coordinate_plane = torch.stack((valid_pts[..., self.matrix_axes[0]], valid_pts[..., self.matrix_axes[1]],
                                            valid_pts[..., self.matrix_axes[2]])).detach().view(3, -1, 1, 2)
            coordinate_line = torch.stack((valid_pts[..., self.vector_axes[0]], valid_pts[..., self.vector_axes[1]],
                                           valid_pts[..., self.vector_axes[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

            sigma_features = torch.zeros((valid_pts.shape[0],), device=valid_pts.device)
            for idx_plane in range(len(self.matrices_density)):
                plane_coef_point = F.grid_sample(self.matrices_density[idx_plane], coordinate_plane[[idx_plane]],
                                                 align_corners=True).view(-1, *valid_pts.shape[:1])
                line_coef_point = F.grid_sample(self.vectors_density[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *valid_pts.shape[:1])
                sigma_features = sigma_features + torch.sum(plane_coef_point * line_coef_point, dim=0)
            valid_sigma = self.density_predictor(sigma_features)[..., None]
            sigma[validity_mask] = valid_sigma
        return sigma

    def get_color(self, input_dict: dict):
        # Read rgb for surface points only
        pts = input_dict['pts']
        # validity_mask_default is not used currently. Added as a fail-safe for future usage.
        validity_mask_default = torch.ones(size=pts.shape[:2], dtype=bool).to(pts.device)
        validity_mask = input_dict.get('pts_validity_mask', validity_mask_default)
        rgb = torch.zeros([*pts.shape[:-1], 3]).to(pts)
        valid_pts = pts[validity_mask]

        if validity_mask.any():
            # plane + line basis
            coordinate_plane = torch.stack((valid_pts[..., self.matrix_axes[0]], valid_pts[..., self.matrix_axes[1]], valid_pts[..., self.matrix_axes[2]])).detach().view(3, -1, 1, 2)
            coordinate_line = torch.stack((valid_pts[..., self.vector_axes[0]], valid_pts[..., self.vector_axes[1]], valid_pts[..., self.vector_axes[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

            plane_coef_point,line_coef_point = [],[]
            for idx_plane in range(len(self.matrices_color)):
                plane_coef_point.append(F.grid_sample(self.matrices_color[idx_plane], coordinate_plane[[idx_plane]],
                                                      align_corners=True).view(-1, *valid_pts.shape[:1]))
                line_coef_point.append(F.grid_sample(self.vectors_color[idx_plane], coordinate_line[[idx_plane]],
                                                     align_corners=True).view(-1, *valid_pts.shape[:1]))
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
            color_features = self.basis_matrix_color((plane_coef_point * line_coef_point).T)
            color_pred_input_dict = {
                'features': color_features,
            }
            if self.tensor_configs['view_dependent_color']:
                color_pred_input_dict['view_dirs'] = input_dict['view_dirs'][validity_mask]
            color_pred_output_dict = self.color_predictor(color_pred_input_dict)
            surface_rgb = color_pred_output_dict['color']
            rgb[validity_mask] = surface_rgb
        return rgb

    def get_visibility(self, input_batch: dict, raw_output_dict: dict, output_dict: dict):
        raise NotImplementedError

    def upsample_model_resolution(self, iter_num):
        super().upsample_model_resolution(iter_num)
        # upsample the tensor elements
        self.matrices_density, self.vectors_density = self.upsample_vectors_and_matrices(self.matrices_density, self.vectors_density, self.resolution)
        self.matrices_color, self.vectors_color = self.upsample_vectors_and_matrices(self.matrices_color, self.vectors_color, self.resolution)
        return

    def upsample_vectors_and_matrices(self, matrices, vectors, resolution_new):
        upsampled_matrices, upsampled_vectors = [], []
        for i in range(len(self.matrix_axes)):
            matrix_axis0, matrix_axis1 = self.matrix_axes[i]
            vector_axis = self.vector_axes[i]

            upsampled_matrix = torch.nn.Parameter(F.interpolate(matrices[i].data,
                                                                size=(resolution_new[matrix_axis1].item(), resolution_new[matrix_axis0].item()),
                                                                mode='bilinear', align_corners=True))
            upsampled_vector = torch.nn.Parameter(F.interpolate(vectors[i].data,
                                                                size=(resolution_new[vector_axis].item(), 1),
                                                                mode='bilinear', align_corners=True))
            upsampled_matrices.append(upsampled_matrix)
            upsampled_vectors.append(upsampled_vector)

        matrices_params = torch.nn.ParameterList(upsampled_matrices)
        vectors_params = torch.nn.ParameterList(upsampled_vectors)
        return matrices_params, vectors_params

    def shrink_tensor(self, bounding_box_new):
        t_l, b_r = super().shrink_tensor(bounding_box_new)
        for i in range(len(self.vector_axes)):
            mode0 = self.vector_axes[i]
            self.vectors_density[i] = torch.nn.Parameter(
                self.vectors_density[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            self.vectors_color[i] = torch.nn.Parameter(
                self.vectors_color[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            mode0, mode1 = self.matrix_axes[i]
            self.matrices_density[i] = torch.nn.Parameter(
                self.matrices_density[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            )
            self.matrices_color[i] = torch.nn.Parameter(
                self.matrices_color[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]]
            )
        return


class AlphaGridMask(torch.nn.Module):
    def __init__(self, alpha_volume, bounding_box):
        super(AlphaGridMask, self).__init__()

        # Register buffers
        self.register_buffer('alpha_volume', torch.tensor(0, dtype=bool))
        self.register_buffer('bounding_box', torch.zeros(size=(2, 3)))
        self.register_buffer('bounding_box_size', torch.zeros(size=(3, )))
        self.register_buffer('resolution', torch.zeros(size=(3, ), dtype=torch.long))

        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])  # TODO: understand why the mask is flipped
        self.bounding_box = bounding_box
        self.bounding_box_size = self.bounding_box[1] - self.bounding_box[0]
        self.resolution = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(alpha_volume.device)

        self._register_state_dict_hook(self.save_state_dict_post_hook)
        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)
        return

    def sample_alpha(self, pts):
        pts_normalized = self.normalize_points(pts)
        alpha_vals = F.grid_sample(self.alpha_volume, pts_normalized.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_points(self, pts):
        norm_points = ((pts - self.bounding_box[0]) / self.bounding_box_size) * 2 - 1
        return norm_points
    
    def save_state_dict_post_hook(self, obj, model_state_dict, prefix, *args, **kwargs):
        """
        alpha_volume is basically a bool value. But can't be stored as bool because grid_sample fails on bool array.
        Saving the 3D grid as float takes more memory. Converting it to bool saves some memory. 
        However, this makes uses of post-hook which can be changed at any point of time. If the code breaks due to this
        change in PyTorch, update the code accordingly or simply disable this. It is only useful to save some MBs.
        """
        model_state_dict[f'{prefix}alpha_volume'] = model_state_dict[f'{prefix}alpha_volume'].bool()
        return 

    def load_state_dict_pre_hook(self, model_state_dict, prefix, *args, **kwargs):
        """
        pre-hook to convert the boolean alpha_volume to float. 
        If code breaks due to change in PyTorch, update the code accordingly or simply disable this.
        """
        model_state_dict[f'{prefix}alpha_volume'] = model_state_dict[f'{prefix}alpha_volume'].float()
        return


class MlpFeaturesColorPredictor(torch.nn.Module):
    def __init__(self, configs, tensor_configs, model_configs, features_dim, features_pe_degree, views_pe_degree, num_units):
        super().__init__()
        self.configs = configs
        self.tensor_configs = tensor_configs
        self.model_configs = model_configs
        self.features_dimension = features_dim
        self.features_pe_degree = features_pe_degree
        self.views_pe_degree = views_pe_degree
        self.num_units = num_units

        self.input_dim = 0
        self.features_pos_enc_fn, features_input_dim = self.get_positional_encoder(self.tensor_configs['features_positional_encoding_degree'], features_dim)
        self.input_dim += features_input_dim
        if self.tensor_configs['use_view_dirs']:
            self.views_pos_enc_fn, views_input_dim = self.get_positional_encoder(self.tensor_configs['views_positional_encoding_degree'], input_dim=3)
            self.input_dim += views_input_dim
        # views_dim = 3
        # self.input_dim = (features_dim * (2 * features_pe_degree + 1)) + (views_dim * (views_pe_degree + 1))
        layer1 = torch.nn.Linear(self.input_dim, num_units)
        layer2 = torch.nn.Linear(num_units, num_units)
        layer3 = torch.nn.Linear(num_units, 3)
        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3, torch.nn.Sigmoid())
        torch.nn.init.constant_(self.mlp[-2].bias, 0)
        return

    @staticmethod
    def get_positional_encoder(degree, input_dim):
        pos_enc_kwargs = {
            'include_input': True,
            'input_dims': input_dim,
            'max_freq_log2': degree - 1,
            'num_freqs': degree,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        pos_enc = PositionalEncoder(pos_enc_kwargs)
        pos_enc_fn = pos_enc.encode
        return pos_enc_fn, pos_enc.out_dim

    def forward(self, input_batch: dict):
        encoded_features = self.features_pos_enc_fn(input_batch['features'])
        input_tensor = encoded_features
        if self.tensor_configs['view_dependent_color']:
            encoded_views = self.views_pos_enc_fn(input_batch['view_dirs'])
            input_tensor = torch.cat([encoded_features, encoded_views], dim=-1)
        color = self.mlp(input_tensor)
        output_batch = {
            'color': color
        }
        return output_batch


class IntrinsicsLearner(torch.nn.Module):
    def __init__(self, initial_intrinsics, learn_focal):
        super().__init__()
        self.name = self.__class__.__name__
        self.initial_intrinsics = torch.from_numpy(initial_intrinsics.astype(numpy.float32))
        self.initial_intrinsics = torch.nn.Parameter(self.initial_intrinsics, requires_grad=False)
        self.learn_focal = learn_focal
        if self.learn_focal:
            raise NotImplementedError
        return

    def forward(self, cam_id):
        """
        cam_id: (nr, )
        """
        intrinsics = self.initial_intrinsics[cam_id]
        return intrinsics

    def get_trainable_parameters(self, optimizer_configs):
        trainable_params = [
            {
                'name': f'{self.name}_params',
                'params': self.parameters(),
                'lr': optimizer_configs['lr_initial'] if optimizer_configs is not None else None
            },
        ]
        return trainable_params


class ExtrinsicsLearner(torch.nn.Module):
    def __init__(self, initial_extrinsics, learn_rotation, learn_translation):
        super().__init__()
        self.name = self.__class__.__name__
        self.num_frames = initial_extrinsics.shape[0]
        self.initial_extrinsics = torch.from_numpy(initial_extrinsics.astype(numpy.float32))
        self.initial_extrinsics = torch.nn.Parameter(self.initial_extrinsics, requires_grad=False)
        self.learn_rotation = learn_rotation
        self.learn_translation = learn_translation

        self.r = torch.nn.Parameter(torch.zeros(size=(self.num_frames, 3), dtype=torch.float32), requires_grad=learn_rotation)  # (N, 3)
        self.t = torch.nn.Parameter(torch.zeros(size=(self.num_frames, 3), dtype=torch.float32), requires_grad=learn_translation)  # (N, 3)
        return

    def forward(self, cam_id):
        """
        cam_id: (nr, )
        """
        r = self.r[cam_id]  # (nr, 3) axis-angle
        t = self.t[cam_id]  # (nr, 3)
        extrinsics = self.make_extrinsics(r, t)  # (nr, 4, 4)

        # learn a delta pose between init pose and target pose
        extrinsics = self.initial_extrinsics[cam_id] @ extrinsics  # TODO: validate this

        return extrinsics

    def get_trainable_parameters(self, optimizer_configs):
        trainable_params = [
            {
                'name': f'{self.name}_params',
                'params': self.parameters(),
                'lr': optimizer_configs['lr_initial'] if optimizer_configs is not None else None
            },
        ]
        return trainable_params

    def make_extrinsics(self, r, t):
        """
        :param r:  (nr, 3) axis-angle             torch tensor
        :param t:  (nr, 3) translation vector     torch tensor
        :return:   (nr, 4, 4)
        """
        # TODO: understand and validate this
        R = self.Exp(r)  # (nr, 3, 3)
        c2w = torch.cat([R, t.unsqueeze(2)], dim=2)  # (nr, 3, 4)
        c2w = self.convert3x4_4x4(c2w)  # (nr, 4, 4)
        extrinsics = torch.linalg.inv(c2w)
        return extrinsics

    @classmethod
    def convert3x4_4x4(cls, input):
        """
        :param input:  (N, 3, 4) or (3, 4) torch or np
        :return:       (N, 4, 4) or (4, 4) torch or np
        """
        if torch.is_tensor(input):
            if len(input.shape) == 3:
                output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
                output[:, 3, 3] = 1.0
            else:
                output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
        else:
            if len(input.shape) == 3:
                output = numpy.concatenate([input, numpy.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
                output[:, 3, 3] = 1.0
            else:
                output = numpy.concatenate([input, numpy.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
                output[3, 3] = 1.0
        return output

    @staticmethod
    def vec2skew(v):
        """
        :param v:  (nr, 3) torch tensor
        :return:   (nr, 3, 3)
        """
        nr = v.shape[0]
        zero = torch.zeros((nr, 1), dtype=torch.float32, device=v.device)
        skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], dim=1)  # (nr, 3, 1)
        skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], dim=1)
        skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], dim=1)
        skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=2)  # (3, 3)
        return skew_v  # (3, 3)

    @classmethod
    def Exp(cls, r):
        """so(3) vector to SO(3) matrix
        :param r: (3, ) axis-angle, torch tensor
        :return:  (3, 3)
        """
        skew_r = cls.vec2skew(r)  # (nr, 3, 3)
        norm_r = r.norm(dim=1) + 1e-15
        eye = torch.eye(3, dtype=torch.float32, device=r.device)
        R = eye[None] + (torch.sin(norm_r) / norm_r)[:, None, None] * skew_r + ((1 - torch.cos(norm_r)) / norm_r ** 2)[:, None, None] * (skew_r @ skew_r)
        return R
