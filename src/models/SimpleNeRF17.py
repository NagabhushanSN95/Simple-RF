# Shree KRISHNAya Namaha
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import numpy
import torch
import torch.nn.functional as F
from torch.nn import ModuleDict
from torch.nn import ModuleList

from utils import CommonUtils04 as CommonUtils


class SimpleNeRF(torch.nn.Module):
    def __init__(self, configs: dict, model_configs: dict):
        super().__init__()
        self.configs = configs
        self.model_configs = model_configs
        self.ndc = self.configs['data_loader']['ndc']
        self.coarse_model_needed = 'coarse_model' in self.configs['model']
        self.fine_model_needed = 'fine_model' in self.configs['model']
        self.predict_visibility = (self.coarse_model_needed and self.configs['model']['coarse_model']['predict_visibility']) or \
                                  (self.fine_model_needed and self.configs['model']['fine_model']['predict_visibility'])

        self.coarse_model = None
        self.fine_model = None
        self.intrinsics_learner = None
        self.extrinsics_learner = None
        self.augmentations_needed = 'augmentations' in configs['model']
        if self.augmentations_needed:
            self.augmented_models = []
            self.augmented_models_nn = []
        self.build_nerf()
        return

    def build_nerf(self):
        if self.coarse_model_needed:
            self.coarse_model = MLP('coarse_model', self.configs, self.configs['model']['coarse_model'], self.model_configs)

        if self.fine_model_needed:
            self.fine_model = MLP('fine_model', self.configs, self.configs['model']['fine_model'], self.model_configs)

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
                    coarse_model = MLP(name, self.configs, augmentation_configs['coarse_model'], self.model_configs)
                    aug_model_nn['coarse_model'] = coarse_model
                if 'fine_model' in augmentation_configs:
                    name = f'{aug_name}_fine_model'
                    fine_model = MLP(name, self.configs, augmentation_configs['fine_model'], self.model_configs)
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
        rays_o, rays_d = CommonUtils.get_rays_tr(pixel_id, intrinsics, extrinsics, mip_nerf_used=False)
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
                _, view_rays_d = CommonUtils.get_rays_tr(pixel_id, view_intrinsics, view_extrinsics, mip_nerf_used=False)
                view_dirs = CommonUtils.get_view_dirs_tr(view_rays_d)
            else:
                view_dirs = CommonUtils.get_view_dirs_tr(rays_d)
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
            z_vals_coarse = self.get_z_vals_coarse(input_dict)

            if not self.ndc:
                pts_coarse = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse[...,:,None] # [num_rays, num_samples, 3]
            else:
                pts_coarse = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_coarse[..., :, None]  # [num_rays, num_samples, 3]

            network_input_coarse = {
                'pts': pts_coarse,
            }
            if self.configs['model']['coarse_model']['use_view_dirs']:
                network_input_coarse['view_dirs'] = view_dirs

            if self.coarse_model.predict_visibility and sec_views_vis:
                view_dirs2 = self.compute_other_view_dirs(z_vals_coarse, rays_o, rays_d, rays_o2)
                network_input_coarse['view_dirs2'] = view_dirs2

            network_output_coarse = self.run_network(network_input_coarse, self.coarse_model)
            if not self.ndc:
                outputs_coarse = self.volume_rendering(network_output_coarse, z_vals=z_vals_coarse, rays_d=rays_d,
                                                       sec_views_vis=sec_views_vis)
            else:
                outputs_coarse = self.volume_rendering(network_output_coarse, z_vals_ndc=z_vals_coarse,
                                                       rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
                                                       sec_views_vis=sec_views_vis)
            weights_coarse = outputs_coarse['weights']

            return_dict['z_vals_coarse'] = z_vals_coarse
            for key in outputs_coarse:
                return_dict[f'{key}_coarse'] = outputs_coarse[key]
            if retraw:
                for key in network_output_coarse.keys():
                    return_dict[f'raw_{key}_coarse'] = network_output_coarse[key]

            if self.augmentations_needed and self.training and (mode != 'test_camera_params_optimization'):
                for augmented_model in self.augmented_models:
                    aug_name = augmented_model['name']
                    if augmented_model['coarse_model'] is not None:
                        network_output_coarse = self.run_network(network_input_coarse, augmented_model['coarse_model'])
                        if not self.ndc:
                            outputs_coarse = self.volume_rendering(network_output_coarse, z_vals=z_vals_coarse,
                                                                   rays_d=rays_d, sec_views_vis=sec_views_vis)
                        else:
                            outputs_coarse = self.volume_rendering(network_output_coarse, z_vals_ndc=z_vals_coarse,
                                                                   rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
                                                                   sec_views_vis=sec_views_vis)
        
                        for key in outputs_coarse:
                            return_dict[f'{aug_name}_{key}_coarse'] = outputs_coarse[key]
                        if retraw:
                            for key in network_output_coarse.keys():
                                return_dict[f'{aug_name}_raw_{key}_coarse'] = network_output_coarse[key]

        if self.fine_model_needed:
            z_vals_fine = self.get_z_vals_fine(z_vals_coarse, weights_coarse)
            if not self.ndc:
                pts_fine = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_fine[...,:,None]  # [num_rays, num_samples, 3]
            else:
                pts_fine = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_fine[..., :, None]  # [num_rays, num_samples, 3]

            network_input_fine = {
                'pts': pts_fine,
            }
            if self.configs['model']['fine_model']['use_view_dirs']:
                network_input_fine['view_dirs'] = view_dirs

            if self.fine_model.predict_visibility and sec_views_vis:
                view_dirs2 = self.compute_other_view_dirs(z_vals_fine, rays_o, rays_d, rays_o2)
                network_input_fine['view_dirs2'] = view_dirs2

            network_output_fine = self.run_network(network_input_fine, self.fine_model)
            if not self.ndc:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals=z_vals_fine, rays_d=rays_d,
                                                     sec_views_vis=sec_views_vis)
            else:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals_ndc=z_vals_fine, rays_d_ndc=rays_d_ndc,
                                                     rays_o=rays_o, rays_d=rays_d, sec_views_vis=sec_views_vis)
            # weights_fine = outputs_fine['weights']

            return_dict['z_vals_fine'] = z_vals_fine
            for key in outputs_fine:
                return_dict[f'{key}_fine'] = outputs_fine[key]
            if retraw:
                for key in network_output_fine.keys():
                    return_dict[f'raw_{key}_fine'] = network_output_fine[key]

            if self.augmentations_needed and self.training and (mode != 'test_camera_params_optimization'):
                for augmented_model in self.augmented_models:
                    aug_name = augmented_model['name']
                    if augmented_model['fine_model'] is not None:
                        network_output_fine = self.run_network(network_input_fine, augmented_model['fine_model'])
                        if not self.ndc:
                            outputs_fine = self.volume_rendering(network_output_fine, z_vals=z_vals_fine, 
                                                                 rays_d=rays_d, sec_views_vis=sec_views_vis)
                        else:
                            outputs_fine = self.volume_rendering(network_output_fine, z_vals_ndc=z_vals_fine,
                                                                 rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
                                                                 sec_views_vis=sec_views_vis)
        
                        for key in outputs_fine:
                            return_dict[f'{aug_name}_{key}_fine'] = outputs_fine[key]
                        if retraw:
                            for key in network_output_fine.keys():
                                return_dict[f'{aug_name}_raw_{key}_fine'] = network_output_fine[key]

        if not retraw:
            if self.coarse_model_needed:
                del return_dict['z_vals_coarse'], return_dict['alpha_coarse'], return_dict['visibility_coarse'], return_dict['weights_coarse']
            if self.fine_model_needed:
                del return_dict['z_vals_fine'], return_dict['alpha_fine'], return_dict['visibility_fine'], return_dict['weights_fine']
            if self.augmentations_needed and self.training:
                for augmented_model in self.augmented_models:
                    aug_name = augmented_model['name']
                    if augmented_model['coarse_model'] is not None:
                        del return_dict[f'{aug_name}_alpha_coarse'], return_dict[f'{aug_name}_visibility_coarse'], return_dict[f'{aug_name}_weights_coarse']
                    if augmented_model['fine_model'] is not None:
                        del return_dict[f'{aug_name}_alpha_fine'], return_dict[f'{aug_name}_visibility_fine'], return_dict[f'{aug_name}_weights_fine']

        return return_dict

    def get_z_vals_coarse(self, input_dict: dict):
        num_rays = input_dict['pixel_id'].shape[0]
        if not self.ndc:
            near, far = self.model_configs['near'], self.model_configs['far']
        else:
            near, far = self.model_configs['near_ndc'], self.model_configs['far_ndc']

        perturb = self.training and self.configs['model']['perturb']
        lindisp = self.configs['model']['lindisp']

        num_samples_coarse = self.configs['model']['coarse_model']['num_samples']
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

    def run_network(self, input_dict, nerf_mlp):
        """
        Prepares inputs and applies network 'nerf_mlp'.
        """
        pts_flat = torch.reshape(input_dict['pts'], [-1, input_dict['pts'].shape[-1]])
        network_input_dict = {
            'pts': pts_flat,
        }

        if nerf_mlp.mlp_configs['use_view_dirs']:
            viewdirs = input_dict['view_dirs']
            if viewdirs.ndim == 2:
                viewdirs = viewdirs[:,None].expand(input_dict['pts'].shape)
            viewdirs_flat = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
            network_input_dict['view_dirs'] = viewdirs_flat

            if nerf_mlp.predict_visibility and ('view_dirs2' in input_dict):
                view_dirs2 = input_dict['view_dirs2']
                view_dirs2_flat = torch.reshape(view_dirs2, [-1, view_dirs2.shape[-2], view_dirs2.shape[-1]])  # (nr*ns, nf-1, 3)
                network_input_dict['view_dirs2'] = view_dirs2_flat

        # nerf_mlp = nerf_mlp.to(pts_flat.device)
        network_output_dict = self.batchify(nerf_mlp)(network_input_dict)

        for k, v in network_output_dict.items():
            if isinstance(v, torch.Tensor):
                network_output_dict[k] = torch.reshape(v, list(input_dict['pts'].shape[:-1]) + list(v.shape[1:]))
            else:
                raise NotImplementedError
        return network_output_dict

    def batchify(self, nerf_mlp):
        """Constructs a version of 'nerf_mlp' that applies to smaller batches.
        """
        chunk = self.configs['model']['netchunk']
        if chunk is None:
            return nerf_mlp

        def ret(input_dict: dict):
            num_pts = input_dict['pts'].shape[0]
            network_output_chunks = {}
            for i in range(0, num_pts, chunk):
                network_input_chunk = {}
                for key in input_dict:
                    if isinstance(input_dict[key], torch.Tensor):
                        network_input_chunk[key] = input_dict[key][i:i+chunk]
                    else:
                        raise RuntimeError(key)

                network_output_chunk = nerf_mlp(network_input_chunk)

                for k in network_output_chunk.keys():
                    if k not in network_output_chunks:
                        network_output_chunks[k] = []
                    if isinstance(network_output_chunk[k], torch.Tensor):
                        network_output_chunks[k].append(network_output_chunk[k])
                    else:
                        raise RuntimeError

            for k in network_output_chunks:
                if isinstance(network_output_chunks[k][0], torch.Tensor):
                    network_output_chunks[k] = torch.cat(network_output_chunks[k], dim=0)
                else:
                    raise NotImplementedError
            return network_output_chunks
        return ret

    def volume_rendering(self, network_output_dict, z_vals=None, rays_o=None, rays_d=None, z_vals_ndc=None,
                         rays_d_ndc=None, sec_views_vis=False):
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

        rgb = network_output_dict['rgb']  # [N_rays, N_samples, 3]
        sigma = network_output_dict['sigma'][..., 0]  # [N_rays, N_samples]

        alpha = 1. - torch.exp(-sigma * delta)  # [N_rays, N_samples]
        visibility = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(rays_d.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * visibility
        rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)  # [N_rays, 3]

        acc_map = torch.sum(weights, dim=-1)
        if not self.ndc:
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)
        else:
            depth_map_ndc = torch.sum(weights * z_vals_ndc, dim=-1) / (acc_map + 1e-6)
            depth_var_map_ndc = torch.sum(weights * torch.square(z_vals_ndc - depth_map_ndc[..., None]), dim=-1)
            z_vals = self.convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d)
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)

        if self.configs['model']['white_bkgd']:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return_dict = {
            'rgb': rgb_map,
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

        if self.predict_visibility and sec_views_vis and ('visibility2' in network_output_dict):
            vis2_point3d = network_output_dict['visibility2']
            vis2_pixel = torch.sum(weights[..., None] * vis2_point3d[..., 0], dim=-2) / (acc_map[..., None] + 1e-6)  # (nr, nf-1)
            return_dict['visibility2'] = vis2_pixel
        return return_dict

    @staticmethod
    def convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d):
        """
        Converts depth in ndc to actual values
        From ndc write up, t' is z_vals_ndc and t is z_vals.
        t' = 1 - oz / (oz + t * dz)
        t = (oz / dz) * (1 / (1-t') - 1)
        But due to the final trick, oz is shifted. So, the actual oz = oz + tn * dz
        Overall t_act = t + tn = ((oz + tn * dz) / dz) * (1 / (1 - t') - 1) + tn
        """
        near = 1  # TODO: do not hard-code
        oz = rays_o[..., 2:3]
        dz = rays_d[..., 2:3]
        tn = -(near + oz) / dz
        constant = torch.where(z_vals_ndc == 1., 1e-3, 0.)
        # depth = (((oz + tn * dz) / (1 - z_vals_ndc + constant)) - oz) / dz
        depth = (oz + tn * dz) / dz * (1 / (1 - z_vals_ndc + constant) - 1) + tn
        return depth

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


class MLP(torch.nn.Module):
    def __init__(self, name, configs, mlp_configs, model_configs):
        """
        """
        super().__init__()
        self.name = name
        self.configs = configs
        self.mlp_configs = mlp_configs
        self.model_configs = model_configs
        self.Dp = self.mlp_configs['points_net_depth']
        self.Dv = self.mlp_configs['views_net_depth']
        self.Wp = self.mlp_configs['points_net_width']
        self.Wv = self.mlp_configs['views_net_width']

        self.pts_pos_enc_fn, self.pts_input_dim = self.get_positional_encoder(self.mlp_configs['points_positional_encoding_degree'])
        self.views_input_dim = 0
        if self.mlp_configs['use_view_dirs']:
            self.views_pos_enc_fn, self.views_input_dim = self.get_positional_encoder(self.mlp_configs['views_positional_encoding_degree'])
        if 'points_sigma_positional_encoding_degree' in self.mlp_configs:
            self.pts_input_dim = (2 * self.mlp_configs['points_sigma_positional_encoding_degree'] + 1) * 3
            self.views_input_dim += ((2 * self.mlp_configs['points_positional_encoding_degree'] + 1) * 3 - self.pts_input_dim)

        self.skips = [4]
        self.view_dep_rgb = self.mlp_configs['view_dependent_rgb']
        self.predict_visibility = self.mlp_configs['predict_visibility']
        self.view_dep_outputs = self.view_dep_rgb or self.predict_visibility
        self.raw_noise_std = self.configs['model']['raw_noise_std']

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.pts_input_dim, self.Wp)] +
            [torch.nn.Linear(self.Wp, self.Wp) if i not in self.skips else torch.nn.Linear(self.Wp + self.pts_input_dim, self.Wp) for i in range(self.Dp - 1)]
        )
        if self.view_dep_outputs:
            self.views_linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.views_input_dim + self.Wp, self.Wv)] +
            [torch.nn.Linear(self.Wv, self.Wv) for _ in range(self.Dv - 1)]
            )

        pts_output_dim = 1  # sigma
        views_output_dim = 0
        if not self.view_dep_rgb:
            pts_output_dim += 3
        else:
            views_output_dim += 3
        if self.predict_visibility:
            views_output_dim += 1

        self.pts_output_linear = torch.nn.Linear(self.Wp, pts_output_dim)
        if self.view_dep_outputs:
            self.feature_linear = torch.nn.Linear(self.Wp, self.Wp)
            self.views_output_linear = torch.nn.Linear(self.Wv, views_output_dim)
        return

    @staticmethod
    def get_positional_encoder(degree):
        pos_enc_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': degree - 1,
            'num_freqs': degree,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

        pos_enc = PositionalEncoder(pos_enc_kwargs)
        pos_enc_fn = pos_enc.encode
        return pos_enc_fn, pos_enc.out_dim

    def get_trainable_parameters(self, optimizer_configs):
        network_trainable_params = torch.nn.ParameterList()
        network_trainable_params.extend(self.parameters())
        trainable_params = [
            {
                'name': f'{self.name}_network_params',
                'params': network_trainable_params,
                'lr': optimizer_configs['lr_initial']
            }
        ]
        return trainable_params

    def forward(self, input_batch: dict):
        input_pts = input_batch['pts']
        output_batch = {}

        encoded_pts = self.pts_pos_enc_fn(input_pts)  # (B, 21)
        pts_outputs = self.get_view_independent_outputs(encoded_pts[:, :self.pts_input_dim])
        if self.view_dep_outputs:
            pts_outputs['feature'] = torch.cat([pts_outputs['feature'], encoded_pts[:, self.pts_input_dim:]], dim=1)
        output_batch.update(pts_outputs)
        if not self.view_dep_rgb:
            rgb = pts_outputs['rgb_view_independent']

        if self.view_dep_outputs:
            input_views = input_batch['view_dirs']
            encoded_views = self.views_pos_enc_fn(input_views)
            view_outputs = self.get_view_dependent_outputs(pts_outputs, encoded_views)
            output_batch.update(view_outputs)
            if self.view_dep_rgb:
                rgb = view_outputs['rgb_view_dependent']

            if 'view_dirs2' in input_batch.keys():
                encoded_views2 = self.views_pos_enc_fn(input_batch['view_dirs2'])
                view_outputs2 = self.get_view_dependent_outputs(pts_outputs, encoded_views2)
                output_batch['visibility2'] = view_outputs2['visibility']
        output_batch['rgb'] = rgb

        if 'feature' in output_batch:
            del output_batch['feature']
        return output_batch

    def get_view_independent_outputs(self, input_pts):
        output_dict = {}
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        pts_output = self.pts_output_linear(h)
        ch_i = 0  # Denotes number of channels which have already been taken output

        sigma = pts_output[..., ch_i:ch_i + 1]
        if self.training and (self.raw_noise_std > 0.):
            noise = torch.randn(sigma.shape).to(sigma.device) * self.raw_noise_std
            sigma = sigma + noise
        sigma = F.relu(sigma)
        output_dict['sigma'] = sigma
        ch_i += 1

        if not self.view_dep_rgb:
            rgb = pts_output[..., ch_i:ch_i+3]
            rgb = torch.sigmoid(rgb)
            output_dict['rgb_view_independent'] = rgb
            ch_i += 3

        if self.view_dep_outputs:
            feature = self.feature_linear(h)
            output_dict['feature'] = feature
        return output_dict

    def get_view_dependent_outputs(self, pts_outputs, input_views):
        output_dict = {}

        feature = pts_outputs['feature']
        if input_views.ndim == 3:
            # For viewdirs2
            nf = input_views.shape[1] + 1
            feature = feature[:, None, :].repeat([1, nf-1, 1])  # (nc, nf-1, cv)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        view_outputs = self.views_output_linear(h)
        ch_i = 0  # Denotes number of channels which have already been taken output

        if self.view_dep_rgb:
            rgb = view_outputs[..., ch_i:ch_i+3]
            rgb = torch.sigmoid(rgb)
            output_dict['rgb_view_dependent'] = rgb
            ch_i += 3

        if self.predict_visibility:
            visibility = view_outputs[..., ch_i:ch_i+1]
            visibility = torch.sigmoid(visibility)
            output_dict['visibility'] = visibility
            ch_i += 1
        return output_dict


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
