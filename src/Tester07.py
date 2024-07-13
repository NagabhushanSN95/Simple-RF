# Shree KRISHNAya Namaha
# Common tester across datasets
# Extended from Tester06.py. Supports camera params optimization.
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy
import simplejson
import skimage.io
import torch
from deepdiff import DeepDiff
from tqdm import tqdm

from data_preprocessors.DataPreprocessorFactory01 import get_data_preprocessor
from loss_functions.LossComputer03 import LossComputer
from lr_decayers.LearningRateDecayerFactory02 import get_lr_decayer
from models.ModelFactory02 import get_model
from optimizers.OptimizerFactory02 import get_optimizer
from utils import CommonUtils04 as CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class NerfTester:
    def __init__(self, train_configs: dict, model_configs: dict, test_configs: dict, root_dirpath: Path):
        train_configs['device'] = test_configs['device']
        self.train_configs = train_configs
        self.test_configs = test_configs
        self.root_dirpath = root_dirpath
        self.database_dirpath = self.root_dirpath / 'data/databases' / self.test_configs['database_dirpath']
        self.data_preprocessor = None
        self.model = None
        self.model_configs = model_configs
        self.device = CommonUtils.get_device(test_configs['device'])

        self.build_model()
        return

    def build_model(self):
        self.data_preprocessor = get_data_preprocessor(self.train_configs, mode='test', model_configs=self.model_configs)
        self.model = get_model(self.train_configs, model_configs=self.model_configs).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.test_configs['device'])
        return

    def load_model(self, model_path: Path):
        checkpoint_state = torch.load(model_path, map_location=self.device)
        iter_num = checkpoint_state['iteration_num']
        self.model.load_state_dict(checkpoint_state['model_state_dict'])
        self.model.eval()

        train_dirname = model_path.parent.parent.parent.stem
        scene_dirname = model_path.parent.parent.stem
        model_name = model_path.stem
        print(f'Loaded Model in {train_dirname}/{scene_dirname}/{model_name} trained for {iter_num} iterations')
        return

    def optimize_test_camera_params(self, test_image: numpy.ndarray, intrinsic, extrinsic, extrinsics_train_gt, extrinsics_train_noisy):
        nearest_train_view_id = numpy.argmin(numpy.linalg.norm(extrinsics_train_gt[:, :3, 3] - extrinsic[:3, 3][None], axis=1))
        extrinsic = extrinsic @ numpy.linalg.inv(extrinsics_train_gt[nearest_train_view_id]) @ extrinsics_train_noisy[nearest_train_view_id]
        test_opt_configs = self.train_configs.copy()
        test_opt_configs['data_loader']['batching'] = False
        if 'augmentations' in test_opt_configs['model']:
            del test_opt_configs['model']['augmentations']
        test_opt_configs['losses'] = test_opt_configs['losses'][:1]
        raw_data_dict = {
            'nerf_data': {
                'images': test_image[None],
                'resolution': test_image.shape[:2],
                'intrinsics': intrinsic[None],
                'extrinsics': extrinsic[None],
                'bounds': numpy.array(self.model_configs['bounds']) / self.model_configs['translation_scale'],
            }
        }
        data_preprocessor = get_data_preprocessor(test_opt_configs, mode='test-optimization', model_configs=self.model_configs, raw_data_dict=raw_data_dict)
        print('Extrinsics before optimization: \n', data_preprocessor.preprocessed_data_dict['nerf_data']['poses'])
        self.model.module.rebuild_camera_params_learners(intrinsics=data_preprocessor.preprocessed_data_dict['nerf_data']['intrinsics'],
                                                         extrinsics=data_preprocessor.preprocessed_data_dict['nerf_data']['poses'], device=self.device)
        loss_computer = LossComputer(test_opt_configs)

        optimizers, lr_decayers = self.get_optimizers_camera_optimization()

        self.model.train()
        for iter_num in range(self.test_configs['optimize_camera_params']['num_iterations']):
            input_batch = data_preprocessor.get_next_batch(iter_num=iter_num, image_num=0)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            actual_batch_size = input_batch['pixel_id'].shape[0]
            sub_batch_size = test_opt_configs.get('sub_batch_size', actual_batch_size)
            for start_idx in range(0, actual_batch_size, sub_batch_size):
                sub_input_batch = {}
                for key in input_batch.keys():
                    if isinstance(input_batch[key], torch.Tensor):
                        sub_input_batch[key] = input_batch[key][start_idx: start_idx+sub_batch_size]
                    elif key == 'common_data':
                        sub_input_batch[key] = input_batch[key].copy()
                    else:
                        sub_input_batch[key] = input_batch[key]
                sub_output_batch = self.model(sub_input_batch, mode='test_camera_params_optimization')
                sub_iter_losses_dict = loss_computer.compute_losses(sub_input_batch, sub_output_batch, self.model)
                sub_batch_loss = sub_iter_losses_dict['TotalLoss']
                sub_batch_loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()

            iter_lrs = {}
            for key in optimizers.keys():
                lr_scale = lr_decayers[key].get_learning_rate_scale(iter_num)
                for param_group in optimizers[key].param_groups:
                    # This is the LR used at current iteration. Updated LR is used in next iteration.
                    iter_lrs[f"lr_{param_group['name']}"] = param_group['lr']
                    param_group['lr'] = param_group['lr'] * lr_scale
        self.model.eval()

        input_batch = data_preprocessor.get_next_batch(iter_num=0, image_num=0)
        output_batch = self.model(input_batch, mode='camera_params_only')
        optimized_intrinsic = output_batch['intrinsics'][0].detach().cpu().numpy()
        optimized_extrinsic = output_batch['extrinsics'][0].detach().cpu().numpy()
        # TODO: try to revert the optimized extrinsic to unprocessed_pose
        print('Extrinsics after optimization: \n', optimized_extrinsic)
        return optimized_intrinsic, optimized_extrinsic

    def get_optimizers_camera_optimization(self):
        optimizers, lr_decayers = {}, {}
        optimizer_intrinsics_configs = next(
            filter(lambda x: x['name'] == 'optimizer_intrinsics', self.train_configs['optimizers']), None)
        parameters_intrinsics = self.model.module.intrinsics_learner.get_trainable_parameters(
            optimizer_intrinsics_configs)
        optimizer_extrinsics_configs = next(
            filter(lambda x: x['name'] == 'optimizer_extrinsics', self.train_configs['optimizers']), None)
        parameters_extrinsics = self.model.module.extrinsics_learner.get_trainable_parameters(
            optimizer_extrinsics_configs)

        optimizer_intrinsics = get_optimizer(self.train_configs, optimizer_name='optimizer_intrinsics',
                                             model_params=parameters_intrinsics)
        lr_decayer_intrinsics = get_lr_decayer(self.train_configs, optimizer_name='optimizer_intrinsics')
        if optimizer_intrinsics is not None:
            optimizers['optimizer_intrinsics'] = optimizer_intrinsics
            lr_decayers['optimizer_intrinsics'] = lr_decayer_intrinsics

        optimizer_extrinsics = get_optimizer(self.train_configs, optimizer_name='optimizer_extrinsics',
                                             model_params=parameters_extrinsics)
        lr_decayer_extrinsics = get_lr_decayer(self.train_configs, optimizer_name='optimizer_extrinsics')
        if optimizer_extrinsics is not None:
            optimizers['optimizer_extrinsics'] = optimizer_extrinsics
            lr_decayers['optimizer_extrinsics'] = lr_decayer_extrinsics
        return optimizers, lr_decayers

    def predict_frame(self, camera_pose: numpy.ndarray, view_camera_pose: numpy.ndarray = None, secondary_poses: List[numpy.ndarray] = None,
                      intrinsic: Optional[numpy.ndarray] = None, view_intrinsic: Optional[numpy.ndarray] = None, secondary_intrinsics: Optional[List[numpy.ndarray]] = None,
                      preprocess_poses: bool = True):
        input_dict = self.data_preprocessor.create_test_data(pose=camera_pose, view_pose=view_camera_pose, secondary_poses=secondary_poses, preprocess_pose=preprocess_poses,
                                                             intrinsic=intrinsic, view_intrinsic=view_intrinsic, secondary_intrinsics=secondary_intrinsics)
        if preprocess_poses:
            camera_pose = input_dict['common_data']['processed_pose'][0].cpu().numpy()
            # view_camera_pose = input_dict['processed_view_pose'].cpu().numpy()
            # secondary_poses = input_dict['processed_secondary_poses'].cpu().numpy()
        if intrinsic is None:
            intrinsic = input_dict['common_data']['intrinsic'][0].cpu().numpy()
        self.model.module.rebuild_camera_params_learners(intrinsics=intrinsic[None], extrinsics=camera_pose[None], device=self.device)

        mode = None
        if view_camera_pose is not None:
            mode = 'static_camera'
        with torch.no_grad():
            output_dict = self.model(input_dict, sec_views_vis=secondary_poses is not None, mode=mode)

        processed_output = self.data_preprocessor.retrieve_inference_outputs(output_dict)
        return processed_output

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), image)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return

    @staticmethod
    def save_depth(path: Path, depth: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        depth_image = numpy.round(depth / depth.max() * 255).astype('uint8')
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), depth_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), depth)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), depth_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return

    @staticmethod
    def save_visibility(path: Path, visibility: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        visibility_image = numpy.round(visibility * 255).astype('uint8')
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), visibility_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), visibility)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), visibility_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return


def save_configs(output_dirpath: Path, configs: dict, filename: Optional[str] = 'Configs.json'):
    # Save configs
    configs_path = output_dirpath / filename
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if 'scene_nums' in old_configs:
            scene_id_key = 'scene_nums'
        elif 'scene_names' in old_configs:
            scene_id_key = 'scene_names'
        else:
            raise RuntimeError
        old_scene_ids = old_configs.get(scene_id_key, [])
        new_scene_ids = configs.get(scene_id_key, [])
        merged_scene_ids = sorted(set(old_scene_ids + new_scene_ids))
        if len(merged_scene_ids) > 0:
            configs[scene_id_key] = merged_scene_ids
            old_configs[scene_id_key] = merged_scene_ids
        old_configs['device'] = configs['device']
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming testing: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_testing(test_configs: dict, scenes_data: dict, output_dir_suffix: str = '', *, save_depth: bool = False,
                  save_depth_var: bool = False, save_visibility: bool = False, optimize_camera_params: bool = False):
    root_dirpath = Path('../')
    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"

    train_num = test_configs['train_num']
    model_name = test_configs['model_name']
    train_dirpath = root_dirpath / f'runs/training/train{train_num:04}'
    train_configs_path = train_dirpath / 'Configs.json'
    if not train_configs_path.exists():
        print(f'Train Configs does not exist at {train_configs_path.as_posix()}. Skipping.')
        return
    with open(train_configs_path.as_posix(), 'r') as configs_file:
        train_configs = simplejson.load(configs_file)

    for scene_id in scenes_data:
        scene_data = scenes_data[scene_id]
        train_configs['data_loader']['scene_id'] = scene_id

        # if scene_id not in ['flower']:
        #     continue

        trained_model_configs_path = train_dirpath / f'{scene_id}/ModelConfigs.json'
        if not trained_model_configs_path.exists():
            print(f'Scene {scene_id}: Trained Model Configs does not exist at {trained_model_configs_path.as_posix()}. Skipping.')
            continue
        with open(trained_model_configs_path.as_posix(), 'r') as configs_file:
            trained_model_configs = simplejson.load(configs_file)
        model_path = train_dirpath / f"{scene_id}/saved_models/{model_name}"
        if not model_path.exists():
            print(f'Scene {scene_id}: Model does not exist at {model_path.as_posix()}. Skipping.')
            continue

        # Build the model
        tester = NerfTester(train_configs, trained_model_configs, test_configs, root_dirpath)
        tester.load_model(model_path)

        # Test and save
        scene_output_dirname = scene_data['output_dirname']
        scene_output_dirpath = output_dirpath / f'{scene_output_dirname}{output_dir_suffix}'

        frame_nums = scene_data['frames_data'].keys()
        if save_visibility or optimize_camera_params:
            train_frame_nums = [frame_num for frame_num in frame_nums if scene_data['frames_data'][frame_num]['is_train_frame']]
        if optimize_camera_params:
            train_extrinsics = [scene_data['frames_data'][frame_num]['extrinsic'] for frame_num in train_frame_nums]
            train_extrinsics_noisy = [scene_data['frames_data'][frame_num]['extrinsic_noisy'] for frame_num in train_frame_nums]
            train_extrinsics = numpy.stack(train_extrinsics, axis=0)
            train_extrinsics_noisy = numpy.stack(train_extrinsics_noisy, axis=0)
        for frame_num in tqdm(frame_nums, desc=f'{scene_id}'):
            frame_data = scene_data['frames_data'][frame_num]
            frame_output_path = scene_output_dirpath / f'predicted_frames/{frame_num:04}.png'
            depth_output_path = scene_output_dirpath / f'predicted_depths/{frame_num:04}.npy'
            depth_var_output_path = scene_output_dirpath / f'predicted_depths_variance/{frame_num:04}.npy'
            depth_ndc_output_path = scene_output_dirpath / f'predicted_depths/{frame_num:04}_ndc.npy'
            depth_var_ndc_output_path = scene_output_dirpath / f'predicted_depths_variance/{frame_num:04}_ndc.npy'
            # visibility_output_path = scene_output_dirpath / f'predicted_visibilities/{frame_num:04}.npy'

            inference_required = not frame_output_path.exists()
            if save_depth:
                inference_required = inference_required or (not depth_output_path.exists())
            if save_depth_var:
                inference_required = inference_required or (not depth_var_output_path.exists())
            # if save_visibility and scene_data['frames_data'][frame_num]['is_train_frame']:
            #     inference_required = inference_required or (not visibility_output_path.exists())

            if inference_required:
                tgt_extrinsic = frame_data['extrinsic']
                tgt_viewcam_extrinsic = frame_data.get('extrinsic_viewcam', None)
                tgt_intrinsic = frame_data.get('intrinsic', None)
                tgt_viewcam_intrinsic = frame_data.get('intrinsic_viewcam', None)
                secondary_poses = None
                secondary_intrinsics = None
                if save_visibility and frame_data['is_train_frame']:
                    secondary_frame_nums = [frame_num1 for frame_num1 in train_frame_nums if frame_num1 != frame_num]
                    secondary_poses = [scene_data['frames_data'][frame2_num]['extrinsic'] for frame2_num in secondary_frame_nums]
                    secondary_intrinsics = [scene_data['frames_data'][frame2_num]['intrinsic'] for frame2_num in secondary_frame_nums]

                if optimize_camera_params:
                    tgt_frame = frame_data['frame']
                    if 'intrinsic_noisy' in frame_data:
                        tgt_intrinsic = frame_data['intrinsic_noisy']
                    tgt_intrinsic, tgt_extrinsic = tester.optimize_test_camera_params(
                        tgt_frame, tgt_intrinsic, tgt_extrinsic, train_extrinsics, train_extrinsics_noisy
                    )
                # optimized extrinsics are already preprocessed
                predictions = tester.predict_frame(tgt_extrinsic, tgt_viewcam_extrinsic, secondary_poses,
                                                   tgt_intrinsic, tgt_viewcam_intrinsic, secondary_intrinsics,
                                                   preprocess_poses=not optimize_camera_params)

                tester.save_image(frame_output_path, predictions['image'])
                if save_depth:
                    tester.save_depth(depth_output_path, predictions['depth'], as_png=True)
                    if 'depth_ndc' in predictions:
                        tester.save_depth(depth_ndc_output_path, predictions['depth_ndc'], as_png=True)
                if save_depth_var:
                    tester.save_depth(depth_var_output_path, predictions['depth_var'], as_png=True)
                    if 'depth_var_ndc' in predictions:
                        tester.save_depth(depth_var_ndc_output_path, predictions['depth_var_ndc'], as_png=True)
                if save_visibility and scene_data['frames_data'][frame_num]['is_train_frame']:
                    if 'visibility2' in predictions:
                        for i in range(len(secondary_frame_nums)):
                            visibility_output_path = scene_output_dirpath / f'predicted_visibilities/{frame_num:04}_{secondary_frame_nums[i]:04}.npy'
                            tester.save_visibility(visibility_output_path, predictions['visibility2'][i], as_png=True)
    return output_dirpath
