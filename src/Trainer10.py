# Shree KRISHNAya Namaha
# Extended from Trainer09.py. Optimizer code updated for optimizing camera parameters
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import datetime
import json
import math
import os
import random
from pathlib import Path
from typing import Optional, List, Dict

import numpy
import pandas
import simplejson
import skimage.io
import skimage.transform
import torch
from deepdiff import DeepDiff
from matplotlib import pyplot
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loaders.DataLoaderFactory import get_data_loader
from data_preprocessors.DataPreprocessorFactory import get_data_preprocessor
from data_preprocessors.DataPreprocessorParent01 import DataPreprocessorParent
from loss_functions.LossComputer03 import LossComputer
from lr_decayers.LearningRateDecayerFactory02 import get_lr_decayer
from lr_decayers.LearningRateDecayerParent02 import LearningRateDecayerParent
from models.ModelFactory02 import get_model
from optimizers.OptimizerFactory02 import get_optimizer
from utils import CommonUtils04 as CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, configs: dict, model_configs: dict, train_data_loader: DataPreprocessorParent,
                 val_data_loader: DataPreprocessorParent, model, loss_computer: LossComputer, optimizers: dict,
                 lr_decayers: Dict[str, LearningRateDecayerParent], output_dirpath: Path, device: str,
                 verbose_log: bool = True):
        self.configs = configs
        self.model_configs = model_configs
        self.train_data_preprocessor = train_data_loader
        self.val_data_preprocessor = val_data_loader
        self.model = model
        self.loss_computer = loss_computer
        self.optimizers = optimizers
        self.lr_decayers = lr_decayers
        self.device = CommonUtils.get_device(device)
        self.output_dirpath = output_dirpath
        self.logger = SummaryWriter((output_dirpath / 'logs').as_posix())
        self.verbose_log = verbose_log

        if hasattr(self.model.module, 'train_data_preprocessor'):
            self.model.module.train_data_preprocessor = self.train_data_preprocessor
        if hasattr(self.model.module, 'optimizers'):
            self.model.module.optimizers = self.optimizers
        return

    def train_one_iter(self, iter_num: int):
        def update_losses_dict_(iter_losses_dict_: dict, sub_iter_losses_dict_: dict, num_samples_: int):
            if iter_losses_dict_ is None:
                iter_losses_dict_ = {}
            for loss_name_ in sub_iter_losses_dict_.keys():
                loss_value_ = sub_iter_losses_dict_[loss_name_]
                loss_value_ = loss_value_['loss_value'] if isinstance(loss_value_, dict) else loss_value_
                loss_value_ = loss_value_.item() if isinstance(loss_value_, torch.Tensor) else loss_value_
                if loss_name_ not in iter_losses_dict_.keys():
                    iter_losses_dict_[loss_name_] = 0
                iter_losses_dict_[loss_name_] += (loss_value_ * num_samples_)
            return iter_losses_dict_

        def delete_dict(dict_data_: dict):
            for key_ in list(dict_data_.keys()):
                del dict_data_[key_]
            return

        input_batch = self.train_data_preprocessor.get_next_batch(iter_num)
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        actual_batch_size = input_batch['pixel_id'].shape[0]
        sub_batch_size = self.configs.get('sub_batch_size', actual_batch_size)
        iter_losses_dict = {}
        for sub_batch_index, start_idx in enumerate(range(0, actual_batch_size, sub_batch_size)):
            sub_input_batch = {}
            for key in input_batch.keys():
                if isinstance(input_batch[key], torch.Tensor):
                    sub_input_batch[key] = input_batch[key][start_idx: start_idx+sub_batch_size]
                elif key == 'common_data':
                    sub_input_batch[key] = input_batch[key].copy()
                else:
                    sub_input_batch[key] = input_batch[key]
            # This is provided to run the operations that need to be run at certain iteration only once
            sub_input_batch['sub_batch_index'] = sub_batch_index
            sub_output_batch = self.model(sub_input_batch)
            sub_iter_losses_dict = self.loss_computer.compute_losses(sub_input_batch, sub_output_batch, self.model)
            sub_batch_loss = sub_iter_losses_dict['TotalLoss']
            sub_batch_loss.backward()

            iter_losses_dict = update_losses_dict_(iter_losses_dict, sub_iter_losses_dict, num_samples_=1)
            delete_dict(sub_output_batch)
            delete_dict(sub_input_batch)
            del sub_output_batch, sub_input_batch
        for optimizer in self.optimizers.values():
            optimizer.step()

        delete_dict(input_batch)
        del input_batch

        return iter_losses_dict

    def run_validation(self, iter_num: int, data_loader, save_dirpath: Path):
        def update_losses_dict_(epoch_losses_dict_: dict, frame_losses_dict_: dict, num_samples_: int):
            if epoch_losses_dict_ is None:
                epoch_losses_dict_ = {}
                for loss_name_ in frame_losses_dict_.keys():
                    loss_value_ = frame_losses_dict_[loss_name_]
                    loss_value_ = loss_value_['loss_value'] if isinstance(loss_value_, dict) else loss_value_
                    epoch_losses_dict_[loss_name_] = loss_value_.item() * num_samples_
            else:
                for loss_name_ in epoch_losses_dict_.keys():
                    loss_value_ = frame_losses_dict_[loss_name_]
                    loss_value_ = loss_value_['loss_value'] if isinstance(loss_value_, dict) else loss_value_
                    epoch_losses_dict_[loss_name_] += (loss_value_.item() * num_samples_)
            return epoch_losses_dict_

        def post_process_output_(tensor_: torch.Tensor, resolution_: tuple):
            h, w = resolution_
            c = tensor_.shape[1] if tensor_.ndim == 2 else 1
            reshaped_array = tensor_.detach().cpu().numpy().reshape((h, w, c)).squeeze()
            return reshaped_array

        def divide_input_batch_(input_batch_: dict, chunk_size_: int):
            num_pixels = input_batch_['rays_o'].shape[0]
            num_batches = int(math.ceil(num_pixels / chunk_size_))
            input_batches_ = [{} for _ in range(num_batches)]
            for key in input_batch_:
                if (isinstance(input_batch_[key], torch.Tensor) or isinstance(input_batch_[key], numpy.ndarray)) and \
                        (input_batch_[key].shape[0] == num_pixels):
                    for i_ in range(num_batches):
                        input_batches_[i_][key] = input_batch_[key][i_ * chunk_size_: (i_ + 1) * chunk_size_]
                elif key == 'common_data':
                    for i_ in range(num_batches):
                        input_batches_[i_][key] = input_batch_[key].copy()
                else:
                    for i_ in range(num_batches):
                        input_batches_[i_][key] = input_batch_[key]
            return input_batches_

        def merge_output_batches_(output_batches_):
            output_batch_ = {}
            for key in output_batches_[0]:
                if isinstance(output_batches_[0][key], torch.Tensor) and (output_batches_[0][key].numel() > 1):
                    output_batch_[key] = torch.cat([output_batch_chunk_[key] for output_batch_chunk_ in output_batches_], dim=0)
                elif isinstance(output_batches_[0][key], torch.Tensor) and (output_batches_[0][key].numel() == 1):  # TotalLoss
                    output_batch_[key] = torch.mean(torch.stack([output_batch_chunk_[key] for output_batch_chunk_ in output_batches_]))
                elif isinstance(output_batches_[0][key], dict):
                    output_batch_[key] = {}
                    inner_dict1 = output_batches_[0][key]
                    for key1 in inner_dict1:
                        if isinstance(inner_dict1[key1], torch.Tensor) and inner_dict1[key1].numel() == 1:  # loss_value
                            output_batch_[key][key1] = torch.mean(torch.stack([output_batch_chunk_[key][key1] for output_batch_chunk_ in output_batches_]))
                        elif isinstance(inner_dict1[key1], dict):  # loss_maps
                            output_batch_[key][key1] = {}
                            inner_dict2 = inner_dict1[key1]
                            for key2 in inner_dict2:
                                if isinstance(inner_dict2[key2], torch.Tensor):
                                    output_batch_[key][key1][key2] = torch.cat([output_batch_chunk_[key][key1][key2] for output_batch_chunk_ in output_batches_], dim=0)
                                else:
                                    raise RuntimeError
                        else:
                            raise RuntimeError
                else:
                    raise RuntimeError
            return output_batch_
        
        def delete_dict_elements(dict_: dict, keys: list):
            for key in keys:
                if key in dict_:
                    del dict_[key]
            return 

        resolution = self.train_data_preprocessor.preprocessed_data_dict['nerf_data']['resolution']
        chunk_size = self.configs['validation_chunk_size']
        total_losses_dict = None
        total_num_samples = 0
        self.model.eval()

        train_data = data_loader.mode == 'train'
        frame_nums = data_loader.preprocessed_data_dict['frame_nums']
        for i, frame_num in enumerate(frame_nums):
            input_batch = data_loader.get_next_batch(iter_num, frame_num)
            input_batches = divide_input_batch_(input_batch, chunk_size)
            output_batches, frame_losses_dicts = [], []
            for input_batch_chunk in input_batches:
                with torch.no_grad():
                    output_batch_chunk = self.model(input_batch_chunk, retraw=True, sec_views_vis=train_data)
                frame_losses_dict_chunk = self.loss_computer.compute_losses(input_batch_chunk, output_batch_chunk, self.model,
                                              return_loss_maps=self.configs['validation_save_loss_maps'])
                useless_output_keys = [
                    'z_vals_coarse', 'z_vals_other_coarse',
                    'raw_sigma_coarse', 'raw_sigma_other_coarse',
                    'raw_rgb_coarse', 'raw_rgb_view_independent_coarse', 'raw_rgb_view_dependent_coarse',
                    'raw_rgb_other_coarse', 'raw_rgb_view_independent_other_coarse', 'raw_rgb_view_dependent_other_coarse',
                    'raw_visibility_coarse', 'raw_visibility2_coarse',
                    'alpha_coarse', 'visibility_coarse', 'weights_coarse',
                    'alpha_other_coarse', 'visibility_other_coarse', 'weights_other_coarse',
                    'rays_d2_coarse', 'rays_o2_ndc_coarse', 'rays_d2_ndc_coarse',
                    'rgb_other_coarse', 'acc_other_coarse',
                    'depth_other_coarse', 'depth_var_other_coarse',
                    'depth_ndc_other_coarse', 'depth_var_ndc_other_coarse',
                    'z_vals_fine', 'z_vals_other_fine',
                    'raw_sigma_fine', 'raw_sigma_other_fine',
                    'raw_rgb_fine', 'raw_rgb_view_independent_fine', 'raw_rgb_view_dependent_fine',
                    'raw_rgb_other_fine', 'raw_rgb_view_independent_other_fine', 'raw_rgb_view_dependent_other_fine',
                    'raw_visibility_fine', 'raw_visibility2_fine',
                    'alpha_fine', 'visibility_fine', 'weights_fine',
                    'alpha_other_fine', 'visibility_other_fine', 'weights_other_fine',
                    'rays_d2_fine', 'rays_o2_ndc_fine', 'rays_d2_ndc_fine',
                    'rgb_other_fine', 'acc_other_fine',
                    'depth_other_fine', 'depth_var_other_fine',
                    'depth_ndc_other_fine', 'depth_var_ndc_other_fine'
                ]
                delete_dict_elements(output_batch_chunk, useless_output_keys)

                output_batches.append(output_batch_chunk)
                frame_losses_dicts.append(frame_losses_dict_chunk)
            output_batch = merge_output_batches_(output_batches)
            frame_losses_dict = merge_output_batches_(frame_losses_dicts)

            # Update losses and the number of samples
            total_losses_dict = update_losses_dict_(total_losses_dict, frame_losses_dict, num_samples_=1)
            total_num_samples += 1

            # Save all predictions
            for mode in ['coarse', 'fine']:
                frame_output_path = save_dirpath / f'predicted_frames/{frame_num:04}_{mode}_Iter{iter_num + 1:05}.png'
                depth_output_path = save_dirpath / f'predicted_depths/{frame_num:04}_{mode}_Iter{iter_num + 1:05}.npy'
                depth_var_output_path = save_dirpath / f'predicted_depths_variance/{frame_num:04}_{mode}_Iter{iter_num + 1:05}.npy'
                depth_ndc_output_path = save_dirpath / f'predicted_depths/{frame_num:04}_{mode}_ndc_Iter{iter_num + 1:05}.npy'
                depth_var_ndc_output_path = save_dirpath / f'predicted_depths_variance/{frame_num:04}_{mode}_ndc_Iter{iter_num + 1:05}.npy'
                self.save_image(frame_output_path, post_process_output_(output_batch[f'rgb_{mode}'], resolution))
                self.save_numpy_array(depth_output_path, post_process_output_(output_batch[f'depth_{mode}'], resolution), as_png=True)
                if f'depth_ndc_{mode}' in output_batch:
                    self.save_numpy_array(depth_ndc_output_path, post_process_output_(output_batch[f'depth_ndc_{mode}'], resolution), as_png=True)
                self.save_numpy_array(depth_var_output_path, post_process_output_(output_batch[f'depth_var_{mode}'], resolution), as_png=True)
                if f'depth_var_ndc_{mode}' in output_batch:
                    self.save_numpy_array(depth_var_ndc_output_path, post_process_output_(output_batch[f'depth_var_ndc_{mode}'], resolution), as_png=True)
                if f'visibility2_{mode}' in output_batch:
                    for j, sec_frame_num in enumerate([x for x in frame_nums if x != frame_num]):
                        vis2_output_path = save_dirpath / f'predicted_visibilities/{frame_num:04}_{sec_frame_num:04}_{mode}_Iter{iter_num + 1:05}.npy'
                        self.save_numpy_array(vis2_output_path, post_process_output_(output_batch[f'visibility2_{mode}'][:, j], resolution), as_png=True)

            # Save all loss maps
            if self.configs['validation_save_loss_maps']:
                for loss_name in frame_losses_dict.keys():
                    if not isinstance(frame_losses_dict[loss_name], dict):
                        continue
                    loss_maps = frame_losses_dict[loss_name]['loss_maps']
                    for loss_fullname, loss_map in loss_maps.items():
                        loss_output_path = save_dirpath / f'Losses/{loss_fullname}_{frame_num:04}_Iter{iter_num + 1:05}.npy'
                        self.save_numpy_array(loss_output_path, post_process_output_(loss_map, resolution), as_png=True)
        for loss_name in total_losses_dict.keys():
            total_losses_dict[loss_name] = total_losses_dict[loss_name] / total_num_samples
        self.model.train()
        return total_losses_dict

    def train(self):
        def update_losses_data_(iter_num_: int, iter_losses_: dict, label: str):
            curr_time = datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p')
            self.logger.add_text(f'{label}/Time', curr_time, iter_num_)
            for key, loss_ in iter_losses_.items():
                loss_value_ = loss_['loss_value'] if isinstance(loss_, dict) else loss_
                self.logger.add_scalar(f'{label}/{key}', loss_value_, iter_num_)
            return

        train_num = self.configs['train_num']
        scene_id = self.configs['data_loader']['scene_id']
        print(f'Training {train_num}/{scene_id} begins...')
        logs_dirpath = self.output_dirpath / 'logs'
        sample_images_dirpath = self.output_dirpath / 'samples'
        saved_models_dirpath = self.output_dirpath / 'saved_models'
        logs_dirpath.mkdir(exist_ok=True)
        sample_images_dirpath.mkdir(exist_ok=True)
        saved_models_dirpath.mkdir(exist_ok=True)

        validation_interval = self.configs['validation_interval']
        # sample_save_interval = self.configs['sample_save_interval']
        model_save_interval = self.configs['model_save_interval']
        total_num_iters = self.configs['num_iterations']

        # self.save_model(0, saved_models_dirpath)
        start_iter_num = self.load_model(saved_models_dirpath)
        for iter_num in tqdm(range(start_iter_num, total_num_iters), initial=start_iter_num, total=total_num_iters,
                             mininterval=1, leave=self.verbose_log):
            iter_losses_dict = self.train_one_iter(iter_num)
            # Update the learning rates
            for key in self.optimizers.keys():
                lr_scale = self.lr_decayers[key].get_learning_rate_scale(iter_num)
                for param_group in self.optimizers[key].param_groups:
                    # This is the LR used at current iteration. Updated LR is used in next iteration.
                    iter_losses_dict[f"lr_{param_group['name']}"] = param_group['lr']
                    param_group['lr'] = param_group['lr'] * lr_scale

            # iter_losses_dict = self.train_one_iter(iter_num)
            update_losses_data_(iter_num + 1, iter_losses_dict, 'train')

            if (iter_num + 1) % validation_interval == 0:
                epoch_val_loss = self.run_validation(iter_num, self.train_data_preprocessor, sample_images_dirpath)
                update_losses_data_(iter_num + 1, epoch_val_loss, 'validation/train_images')
                epoch_val_loss = self.run_validation(iter_num, self.val_data_preprocessor, sample_images_dirpath)
                update_losses_data_(iter_num + 1, epoch_val_loss, 'validation/val_images')

            # if (iter_num + 1) % sample_save_interval == 0:
            #     self.save_sample_images(iter_num + 1, sample_images_dirpath)

            if (iter_num + 1) % model_save_interval == 0:
                self.save_model(iter_num + 1, saved_models_dirpath)

            if (iter_num + 1) >= total_num_iters:
                break

        # save_plots(logs_dirpath)
        return

    def save_sample_images(self, iter_num, save_dirpath):
        self.model.eval()

        def render_frame(camera_pose: numpy.ndarray):
            if camera_pose.shape[0] == 3:
                bottom = numpy.array([0, 0, 0, 1])[None]
                camera_pose = numpy.concatenate([camera_pose, bottom], axis=0)
            input_dict = self.train_data_preprocessor.create_test_data(camera_pose, None, preprocess_pose=False)
            with torch.no_grad():
                output_batch = self.model(input_dict)
            processed_output = self.train_data_preprocessor.retrieve_inference_outputs(output_batch)
            return processed_output

        for data_loader in [self.train_data_preprocessor, self.val_data_preprocessor]:
            for frame_num, pose in zip(data_loader.preprocessed_data_dict['frame_nums'],
                                       data_loader.preprocessed_data_dict['nerf_data']['poses']):
                predictions = render_frame(pose)
                frame_output_path = save_dirpath / f'predicted_frames/{frame_num:04}_Iter{iter_num:05}.png'
                depth_output_path = save_dirpath / f'predicted_depths/{frame_num:04}_Iter{iter_num:05}.png'
                depth_var_output_path = save_dirpath / f'predicted_depths_variance/{frame_num:04}_Iter{iter_num:05}.png'
                depth_ndc_output_path = save_dirpath / f'predicted_depths/{frame_num:04}_ndc_Iter{iter_num:05}.png'
                depth_var_ndc_output_path = save_dirpath / f'predicted_depths_variance/{frame_num:04}_ndc_Iter{iter_num:05}.png'
                self.save_image(frame_output_path, predictions['image'])
                self.save_numpy_array(depth_output_path, predictions['depth'], as_png=True)
                if 'depth_ndc' in predictions:
                    self.save_numpy_array(depth_ndc_output_path, predictions['depth_ndc'], as_png=True)
                self.save_numpy_array(depth_var_output_path, predictions['depth_var'], as_png=True)
                if 'depth_var_ndc' in predictions:
                    self.save_numpy_array(depth_var_ndc_output_path, predictions['depth_var_ndc'], as_png=True)

        self.model.train()
        return

    def save_model(self, iter_num: int, save_dirpath: Path, label: str = None):
        if label is None:
            label = f'Iter{iter_num:06}'
        save_path1 = save_dirpath / f'Model_{label}.tar'
        save_path2 = save_dirpath / f'Model_Latest.tar'
        checkpoint_state = {
            'iteration_num': iter_num,
            'model_state_dict': self.model.state_dict(),
            'optimizers_state_dict': {key: self.optimizers[key].state_dict() for key in self.optimizers.keys()},
        }
        torch.save(checkpoint_state, save_path1)
        if save_path2.exists():
            os.remove(save_path2)
        os.system(f'ln -s {os.path.relpath(save_path1, save_path2.parent)} {save_path2.as_posix()}')
        return

    def load_model(self, saved_models_dirpath: Path):
        latest_model_path = saved_models_dirpath / 'Model_Latest.tar'
        if latest_model_path.exists():
            if self.device.type == 'cpu':
                checkpoint_state = torch.load(latest_model_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(latest_model_path)
            iter_num = checkpoint_state['iteration_num']
            self.model.load_state_dict(checkpoint_state['model_state_dict'])
            for key in self.optimizers.keys():
                self.optimizers[key].load_state_dict(checkpoint_state['optimizers_state_dict'][key])
            print(f'Resuming Training from iteration {iter_num + 1}')
        else:
            iter_num = 0
        return iter_num

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image.flat[0], numpy.floating):
            image = numpy.round(image * 255).astype('uint8')
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), image)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return

    @staticmethod
    def save_numpy_array(path: Path, data_array: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        data_image = numpy.round(data_array / data_array.max() * 255).astype('uint8')
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), data_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), data_array)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), data_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown data format: {path.as_posix()}')
        return


def save_plots(logs_dirpath: Path):
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(logs_dirpath.as_posix(), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    for loss_key in ea.scalars.Keys():
        prefix, *loss_name_parts = loss_key.split('/')
        loss_name = '_'.join(loss_name_parts)
        loss_data = pandas.DataFrame(ea.Scalars(loss_key))
        iter_nums = loss_data['step'].to_numpy()
        loss_values = loss_data['value'].to_numpy()
        save_path = logs_dirpath / f'{prefix}_{loss_name}.png'
        pyplot.plot(iter_nums, loss_values)
        pyplot.savefig(save_path)
        pyplot.close()
    return


def init_seeds(seed: int = 0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def update_missing_configs(new_configs: dict, old_configs: dict):
    for key in old_configs.keys():
        if key not in new_configs.keys():
            new_configs[key] = old_configs[key]
        elif isinstance(old_configs[key], dict):
            new_configs[key] = update_missing_configs(new_configs[key], old_configs[key])
    return new_configs


def save_configs(output_dirpath: Path, configs: dict, filename: Optional[str] = 'Configs.json'):
    # Save configs
    configs_path = output_dirpath / filename
    if configs_path.exists():
        # If resume_training is false, an error would've been raised when creating output directory. No need to handle
        # it here.
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        configs['seed'] = old_configs['seed']
        configs = update_missing_configs(configs, old_configs)
        if 'scene_nums' in old_configs['data_loader']:
            scene_id_key = 'scene_nums'
        elif 'scene_names' in old_configs['data_loader']:
            scene_id_key = 'scene_names'
        else:
            raise RuntimeError
        old_scene_ids = old_configs['data_loader'].get(scene_id_key, [])
        new_scene_ids = configs['data_loader'].get(scene_id_key, [])
        merged_scene_ids = sorted(set(old_scene_ids + new_scene_ids))
        if len(merged_scene_ids) > 0:
            configs['data_loader'][scene_id_key] = merged_scene_ids
            old_configs['data_loader'][scene_id_key] = merged_scene_ids
        if configs['num_iterations'] > old_configs['num_iterations']:
            old_configs['num_iterations'] = configs['num_iterations']
        old_configs['device'] = configs['device']
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming training: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def save_model_configs(output_dirpath: Path, configs: dict, filename: Optional[str] = 'Configs.json'):
    # Save model configs
    configs_path = output_dirpath / filename
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming training: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def get_optimizers(configs, model):
    # get parameters to optimize
    optimizers, lr_decayers = {}, {}
    optimizer_main_configs = next(filter(lambda x: x['name'] == 'optimizer_main', configs['optimizers']))
    parameters_nerf = model.module.get_trainable_parameters(optimizer_main_configs)
    # if model.module.intrinsics_learner is not None:
    optimizer_intrinsics_configs = next(filter(lambda x: x['name'] == 'optimizer_intrinsics', configs['optimizers']), None)
    parameters_intrinsics = model.module.intrinsics_learner.get_trainable_parameters(optimizer_intrinsics_configs)
    # if model.module.extrinsics_learner is not None:
    optimizer_extrinsics_configs = next(filter(lambda x: x['name'] == 'optimizer_extrinsics', configs['optimizers']), None)
    parameters_extrinsics = model.module.extrinsics_learner.get_trainable_parameters(optimizer_extrinsics_configs)

    optimizer_nerf = get_optimizer(configs, optimizer_name='optimizer_main', model_params=parameters_nerf)
    lr_decayer = get_lr_decayer(configs, optimizer_name='optimizer_main')
    optimizers['optimizer_nerf'] = optimizer_nerf
    lr_decayers['optimizer_nerf'] = lr_decayer

    optimizer_intrinsics = get_optimizer(configs, optimizer_name='optimizer_intrinsics',
                                         model_params=parameters_intrinsics)
    lr_decayer_intrinsics = get_lr_decayer(configs, optimizer_name='optimizer_intrinsics')
    if optimizer_intrinsics is not None:
        optimizers['optimizer_intrinsics'] = optimizer_intrinsics
        lr_decayers['optimizer_intrinsics'] = lr_decayer_intrinsics

    optimizer_extrinsics = get_optimizer(configs, optimizer_name='optimizer_extrinsics',
                                         model_params=parameters_extrinsics)
    lr_decayer_extrinsics = get_lr_decayer(configs, optimizer_name='optimizer_extrinsics')
    if optimizer_extrinsics is not None:
        optimizers['optimizer_extrinsics'] = optimizer_extrinsics
        lr_decayers['optimizer_extrinsics'] = lr_decayer_extrinsics
    return optimizers, lr_decayers


def start_training(configs: dict):
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / 'data/databases' / configs['database_dirpath']
    output_dirpath = root_dirpath / f'runs/training/train{configs["train_num"]:04}'
    
    scene_ids = configs['data_loader']['scene_ids']
    for scene_id in scene_ids:
        init_seeds(configs['seed'])
        scene_output_dirpath = output_dirpath / f'{scene_id}'
        scene_output_dirpath.mkdir(parents=True, exist_ok=configs['resume_training'])

        # Create data_loaders, models, optimizers etc
        configs['data_loader']['scene_id'] = scene_id
        configs['root_dirpath'] = root_dirpath
        configs['output_dirpath'] = scene_output_dirpath
        train_data_loader = get_data_loader(configs, database_dirpath, mode='train')
        train_data_preprocessor = get_data_preprocessor(configs,
                                                        mode='train',
                                                        raw_data_dict=train_data_loader.load_data())
        val_data_loader = get_data_loader(configs, database_dirpath, mode='validation')
        val_data_preprocessor = get_data_preprocessor(configs,
                                                      mode='validation',
                                                      raw_data_dict=val_data_loader.load_data(),
                                                      model_configs=train_data_preprocessor.get_model_configs())
        model_configs = train_data_preprocessor.get_model_configs()
        model = get_model(configs, model_configs=model_configs)
        model = torch.nn.DataParallel(model, device_ids=configs['device'])
        loss_computer = LossComputer(configs)

        optimizers, lr_decayers = get_optimizers(configs, model)

        save_model_configs(scene_output_dirpath, model_configs, 'ModelConfigs.json')

        # Start training
        trainer = Trainer(configs, model_configs, train_data_preprocessor, val_data_preprocessor, model, loss_computer,
                          optimizers, lr_decayers, scene_output_dirpath, configs['device'])
        trainer.train()

        del trainer, optimizers, lr_decayers, loss_computer, model
        del train_data_loader, train_data_preprocessor, val_data_loader, val_data_preprocessor
        torch.cuda.empty_cache()
    return
