# Shree KRISHNAya Namaha
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import datetime
import os
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skimage.io
import skvideo.io

import Tester07 as Tester
import Trainer10 as Trainer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_video(path: Path, video: numpy.ndarray):
    if path.exists():
        return
    try:
        skvideo.io.vwrite(path.as_posix(), video,
                          inputdict={'-r': str(15)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p'}, verbosity=1)
    except (OSError, NameError):
        pass
    return


def start_training(train_configs: dict) -> None:
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / 'data/databases' / train_configs['database_dirpath']

    # Setup output dirpath
    output_dirpath = root_dirpath / f'runs/training/train{train_configs["train_num"]:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    scene_nums = train_configs['data_loader'].get('scene_nums', None)
    Trainer.save_configs(output_dirpath, train_configs)
    train_configs['data_loader']['scene_nums'] = scene_nums

    if train_configs['data_loader']['scene_nums'] is None:
        set_num = train_configs['data_loader']['train_set_num']
        video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        scene_nums = numpy.unique(video_data['scene_num'].to_numpy())
    scene_ids = [f'{scene_num:05}' for scene_num in scene_nums]
    train_configs['data_loader']['scene_ids'] = scene_ids
    Trainer.start_training(train_configs)
    return


def start_testing(test_configs: dict):
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / 'data/databases' / test_configs['database_dirpath']
    optimize_camera_params = 'optimize_camera_params' in test_configs

    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    train_video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TrainVideosData.csv'
    test_video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    train_video_data = pandas.read_csv(train_video_datapath)
    test_video_data = pandas.read_csv(test_video_datapath)
    scene_nums = test_configs.get('scene_nums', test_video_data['scene_num'].to_numpy())
    scene_nums = numpy.unique(scene_nums)
    scenes_data = {}
    for scene_num in scene_nums:
        scene_id = f'{scene_num:05}'
        scenes_data[scene_id] = {
            'output_dirname': scene_id,
            'frames_data': {}
        }

        extrinsics_path = database_dirpath / f'test/database_data/{scene_id}/CameraExtrinsics.csv'
        extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        # Intrinsics and frames required to compute plane sweep volume for conv visibility prediction
        intrinsics_path = database_dirpath / f'test/database_data/{scene_id}/CameraIntrinsics.csv'
        intrinsics = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))
        frames_dirpath = database_dirpath / f'test/database_data/{scene_id}/rgb'

        if optimize_camera_params:
            camera_params_dirname = test_configs['optimize_camera_params']['dirname']
            intrinsics_noisy_path = database_dirpath / f'test/estimated_camera_params/{camera_params_dirname}/{scene_num:05}/CameraIntrinsics{test_configs["resolution_suffix"]}.csv'
            extrinsics_noisy_path = database_dirpath / f'test/estimated_camera_params/{camera_params_dirname}/{scene_num:05}/CameraExtrinsics.csv'
            intrinsics_noisy = numpy.loadtxt(intrinsics_noisy_path.as_posix(), delimiter=',').reshape((-1, 3, 3))
            extrinsics_noisy = numpy.loadtxt(extrinsics_noisy_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

        test_frame_nums = test_video_data.loc[test_video_data['scene_num'] == scene_num]['pred_frame_num'].to_list()
        train_frame_nums = train_video_data.loc[train_video_data['scene_num'] == scene_num]['pred_frame_num'].to_list()
        frame_nums = numpy.unique(sorted(test_frame_nums + train_frame_nums))
        for frame_num in frame_nums:
            is_train_frame = frame_num in train_frame_nums
            scenes_data[scene_id]['frames_data'][frame_num] = {
                'extrinsic': extrinsics[frame_num],
                'intrinsic': intrinsics[frame_num],
                'is_train_frame': is_train_frame,
            }
            if optimize_camera_params:
                frame_path = frames_dirpath / f'{frame_num:04}.png'
                frame = read_image(frame_path)
                scenes_data[scene_id]['frames_data'][frame_num]['frame'] = frame
                if is_train_frame:
                    scenes_data[scene_id]['frames_data'][frame_num]['intrinsic_noisy'] = intrinsics_noisy[train_frame_nums.index(frame_num)]
                    scenes_data[scene_id]['frames_data'][frame_num]['extrinsic_noisy'] = extrinsics_noisy[train_frame_nums.index(frame_num)]
    Tester.start_testing(test_configs, scenes_data, save_depth=True, save_depth_var=True, save_visibility=False, optimize_camera_params=optimize_camera_params)

    # Run QA
    qa_filepath = Path('qa/00_Common/src/AllMetrics01_RealEstate.py')
    gt_depth_dirpath = Path('../data/dense_input_radiance_fields/NeRF/runs/testing/test0001')
    cmd = f'python {qa_filepath.absolute().as_posix()} ' \
          f'--demo_function_name demo2 ' \
          f'--pred_videos_dirpath {output_dirpath.absolute().as_posix()} ' \
          f'--database_dirpath {database_dirpath.absolute().as_posix()} ' \
          f'--gt_depth_dirpath {gt_depth_dirpath.absolute().as_posix()} ' \
          f'--frames_datapath {test_video_datapath.absolute().as_posix()} ' \
          f'--pred_frames_dirname predicted_frames ' \
          f'--pred_depths_dirname predicted_depths ' \
          f'--mask_folder_name {test_configs["qa_masks_dirname"]}'
    os.system(cmd)
    return


def start_testing_videos(test_configs: dict):
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / 'data/databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_nums = test_configs.get('scene_nums', video_data['scene_num'].to_numpy())
    scene_nums = numpy.unique(scene_nums)

    videos_data = [1, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_num in scene_nums:
            scenes_data = {}
            scene_id = f'{scene_num:05}'
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/{scene_id}.csv'
            if not extrinsics_path.exists():
                continue
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[frame_num + 1]
                }
            output_dir_suffix = f'_video{video_num:02}'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'predicted_frames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'PredictedVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def start_testing_static_videos(test_configs: dict):
    """
    This is for view_dirs visualization
    :param test_configs:
    :return:
    """
    root_dirpath = Path('../')
    database_dirpath = root_dirpath / 'data/databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"runs/testing/test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_nums = test_configs.get('scene_nums', video_data['scene_num'].to_numpy())
    scene_nums = numpy.unique(scene_nums)

    videos_data = [4, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_num in scene_nums:
            scenes_data = {}
            scene_id = f'{scene_num:05}'
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'train_test_sets/set{set_num:02}/video_poses{video_num:02}/{scene_id}.csv'
            if not extrinsics_path.exists():
                continue
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[0],
                    'extrinsic_viewcam': extrinsics[frame_num + 1],
                }
            output_dir_suffix = f'_video{video_num:02}_static_camera'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'predicted_frames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'StaticCameraVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def demo1a() -> None:
    train_num = 12
    test_num = 12
    scene_nums = [0, 1, 3, 4, 6]
    # scene_nums = [0]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'description': 'Simple-NeRF: 2 views',
            'database': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader02',
                'data_preprocessor_name': 'DataPreprocessor10',
                'train_set_num': 12,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'camera_params': {
                    'load_database_intrinsics': True,
                    'load_database_extrinsics': True,
                    # 'load_noisy_intrinsics': True,
                    # 'load_noisy_extrinsics': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'scene_data': {
                    'load_database_bounds': True,
                    # 'load_noisy_bounds': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'sparse_depth': {
                    'dirname': 'DE12',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'SimpleNeRF17',
                'coarse_model': {
                    'num_samples': 64,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'fine_model': {
                    'num_samples': 128,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'augmentations': [
                    {
                        'name': 'points_augmentation',
                        'coarse_model': {
                            'points_net_depth': 8,
                            'views_net_depth': 1,
                            'points_net_width': 256,
                            'views_net_width': 128,
                            'points_positional_encoding_degree': 10,
                            'points_sigma_positional_encoding_degree': 3,
                            'views_positional_encoding_degree': 4,
                            'use_view_dirs': True,
                            'view_dependent_rgb': True,
                            'predict_visibility': False,
                        },
                        # 'fine_model': {
                        #     'points_net_depth': 8,
                        #     'views_net_depth': 1,
                        #     'points_net_width': 256,
                        #     'views_net_width': 128,
                        #     'points_positional_encoding_degree': 10,
                        #     'points_sigma_positional_encoding_degree': 3,
                        #     'views_positional_encoding_degree': 4,
                        #     'use_view_dirs': True,
                        #     'view_dependent_rgb': True,
                        #     'predict_visibility': False,
                        # }
                    },
                    {
                        'name': 'views_augmentation',
                        'coarse_model': {
                            'points_net_depth': 8,
                            'views_net_depth': 1,
                            'points_net_width': 256,
                            'views_net_width': 128,
                            'points_positional_encoding_degree': 10,
                            'use_view_dirs': False,
                            'view_dependent_rgb': False,
                            'predict_visibility': False,
                        },
                        # 'fine_model': {
                        #     'points_net_depth': 8,
                        #     'views_net_depth': 1,
                        #     'points_net_width': 256,
                        #     'views_net_width': 128,
                        #     'points_positional_encoding_degree': 10,
                        #     'use_view_dirs': False,
                        #     'view_dependent_rgb': False,
                        #     'predict_visibility': False,
                        # }
                    },
                ],
                'learn_camera_focal_length': False,
                'learn_camera_rotation': False,
                'learn_camera_translation': False,
                'chunk': 4*1024,
                'lindisp': False,
                'netchunk': 16*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE14',
                    'weight': 1,
                    "models_to_regularize": {
                        "main_coarse": 1,
                        "main_fine": 1,
                        "points_augmentation_coarse": 1,
                        "views_augmentation_coarse": 1
                    }
                },
                {
                    'name': 'SparseDepthMSE14',
                    'weight': 0.1,
                    "models_to_regularize": {
                        "main_coarse": 1,

                        "points_augmentation_coarse": 1,
                        "views_augmentation_coarse": 1
                    }
                },
                {
                    'name': 'AugmentationsDepthLoss11',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'CoarseFineConsistencyLoss34',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
            ],
            'optimizers': [
                {
                    'name': 'optimizer_main',
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'lr_decayer_name': 'NeRFLearningRateDecayer03',
                    'lr_initial': 5e-4,
                    'lr_decay': 250,
                },
            ],
            'resume_training': True,
            'sub_batch_size': 4096,
            'num_iterations': 100000,
            'validation_interval': 1000000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 25000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            # 'seed': 0,
            'device': [0, ],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 12,
            'train_num': train_num,
            'model_name': f'Model_Iter{train_configs["num_iterations"]:06}.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'qa_masks_dirname': 'VM12',
            'scene_nums': [scene_num],
            'device': [0, ],
        }
        start_training(train_configs)
        start_testing(test_configs)
        # start_testing_videos(test_configs)
        # start_testing_static_videos(test_configs)
    return


def demo1b() -> None:
    train_num = 42
    test_num = 42
    scene_nums = [0, 1, 3, 4, 6]
    # scene_nums = [0]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'description': 'Simple-NeRF: 3 views',
            'database': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader02',
                'data_preprocessor_name': 'DataPreprocessor10',
                'train_set_num': 13,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'camera_params': {
                    'load_database_intrinsics': True,
                    'load_database_extrinsics': True,
                    # 'load_noisy_intrinsics': True,
                    # 'load_noisy_extrinsics': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'scene_data': {
                    'load_database_bounds': True,
                    # 'load_noisy_bounds': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'sparse_depth': {
                    'dirname': 'DE13',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'SimpleNeRF17',
                'coarse_model': {
                    'num_samples': 64,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'fine_model': {
                    'num_samples': 128,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'augmentations': [
                    {
                        'name': 'points_augmentation',
                        'coarse_model': {
                            'points_net_depth': 8,
                            'views_net_depth': 1,
                            'points_net_width': 256,
                            'views_net_width': 128,
                            'points_positional_encoding_degree': 10,
                            'points_sigma_positional_encoding_degree': 3,
                            'views_positional_encoding_degree': 4,
                            'use_view_dirs': True,
                            'view_dependent_rgb': True,
                            'predict_visibility': False,
                        },
                        # 'fine_model': {
                        #     'points_net_depth': 8,
                        #     'views_net_depth': 1,
                        #     'points_net_width': 256,
                        #     'views_net_width': 128,
                        #     'points_positional_encoding_degree': 10,
                        #     'points_sigma_positional_encoding_degree': 3,
                        #     'views_positional_encoding_degree': 4,
                        #     'use_view_dirs': True,
                        #     'view_dependent_rgb': True,
                        #     'predict_visibility': False,
                        # }
                    },
                    {
                        'name': 'views_augmentation',
                        'coarse_model': {
                            'points_net_depth': 8,
                            'views_net_depth': 1,
                            'points_net_width': 256,
                            'views_net_width': 128,
                            'points_positional_encoding_degree': 10,
                            'use_view_dirs': False,
                            'view_dependent_rgb': False,
                            'predict_visibility': False,
                        },
                        # 'fine_model': {
                        #     'points_net_depth': 8,
                        #     'views_net_depth': 1,
                        #     'points_net_width': 256,
                        #     'views_net_width': 128,
                        #     'points_positional_encoding_degree': 10,
                        #     'use_view_dirs': False,
                        #     'view_dependent_rgb': False,
                        #     'predict_visibility': False,
                        # }
                    },
                ],
                'learn_camera_focal_length': False,
                'learn_camera_rotation': False,
                'learn_camera_translation': False,
                'chunk': 4*1024,
                'lindisp': False,
                'netchunk': 16*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE14',
                    'weight': 1,
                    "models_to_regularize": {
                        "main_coarse": 1,
                        "main_fine": 1,
                        "points_augmentation_coarse": 1,
                        "views_augmentation_coarse": 1
                    }
                },
                {
                    'name': 'SparseDepthMSE14',
                    'weight': 0.1,
                    "models_to_regularize": {
                        "main_coarse": 1,
                        "points_augmentation_coarse": 1,
                        "views_augmentation_coarse": 1
                    }
                },
                {
                    'name': 'AugmentationsDepthLoss11',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'CoarseFineConsistencyLoss34',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
            ],
            'optimizers': [
                {
                    'name': 'optimizer_main',
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'lr_decayer_name': 'NeRFLearningRateDecayer03',
                    'lr_initial': 5e-4,
                    'lr_decay': 250,
                },
            ],
            'resume_training': True,
            'sub_batch_size': 4096,
            'num_iterations': 100000,
            'validation_interval': 1000000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 25000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            # 'seed': 0,
            'device': [0, ],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 13,
            'train_num': train_num,
            'model_name': f'Model_Iter{train_configs["num_iterations"]:06}.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'qa_masks_dirname': 'VM13',
            'scene_nums': [scene_num],
            'device': [0, ],
        }
        start_training(train_configs)
        start_testing(test_configs)
        # start_testing_videos(test_configs)
        # start_testing_static_videos(test_configs)
    return


def demo1c() -> None:
    train_num = 43
    test_num = 43
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'description': 'Simple-NeRF: 4 views',
            'database': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader02',
                'data_preprocessor_name': 'DataPreprocessor10',
                'train_set_num': 14,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'camera_params': {
                    'load_database_intrinsics': True,
                    'load_database_extrinsics': True,
                    # 'load_noisy_intrinsics': True,
                    # 'load_noisy_extrinsics': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'scene_data': {
                    'load_database_bounds': True,
                    # 'load_noisy_bounds': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'sparse_depth': {
                    'dirname': 'DE14',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'SimpleNeRF17',
                'coarse_model': {
                    'num_samples': 64,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'fine_model': {
                    'num_samples': 128,
                    'points_net_depth': 8,
                    'views_net_depth': 1,
                    'points_net_width': 256,
                    'views_net_width': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': False,
                },
                'augmentations': [
                    {
                        'name': 'points_augmentation',
                        'coarse_model': {
                            'points_net_depth': 8,
                            'views_net_depth': 1,
                            'points_net_width': 256,
                            'views_net_width': 128,
                            'points_positional_encoding_degree': 10,
                            'points_sigma_positional_encoding_degree': 3,
                            'views_positional_encoding_degree': 4,
                            'use_view_dirs': True,
                            'view_dependent_rgb': True,
                            'predict_visibility': False,
                        },
                        # 'fine_model': {
                        #     'points_net_depth': 8,
                        #     'views_net_depth': 1,
                        #     'points_net_width': 256,
                        #     'views_net_width': 128,
                        #     'points_positional_encoding_degree': 10,
                        #     'points_sigma_positional_encoding_degree': 3,
                        #     'views_positional_encoding_degree': 4,
                        #     'use_view_dirs': True,
                        #     'view_dependent_rgb': True,
                        #     'predict_visibility': False,
                        # }
                    },
                    {
                        'name': 'views_augmentation',
                        'coarse_model': {
                            'points_net_depth': 8,
                            'views_net_depth': 1,
                            'points_net_width': 256,
                            'views_net_width': 128,
                            'points_positional_encoding_degree': 10,
                            'use_view_dirs': False,
                            'view_dependent_rgb': False,
                            'predict_visibility': False,
                        },
                        # 'fine_model': {
                        #     'points_net_depth': 8,
                        #     'views_net_depth': 1,
                        #     'points_net_width': 256,
                        #     'views_net_width': 128,
                        #     'points_positional_encoding_degree': 10,
                        #     'use_view_dirs': False,
                        #     'view_dependent_rgb': False,
                        #     'predict_visibility': False,
                        # }
                    },
                ],
                'learn_camera_focal_length': False,
                'learn_camera_rotation': False,
                'learn_camera_translation': False,
                'chunk': 4*1024,
                'lindisp': False,
                'netchunk': 16*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE14',
                    'weight': 1,
                    "models_to_regularize": {
                        "main_coarse": 1,
                        "main_fine": 1,
                        "points_augmentation_coarse": 1,
                        "views_augmentation_coarse": 1
                    }
                },
                {
                    'name': 'SparseDepthMSE14',
                    'weight': 0.1,
                    "models_to_regularize": {
                        "main_coarse": 1,
                        "points_augmentation_coarse": 1,
                        "views_augmentation_coarse": 1
                    }
                },
                {
                    'name': 'AugmentationsDepthLoss11',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
                {
                    'name': 'CoarseFineConsistencyLoss34',
                    'iter_weights': {
                        '0': 0, '10000': 0.1
                    },
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
            ],
            'optimizers': [
                {
                    'name': 'optimizer_main',
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'lr_decayer_name': 'NeRFLearningRateDecayer03',
                    'lr_initial': 5e-4,
                    'lr_decay': 250,
                },
            ],
            'resume_training': True,
            'sub_batch_size': 4096,
            'num_iterations': 100000,
            'validation_interval': 1000000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 25000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            # 'seed': 0,
            'device': [0, ],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 14,
            'train_num': train_num,
            'model_name': f'Model_Iter{train_configs["num_iterations"]:06}.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'qa_masks_dirname': 'VM14',
            'scene_nums': [scene_num],
            'device': [0, ],
        }
        start_training(train_configs)
        start_testing(test_configs)
        # start_testing_videos(test_configs)
        # start_testing_static_videos(test_configs)
    return


def demo2b() -> None:
    train_num = 212
    test_num = 212
    scene_nums = [0, 1, 3, 4, 6]

    for scene_num in scene_nums:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'description': 'Simple-TensoRF: 3 views',
            'database': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'data_loader': {
                'data_loader_name': 'RealEstateDataLoader02',
                'data_preprocessor_name': 'DataPreprocessor10',
                'train_set_num': 13,
                'scene_nums': [scene_num],
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'camera_params': {
                    'load_database_intrinsics': True,
                    'load_database_extrinsics': True,
                    # 'load_noisy_intrinsics': True,
                    # 'load_noisy_extrinsics': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'scene_data': {
                    'load_database_bounds': True,
                    # 'load_noisy_bounds': True,
                    # 'dirname': 'PEL001_PDE03',
                },
                'sparse_depth': {
                    'dirname': 'DE13',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'SimpleTensoRF09',
                'coarse_model': {
                    'decomposition_type': 'VectorMatrix',
                    'num_samples_max': 1e6,
                    'num_components_density': [16, 4, 4],
                    'num_components_color': [48, 12, 12],
                    'bounding_box': [
                        [-1.5, -1.67, -1.0],
                        [+1.5, +1.67, +1.0],
                    ],
                    'num_voxels_initial': 2097156,  # 128**3 + 4
                    'num_voxels_final': 640 ** 3,
                    'tensor_upsampling_iters': [2000, 3000, 4000, 5500],
                    'num_voxels_per_sample': 0.5,
                    'alpha_mask_update_iters': [2500],
                    'alpha_mask_threshold': 0.0001,
                    'ray_marching_weight_threshold': 0.0001,
                    'use_view_dirs': True,
                    'view_dependent_color': True,
                    # 'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 0,
                    'features_positional_encoding_degree': 0,
                    'features_dimension_color': 27,
                    'density_offset': -10,
                    'distance_scale': 25,
                    'density_predictor': 'ReLU',
                    'color_predictor': 'MLP_Features',
                    'num_units_color_predictor': 128,
                    'predict_visibility': False,
                },
                'augmentations': [
                    {
                        'name': 'points_augmentation',
                        'coarse_model': {
                            'decomposition_type': 'VectorMatrix',
                            'num_samples_max': 1e6,
                            'num_components_density': [4, 4, 4],
                            'num_components_color': [48, 12, 12],
                            'bounding_box': [
                                [-1.5, -1.67, -0.5],
                                [+1.5, +1.67, +1.0],
                            ],
                            'num_voxels_initial': 64**3,
                            'num_voxels_final': 160 ** 3,
                            'tensor_upsampling_iters': [2000, 3000, 4000, 5500],
                            'num_voxels_per_sample': 0.5,
                            'alpha_mask_update_iters': [2500],
                            'alpha_mask_threshold': 0.0001,
                            'ray_marching_weight_threshold': 0.0001,
                            'use_view_dirs': True,
                            'view_dependent_color': True,
                            # 'points_positional_encoding_degree': 10,
                            'views_positional_encoding_degree': 0,
                            'features_positional_encoding_degree': 0,
                            'features_dimension_color': 27,
                            'density_offset': -10,
                            'distance_scale': 25,
                            'density_predictor': 'ReLU',
                            'color_predictor': 'MLP_Features',
                            'num_units_color_predictor': 128,
                            'predict_visibility': False,
                        },
                    },
                ],
                'learn_camera_focal_length': False,
                'learn_camera_rotation': False,
                'learn_camera_translation': False,
                'chunk': 4 * 1024,
                'lindisp': False,
                'perturb': True,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE14',
                    'weight': 1,
                    'models_to_regularize': {'main_coarse': 1, 'main_fine': 1, 'points_augmentation_coarse': 1, 'views_augmentation_coarse': 1},
                },
                {
                    'name': 'SparseDepthMSE14',
                    'weight': 0.1,
                    'models_to_regularize': {'main_coarse': 1, 'points_augmentation_coarse': 1, 'views_augmentation_coarse': 1},
                },
                {
                    'name': 'TotalVariationLoss05',
                    'weight': 1e-2,
                    'weight_density': 1,
                    'weight_color': 1,
                    'models_to_regularize': {'main_coarse': 1, 'main_fine': 1, 'points_augmentation_coarse': 1, 'views_augmentation_coarse': 1},
                },
                {
                    'name': 'MassConcentrationLoss07',
                    'iter_weights': {'0': 0, '5000': 0.01},
                    'num_bins': 5,
                    'models_to_regularize': {'points_augmentation_coarse': 1, 'views_augmentation_coarse': 1}
                },
                {
                    'name': 'AugmentationsDepthLoss11',
                    'iter_weights': {'0': 0, '1000': 0.1},
                    'rmse_threshold': 0.1,
                    'patch_size': [5, 5],
                },
            ],
            'optimizers': [
                {
                    'name': 'optimizer_main',
                    'beta1': 0.9,
                    'beta2': 0.99,
                    'lr_decayer_name': 'TensoRFLearningRateDecayer01',
                    'lr_initial_tensor': 2e-2,
                    'lr_initial_network': 1e-3,
                    'lr_decay_ratio': 0.1,
                    'lr_decay_iters': None,
                },
            ],
            'resume_training': True,
            'sub_batch_size': 4096,
            'num_iterations': 25000,
            'validation_interval': 100000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 5000,
            'mixed_precision_training': False,
            'seed': numpy.random.randint(1000),
            # 'seed': 0,
            'device': [0, ],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 13,
            'train_num': train_num,
            'model_name': f'Model_Iter0{train_configs["num_iterations"]:05}.tar',
            'database_name': 'RealEstate10K',
            'database_dirpath': 'RealEstate10K/data',
            'qa_masks_dirname': 'VM13',
            'scene_nums': [scene_num],
            'device': [0, ],
        }
        start_training(train_configs)
        start_testing(test_configs)
        # start_testing_videos(test_configs)
        # start_testing_static_videos(test_configs)
    return


def demo_resume_testing():
    train_num = 12
    test_num = 12
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 12,
        'train_num': train_num,
        'model_name': 'Model_Iter100000.tar',
        'database_name': 'RealEstate10K',
        'database_dirpath': 'RealEstate10K/data',
        'qa_masks_dirname': 'VM12',
        'scene_nums': [0, 1, 3, 4, 6],
        'device': [0, ],
    }
    start_testing(test_configs)
    # start_testing_videos(test_configs)
    # start_testing_static_videos(test_configs)
    return


def main() -> None:
    demo1a()
    demo1b()
    demo1c()
    demo2b()
    return


if __name__ == "__main__":
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
