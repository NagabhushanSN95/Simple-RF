# Shree KRISHNAya Namaha
# Runs all metrics serially
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import argparse
import datetime
import importlib.util
import time
import traceback
from pathlib import Path
from typing import List

import SceneWiseGrouper

this_filepath = Path(__file__)
this_filename = Path(__file__).stem


def run_all_specified_qa(metric_filepaths: List[Path], pred_videos_dirpath: Path, database_dirpath: Path,
                         gt_depth_dirpath: Path, frames_datapath: Path, pred_frames_dirname: str,
                         pred_depths_dirname: str, downsampling_factor: int,
                         mask_folder_name: str):
    args_values = locals()
    qa_scores = {}
    for metric_file_path in metric_filepaths:
        spec = importlib.util.spec_from_file_location('module.name', metric_file_path.absolute().resolve().as_posix())
        qa_module = importlib.util.module_from_spec(spec)
        # noinspection PyUnresolvedReferences
        spec.loader.exec_module(qa_module)
        function_arguments = {}
        for arg_name in run_all_specified_qa.__code__.co_varnames[:run_all_specified_qa.__code__.co_argcount]:
            # noinspection PyUnresolvedReferences
            if arg_name in qa_module.start_qa.__code__.co_varnames[:qa_module.start_qa.__code__.co_argcount]:
                function_arguments[arg_name] = args_values[arg_name]
        # noinspection PyUnresolvedReferences
        qa_score = qa_module.start_qa(**function_arguments)
        # noinspection PyUnresolvedReferences
        qa_name = qa_module.this_metric_name
        qa_scores[qa_name] = qa_score
    return qa_scores


def run_all_qa(pred_videos_dirpath: Path, database_dirpath: Path, gt_depth_dirpath: Path, frames_datapath: Path,
               pred_frames_dirname: str, pred_depths_dirname: str, 
               downsampling_factor: int, mask_folder_name: str):
    frame_metric_filepaths = [
        this_filepath.parent / '../../01_RMSE/src/RMSE01_RealEstate.py',
        this_filepath.parent / '../../02_PSNR/src/PSNR01_RealEstate.py',
        this_filepath.parent / '../../03_SSIM/src/SSIM01_RealEstate.py',
        this_filepath.parent / '../../04_LPIPS/src/LPIPS01_RealEstate.py',
        this_filepath.parent / '../../11_MaskedRMSE/src/MaskedRMSE11_RealEstate.py',
        this_filepath.parent / '../../12_MaskedPSNR/src/MaskedPSNR11_RealEstate.py',
        this_filepath.parent / '../../13_MaskedSSIM/src/MaskedSSIM11_RealEstate.py',
        this_filepath.parent / '../../14_MaskedLPIPS/src/MaskedLPIPS11_RealEstate.py',

        this_filepath.parent / '../../05_DepthRMSE/src/DepthRMSE01_RealEstate.py',
        this_filepath.parent / '../../07_DepthMAE/src/DepthMAE01_RealEstate.py',
        this_filepath.parent / '../../09_DepthSROCC/src/DepthSROCC01_RealEstate.py',
        this_filepath.parent / '../../15_MaskedDepthRMSE/src/MaskedDepthRMSE11_RealEstate.py',
        this_filepath.parent / '../../17_MaskedDepthMAE/src/MaskedDepthMAE11_RealEstate.py',
        this_filepath.parent / '../../19_MaskedDepthSROCC/src/MaskedDepthSROCC11_RealEstate.py',
    ]
    qa_scores = run_all_specified_qa(frame_metric_filepaths, pred_videos_dirpath, database_dirpath, gt_depth_dirpath,
                                     frames_datapath, pred_frames_dirname, pred_depths_dirname,
                                     downsampling_factor, mask_folder_name)
    test_num = int(pred_videos_dirpath.stem[4:])
    SceneWiseGrouper.group_qa_scores(pred_videos_dirpath.parent, [test_num])
    return qa_scores


def demo1():
    root_dirpath = Path('../../../../')
    pred_videos_dirpath = root_dirpath / 'runs/testing/test0011'
    database_dirpath = root_dirpath / 'data/databases/RealEstate10K/data'
    gt_depth_dirpath = root_dirpath / 'data/dense_input_radiance_fields/NeRF/runs/testing/test0001'
    frames_data_path = database_dirpath / 'train_test_sets/set12/TestVideosData.csv'
    pred_frames_dirname = 'predicted_frames'
    pred_depths_dirname = 'predicted_depths'
    downsampling_factor = 1
    mask_folder_name = 'VM12'
    qa_scores = run_all_qa(pred_videos_dirpath, database_dirpath, gt_depth_dirpath, frames_data_path,
                           pred_frames_dirname, pred_depths_dirname, downsampling_factor, mask_folder_name)
    return qa_scores


def demo2(args: dict):
    pred_videos_dirpath = args['pred_videos_dirpath']
    if pred_videos_dirpath is None:
        raise RuntimeError(f'Please provide pred_videos_dirpath')
    pred_videos_dirpath = Path(pred_videos_dirpath)

    database_dirpath = args['database_dirpath']
    if database_dirpath is None:
        raise RuntimeError(f'Please provide database_dirpath')
    database_dirpath = Path(database_dirpath)

    gt_depth_dirpath = args['gt_depth_dirpath']
    if gt_depth_dirpath is None:
        raise RuntimeError(f'Please provide gt_depth_dirpath')
    gt_depth_dirpath = Path(gt_depth_dirpath)

    frames_datapath = args['frames_datapath']
    if frames_datapath is None:
        raise RuntimeError(f'Please provide frames_datapath')
    frames_datapath = Path(frames_datapath)

    pred_frames_dirname = args['pred_frames_dirname']
    pred_depths_dirname = args['pred_depths_dirname']
    downsampling_factor = args['downsampling_factor']
    mask_folder_name = args['mask_folder_name']

    qa_scores = run_all_qa(pred_videos_dirpath, database_dirpath, gt_depth_dirpath, frames_datapath,
                           pred_frames_dirname, pred_depths_dirname,  downsampling_factor, mask_folder_name)
    return qa_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_videos_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--gt_depth_dirpath')
    parser.add_argument('--frames_datapath')
    parser.add_argument('--pred_frames_dirname', default='predicted_frames')
    parser.add_argument('--pred_depths_dirname', default='predicted_depths')
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--mask_folder_name', default=None)
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_videos_dirpath': args.pred_videos_dirpath,
        'database_dirpath': args.database_dirpath,
        'gt_depth_dirpath': args.gt_depth_dirpath,
        'frames_datapath': args.frames_datapath,
        'pred_frames_dirname': args.pred_frames_dirname,
        'pred_depths_dirname': args.pred_depths_dirname,
        'downsampling_factor': args.downsampling_factor,
        'mask_folder_name': args.mask_folder_name,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        qa_scores = demo1()
    elif args['demo_function_name'] == 'demo2':
        qa_scores = demo2(args)
    else:
        raise RuntimeError(f'Unknown demo function: {args["demo_function_name"]}')
    return qa_scores


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    parsed_args = parse_args()
    try:
        qa_scores_dict = main(parsed_args)
        qa_scores_str = '\n'.join([f'{key}: {value}' for key, value in qa_scores_dict.items()])
        run_result = f'Program completed successfully!\n\n{parsed_args["pred_videos_dirpath"]}\n{qa_scores_str}'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
