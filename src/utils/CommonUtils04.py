# Shree KRISHNAya Namaha
# Common Utility Functions
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

from pathlib import Path
from typing import Union

from matplotlib import pyplot

import numpy
import torch
from torch import Tensor

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_device(device):
    """
    Returns torch device object
    :param device: None//0/[0,],[0,1]. If multiple gpus are specified, first one is chosen
    :return:
    """
    if (device is None) or (device == '') or (not torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device0 = device[0] if isinstance(device, list) else device
        device = torch.device(f'cuda:{device0}')
    return device


def move_to_device(tensor_data: Union[torch.Tensor, list, dict], device):
    if isinstance(tensor_data, torch.Tensor):
        moved_tensor_data = tensor_data.to(device, non_blocking=True)
    elif isinstance(tensor_data, list):
        moved_tensor_data = []
        for tensor_elem in tensor_data:
            moved_tensor_data.append(move_to_device(tensor_elem, device))
    elif isinstance(tensor_data, dict):
        moved_tensor_data = {}
        for key in tensor_data:
            moved_tensor_data[key] = move_to_device(tensor_data[key], device)
    else:
        moved_tensor_data = tensor_data
    return moved_tensor_data


# TODO: Remove this if not required during inference
def get_rays_np(pixel_id, intrinsic, extrinsics, mip_nerf_used, nerf_synthetic=False):
    if nerf_synthetic:
        p = numpy.eye(4)
        p[1,1] = -1
        p[2,2] = -1
        extrinsics = numpy.matmul(numpy.matmul(p, extrinsics), p)
    x, y = pixel_id[:, 1], pixel_id[:, 2]
    if mip_nerf_used:
        x += 0.5
        y += 0.5
    ones = numpy.ones_like(x)
    points_homo = numpy.stack([x, y, ones], axis=1)  # (nr, 3)
    dirs = (numpy.linalg.inv(intrinsic) @ points_homo[:, :, None])[:, :, 0]  # (nr, 3)
    dirs[:, 1:] *= -1
    if nerf_synthetic:
        dirs[:, 1:] *= -1
    # Rotate ray directions from camera frame to the world frame
    rays_d = numpy.sum(dirs[..., numpy.newaxis, :] * extrinsics[:, :3, :3], axis=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = extrinsics[:, :3, -1]  # (nr, 3)
    return rays_o, rays_d


def get_rays_tr(pixel_id, intrinsic, extrinsics, mip_nerf_used, nerf_synthetic=False):
    if nerf_synthetic:
        p = torch.eye(4).to(extrinsics)
        p[1,1] = -1
        p[2,2] = -1
        extrinsics = torch.matmul(torch.matmul(p, extrinsics), p)
    # TODO: rename mip_nerf_used variable to something more meaningful
    pixel_id = pixel_id.float()
    x, y = pixel_id[:, 1], pixel_id[:, 2]
    if mip_nerf_used:
        x += 0.5
        y += 0.5
    ones = torch.ones_like(x)
    points_homo = torch.stack([x, y, ones], dim=1)  # (nr, 3)
    dirs = (torch.linalg.inv(intrinsic) @ points_homo[:, :, None])[:, :, 0]  # (nr, 3)
    dirs[:, 1:] *= -1
    if nerf_synthetic:
        dirs[:, 1:] *= -1
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[:, numpy.newaxis, :] * extrinsics[:, :3, :3], dim=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = extrinsics[:, :3, -1]  # (nr, 3)
    return rays_o, rays_d


# TODO: Remove this if not required during inference
def get_ndc_rays_np(rays_o, rays_d, resolution, intrinsic, near):
    h, w = resolution
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (w / (2. * fx)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (h / (2. * fy)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (w / (2. * fx)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (h / (2. * fy)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o_ndc = numpy.stack([o0, o1, o2], -1)
    rays_d_ndc = numpy.stack([d0, d1, d2], -1)
    return rays_o_ndc, rays_d_ndc


def get_ndc_rays_tr(rays_o, rays_d, resolution, intrinsic, near):
    h, w = resolution
    fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    # Shift ray origins to near plane
    t = -(near + rays_o[:, 2]) / rays_d[:, 2]
    rays_o = rays_o + t[:, None] * rays_d

    # Projection
    o0 = -1. / (w / (2. * fx)) * rays_o[:, 0] / rays_o[:, 2]
    o1 = -1. / (h / (2. * fy)) * rays_o[:, 1] / rays_o[:, 2]
    o2 = 1. + 2. * near / rays_o[:, 2]

    d0 = -1. / (w / (2. * fx)) * (rays_d[:, 0] / rays_d[:, 2] - rays_o[:, 0] / rays_o[:, 2])
    d1 = -1. / (h / (2. * fy)) * (rays_d[:, 1] / rays_d[:, 2] - rays_o[:, 1] / rays_o[:, 2])
    d2 = -2. * near / rays_o[:, 2]

    rays_o_ndc = torch.stack([o0, o1, o2], -1)
    rays_d_ndc = torch.stack([d0, d1, d2], -1)
    return rays_o_ndc, rays_d_ndc


# TODO: Remove this if not required during inference
def get_view_dirs_np(rays_d):
    view_dirs = rays_d / numpy.linalg.norm(rays_d, ord=2, axis=-1, keepdims=True)
    return view_dirs


def get_view_dirs_tr(rays_d):
    view_dirs = rays_d / torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True)
    return view_dirs


# TODO: Remove this if not required during inference
def get_radii_np(rays_d):
    dx = numpy.sqrt(numpy.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :]) ** 2, -1))
    dx = numpy.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = dx[..., None] * 2 / numpy.sqrt(12)
    return radii


def get_radii_tr(rays_d):
    dx = torch.sqrt(torch.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :]) ** 2, -1))
    dx = torch.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = dx[..., None] * 2 / torch.sqrt(12)
    return radii


# TODO: Remove this if not required during inference
def get_radii_ndc_np(rays_o_ndc):
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = numpy.sqrt(numpy.sum((rays_o_ndc[:, :-1, :, :] - rays_o_ndc[:, 1:, :, :]) ** 2, -1))
    dx = numpy.concatenate([dx, dx[:, -2:-1, :]], 1)

    dy = numpy.sqrt(numpy.sum((rays_o_ndc[:, :, :-1, :] - rays_o_ndc[:, :, 1:, :]) ** 2, -1))
    dy = numpy.concatenate([dy, dy[:, :, -2:-1]], 2)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii_ndc = (0.5 * (dx + dy))[..., None] * 2 / numpy.sqrt(12)
    return radii_ndc


def get_radii_ndc_tr(rays_o_ndc):
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(torch.sum((rays_o_ndc[:, :-1, :, :] - rays_o_ndc[:, 1:, :, :]) ** 2, -1))
    dx = torch.concatenate([dx, dx[:, -2:-1, :]], 1)

    dy = torch.sqrt(torch.sum((rays_o_ndc[:, :, :-1, :] - rays_o_ndc[:, :, 1:, :]) ** 2, -1))
    dy = torch.concatenate([dy, dy[:, :, -2:-1]], 2)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii_ndc = (0.5 * (dx + dy))[..., None] * 2 / torch.sqrt(12)
    return radii_ndc


def convert_depth_to_ndc(depths, rays_o, rays_d, near):
    oz = rays_o[:, 2:]
    dz = rays_d[:, 2:]
    tn = -(near + oz) / dz

    oz_prime = oz + tn * dz
    depths_ndc = 1 - oz_prime / (oz_prime + (depths - tn)*dz)
    return depths_ndc


def convert_depth_from_ndc(depths_ndc, rays_o, rays_d):
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
    constant = torch.where(depths_ndc == 1., 1e-3, 0.)
    # depth = (((oz + tn * dz) / (1 - z_vals_ndc + constant)) - oz) / dz
    depth = (oz + tn * dz) / dz * (1 / (1 - depths_ndc + constant) - 1) + tn
    return depth


def reproject(
        points_to_reproject: Tensor,
        poses_to_reproject_to: Tensor,
        intrinsics: Tensor,
) -> Tensor:
    """

    Args:
        points_to_reproject: (num_rays, )
        poses_to_reproject_to: (num_poses, 4, 4)
        intrinsics: (num_poses, 3, 3)

    Returns:

    """
    other_views_origins = poses_to_reproject_to[:, :3, 3]
    other_views_rotations = poses_to_reproject_to[:, :3, :3]
    reprojected_rays_d = points_to_reproject - other_views_origins

    # for changing coordinate system conventions
    permuter = torch.eye(3).to(points_to_reproject.device)
    permuter[1:] *= -1
    intrinsics = intrinsics[:1]  # TODO: Do not hard-code. Take intrinsic corresponding to each ray

    pos_2 = (intrinsics @ permuter[None] @ other_views_rotations.transpose(1, 2) @ reprojected_rays_d[..., None]).squeeze()
    pos_2 = pos_2[:, :2] / pos_2[:, 2:]
    return pos_2
