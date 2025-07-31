import numpy as np
import torch
import math

from math import cos, atan2, asin

def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z



def posenc_nerf(x, min_deg=0, max_deg=15):
    """Concatenate x and its positional encodings, following NeRF."""
    device = x.device
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)
    xb = torch.reshape(
    (x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    emb = torch.sin(torch.cat([xb, xb + np.pi / 2.], dim=-1))
    return torch.cat([x, emb], dim=-1)

def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions

def get_rot_x(angle,device='cpu'):
    '''
    transformation matrix that rotates a point about the standard X axis
    '''
    Rx = torch.zeros((3, 3),device=device)
    Rx[0, 0] = 1
    Rx[1, 1] = torch.cos(angle)
    Rx[1, 2] = -torch.sin(angle)
    Rx[2, 1] = torch.sin(angle)
    Rx[2, 2] = torch.cos(angle)

    return Rx

def get_rot_y(angle,device='cpu'):
    '''
    transformation matrix that rotates a point about the standard Y axis
    '''
    Ry = torch.zeros((3, 3),device=device)
    Ry[0, 0] = torch.cos(angle)
    Ry[0, 2] = -torch.sin(angle)
    Ry[2, 0] = torch.sin(angle)
    Ry[2, 2] = torch.cos(angle)
    Ry[1, 1] = 1

    return Ry.to(device)

def get_rot_z(angle,device='cpu'):
    '''
    transformation matrix that rotates a point about the standard Z axis
    '''
    Rz = torch.zeros((3, 3),device=device)
    Rz[0, 0] = torch.cos(angle)
    Rz[0, 1] = -torch.sin(angle)
    Rz[1, 0] = torch.sin(angle)
    Rz[1, 1] = torch.cos(angle)
    Rz[2, 2] = 1

    return Rz

def calculate_rotation(camera_angles,device='cpu'):
    rot_z = get_rot_z(camera_angles[2],device=device)
    rot_y = get_rot_y(camera_angles[1],device=device)
    rot_x = get_rot_x(camera_angles[0],device=device)
    final_rot = torch.mm(torch.mm(rot_z, rot_y), rot_x)
    return  final_rot[None,...]

def get_pose_map(angles, translation, fov_deg=18.837,
                 height=64, width=64, degrees=True,
                 max_deg_pos=15, max_deg_dir=8,
                 device='cpu'):
    if degrees:
        angles = angles * np.pi/180
    rotation = calculate_rotation(angles)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    M = torch.eye(4, device=device)
    M[:3, :3] = rotation
    M[:3, 3] = translation
    M_inv = M.inverse()
    ray_origins, ray_dirs, = get_ray_bundle(height, width, focal_length=intrinsics[0,0], tform_cam2world=M_inv)
    pos_emb_pos = posenc_nerf(ray_origins, min_deg=0, max_deg=max_deg_pos)
    pos_emb_dir = posenc_nerf(ray_dirs, min_deg=0, max_deg=max_deg_dir)
    pos_emb = torch.cat([pos_emb_pos, pos_emb_dir], dim=-1)
    return pos_emb, M