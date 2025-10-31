from typing import Optional, Union
import torch
from torch import nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gaussian-splatting'))

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.matrix.intrinsics_numpy import fov_to_focal_length
from scene.cameras import Camera



class RenderCam:
    def __init__(self, width, height, R, T, FoVx, FoVy, cx: Optional[float] = None, cy: Optional[float] = None,
                 znear: float = 0.01, zfar: float = 100,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 device: torch.device = torch.device('cuda'),
                 ):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.image_width = width
        self.image_height = height

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), device=device).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar,
                                                fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_pose(self) -> Pose:
        return GS_camera_to_pose(self)

    def to_intrinsics(self) -> Intrinsics:
        return GS_camera_to_intrinsics(self)

    @staticmethod
    def from_pose(pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int, znear: float = 0.01, zfar: float = 100) -> 'RenderCam':
        return pose_to_rendercam(pose, intrinsics, img_w, img_h, znear=znear, zfar=zfar)


# ==========================================================
# Conversion between Gaussian Splatting camera and dreifus Pose
# ==========================================================


def GS_camera_to_pose(camera: Union[Camera, RenderCam]) -> Pose:
    pose = Pose(camera.R.transpose(), camera.T, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
    return pose


def GS_camera_to_intrinsics(camera: Union[Camera, RenderCam]) -> Intrinsics:
    fx = fov_to_focal_length(camera.FoVx, camera.image_width)
    fy = fov_to_focal_length(camera.FoVy, camera.image_height)
    # Camera class doesn't have cx/cy, use image center
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0
    intrinsics = Intrinsics(fx, fy, cx=cx, cy=cy)
    return intrinsics


def pose_to_GS_camera(pose: Pose, intrinsics: Intrinsics, img_w: int, img_h: int) -> Camera:
    fov_x = intrinsics.get_fovx(img_w)
    fov_y = intrinsics.get_fovy(img_h)
    cx = intrinsics.cx
    cy = intrinsics.cy
    dummy_img = torch.empty((3, img_h, img_w))  # TODO: Find another way, s.t. we do not have to allocate this all the time

    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)
    T = pose.get_translation()
    R = pose.get_rotation_matrix().transpose()

    camera = Camera(0, R, T, fov_x, fov_y, dummy_img, None, None, None, cx=cx, cy=cy)

    return camera


def pose_to_rendercam(pose: Pose,
                      intrinsics: Intrinsics,
                      img_w: int,
                      img_h: int,
                      znear: float = 0.01,
                      zfar: float = 100,
                      device: torch.device = torch.device('cuda')) -> RenderCam:
    fov_x = intrinsics.get_fovx(img_w)
    fov_y = intrinsics.get_fovy(img_h)
    cx = intrinsics.cx
    cy = intrinsics.cy

    pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
    pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
    pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)
    T = pose.get_translation()
    R = pose.get_rotation_matrix().transpose()

    camera = RenderCam(img_w, img_h, R, T, fov_x, fov_y, cx=cx, cy=cy, znear=znear, zfar=zfar, device=device)

    return camera
