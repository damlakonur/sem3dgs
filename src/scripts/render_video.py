#!/usr/bin/env python3
"""
Render video from Gaussian Splatting model using GT or synthetic camera poses.
"""
import torch
import sys
import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gaussian-splatting'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scene import Scene, GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import argparse
from arguments import get_combined_args


def create_circle_trajectory(cameras, num_frames, height_offset=0.0, radius_scale=1.0):
    """
    Create a circular camera trajectory around the scene center.
    Horizontal orbit at fixed height.
    """
    cam_positions = []
    for cam in cameras:
        pos = -cam.R.T @ cam.T
        cam_positions.append(pos)
    
    cam_positions = np.array(cam_positions)
    center = cam_positions.mean(axis=0)
    
    distances = np.linalg.norm(cam_positions - center, axis=1)
    avg_distance = distances.mean() * radius_scale
    avg_height = cam_positions[:, 1].mean()
    
    up = np.array([0, -1, 0])
    
    ref_cam = cameras[0]
    znear, zfar = 0.01, 100.0
    circle_cameras = []
    
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * np.pi
        
        x = center[0] + avg_distance * np.cos(angle)
        y = avg_height + height_offset
        z = center[2] + avg_distance * np.sin(angle)
        cam_pos = np.array([x, y, z])
        
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up_new = np.cross(right, forward)
        
        R = np.stack([right, -up_new, -forward], axis=1)
        T = -R @ cam_pos
        
        world_view = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1)
        projection = getProjectionMatrix(
            znear=znear, zfar=zfar,
            fovX=ref_cam.FoVx, fovY=ref_cam.FoVy
        ).transpose(0, 1)
        full_proj = world_view @ projection
        
        new_cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovy=ref_cam.FoVy,
            fovx=ref_cam.FoVx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view.cuda(),
            full_proj_transform=full_proj.cuda()
        )
        circle_cameras.append(new_cam)
    
    return circle_cameras


def create_interpolate_trajectory(cameras, num_frames, loop=True):
    """
    Smooth interpolation between GT camera poses.
    """
    n_cams = len(cameras)
    
    quaternions = []
    translations = []
    
    for cam in cameras:
        R = cam.R
        rot = Rotation.from_matrix(R)
        quaternions.append(rot.as_quat())
        translations.append(cam.T)
    
    if loop:
        quaternions.append(quaternions[0])
        translations.append(translations[0])
        key_times = np.linspace(0, 1, n_cams + 1)
    else:
        key_times = np.linspace(0, 1, n_cams)
    
    quaternions = np.array(quaternions)
    translations = np.array(translations)
    
    trans_interp = interp1d(key_times, translations, axis=0, kind='cubic')
    
    ref_cam = cameras[0]
    znear, zfar = 0.01, 100.0
    
    interp_times = np.linspace(0, 1, num_frames, endpoint=not loop)
    interp_cameras = []
    
    for t in interp_times:
        idx = np.searchsorted(key_times, t, side='right') - 1
        idx = max(0, min(idx, len(key_times) - 2))
        
        t0, t1 = key_times[idx], key_times[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
        
        key_rots = Rotation.from_quat(np.stack([quaternions[idx], quaternions[idx + 1]]))
        slerp = Slerp([0, 1], key_rots)
        R_interp = slerp(alpha).as_matrix()
        T_interp = trans_interp(t)
        
        world_view = torch.tensor(getWorld2View2(R_interp, T_interp)).float().transpose(0, 1)
        projection = getProjectionMatrix(
            znear=znear, zfar=zfar,
            fovX=ref_cam.FoVx, fovY=ref_cam.FoVy
        ).transpose(0, 1)
        full_proj = world_view @ projection
        
        new_cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovy=ref_cam.FoVy,
            fovx=ref_cam.FoVx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view.cuda(),
            full_proj_transform=full_proj.cuda()
        )
        interp_cameras.append(new_cam)
    
    return interp_cameras


def frames_to_video(frames_dir, output_path, fps=30):
    """Convert frames to video using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        return False


def render_video(model_path, output_path, fps=30, skip=1, use_test=False, 
                  trajectory="gt", num_frames=None, height_offset=0.0, radius_scale=1.0):
    """Render video using GT or synthetic camera poses."""
    
    # Load model
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    
    # Override with our values
    sys.argv = ['render_video.py', '-m', model_path]
    args = get_combined_args(parser)
    
    lp = lp.extract(args)
    pp = pp.extract(args)
    
    with torch.no_grad():
        gaussians = GaussianModel(lp.sh_degree)
        scene = Scene(lp, gaussians, load_iteration=-1, shuffle=False)
        
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # Get base cameras (sorted by name)
        base_cameras = scene.getTestCameras() if use_test else scene.getTrainCameras()
        base_cameras = sorted(base_cameras, key=lambda c: c.image_name)
        base_cameras = base_cameras[::skip]
        
        # Generate trajectory
        if trajectory == "gt":
            cameras = base_cameras
            print(f"Using {len(cameras)} GT cameras")
        elif trajectory == "circle":
            n_frames = num_frames or 180
            cameras = create_circle_trajectory(base_cameras, n_frames, height_offset, radius_scale)
            print(f"Using {len(cameras)} circular trajectory cameras (height_offset={height_offset}, radius_scale={radius_scale})")
        elif trajectory == "interpolate":
            n_frames = num_frames or len(base_cameras) * 3
            cameras = create_interpolate_trajectory(base_cameras, n_frames, loop=True)
            print(f"Using {len(cameras)} interpolated cameras")
        
        # Create temp frames directory
        frames_dir = Path("_temp_rgb_frames")
        frames_dir.mkdir(exist_ok=True)
        
        for i, cam in enumerate(tqdm(cameras, desc="Rendering")):
            output = render(cam, gaussians, pp, bg_color)
            rgb = output['render']  # [3, H, W]
            
            # Convert to numpy
            rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure even dimensions for libx264
            H, W = rgb_np.shape[:2]
            new_H = H if H % 2 == 0 else H + 1
            new_W = W if W % 2 == 0 else W + 1
            if new_H != H or new_W != W:
                padded = np.ones((new_H, new_W, 3), dtype=np.uint8) * 255
                padded[:H, :W] = rgb_np
                rgb_np = padded
            
            Image.fromarray(rgb_np).save(frames_dir / f"frame_{i:04d}.png")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Convert to video
        print(f"\nCreating video at {fps} FPS...")
        if frames_to_video(frames_dir, output_path, fps):
            print(f"✓ Video saved to: {output_path}")
        else:
            print(f"✗ Failed to create video. Frames saved in {frames_dir}/")
        
        # Cleanup
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render video from GT or synthetic camera poses")
    parser.add_argument("-m", "--model", required=True, help="Path to model directory")
    parser.add_argument("-o", "--output", default="output/video.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--skip", type=int, default=1, help="Use every Nth camera (for GT trajectory)")
    parser.add_argument("--test", action="store_true", help="Use test cameras instead of train")
    parser.add_argument("--trajectory", type=str, default="gt", 
                       choices=["gt", "circle", "interpolate"],
                       help="Camera trajectory: 'gt', 'circle', or 'interpolate'")
    parser.add_argument("--num_frames", type=int, default=None, 
                       help="Number of frames for circle/interpolate trajectory")
    parser.add_argument("--height_offset", type=float, default=0.0,
                       help="Height offset for circle trajectory (negative = lower)")
    parser.add_argument("--radius_scale", type=float, default=1.0,
                       help="Radius scale for circle trajectory (>1 = further, <1 = closer)")
    args = parser.parse_args()
    
    render_video(args.model, args.output, args.fps, args.skip, args.test,
                 args.trajectory, args.num_frames, args.height_offset, args.radius_scale)
