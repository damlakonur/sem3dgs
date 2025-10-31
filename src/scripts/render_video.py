#!/usr/bin/env python3
import torch
import sys
import os
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gaussian-splatting'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams

from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from dreifus.image import Img
from mediapy import VideoWriter
import argparse
from arguments import get_combined_args

from src.utils.camera_utils import pose_to_rendercam, GS_camera_to_intrinsics


def render_video(model_path, output_path, duration=10, fps=30, radius=3.0):
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
        
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        ref_camera = scene.getTrainCameras()[0]
        W = ref_camera.image_width
        H = ref_camera.image_height
        
        intrinsics = GS_camera_to_intrinsics(ref_camera)
        
        num_frames = int(duration * fps)
        trajectory = circle_around_axis(
            num_frames,
            axis=Vec3(0, 1, 0),
            up=Vec3(0, -1, 0),
            move=Vec3(0, 0, 0),
            distance=radius
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with VideoWriter(output_path, (H, W), fps=fps) as video_writer:
            for pose in tqdm(trajectory, desc="Rendering"):
                gs_camera = pose_to_rendercam(pose, intrinsics, W, H)
                output = render(gs_camera, gaussians, pp, bg_color)
                rendered_image = Img.from_torch(output['render']).to_numpy().img
                video_writer.add_image(rendered_image)
        
        print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", default="output/video.mp4")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()
    
    render_video(args.model, args.output, args.duration, args.fps, args.radius)
