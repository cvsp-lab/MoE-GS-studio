#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene_spline_rot import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_hash import render, all_t_gaussians, render_frames, render_experts
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments_MoDE import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, ModelEmbParams
from scene_spline_rot import GaussianModel, DeformModel
from time import time
import threading
import concurrent.futures

from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except (IOError, OSError):
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, deform, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    rbfbasefunction = trbfunction

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_images = []

    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])

    psnrs= []
    ssims= []
    lpipss= []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        cam_no = view.cam_no
        frame_no = view.frame_no

        # fid = view.fid
        fid = torch.Tensor(np.array([view.time])).to(gaussians.get_xyz.device)
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = deform.step(xyz.detach(), time_input, fixed_attention=True)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type, basicfunction=rbfbasefunction, cam_no=cam_no, frame_no=frame_no, iter=iteration, num_down_emb_c=emb.emb_min_embeddings, num_down_emb_f=emb.emb_min_embeddings, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

        gt = view.original_image[0:3, :, :].cuda()
        gt_images.append(to8b(gt).transpose(1,2,0))
        gt_list.append(gt)

        psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        ssims.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0))) 
        lpipss.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='alex'))

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    mean_results = {
        "PSNR": torch.tensor(psnrs).mean().item(),
        "SSIM": torch.tensor(ssims).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        }
    
    with open(model_path + "/" + name + "/" + "ours_{}".format(iteration) + "/" + "mean_metrics.json", 'w') as fp:
        json.dump(mean_results, fp, indent=True)

    # multithread_write(gt_list, gts_path)

    # multithread_write(render_list, render_path)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gt.mp4'), gt_images, fps=30)

def render_gate(model_path, name, iteration, views, gaussians, deform, pipeline, background, cam_type):
    render_images = []
    rbfbasefunction = trbfunction

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        cam_no = view.cam_no
        frame_no = view.frame_no

        fid = torch.Tensor(np.array([view.time])).to(gaussians.get_xyz.device)
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = deform.step(xyz.detach(), time_input, fixed_attention=True)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type, basicfunction=rbfbasefunction, cam_no=cam_no, frame_no=frame_no, iter=iteration, num_down_emb_c=emb.emb_min_embeddings, num_down_emb_f=emb.emb_min_embeddings, gate_viz=True, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))


    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gate.mp4'), render_images, fps=30)

def render_trajectory(model_path, name, iteration, views, gaussians, deform, pipeline, background, cam_type, N=50, step=2):
    render_images = []
    gate_images = []

    rbfbasefunction = trbfunction
    
    means3D_list = []
    means2D_list = []
    shs_list = []
    shs_gate_list = []
    opacities_list = []
    scales_list = []
    rotations_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        cam_no = view.cam_no
        frame_no = view.frame_no

        # fid = view.fid
        fid = torch.Tensor(np.array([view.time])).to(gaussians.get_xyz.device)
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = deform.step(xyz.detach(), time_input, fixed_attention=True)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

        rendering = all_t_gaussians(view, gaussians, pipeline, background,cam_type=cam_type, basicfunction=rbfbasefunction, cam_no=cam_no, frame_no=frame_no, iter=iteration, num_down_emb_c=emb.emb_min_embeddings, num_down_emb_f=emb.emb_min_embeddings, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)
        means3D, means2D, shs, shs_gate, opacities, scales, rotations = rendering['means3D'], rendering['means2D'], rendering['shs'], rendering['shs_gate'], rendering['opacities'], rendering['scales'], rendering['rotations']
        
        means3D_list.append(means3D)
        means2D_list.append(means2D)
        shs_list.append(shs)
        shs_gate_list.append(shs_gate)
        opacities_list.append(opacities)
        scales_list.append(scales)
        rotations_list.append(rotations)

    T = len(means3D_list)
    tau = 4.0  # 감쇠 정도 (작을수록 빨리 사라짐)
    w = torch.exp(-torch.arange(N).float() / tau)
    w = w / w.max()  # 마지막이 1이 되도록
    w = w.to(opacities_list[0].device)

    for i in range(0, T - N + 1, step):  # 2칸씩 슬라이드
        opacities_cat = torch.cat([opacities_list[i+j] * w[j] for j in range(N)], dim=0)
        means3D_cat   = torch.cat(means3D_list[i:i+N],   dim=0)
        means2D_cat   = torch.cat(means2D_list[i:i+N],   dim=0)
        shs_cat       = torch.cat(shs_list[i:i+N],       dim=0)
        scales_cat    = torch.cat(scales_list[i:i+N],    dim=0)
        rotations_cat = torch.cat(rotations_list[i:i+N], dim=0)
        
        rendering = render_frames(view, gaussians, pipeline, means3D_cat, means2D_cat, shs_cat, opacities_cat, scales_cat, rotations_cat, background)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))

    for i in range(0, T - N + 1, step):  # 2칸씩 슬라이드
        opacities_cat = torch.cat([opacities_list[i+j] * w[j] for j in range(N)], dim=0)
        means3D_cat   = torch.cat(means3D_list[i:i+N],   dim=0)
        means2D_cat   = torch.cat(means2D_list[i:i+N],   dim=0)
        shs_gate_cat       = torch.cat(shs_gate_list[i:i+N],       dim=0)
        scales_cat    = torch.cat(scales_list[i:i+N],    dim=0)
        rotations_cat = torch.cat(rotations_list[i:i+N], dim=0)
        
        gate_rendering = render_frames(view, gaussians, pipeline, means3D_cat, means2D_cat, shs_gate_cat, opacities_cat, scales_cat, rotations_cat, background)["render"]
        gate_images.append(to8b(gate_rendering).transpose(1,2,0))

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_trajectory.mp4'), render_images, fps=15)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gate_trajectory.mp4'), gate_images, fps=15)


def render_trajectory_expert(expert_data_list, view, model_path, name, iteration, views, gaussians, deform, pipeline, background, cam_type, N=50, step=2):
    expert_path = os.path.join(model_path, name, "ours_{}".format(iteration), "experts")

    render_images = []
    gate_images = []

    rbfbasefunction = trbfunction
    

    T = len(expert_data_list)
    tau = 4.0  # 감쇠 정도 (작을수록 빨리 사라짐)
    w = torch.exp(-torch.arange(N).float() / tau)
    w = w / w.max()  # 마지막이 1이 되도록
    w = w.cuda()

    expert_names = ["deform", "grid"]
    
    for name in expert_names:
        render_images = []

        for i in range(0, T - N + 1, step):  # 2칸씩 슬라이드
            opacities_cat = torch.cat([expert_data_list[i+j][name]["opacity"] * w[j] for j in range(N)], dim=0)
            means3D_cat = torch.cat([expert_data_list[i+j][name]["means3D"] for j in range(N)], dim=0)
            means2D_cat = torch.cat([expert_data_list[i+j][name]["means2D"] for j in range(N)], dim=0)
            shs_cat = torch.cat([expert_data_list[i+j][name]["shs"] for j in range(N)], dim=0)
            scales_cat = torch.cat([expert_data_list[i+j][name]["tiny_scales"] for j in range(N)], dim=0)
            rotations_cat = torch.cat([expert_data_list[i+j][name]["rotations"] for j in range(N)], dim=0)
            
            rendering = render_frames(view, gaussians, pipeline, means3D_cat, means2D_cat, shs_cat, opacities_cat, scales_cat, rotations_cat, background)["render"]
            render_images.append(to8b(rendering).transpose(1,2,0))

        imageio.mimwrite(os.path.join(expert_path, f'{name}_trajectory.mp4'), render_images, fps=15)

def render_expert(model_path, name, iteration, views, gaussians, deform, pipeline, background, cam_type):
    expert_path = os.path.join(model_path, name, "ours_{}".format(iteration), "experts")
    makedirs(expert_path, exist_ok=True)

    rbfbasefunction = trbfunction

    expert_data_list = []
    deform_images = []
    poly_images = []
    emb_images = []
    grid_images = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        cam_no = view.cam_no
        frame_no = view.frame_no

        fid = torch.Tensor(np.array([view.time])).to(gaussians.get_xyz.device)
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        deform_pkgs = deform.step(xyz.detach(), time_input, fixed_attention=True)
        d_xyz, d_rotation, d_scaling = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling']

        rendering, expert_data = render_experts(view, gaussians, pipeline, background,cam_type=cam_type, basicfunction=rbfbasefunction, cam_no=cam_no, frame_no=frame_no, iter=iteration, num_down_emb_c=emb.emb_min_embeddings, num_down_emb_f=emb.emb_min_embeddings, gate_viz=True, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)
        deform_image, grid_image = rendering["deform_image"], rendering["grid_image"]   

        deform_images.append(to8b(deform_image).transpose(1,2,0))
        grid_images.append(to8b(grid_image).transpose(1,2,0))
        
        expert_data_list.append(expert_data)


    imageio.mimwrite(os.path.join(expert_path, 'video_deform.mp4'), deform_images, fps=30)
    imageio.mimwrite(os.path.join(expert_path, 'video_grid.mp4'), grid_images, fps=30)

    return expert_data_list, view

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, embparam : ModelEmbParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam, embparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        deform = DeformModel(dataset.grid_args, dataset.network_args, scale_xyz=dataset.scale_xyz, reg_spatial_able=False, reg_temporal_able=False)
        deform.load_weights(dataset.model_path, iteration=iteration)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, deform, pipeline, background,cam_type)
        # render_gate(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, deform, pipeline, background,cam_type)
        
        # render_trajectory(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, deform, pipeline, background,cam_type)
        expert_data_list, view = render_expert(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, deform, pipeline, background,cam_type)
        render_trajectory_expert(expert_data_list, view, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, deform, pipeline, background,cam_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    emb = ModelEmbParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), emb.extract(args), args.skip_train, args.skip_test, args.skip_video)