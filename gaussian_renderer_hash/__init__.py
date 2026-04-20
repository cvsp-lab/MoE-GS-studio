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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene_spline_rot.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
import torch.nn.functional as F
from utils.general_utils import quaternion_to_matrix

def interpolate_cubic_hermite(signal, times, N):
    """
    signal: (B, E, N)  # B=batch, E=experts, N=timeline length
    times:  (B, E, 1)  # sampling times normalized to [0,1]
    N: int or tensor   # number of timeline points
    """
    E = signal.shape[1]  # expert 수 (기존엔 3으로 고정)

    times_scaled = times * (N - 1)[:, None]
    indices = torch.floor(times_scaled).long()
    # Clamping to avoid out-of-bounds indices

    indices = torch.clamp(
        indices, torch.zeros_like(N)[:, None].expand(-1, E, -1), (N - 2)[:, None].expand(-1, E, -1)
    ).long()
    left_indices = torch.clamp(
        indices - 1, torch.zeros_like(N)[:, None].expand(-1, E, -1), (N - 1)[:, None].expand(-1, E, -1)
    ).long()
    right_indices = torch.clamp(
        indices + 1, torch.zeros_like(N)[:, None].expand(-1, E, -1), (N - 1)[:, None].expand(-1, E, -1)
    ).long()
    right_right_indices = torch.clamp(
        indices + 2, torch.zeros_like(N)[:, None].expand(-1, E, -1), (N - 1)[:, None].expand(-1, E, -1)
    ).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    # if len(signal.shape) == 3:  # remove extra singleton dimension
    interpolation = interpolation.squeeze(-1)
    # end.record()
    # torch.cuda.synchronize()
    # print('v1:', start.elapsed_time(end))
    return interpolation

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None, basicfunction=None,
    cam_no=None, frame_no=None, iter=None, num_down_emb_c=5, num_down_emb_f=5, gate_viz=False, random_gate=False,
    d_xyz=None, d_rotation=None, d_scaling=None):
    
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        deform_means3D, deform_scales, deform_rotations, opacity_final, deform_shs = pc._deformation(means3D, scales,
                                                                 rotations, opacity, shs,
                                                                 time)
        
        m = means3D.detach()
        s = scales.detach()
        r = rotations.detach()
        o = opacity.detach()
        sh = shs.detach()
        
        # Embedding based deformation Expert
        # emb_means3D, emb_scales, emb_rotations, emb_opacity, emb_shs, emb_extras =\
        #     pc._emb_network(m, s, r, o, time, cam_no, pc, None, sh, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)

        # hash gride based deformation Expert
        xyz_r = quaternion_to_matrix(d_xyz[..., :4])
        xyz_t = d_xyz[..., 4:, None]
        grid_means3D = xyz_r @ m[..., None] + xyz_t
        grid_means3D = grid_means3D.squeeze(-1)
        # grid_scales = s + d_scaling
        grid_rotations = r + d_rotation

        # Interpolation based temporal Gating
        control_xyz = pc.get_control_weight.cuda()
        curr_time = torch.tensor(viewpoint_camera.time).cuda()

        spline_weight = interpolate_cubic_hermite(
            control_xyz.permute(0, 2, 1),
            curr_time[None, None].expand(control_xyz.shape[0], control_xyz.shape[2], 1),
            N=pc.current_control_num,
        )
        
        if random_gate:
            spline_weight = torch.softmax(torch.randn(spline_weight.shape, device=spline_weight.device), dim=-1)
            
        pos_g = gating(spline_weight[:,:2], moe_top_k=pc.moe_top_k, mixtral_style=False)
        pos_gu = pos_g.unsqueeze(-1)

        rot_g = gating(spline_weight[:,2:4], moe_top_k=pc.moe_top_k, mixtral_style=False)
        rot_gu = rot_g.unsqueeze(-1)

        if pc.MoE_mean:
            means_stack    = torch.stack([deform_means3D, grid_means3D], dim=1)
            means3D_final = torch.sum(means_stack * pos_gu, dim=1)
        else:
            means3D_final = deform_means3D

        if pc.MoE_rotation:
            rotation_stack = torch.stack([deform_rotations, grid_rotations], dim=1)
            rotations_final = torch.sum(rotation_stack * rot_gu, dim=1)
        else:
            rotations_final = deform_rotations
            
        # scale,shs 는 non-moe attributes로 정의함함
        scales_final = deform_scales
        shs_final = deform_shs

        if gate_viz:
            shs_final = torch.zeros_like(shs_final)  # shape: (B, 3)
            color_map_tensor = torch.tensor([
                [1.0, 0.0, 0.0],  # Expert 0 → red
                [0.0, 1.0, 0.0],  # Expert 1 → green
                [0.0, 0.0, 1.0],  # Expert 2 → blue
                [1.0, 1.0, 0.0],  # Expert 3 → yellow
            ], device=means3D.device)

            # assignment = torch.argmax(g, dim=1)  # shape: (B,)
            # rgb_values = color_map_tensor[assignment]
            blended_colors = torch.matmul(g, color_map_tensor)
            shs_final[:, 0, :] = blended_colors

    else:
        raise NotImplementedError


    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}

def gating(
    logits: torch.Tensor,                 # shape: [B, E]
    moe_top_k: int = 1,
    mixtral_style: bool = True,           # True: top-k on logits → softmax(k); False: softmax(all) → top-k pick
    use_logits_std_norm: bool = True,    # 행별 표준편차로 스케일 정규화
    target_std: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns:
        gates_full: [B, E] — 상위 k개 위치만 양수(합=1), 나머지 0 (sparse routing)
    """
    num_gaussians, num_experts = logits.shape
    assert 1 <= moe_top_k <= num_experts, "moe_top_k must be within [1, num_experts]"

    if use_logits_std_norm:
        std = logits.std(dim=1, keepdim=True).clamp_min(eps)
        logits = logits / (std / target_std)  # 토큰별 스케일 정렬

    if mixtral_style:
        # Mixtral: top-k on logits → softmax over those k only
        gates, indices = torch.topk(logits, k=moe_top_k, dim=1)            # [B, k], [B, k]
        gates = F.softmax(gates, dim=1)                           # [B, k], 행합=1
    else:
        # Non-Mixtral: softmax over all experts → then pick top-k
        gates = F.softmax(logits, dim=1)                                      # [B, E], 행합=1
        gates, indices = torch.topk(gates, k=moe_top_k, dim=1)

        # zero expert(인덱스 = num_experts-1)가 top-k에 포함되면 해당 확률을 0으로 무효화
        # gates = torch.where(indices==(num_experts-1),
        #                     torch.zeros_like(gates),
        #                     gates
        # )
        gates /= torch.sum(gates, dim=1, keepdim=True)

    # 상위 k gate들을 원래 [B, E] 위치로 scatter
    gates_full = torch.zeros_like(logits)                 # [B, E]
    gates_full.scatter_(dim=1, index=indices, src=gates)  # 선택된 k 위치에만 값 채움

    return gates_full

def all_t_gaussians(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None, basicfunction=None,
    cam_no=None, frame_no=None, iter=None, num_down_emb_c=5, num_down_emb_f=5, gate_viz=False, random_gate=False,
    d_xyz=None, d_rotation=None, d_scaling=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc._scaling
    rotations = pc._rotation

    deform_means3D, deform_scales, deform_rotations, deform_opacity, deform_shs = pc._deformation(means3D, scales,
                                                                rotations, opacity, shs,
                                                                time)
    
    m = means3D.detach()
    s = scales.detach()
    r = rotations.detach()
    o = opacity.detach()
    sh = shs.detach()
    
    # Embedding based deformation Expert
    emb_means3D, emb_scales, emb_rotations, emb_opacity, emb_shs, emb_extras =\
        pc._emb_network(m, s, r, o, time, cam_no, pc, None, sh, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)

    # hash gride based deformation Expert
    xyz_r = quaternion_to_matrix(d_xyz[..., :4])
    xyz_t = d_xyz[..., 4:, None]
    grid_means3D = xyz_r @ m[..., None] + xyz_t
    grid_means3D = grid_means3D.squeeze(-1)

    grid_scales = s + d_scaling
    grid_rotations = r + d_rotation

    # parametric function
    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    trbfdistanceoffset = time - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)

    poly_opacity = o * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    tforpoly = trbfdistanceoffset.detach()
    poly_means3D = m +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    poly_rotations = pc.get_poly_rotation(tforpoly) # to try use 


    # Interpolation based temporal Gating
    control_xyz = pc.get_control_weight.cuda()
    curr_time = torch.tensor(viewpoint_camera.time).cuda()

    spline_weight = interpolate_cubic_hermite(
        control_xyz.permute(0, 2, 1),
        curr_time[None, None].expand(control_xyz.shape[0], control_xyz.shape[2], 1),
        N=pc.current_control_num,
    )
        
    g = gating(spline_weight, moe_top_k=pc.moe_top_k, mixtral_style=False)
    gu = g.unsqueeze(-1)

    if pc.MoE_mean:
        means_stack    = torch.stack([deform_means3D, poly_means3D, emb_means3D, grid_means3D], dim=1)
        means3D_final = torch.sum(means_stack * gu, dim=1)
    else:
        means3D_final = deform_means3D

    rotations_final = deform_rotations
    opacity_final = deform_opacity

    # scale,shs 는 non-moe attributes로 정의함함
    scales_final = deform_scales
    shs_final = deform_shs

    # shs_gate = torch.zeros_like(shs_final.clone().detach())  # shape: (B, 3)
    
    color_map_tensor = torch.tensor([
            [1.0, 0.0, 0.0],  # Expert 0 → red
            [0.0, 1.0, 0.0],  # Expert 1 → green
            [0.0, 0.0, 1.0],  # Expert 2 → blue
            [1.0, 1.0, 0.0],  # Expert 3 → yellow
        ], device=means3D.device)

    blended_colors = torch.matmul(g, color_map_tensor)
    shs_gate = torch.zeros_like(shs_final)
    shs_gate[:, 0, :] = blended_colors
        
    tiny_world_scale = 1e-2
    scales_final = torch.full_like(scales_final, math.log(tiny_world_scale))

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    return {"means3D": means3D_final,
            "means2D": means2D,
            "shs": shs_final,
            "shs_gate": shs_gate,
            "opacities": opacity,
            "scales": scales_final,
            "rotations": rotations_final}


def render_frames(viewpoint_camera, pc : GaussianModel, pipe, means3D_final, means2D, shs_final, opacity, scales_final, rotations_final, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # shs = None
    colors_precomp = None
    cov3D_precomp = None

    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}


def render_experts(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None, basicfunction=None,
    cam_no=None, frame_no=None, iter=None, num_down_emb_c=5, num_down_emb_f=5, gate_viz=False, random_gate=False,
    d_xyz=None, d_rotation=None, d_scaling=None):
    
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        deform_means3D, deform_scales, deform_rotations, deform_opacity, deform_shs = pc._deformation(means3D, scales,
                                                                 rotations, opacity, shs,
                                                                 time)
        
        m = means3D.detach()
        s = scales.detach()
        r = rotations.detach()
        o = opacity.detach()
        sh = shs.detach()
        
        # hash gride based deformation Expert
        xyz_r = quaternion_to_matrix(d_xyz[..., :4])
        xyz_t = d_xyz[..., 4:, None]
        grid_means3D = xyz_r @ m[..., None] + xyz_t
        grid_means3D = grid_means3D.squeeze(-1)

        # Interpolation based temporal Gating
        control_xyz = pc.get_control_weight.cuda()
        curr_time = torch.tensor(viewpoint_camera.time).cuda()

        spline_weight = interpolate_cubic_hermite(
            control_xyz.permute(0, 2, 1),
            curr_time[None, None].expand(control_xyz.shape[0], control_xyz.shape[2], 1),
            N=pc.current_control_num,
        )
        
        g = gating(spline_weight, moe_top_k=pc.moe_top_k, mixtral_style=False)
        means_stack    = torch.stack([deform_means3D, grid_means3D], dim=1)
        # means3D_final = torch.sum(means_stack * gu, dim=1)

        experts = ["deform", "grid"]
        expert_data = {}

        
        tiny_world_scale = 1e-2
        tiny_scales = torch.full_like(deform_scales, math.log(tiny_world_scale))

        for e_idx, name in enumerate(experts):
            mask = (g[:, e_idx] > 1e-3)

            scales_final = pc.scaling_activation(deform_scales)
            tiny_scales_final = pc.scaling_activation(tiny_scales)
            rotations_final = pc.rotation_activation(deform_rotations)
            opacity_final = pc.opacity_activation(deform_opacity)

            expert_data[name] = {
                "means3D": means_stack[:, e_idx, :][mask],
                "means2D": means2D[mask],
                "scales": scales_final[mask],
                "tiny_scales": tiny_scales_final[mask],
                "rotations": rotations_final[mask],
                "opacity": opacity_final[mask],
                "shs": deform_shs[mask],
            }

    else:
        raise NotImplementedError

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color


    expert_images = {}
    expert_names = ["deform", "grid"]

    for name in expert_names:
        expert_images[f"{name}_image"], _, _ = rasterizer(
            means3D        = expert_data[name]["means3D"],
            means2D        = expert_data[name]["means2D"],
            shs            = expert_data[name]["shs"],
            colors_precomp = colors_precomp,
            opacities      = expert_data[name]["opacity"],
            scales         = expert_data[name]["scales"],
            rotations      = expert_data[name]["rotations"],
            cov3D_precomp  = cov3D_precomp,
        )

    return expert_images, expert_data