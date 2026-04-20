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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from ply_io import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene_spline_rot.deformation import deform_network
from scene_spline_rot.regulation import compute_plane_smoothness
from thirdparty.embedding.scene.deformation import deform_network as emb_network

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args, args2):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._deformation = deform_network(args)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()

        self._omega = torch.empty(0)
        self._motion = torch.empty(0)

        self._embedding = torch.empty(0)
        self._emb_network = emb_network(W=args2.emb_net_width, D=args2.emb_defor_depth, 
                                    min_embeddings=args2.emb_min_embeddings, max_embeddings=args2.emb_max_embeddings, 
                                    num_frames=args2.emb_total_num_frames,
                                    args=args2)

        self.moe_top_k = args.moe_top_k
        # self._rout_interv = args.rout_interv
        self.control_num = args.control_num
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(1)
        print(f"MoE Topk : {self.moe_top_k}")
        print(f"Init control num : {self.control_num}")

        self.MoE_mean = args.MoE_mean
        self.MoE_rotation = args.MoE_rotation
        self.MoE_opacity = args.MoE_opacity

        print(f"MoE on position: {self.MoE_mean}")
        print(f"MoE on rotation: {self.MoE_rotation}")
        print(f"MoE on opacity: {self.MoE_opacity}")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._deformation_table,
        
        # self.grid,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_embedding(self):
        return self._embedding

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def get_trbfcenter(self):
        return self._trbf_center
    @property
    def get_trbfscale(self):
        return self._trbf_scale
    
    def get_poly_rotation(self, delta_t):
        r = self._rotation.detach()
        rotation = r + delta_t * self._omega 
        self.delta_t = delta_t
        return rotation
    
    @property
    def get_control_weight(self):
        return self.control_weight

    # def get_weight(self, frame_num):
    #     index = frame_num // self._rout_interv
    #     index = min(index, self._weight.shape[0] - 1)
    #     return self._weight[:, index, :]
    
    @property
    def get_weight(self):
        return self._weight

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int, duration):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

        motion = torch.zeros((self._xyz.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))
        omega = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        
        # teporal center for basis function
        self._trbf_center = nn.Parameter((torch.zeros((self._xyz.shape[0], 1), device="cuda").requires_grad_(True)))
        self._trbf_scale = nn.Parameter(torch.ones((self._xyz.shape[0], 1), device="cuda").requires_grad_(True)) 
        self.trbfoutput = None 

        # embedding & network initialize
        embedding = torch.zeros((fused_color.shape[0], self._emb_network.gaussian_embedding_dim)).float().cuda()  # [jm]
        self._embedding = nn.Parameter(embedding.requires_grad_(True))  # [jm]
        self._emb_network = self._emb_network.to("cuda") 

        # Routing weight for polynomial representation
        # rout_weights = 0.01 * torch.randn((self._xyz.shape[0], 3), device="cuda")  # 작은 분산
        # self._weight = nn.Parameter(rout_weights.requires_grad_(True))
        # weight_motion = torch.zeros((self._xyz.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        # self._weight_motion = nn.Parameter(weight_motion.requires_grad_(True))

        # time_step = 1 / (dyn_tracjectory.shape[1] - 1.0) # [num_gaussian, trajectory, 3]
        # t_step = torch.arange(0, 1 + time_step, time_step).cuda().float()
        # t_step = t_step[None, :, None].expand(dyn_tracjectory.shape[0], -1, -1)
        # init_control_pts = inverse_cubic_hermite(dyn_tracjectory / self.deform_spatial_scale, t_step, N_pts=self.control_num)

        rout_weights = 0.01 * torch.randn((self._xyz.shape[0], self.control_num, 8), device="cuda")  # E0_p, E1_p, E2_p, E3_p, E0_r, E1_r, E2_r, E3_r
        self.control_weight = nn.Parameter(rout_weights.requires_grad_(True))
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(fused_point_cloud.shape[0])[
            ..., None
        ]


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            
            {'params': [self._omega], 'lr': training_args.omega_lr, "name": "omega"},
            {'params': [self._motion], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion"},
            {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
            # {'params': [self._weight], 'lr': training_args.weight_lr, "name": "weight"},
            # {'params': [self._weight_motion], 'lr': training_args.weight_movelr, "name": "weight_motion"},

            {
                "params": [self.control_weight],
                "lr": 10 * training_args.weight_movelr,
                "name": "control_weight",
            },
            {"params": [self.current_control_num], "lr": 0.0, "name": "current_control_num"},

            {'params': list(self._emb_network.get_mlp_parameters()), 'lr': training_args.embnet_lr_init * self.spatial_lr_scale, "name": "embnet"},
            {'params': [self._emb_network.offsets], 'lr': training_args.offsets_lr, "name": "offsets"},
            {'params': [self._embedding], 'lr': training_args.feature_lr, "name": "embedding"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.weight_scheduler_args = get_expon_lr_func(lr_init=training_args.weight_movelr,
                                                    lr_final=training_args.weight_movelr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.embenet_scheduler_args = get_expon_lr_func(lr_init=training_args.embnet_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.embnet_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.embnet_lr_delay_mult,
                                                    max_steps=training_args.embnet_lr_max_steps) 

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if param_group["name"] == "control_weight":
                lr = self.weight_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "embnet":
                lr = self.embenet_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'trbf_center', 'trbf_scale', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))

        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._embedding.shape[1]):
            l.append('embedding_{}'.format(i))
        # for i in range(self._weight.shape[1]):
        #     l.append('weight_{}'.format(i))
        # for i in range(self._weight_motion.shape[1]):
        #     l.append('weight_motion_{}'.format(i))
        for i in range(self.control_weight.shape[1]):
            for j in range(self.control_weight.shape[2]):
                if j == 0:
                    l.append("control_x_{}".format(i))
                elif j == 1:
                    l.append("control_y_{}".format(i))
                elif j == 2:
                    l.append("control_z_{}".format(i))
                elif j == 3:
                    l.append("control_l_{}".format(i))
                if j == 4:
                    l.append("control_rx_{}".format(i))
                elif j == 5:
                    l.append("control_ry_{}".format(i))
                elif j == 6:
                    l.append("control_rz_{}".format(i))
                elif j == 7:
                    l.append("control_rl_{}".format(i))
                
        l.append("current_control_num")
        return l
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))

    def load_embnet(self, path):
        print("loading embedding model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"embnet.pth"),map_location="cuda")
        self._emb_network.load_state_dict(weight_dict)
        self._emb_network = self._emb_network.to("cuda")

    def save_embnet(self, path):
        torch.save(self._emb_network.state_dict(),os.path.join(path, "embnet.pth"))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        control_weight = self.control_weight.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        current_control_num = self.current_control_num.detach().cpu().numpy()

        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        trbf_center= self._trbf_center.detach().cpu().numpy()
        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()
        embedding = self._embedding.detach().cpu().numpy()

        # weight = self._weight.detach().cpu().numpy()
        # weight_motion = self._weight_motion.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, omega, f_dc, f_rest, opacities, scale, rotation, embedding, control_weight, current_control_num), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        # weight = self._weight.detach().cpu().numpy()
        # torch.save(weight, os.path.join(os.path.dirname(path), "weight.pth"))

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        control_weight_list = []
        for i in range(self.control_num):
            control_weight_list.append(
                np.stack(
                    (
                        np.asarray(plydata.elements[0][f"control_x_{i}"]),
                        np.asarray(plydata.elements[0][f"control_y_{i}"]),
                        np.asarray(plydata.elements[0][f"control_z_{i}"]),
                        np.asarray(plydata.elements[0][f"control_l_{i}"]),
                        np.asarray(plydata.elements[0][f"control_rx_{i}"]),
                        np.asarray(plydata.elements[0][f"control_ry_{i}"]),
                        np.asarray(plydata.elements[0][f"control_rz_{i}"]),
                        np.asarray(plydata.elements[0][f"control_rl_{i}"]),
                    ),
                    axis=1,
                )
            )
        control_weight = np.stack(control_weight_list, axis=1)
        current_control_num = np.asarray(plydata.elements[0]["current_control_num"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))  # pylint: disable=unsubscriptable-object

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("embedding")]
        embedding_names = sorted(embedding_names, key = lambda x: int(x.split('_')[-1]))
        embeddings = np.zeros((xyz.shape[0], len(embedding_names)))
        for idx, attr_name in enumerate(embedding_names):
            embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # weights = torch.load(os.path.join(os.path.dirname(path), "weight.pth"))
        # weight_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("weight_") and not p.name.startswith("weight_motion_")]
        # weight_names = sorted(weight_names, key = lambda x: int(x.split('_')[-1]))
        # weights = np.zeros((xyz.shape[0], len(weight_names)))
        # for idx, attr_name in enumerate(weight_names):
        #     weights[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # nummotion = 9
        # weight_motion = np.zeros((xyz.shape[0], nummotion))
        # for i in range(nummotion):
        #     weight_motion[:, i] = np.asarray(plydata.elements[0]["weight_motion_"+str(i)])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.control_weight = nn.Parameter(
            torch.tensor(control_weight, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.current_control_num = nn.Parameter(
            torch.tensor(current_control_num, dtype=torch.int64, device="cuda"), requires_grad=False
        )

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._embedding = nn.Parameter(torch.tensor(embeddings, dtype=torch.float, device="cuda").requires_grad_(True))

        # self._weight = nn.Parameter(torch.tensor(weights, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._weight_motion = nn.Parameter(torch.tensor(weight_motion, dtype=torch.float, device="cuda").requires_grad_(True))
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "offsets":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "current_control_num":
                    group["params"][0] = nn.Parameter(group["params"][0][mask], requires_grad=False)
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._embedding = optimizable_tensors["embedding"]
        # self._weight = optimizable_tensors["weight"]
        # self._weight_motion = optimizable_tensors["weight_motion"]
        self.control_weight = optimizable_tensors["control_weight"]
        self.current_control_num = optimizable_tensors["current_control_num"]

        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1 or group["name"] == "offsets":continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "current_control_num":
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0), requires_grad=False
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbfscale, new_motion, new_omega, new_embedding, new_control_weight, new_current_control_num):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "trbf_center" : new_trbf_center,
        "trbf_scale" : new_trbfscale,
        "motion": new_motion,
        "omega": new_omega,
        "embedding" : new_embedding,
        "control_weight": new_control_weight,
        "current_control_num": new_current_control_num,
        # "deformation": new_deformation
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.control_weight = optimizable_tensors["control_weight"]
        self.current_control_num = optimizable_tensors["current_control_num"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._deformation = optimizable_tensors["deformation"]
        
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._embedding = optimizable_tensors["embedding"]
        # self._weight = optimizable_tensors["weight"]
        # self._weight_motion = optimizable_tensors["weight_motion"]

        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        new_trbfscale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_embedding = self._embedding[selected_pts_mask].repeat(N,1)
        # new_weight = self._weight[selected_pts_mask].repeat(N,1)
        # new_weight_motion = self._weight_motion[selected_pts_mask].repeat(N,1)
        
        new_control_weight = self.control_weight[selected_pts_mask].repeat(N, 1, 1)
        new_current_control_num = self.current_control_num[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbfscale, new_motion, new_omega, new_embedding, new_control_weight, new_current_control_num)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_control_weight = self.control_weight[selected_pts_mask]
        new_current_control_num = self.current_control_num[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        new_trbf_center = self._trbf_center[selected_pts_mask]
        new_trbf_scale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_embedding = self._embedding[selected_pts_mask]
        # new_weight = self._weight[selected_pts_mask]
        # new_weight_motion = self._weight_motion[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbf_scale, new_motion, new_omega, new_embedding, new_control_weight, new_current_control_num)

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
    

        return final_point, mask_d    
    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)

        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        new_trbf_center = self._trbf_center[selected_pts_mask][mask]
        new_trbf_scale = self._trbf_scale[selected_pts_mask][mask]
        new_motion = self._motion[selected_pts_mask][mask]
        new_omega = self._omega[selected_pts_mask][mask]
        new_embedding = self._embedding[selected_pts_mask][mask]
        new_control_weight = self.control_weight[selected_pts_mask][mask]
        new_current_control_num = self.current_control_num[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbf_scale, new_motion, new_omega, new_embedding, new_control_weight, new_current_control_num)
        return selected_xyz, new_xyz

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
