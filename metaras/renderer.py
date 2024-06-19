# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import metaras


class MetaRas(nn.Module):
    def __init__(self,
                 image_size=256,
                 background_color=[0, 0, 0],
                 anti_aliasing=False,
                 #
                 dist_func='uniform',
                 dist_scale=-2,
                 dist_scale_rgb=-2,
                 dist_squared=False,
                 dist_shape=0.0,
                 dist_shift=0.0,
                 dist_eps=1e4,
                 #
                 aggr_alpha_func='probabilistic',
                 aggr_alpha_t_conorm_p=0.0,
                 #
                 aggr_rgb_func='softmax',
                 aggr_rgb_eps=1e-3,
                 aggr_rgb_gamma=1e-3,
                 #
                 near=1,
                 far=100,
                 double_side=False,
                 texture_type='surface',
                 #
                 w1=torch.zeros((4*1),dtype=torch.float32,device='cuda'),
                 w2=torch.zeros((4*4),dtype=torch.float32,device='cuda'),
                 w3=torch.zeros((4*4),dtype=torch.float32,device='cuda'),
                 w4=torch.zeros((4*4),dtype=torch.float32,device='cuda'),
                 w5=torch.zeros((4*1),dtype=torch.float32,device='cuda'),
                 #
                 wrgb1=torch.zeros((4*1),dtype=torch.float32,device='cuda'),
                 wrgb2=torch.zeros((4*4),dtype=torch.float32,device='cuda'),
                 wrgb3=torch.zeros((4*4),dtype=torch.float32,device='cuda'),
                 wrgb4=torch.zeros((4*4),dtype=torch.float32,device='cuda'),
                 wrgb5=torch.zeros((4*1),dtype=torch.float32,device='cuda'),
                 ):
        super(MetaRas, self).__init__()

        if aggr_rgb_func not in ['hard', 'softmax', 'mlp', 'gaussian', 'logistic', 'exponential_rev', 'gamma_rev']: 
            # softmax_in not mentioned since its functionality is not tested
            # mlp index = 3
            raise ValueError('Aggregate function (RGB) currently supports the following values: hard, softmax, mlp, gaussian, logistic, exponential_rev, gamma_rev')
        if texture_type not in ['surface', 'vertex']:
            raise ValueError('Texture type only support surface and vertex.')

        self.image_size = image_size
        self.background_color = background_color
        self.anti_aliasing = anti_aliasing

        self.dist_func = dist_func
        self.dist_scale = dist_scale
        self.dist_scale_rgb = dist_scale_rgb
        self.dist_squared = dist_squared
        self.dist_shape = dist_shape
        self.dist_shift = dist_shift
        self.dist_eps = dist_eps

        self.aggr_alpha_func = aggr_alpha_func
        self.aggr_alpha_t_conorm_p = aggr_alpha_t_conorm_p

        self.aggr_rgb_func = aggr_rgb_func
        self.aggr_rgb_eps = aggr_rgb_eps
        self.aggr_rgb_gamma = aggr_rgb_gamma

        self.near = near
        self.far = far
        self.double_side = double_side
        self.texture_type = texture_type

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        
        self.wrgb1 = wrgb1
        self.wrgb2 = wrgb2
        self.wrgb3 = wrgb3
        self.wrgb4 = wrgb4
        self.wrgb5 = wrgb5

    def forward(self, mesh):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)

        images = metaras.functional.render(
            face_vertices=mesh.face_vertices,
            textures=mesh.face_textures,
            image_size=image_size,
            background_color=self.background_color,
            dist_func=self.dist_func,
            dist_scale=self.dist_scale,
            dist_scale_rgb=self.dist_scale_rgb,
            dist_squared=self.dist_squared,
            dist_shape=self.dist_shape,
            dist_shift=self.dist_shift,
            dist_eps=self.dist_eps,
            aggr_alpha_func=self.aggr_alpha_func,
            aggr_alpha_t_conorm_p=self.aggr_alpha_t_conorm_p,
            aggr_rgb_func=self.aggr_rgb_func,
            aggr_rgb_eps=self.aggr_rgb_eps,
            aggr_rgb_gamma=self.aggr_rgb_gamma,
            near=self.near,
            far=self.far,
            double_side=self.double_side,
            texture_type=self.texture_type,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4,
            w5=self.w5,
            wrgb1=self.wrgb1,
            wrgb2=self.wrgb2,
            wrgb3=self.wrgb3,
            wrgb4=self.wrgb4,
            wrgb5=self.wrgb5,
        )

        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images

    def forward_tensors(self, face_vertices, face_textures):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)

        images = metaras.functional.render(
            face_vertices=face_vertices,
            textures=face_textures,
            image_size=image_size,
            background_color=self.background_color,
            dist_func=self.dist_func,
            dist_scale=self.dist_scale,
            dist_scale_rgb=self.dist_scale_rgb,
            dist_squared=self.dist_squared,
            dist_shape=self.dist_shape,
            dist_shift=self.dist_shift,
            dist_eps=self.dist_eps,
            aggr_alpha_func=self.aggr_alpha_func,
            aggr_alpha_t_conorm_p=self.aggr_alpha_t_conorm_p,
            aggr_rgb_func=self.aggr_rgb_func,
            aggr_rgb_eps=self.aggr_rgb_eps,
            aggr_rgb_gamma=self.aggr_rgb_gamma,
            near=self.near,
            far=self.far,
            double_side=self.double_side,
            texture_type=self.texture_type,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4,
            w5=self.w5,
            wrgb1=self.wrgb1,
            wrgb2=self.wrgb2,
            wrgb3=self.wrgb3,
            wrgb4=self.wrgb4,
            wrgb5=self.wrgb5,
        )

        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images
    
