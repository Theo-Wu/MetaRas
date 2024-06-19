/* Copyright (c) Felix Petersen.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>

#include <vector>
#include <tuple>
#include <iostream>

// CUDA forward declarations

std::vector<at::Tensor> forward_render_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        int dist_func,
        at::Tensor dist_scale,
        at::Tensor dist_scale_rgb,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type,
        at::Tensor w1,
        at::Tensor w2,
        at::Tensor w3,
        at::Tensor w4,
        at::Tensor w5,
        at::Tensor wrgb1,
        at::Tensor wrgb2,
        at::Tensor wrgb3,
        at::Tensor wrgb4,
        at::Tensor wrgb5
);


std::vector<at::Tensor> backward_render_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,        
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        int dist_func,
        at::Tensor dist_scale,
        at::Tensor dist_scale_rgb,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type,
        at::Tensor w1,
        at::Tensor w2,
        at::Tensor w3,
        at::Tensor w4,
        at::Tensor w5,
        at::Tensor wrgb1,
        at::Tensor wrgb2,
        at::Tensor wrgb3,
        at::Tensor wrgb4,
        at::Tensor wrgb5,
        at::Tensor grad_w1,
        at::Tensor grad_w2,
        at::Tensor grad_w3,
        at::Tensor grad_w4,
        at::Tensor grad_w5,
        at::Tensor grad_wrgb1,
        at::Tensor grad_wrgb2,
        at::Tensor grad_wrgb3,
        at::Tensor grad_wrgb4,
        at::Tensor grad_wrgb5,
        at::Tensor grad_scale,
        at::Tensor grad_scale_rgb
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> forward_render(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        int dist_func,
        at::Tensor dist_scale,
        at::Tensor dist_scale_rgb,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type,
        at::Tensor w1,
        at::Tensor w2,
        at::Tensor w3,
        at::Tensor w4,
        at::Tensor w5,
        at::Tensor wrgb1,
        at::Tensor wrgb2,
        at::Tensor wrgb3,
        at::Tensor wrgb4,
        at::Tensor wrgb5
        ) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(soft_colors);
    CHECK_INPUT(dist_scale);
    CHECK_INPUT(dist_scale_rgb);
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
    CHECK_INPUT(w3);
    CHECK_INPUT(w4);
    CHECK_INPUT(w5);
    CHECK_INPUT(wrgb1);
    CHECK_INPUT(wrgb2);
    CHECK_INPUT(wrgb3);
    CHECK_INPUT(wrgb4);
    CHECK_INPUT(wrgb5);

    return forward_render_cuda(
        faces,
        textures,
        faces_info,
        aggrs_info,
        soft_colors,
        image_size,
        dist_func,
        dist_scale,
        dist_scale_rgb,
        dist_squared,
        dist_shape,
        dist_shift,
        dist_eps,
        aggr_alpha_func,
        aggr_alpha_t_conorm_p,
        aggr_rgb_func,
        aggr_rgb_eps,
        aggr_rgb_gamma,
        near,
        far,
        double_side,
        texture_type,
        w1,
        w2,
        w3,
        w4,
        w5,
        wrgb1,
        wrgb2,
        wrgb3,
        wrgb4,
        wrgb5
    );
}


std::vector<at::Tensor> backward_render(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        int dist_func,
        at::Tensor dist_scale,
        at::Tensor dist_scale_rgb,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type,
        at::Tensor w1,
        at::Tensor w2,
        at::Tensor w3,
        at::Tensor w4,
        at::Tensor w5,
        at::Tensor wrgb1,
        at::Tensor wrgb2,
        at::Tensor wrgb3,
        at::Tensor wrgb4,
        at::Tensor wrgb5,
        at::Tensor grad_w1,
        at::Tensor grad_w2,
        at::Tensor grad_w3,
        at::Tensor grad_w4,
        at::Tensor grad_w5,
        at::Tensor grad_wrgb1,
        at::Tensor grad_wrgb2,
        at::Tensor grad_wrgb3,
        at::Tensor grad_wrgb4,
        at::Tensor grad_wrgb5,
        at::Tensor grad_scale,
        at::Tensor grad_scale_rgb
        ) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(soft_colors);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(grad_faces);
    CHECK_INPUT(grad_textures);
    CHECK_INPUT(grad_soft_colors);
    CHECK_INPUT(dist_scale);
    CHECK_INPUT(dist_scale_rgb);
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
    CHECK_INPUT(w3);
    CHECK_INPUT(w4);
    CHECK_INPUT(w5);
    CHECK_INPUT(wrgb1);
    CHECK_INPUT(wrgb2);
    CHECK_INPUT(wrgb3);
    CHECK_INPUT(wrgb4);
    CHECK_INPUT(wrgb5);
    CHECK_INPUT(grad_w1);
    CHECK_INPUT(grad_w2);
    CHECK_INPUT(grad_w3);
    CHECK_INPUT(grad_w4);
    CHECK_INPUT(grad_w5);
    CHECK_INPUT(grad_wrgb1);
    CHECK_INPUT(grad_wrgb2);
    CHECK_INPUT(grad_wrgb3);
    CHECK_INPUT(grad_wrgb4);
    CHECK_INPUT(grad_wrgb5);
    CHECK_INPUT(grad_scale);
    CHECK_INPUT(grad_scale_rgb);

    return backward_render_cuda(
        faces,
        textures,
        soft_colors,
        faces_info,
        aggrs_info,
        grad_faces,
        grad_textures,
        grad_soft_colors,
        image_size,
        dist_func,
        dist_scale,
        dist_scale_rgb,
        dist_squared,
        dist_shape,
        dist_shift,
        dist_eps,
        aggr_alpha_func,
        aggr_alpha_t_conorm_p,
        aggr_rgb_func,
        aggr_rgb_eps,
        aggr_rgb_gamma,
        near,
        far,
        double_side,
        texture_type,
        w1,
        w2,
        w3,
        w4,
        w5,
        wrgb1,
        wrgb2,
        wrgb3,
        wrgb4,
        wrgb5,
        grad_w1,
        grad_w2,
        grad_w3,
        grad_w4,
        grad_w5,
        grad_wrgb1,
        grad_wrgb2,
        grad_wrgb3,
        grad_wrgb4,
        grad_wrgb5,
        grad_scale,
        grad_scale_rgb
    );
}


float mlp_forward_float(
        float* signed_x,
        float* w1,
        float* w2,
        float* w3,
        float* w4,
        float* w5
);

std::tuple<float,float*,float*,float*,float*,float*> mlp_backward_float(
        float* signed_x,
        float* w1,
        float* w2,
        float* w3,
        float* w4,
        float* w5,
        float* grad_w1,
        float* grad_w2,
        float* grad_w3,
        float* grad_w4,
        float* grad_w5
);

float sigmoid_forward_float(
        int function_id,
        float sign,
        float x,
        float scale,
        float dist_shape,
        float dist_shift
);

std::tuple<float,float*> sigmoid_backward_float(
        int function_id,
        float sign,
        float x,
        float scale,
        float dist_shape,
        float dist_shift,
        float* grad_scale
);

float t_conorm_forward_float(
        int t_conorm_id,
        float a_existing,
        float b_new,
        int face_id,
        float t_conorm_p
);

float t_conorm_backward_float(
        int t_conorm_id,
        float a_all,
        float b_current,
        int number_of_faces,
        float t_conorm_p
);


PYBIND11_MODULE(generalized_renderer, m) {
    m.def("forward_render", &forward_render, "FORWARD_RENDER (CUDA)");
    m.def("backward_render", &backward_render, "BACKWARD_RENDER (CUDA)");
    m.def("mlp_forward", &mlp_forward_float);
    m.def("mlp_backward", &mlp_backward_float);
    m.def("sigmoid_forward", &sigmoid_forward_float);
    m.def("sigmoid_backward", &sigmoid_backward_float);
    m.def("t_conorm_forward", &t_conorm_forward_float);
    m.def("t_conorm_backward", &t_conorm_backward_float);
}
