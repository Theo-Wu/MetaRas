/* Copyright (c) Felix Petersen.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "mlp.h"
#include <tuple>

#define PROBABILITY_THRESHOLD 0.000001
//#define PROBABILITY_THRESHOLD 0.001

#define NUM_STEPS_GAMMA  32
#define GAMMA_THRESHOLD  15.
#define NUM_THREADS      256

// for the older gpus atomicAdd with double arguments does not exist
// if using CUDA 8.0 or older please uncomment this 

// #if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
// static __inline__ __device__ double atomicAdd(double* address, double val) {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                 __double_as_longlong(val + __longlong_as_double(assumed)));
//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }
// #endif


namespace{

template <typename scalar_t>
__device__ __forceinline__ void barycentric_coordinate(scalar_t *w, const scalar_t x, const scalar_t y, const scalar_t *face_info) {
    w[0] = face_info[3 * 0 + 0] * x + face_info[3 * 0 + 1] * y + face_info[3 * 0 + 2];
    w[1] = face_info[3 * 1 + 0] * x + face_info[3 * 1 + 1] * y + face_info[3 * 1 + 2];
    w[2] = face_info[3 * 2 + 0] * x + face_info[3 * 2 + 1] * y + face_info[3 * 2 + 2];
}


template <typename scalar_t>
__device__ __forceinline__ bool check_border(const scalar_t x, const scalar_t y, const scalar_t *face, const scalar_t threshold) {
    return (x > max(max(face[0], face[3]), face[6]) + threshold ||
            x < min(min(face[0], face[3]), face[6]) - threshold ||
            y > max(max(face[1], face[4]), face[7]) + threshold ||
            y < min(min(face[1], face[4]), face[7]) - threshold);
}


template <typename scalar_t>
__device__ __forceinline__ bool check_face_frontside(const scalar_t *face) {
    return (face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]);
}


template <typename scalar_t>
__device__ __forceinline__ bool check_pixel_inside(const scalar_t *w) {
    return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
}


template <typename scalar_t>
__device__ __forceinline__ void barycentric_clip(scalar_t *w) {
    for (int k = 0; k < 3; k++) w[k] = max(min(w[k], 1.), 0.);
    const scalar_t w_sum = max(w[0] + w[1] + w[2], 1e-5);
    for (int k = 0; k < 3; k++) w[k] /= w_sum;
}


template <typename scalar_t>
__device__ __forceinline__ void euclidean_p2f_distance(scalar_t &sign, scalar_t &dis_x, scalar_t &dis_y,
                                                       scalar_t *w, scalar_t *t, 
                                                       const scalar_t* face, const scalar_t *face_info,
                                                       const scalar_t xp, const scalar_t yp) {
    const scalar_t *face_sym = face_info + 9;
    const scalar_t *face_obt = face_info + 18;

    if (w[0] > 0 && w[1] > 0 && w[2] > 0 &&
        w[0] < 1 && w[1] < 1 && w[2] < 1) {
        // inside the triangle, w[0] + w[1] + w[2] = 0
        scalar_t dis_min = 100000000;
        scalar_t dis_x_min = 0;
        scalar_t dis_y_min = 0;
        scalar_t a0[3];
        scalar_t t0[3];
        for (int k = 0; k < 3; k++) {
            int v0 = k;
            int v1 = (k + 1) % 3;
            int v2 = (k + 2) % 3;
            a0[0] = face_sym[3 * v0 + 0] - face_sym[3 * v1 + 0];
            a0[1] = face_sym[3 * v0 + 1] - face_sym[3 * v1 + 1];
            a0[2] = face_sym[3 * v0 + 2] - face_sym[3 * v1 + 2];

            t0[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] - a0[v1]) / (a0[v0] - a0[v1]);
            t0[v1] = 1 - t0[v0];
            t0[v2] = 0;

            t0[0] -= w[0];
            t0[1] -= w[1];
            t0[2] -= w[2];

            // calculate distance
            dis_x = t0[0] * face[0] + t0[1] * face[3] + t0[2] * face[6];
            dis_y = t0[0] * face[1] + t0[1] * face[4] + t0[2] * face[7];
            scalar_t dis = dis_x * dis_x + dis_y * dis_y;

            if (dis < dis_min) {
                dis_min = dis;
                dis_x_min = dis_x;
                dis_y_min = dis_y;
                t[0] = t0[0];
                t[1] = t0[1];
                t[2] = t0[2];
            }
        }
        dis_x = dis_x_min;
        dis_y = dis_y_min;
        sign = 1;
    } else {
        int v0 = -1;

        if (w[1] <= 0 && w[2] <= 0) {
            v0 = 0;
            if (face_obt[0] == 1 && (xp - face[0]) * (face[6] - face[0]) + (yp - face[1]) * (face[7] - face[1]) > 0) v0 = 2;
        } else if (w[2] <= 0 && w[0] <= 0) {
            v0 = 1;
            if (face_obt[1] == 1 && (xp - face[3]) * (face[0] - face[3]) + (yp - face[4]) * (face[1] - face[4]) > 0) v0 = 0;
        } else if (w[0] <= 0 && w[1] <= 0) {
            v0 = 2;
            if (face_obt[2] == 1 && (xp - face[6]) * (face[3] - face[6]) + (yp - face[7]) * (face[4] - face[7]) > 0) v0 = 1;
        } else
        if (w[0] <= 0) v0 = 1;
        else if (w[1] <= 0) v0 = 2;
        else if (w[2] <= 0) v0 = 0;

        const int v1 = (v0 + 1) % 3;
        const int v2 = (v0 + 2) % 3;

        scalar_t a0[3];

        a0[0] = face_sym[3 * v0 + 0] - face_sym[3 * v1 + 0];
        a0[1] = face_sym[3 * v0 + 1] - face_sym[3 * v1 + 1];
        a0[2] = face_sym[3 * v0 + 2] - face_sym[3 * v1 + 2];

        t[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] - a0[v1]) / (a0[v0] - a0[v1]);
        t[v1] = 1 - t[v0];
        t[v2] = 0;

        // clamp to [0, 1]
        for (int k = 0; k < 3; k++) {
            t[k] = min(max(t[k], 0.), 1.);
            t[k] -= w[k];
        }

        // calculate distance
        dis_x = t[0] * face[0] + t[1] * face[3] + t[2] * face[6];
        dis_y = t[0] * face[1] + t[1] * face[4] + t[2] * face[7];
        sign = -1;
    }
}


template <typename scalar_t>
__device__ __forceinline__ void forward_barycentric_p2f_distance(scalar_t &dis, const scalar_t *w) {
    dis = w[0] > w[1] ? (w[1] > w[2] ? w[2] : w[1]) : (w[0] > w[2] ? w[2] : w[0]);
    dis = dis > 0 ? pow(dis, 2) : -pow(dis, 2);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t forward_sample_texture(const scalar_t *texture, const scalar_t *w, const int R, const int k, const int texture_type) {
    scalar_t texture_k = 0;
    if (texture_type == 0) { // sample surface color with resolution as R
        const int w_x = w[0] * R;
        const int w_y = w[1] * R;
        if ((w[0] + w[1]) * R - w_x - w_y <= 1) {
            texture_k = texture[(w_y * R + w_x) * 3 + k];
        } else {
            texture_k = texture[((R - 1 - w_y) * R + (R - 1 - w_x)) * 3 + k];
        }
    } else
    if (texture_type == 1) { // sample vertex color
        texture_k = w[0] * texture[k] + w[1] * texture[3+k] + w[2] * texture[6+k];
    }
    return texture_k;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t backward_sample_texture(const scalar_t grad_color, const scalar_t *w, const int R, const int k, const int texture_type) {
    scalar_t grad_texture_k = 0;
    if (texture_type == 0) { // sample surface color with resolution as R
        const int w_x = w[0] * R;
        const int w_y = w[1] * R;
        if ((w[0] + w[1]) * R - w_x - w_y <= 1) {
            if (k == w_y * R + w_x) {
                grad_texture_k = grad_color;
            }
        } else {
            if (k == (R - 1 - w_y) * R + (R - 1 - w_x)) {
                grad_texture_k = grad_color;
            }
        }
    } else
    if (texture_type == 1) {
        grad_texture_k = w[k] * grad_color;
    }
    return grad_texture_k;
}


// hard
#define HEAVISIDE_ID            0
// finite support
#define UNIFORM_ID              1
#define CUBIC_HERMITE_ID        2
#define WIGNER_SEMICIRCLE_ID    3
// exponential convergence
#define GAUSSIAN_ID             4
#define LAPLACE_ID              5
#define LOGISTIC_ID             6
#define GUDERMANNIAN_ID         7  // aka. hyperbolic secant
// linear convergence
#define CAUCHY_ID               8
#define RECIPROCAL_ID           9
// asymmetrical
#define GUMBEL_MAX_ID           10
#define GUMBEL_MIN_ID           11
#define EXPONENTIAL_ID          12
#define NEG_EXPONENTIAL_ID      13
#define GAMMA_ID                14  // scale corresponds to 1/b, dist_shape corresponds to p, dist_shift corresponds to a shift by dist_shift before applying scale
#define NEG_GAMMA_ID            15
#define LEVY_ID                 16
#define NEG_LEVY_ID             17
// mlp
#define MLP_ID                  18

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t sigmoid_forward_cuda(
    const int function_id,
    const scalar_t sign,
    const scalar_t x,
    const scalar_t scale,
    const scalar_t dist_shape,
    const scalar_t dist_shift
    ) {
    if (function_id == HEAVISIDE_ID) {
        return (sign > 0) ? 1. : 0.;
    } else
    if (function_id == LOGISTIC_ID) {
        return 1. / (1. + exp(- sign * x / scale));
    } else
    if (function_id == CAUCHY_ID) {
        return atanf(sign * x / scale) / M_PI + 0.5;
    } else
    if (function_id == RECIPROCAL_ID) {
        return sign * x / scale / (1 + x / scale) / 2. + 0.5;
    } else
    if (function_id == LAPLACE_ID) {  // https://en.wikipedia.org/wiki/Laplace_distribution
        if (sign < 0) {
            return 0.5 * exp(- x / scale);
        } else {
            return 1. - 0.5 * exp(- x / scale);
        }
    } else
    if (function_id == UNIFORM_ID) {
        if (sign * x / scale < -1) {
            return 0.;
        } else if (sign * x / scale < 1)  {
            return (sign * x) * 0.5 / scale + 0.5;
        } else {
            return 1.;
        }
    } else
    if (function_id == GUDERMANNIAN_ID) {
        return atan(tanh(sign * x / scale / 2.)) * 2. / M_PI + 0.5;
    } else
    if (function_id == CUBIC_HERMITE_ID) {
        if (sign * x / scale < -1) {
            return 0.;
        } else if (sign * x / scale < 1)  {
            const scalar_t y = (sign * x) * 0.5 / scale + 0.5;
            return 3 * y * y - 2 * y * y * y;
        } else {
            return 1.;
        }
    } else
    if (function_id == GAUSSIAN_ID) {
        return normcdf(sign * x / scale);
    } else
    if (function_id == GAMMA_ID || function_id == NEG_GAMMA_ID) {  // https://de.wikipedia.org/wiki/Gammaverteilung
        if (dist_shape < 0.) {
            printf("Error in sigmoid_forward_cuda; invalid param p (dist_shape): %g\n", dist_shape);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        scalar_t xs;
        if (function_id == GAMMA_ID) {
            if (sign * x + dist_shift * scale <= 0.) return 0.;
            xs = sign * x + dist_shift * scale;
            if (xs / scale > GAMMA_THRESHOLD) return 1.;
        } else {
            if (sign * x - dist_shift * scale >= 0.) return 1.;
            xs = - (sign * x - dist_shift * scale);
            if (xs / scale > GAMMA_THRESHOLD) return 0.;
        }
        scalar_t kummers = 1. / tgamma(dist_shape + 1.);
        scalar_t factor = kummers;
    #pragma unroll
        for (int i=1; i < NUM_STEPS_GAMMA; i++) {
            factor *= xs / scale / (dist_shape + i);
            kummers += factor;
        }
        const scalar_t y = pow(xs / scale, dist_shape) * exp(- xs / scale) * kummers;
        return (function_id == GAMMA_ID) ? y : 1. - y;
    } else
    if (function_id == WIGNER_SEMICIRCLE_ID) {
        if (sign * x / scale < -1) {
            return 0.;
        } else if (sign * x / scale < 1)  {
            return 0.5 + (sign * x * sqrt(scale * scale - x * x)) / (M_PI * scale * scale) + asin(sign * x / scale) / M_PI;
        } else {
            return 1.;
        }
    } else
    if (function_id == GUMBEL_MAX_ID) {
        const scalar_t y = exp(-exp(- sign * x / scale));
        return y;
    } else
    if (function_id == GUMBEL_MIN_ID) {
        const scalar_t y = exp(-exp(sign * x / scale));
        return 1. - y;
    } else
    if (function_id == LEVY_ID || function_id == NEG_LEVY_ID) {
        scalar_t xs;
        if (function_id == LEVY_ID) {
            if (sign * x + dist_shift * scale <= 1e-6) return 0.;
            xs = sign * x + dist_shift * scale;
        } else {
            if (sign * x - dist_shift * scale >= -1e-6) return 1.;
            xs = - (sign * x - dist_shift * scale);
        }
        const scalar_t y = erfc(sqrt(scale / 2. / xs));
        return (function_id == LEVY_ID) ? y : 1. - y;
    } else
    if (function_id == EXPONENTIAL_ID || function_id == NEG_EXPONENTIAL_ID) {
        scalar_t xs;
        if (function_id == EXPONENTIAL_ID) {
            if (sign * x + dist_shift * scale < 0.) return 0.;
            xs = sign * x + dist_shift * scale;
        } else {
            if (sign * x - dist_shift * scale > 0.) return 1.;
            xs = - (sign * x - dist_shift * scale);
        }
        const scalar_t y = 1. - exp(- xs / scale);
        return (function_id == EXPONENTIAL_ID) ? y : 1. - y;
    }
    printf("Error in sigmoid_forward_cuda; the following ID is unknown: %d\n", function_id);
    return std::numeric_limits<scalar_t>::signaling_NaN();
}

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t sigmoid_backward_cuda(
    const int function_id,
    const scalar_t sign,
    const scalar_t x,
    const scalar_t scale,
    const scalar_t dist_shape,
    const scalar_t dist_shift,
    scalar_t* grad_scale
    ) {
    if (function_id == HEAVISIDE_ID) {
        grad_scale[0] = 0.;
        return 0.;
    } else
    if (function_id == LOGISTIC_ID) {
        const scalar_t y = 1. / (1. + exp(- sign * x / scale));
        grad_scale[0] = y * (1-y) * (- sign * x / (scale * scale));
        return y * (1-y) / scale;
    } else
    if (function_id == CAUCHY_ID) {  // https://www.wolframalpha.com/input/?i=%281%2Fpi+*+arctan%28x%2Fa%29%29%27
        grad_scale[0] = - sign * x / (M_PI * scale * scale + M_PI * x * x); // https://www.wolframalpha.com/input?i=d%5B+atanf%28x+%2F+s%29+%2F+PI+%2B+0.5+%2C+s+%5D
        return 1. / (M_PI * scale + M_PI / scale * x * x);
    } else
    if (function_id == RECIPROCAL_ID) {  // https://www.wolframalpha.com/input/?i=%28x%2Fa%2F%281%2B%7Cx%7C%2Fa%29+%2F+2+%2B+0.5%29%27
        grad_scale[0] = - sign * x / (2. * (scale + x) * (scale + x)); // https://www.wolframalpha.com/input?i=d%5B+x+%2F+s%2F+%281+%2B+x+%2F+s%29+%2F+2.+%2B+0.5+%2C+s+%5D
        return scale / (2. * (scale + x) * (scale + x));
    } else
    if (function_id == LAPLACE_ID) {
        grad_scale[0] = 0.5 / scale / scale * x * exp(- x / scale);
        return 0.5 / scale * exp(- x / scale);
    } else
    if (function_id == UNIFORM_ID) {
        grad_scale[0] = (sign * x / scale > -1 && sign * x / scale < 1) ? - 0.5 * sign * x / scale / scale : 0.;
        return (sign * x / scale > -1 && sign * x / scale < 1) ? 0.5 / scale : 0.;
    } else
    if (function_id == GUDERMANNIAN_ID) {  // https://www.wolframalpha.com/input/?i=%28atan%28tanh%28x+%2F+a%2F+2%29%29+*+2+%2F+pi+%2B.5%29%27
        grad_scale[0] = - sign * x / cosh(sign * x / scale) / M_PI / scale / scale; // https://www.wolframalpha.com/input?i=d%5B%28atan%28tanh%28x+%2F+a%2F+2%29%29+*+2+%2F+pi+%2B.5%29%2Ca%5D
        return 1. / cosh(sign * x / scale) / M_PI / scale;
    } else
    if (function_id == CUBIC_HERMITE_ID) {  // https://www.wolframalpha.com/input/?i=%283+*+%28x%2F2%2Fa%2B.5%29%5E2+-+2+*+%28x%2F2%2Fa%2B.5%29%5E3%29%27
        if ((sign * x / scale < -1.) || (sign * x / scale > 1.)) {
            grad_scale[0] = 0;
            return 0.;
        } else {
            grad_scale[0] = sign * x * (0.75 * (x * x) / pow(scale, 4.) - 0.75 * x / (scale * scale)); // https://www.wolframalpha.com/input?i=d%5B+3+*+%28%28x%29+*+0.5+%2F+s%2B+0.5%29+*+%28%28x%29+*+0.5+%2F+s%2B+0.5%29+-+2+*+%28%28x%29+*+0.5+%2F+s%2B+0.5%29%5E3+%2C+s+%5D
            return 0.75 / scale - 0.75 * (x * x) / pow(scale, 3.);
        }
    } else
    if (function_id == GAUSSIAN_ID) {
        grad_scale[0] = - 1. / scale / scale / sqrt(2. * M_PI) * sign * x * exp(-0.5 * (x / scale) * (x / scale)); // https://www.wolframalpha.com/input?i=d%5B1%2F2+erfc%28-%28x%2Fs%29%2F%28sqrt%282%29+%29%29%2Cs%5D
        return 1. / scale / sqrt(2. * M_PI) * exp(- 0.5 * (x / scale) * (x / scale));
    } else
    if (function_id == GAMMA_ID || function_id == NEG_GAMMA_ID) {  // https://de.wikipedia.org/wiki/Gammaverteilung
        if (dist_shape < 0.) {
            printf("Error in sigmoid_backward_cuda; invalid param p (dist_shape): %g\n", dist_shape);
            grad_scale[0] = 0;
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        double xs;
        if (function_id == GAMMA_ID) {
            if (sign * x + dist_shift * scale <= 0.){
                grad_scale[0] = 0.;
                return 0.;
            }
            xs = (double) sign * (double) x + (double) dist_shift * (double) scale;
        } else {
            if (sign * x - dist_shift * scale >= 0.){
                grad_scale[0] = 0.;
                return 0.;
            }
            xs = - ((double) sign * (double) x - (double) dist_shift * (double) scale);
        }
        grad_scale[0] = (scalar_t) ( // https://www.wolframalpha.com/input?i=d%5B+pow%28%28x+%2B+t*+s%29%2F+s%2C+d%29+*+exp%28-+%28x+%2B+t*+s%29%2F+s%29+*+k%2C+s+%5D
                sign * x * pow(1. / (double) scale, (double) dist_shape + 1.) / tgamma((double) dist_shape)
                * pow(xs, (double) dist_shape - 1.) * exp(- xs / (double) scale)
        );
        return (scalar_t) (
                pow(1. / (double) scale, (double) dist_shape) / tgamma((double) dist_shape)
                * pow(xs, (double) dist_shape - 1.) * exp(- xs / (double) scale)
        );
    } else
    if (function_id == WIGNER_SEMICIRCLE_ID) {
        if (x / scale > 1) {
            grad_scale[0] = 0.;
            return 0.;
        }
        grad_scale[0] = - 2. * sign * x / M_PI / scale / scale / scale * sqrt(scale * scale - x * x); // https://www.wolframalpha.com/input?i=d%5B+0.5+%2B+%28x+*+sqrt%28s%5E2-+x%5E2%29%29+%2F+%28PI+*+s%5E2%29+%2B+asin%28x+%2F+s%29+%2F+PI%2C+s+%5D
        return 2. / M_PI / scale / scale * sqrt(scale * scale - x * x);
    } else
    if (function_id == GUMBEL_MAX_ID) {
        grad_scale[0] = - sign * x * exp(-((sign * x / scale) + exp(-sign * x / scale))) / scale / scale; // https://www.wolframalpha.com/input?i=d%5B+exp%28-exp%28-+x+%2F+s%29%29%2C+s+%5D
        return exp(- ((sign * x / scale) + exp(-(sign * x / scale)))) / scale;
    } else
    if (function_id == GUMBEL_MIN_ID) {
        grad_scale[0] = - sign * x * exp((sign * x / scale) - exp((sign * x / scale))) / scale / scale; // https://www.wolframalpha.com/input?i=d%5B+1-exp%28-exp%28x+%2F+s%29%29%2C+s+%5D
        return exp(- ((- sign * x / scale) + exp(sign * x / scale))) / scale;
    } else
    if (function_id == LEVY_ID || function_id == NEG_LEVY_ID) {
        scalar_t xs;
        if (function_id == LEVY_ID) {
            if (sign * x + dist_shift * scale <= 1e-6) return 0.;
            xs = sign * x + dist_shift * scale;
        } else {
            if (sign * x - dist_shift * scale >= -1e-6) return 0.;
            xs = - (sign * x - dist_shift * scale);
        }
        grad_scale[0] = - sqrt(1. / 2. / M_PI) * sign * x * exp(- scale / 2. / xs) / (xs * xs) / sqrt(scale / xs); // https://www.wolframalpha.com/input?i=d%5B+erfc%28sqrt%28s%2F+2.+%2F+%28x+%2B+t*+s%29%29%29%2C+s+%5D
        return sqrt(scale / 2. / M_PI) * exp(- scale / 2. / xs) / pow(xs, 3./2.);
    } else
    if (function_id == EXPONENTIAL_ID || function_id == NEG_EXPONENTIAL_ID) {
        scalar_t xs;
        if (function_id == EXPONENTIAL_ID) {
            if (sign * x + dist_shift * scale < 0.) return 0.;
            xs = sign * x + dist_shift * scale;
        } else {
            if (sign * x - dist_shift * scale > 0.) return 0.;
            xs = - (sign * x - dist_shift * scale);
        }
        grad_scale[0] = 1. / scale / scale * sign * x * exp(- xs / scale); // https://www.wolframalpha.com/input?i=d%5B+1.+-+exp%28-+%28x+%2B+t*+s%29%2F+s%29+%2C+s+%5D
        return 1. / scale * exp(- xs / scale);
    }
    printf("Error in sigmoid_backward_cuda; the following ID is unknown: %d\n", function_id);
    grad_scale[0] = 0.;
    return std::numeric_limits<scalar_t>::signaling_NaN();
}

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t mlp_forward_cuda(
    scalar_t* signed_x,
    scalar_t* w1,
    scalar_t* w2,
    scalar_t* w3,
    scalar_t* w4,
    scalar_t* w5,
    char mode = 'd'     // char 'd' = 'distance' . char 'z' = 'depth'
    ){
    scalar_t res5[1] = {0};
    scalar_t res4[4] = {0};
    scalar_t res3[4] = {0};
    scalar_t res2[4] = {0};
    scalar_t res1[4] = {0};

    linearforward(w1,signed_x,res1,4,1,1,1);
    activationforward(res1,4*1,mode);
    linearforward(w2,res1,res2,4,4,4,1);
    activationforward(res2,4*1,mode);
    linearforward(w3,res2,res3,4,4,4,1);
    activationforward(res3,4*1,mode);
    linearforward(w4,res3,res4,4,4,4,1);
    linearAddEqual(res4,res1,4*1); // residual connection
    activationforward(res4,4*1,mode);
    linearforward(w5,res4,res5,1,4,4,1);
    if (mode == 'd')
        sigmoidforward(res5,1*1);
    return res5[0];
}

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t mlp_backward_cuda(
    scalar_t* signed_x,
    scalar_t* w1,
    scalar_t* w2,
    scalar_t* w3,
    scalar_t* w4,
    scalar_t* w5,
    scalar_t* d_w1,
    scalar_t* d_w2,
    scalar_t* d_w3,
    scalar_t* d_w4,
    scalar_t* d_w5,
    char mode = 'd'
    ){
    scalar_t d_res5[1] = {1};
    scalar_t d_res4[4] = {0};
    scalar_t d_res3[4] = {0};
    scalar_t d_res2[4] = {0};
    scalar_t d_res1[4] = {0};
    scalar_t d_signed_x[1] = {0};

    scalar_t res5[1] = {0};
    scalar_t res4[4] = {0};
    scalar_t res3[4] = {0};
    scalar_t res2[4] = {0};
    scalar_t res1[4] = {0};
    scalar_t res4_res1[4] = {0};

    scalar_t d_tanh_res4[4] = {0};
    scalar_t d_tanh_res3[4] = {0};
    scalar_t d_tanh_res2[4] = {0};
    scalar_t d_tanh_res1[4] = {0};
    
    // forward
    linearforward(w1,signed_x,res1,4,1,1,1);
    activationforward(res1,4*1,mode);
    linearforward(w2,res1,res2,4,4,4,1);
    activationforward(res2,4*1,mode);
    linearforward(w3,res2,res3,4,4,4,1);
    activationforward(res3,4*1,mode);
    linearforward(w4,res3,res4,4,4,4,1);
    linearAdd(res4,res1,res4_res1,4*1); // residual connection
    activationforward(res4_res1,4*1,mode);
    linearforward(w5,res4_res1,res5,1,4,4,1);

    if (mode == 'd')
        sigmoidforward(res5,1*1);

    // backward

    // Gradients with respect to res4 and w5
    if (mode == 'd')
        sigmoidbackward(res5,d_res5,1*1); //d_res5 = d_sigmoid(res5)
    //d_w5 = d_res5 * res4_res1
    //d_res4 = d_res5 * w5
    linearbackward(w5,res4_res1,d_w5,d_res4,d_res5,1,4,4,1);

    // Gradient with respect to res3 and w4
    activationbackward(res4_res1,d_tanh_res4,1*4,mode); //d_tanh_res4 = d_tanh(res4_res1)
    linearMulEqual(d_res4,d_tanh_res4,1*4); //d_res4 *= d_tanh_res4

    //d_res3 = d_res4 * w4
    //d_w4 = d_res4 * res3
    linearbackward(w4,res3,d_w4,d_res3,d_res4,4,4,4,1);

    // Gradient with respect to res2 and w3
    activationbackward(res3,d_tanh_res3,4*1,mode); //d_tanh_res3 =  d_tanh(res3)
    linearMulEqual(d_res3,d_tanh_res3,4*1); //d_res3 *= d_tanh_res3

    //d_res2 = d_res3 * w3
    //d_w3 = d_res3 * res2
    linearbackward(w3,res2,d_w3,d_res2,d_res3,4,4,4,1);

    // Gradient with respect to res1 and w2
    activationbackward(res2,d_tanh_res2,4*1,mode); //d_tanh_res2 = d_tanh(res2)
    linearMulEqual(d_res2,d_tanh_res2,4*1); //d_res2 *= d_tanh_res2

    //d_res1 = d_res2 * w2 + d_res4
    //d_w2 = d_res2 * res1
    linearbackward(w2,res1,d_w2,d_res1,d_res2,4,4,4,1);
    linearAddEqual(d_res1,d_res4,4*1); // residual connection

    // Gradient with respect to x and w1
    activationbackward(res1,d_tanh_res1,4*1,mode); //d_tanh_res1 = d_tanh(res1)
    linearMulEqual(d_res1,d_tanh_res1,4*1); //d_res1 *= d_tanh_res1
    
    //d_signed_x = d_res1 * w1
    //d_w1 = d_res1 * signed_x
    linearbackward(w1,signed_x,d_w1,d_signed_x,d_res1,4,1,1,1);
    return d_signed_x[0];
}

#define MAX_TCN                                 1
#define PROBABILISTIC_SUM_TCN                   2
#define EINSTEIN_SUM_TCN                        3
#define HAMACHER_TCN                            4
#define FRANK_TCN                               5
#define YAGER_TCN                               6
#define ACZEL_ALSINA_TCN                        7
#define DOMBI_TCN                               8
#define SCHWEIZER_SKLAR_TCN                     9

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t t_conorm_forward_cuda(
        const int t_conorm_id,
        const scalar_t a_existing,
        const scalar_t b_new,
        const int face_id,
        const scalar_t t_conorm_p
    ) {
    if (t_conorm_id == MAX_TCN) {
        return max(a_existing, b_new);
    } else
    if (t_conorm_id == PROBABILISTIC_SUM_TCN) {
        return a_existing + b_new - a_existing * b_new;
    } else
    if (t_conorm_id == EINSTEIN_SUM_TCN) {
        return (a_existing + b_new) / (1 + a_existing * b_new);
    } else
    if (t_conorm_id == HAMACHER_TCN) {
        if (t_conorm_p < 0.) {
            printf("Error in t_conorm_forward_cuda; invalid param p (t_conorm_p): %g\n", t_conorm_p);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        const scalar_t a = 1. - a_existing;
        const scalar_t b = 1. - b_new;
        const scalar_t c = (a * b) / max(t_conorm_p + (1. - t_conorm_p) * (a + b - a * b), 1e-6);
        return 1. - c;
    } else
    if (t_conorm_id == FRANK_TCN) {
        if (t_conorm_p <= 0. || t_conorm_p == 1.) {
            printf("Error in t_conorm_forward_cuda; invalid param p (t_conorm_p): %g\n", t_conorm_p);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        // log(x, basis) = log(x) / log(basis)
        const scalar_t a = 1. - a_existing;
        const scalar_t b = 1. - b_new;
        const scalar_t c = log1p( (pow(t_conorm_p, a) - 1.) * (pow(t_conorm_p, b) - 1.) / (t_conorm_p - 1.) ) / log(t_conorm_p);
        return 1. - c;
    } else
    if (t_conorm_id == YAGER_TCN) {
        if (t_conorm_p <= 0.) {
            printf("Error in t_conorm_forward_cuda; invalid param p (t_conorm_p): %g\n", t_conorm_p);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        const scalar_t a = 1. - a_existing;
        const scalar_t b = 1. - b_new;
        const scalar_t c = max(0., 1. - pow(pow(1. - a, t_conorm_p) + pow(1. - b, t_conorm_p), 1./t_conorm_p));
        return 1. - c;
    } else
    if (t_conorm_id == ACZEL_ALSINA_TCN) {
        if (t_conorm_p <= 0.) {
            printf("Error in t_conorm_forward_cuda; invalid param p (t_conorm_p): %g\n", t_conorm_p);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        const scalar_t a = 1. - a_existing;
        const scalar_t b = 1. - b_new;
        if (a < 1e-8) return 1.;  // note that 1. because of 1-c with c=0;
        if (b < 1e-8) return 1.;  // note that 1. because of 1-c with c=0;
        const scalar_t c = exp(- pow(  pow(- log(a), t_conorm_p) + pow(- log(b), t_conorm_p)  , 1. / t_conorm_p));
        return 1. - c;
    } else
    if (t_conorm_id == DOMBI_TCN) {
        if (t_conorm_p <= 0.) {
            printf("Error in t_conorm_forward_cuda; invalid param p (t_conorm_p): %g\n", t_conorm_p);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        const scalar_t a = 1. - a_existing;
        const scalar_t b = 1. - b_new;
        const scalar_t p = t_conorm_p;
        if (a < 1e-8) return 1.;  // note that 1. because of 1-c with c=0;
        if (b < 1e-8) return 1.;  // note that 1. because of 1-c with c=0;
        const scalar_t c = 1. / (1. +
                pow(
                        pow((1. - a) / a, p)
                        + pow((1. - b) / b, p)
                    , 1./p)
                    );
        return 1. - c;
    } else
    if (t_conorm_id == SCHWEIZER_SKLAR_TCN) {
        if (t_conorm_p >= 0.) {
            printf("Error in t_conorm_forward_cuda; invalid param p (t_conorm_p): %g\n", t_conorm_p);
            return std::numeric_limits<scalar_t>::signaling_NaN();
        }
        const scalar_t a = 1. - a_existing;
        const scalar_t b = 1. - b_new;
        const scalar_t c = pow(pow(a, t_conorm_p) + pow(b, t_conorm_p) - 1., 1./t_conorm_p);
        return 1. - c;
    }
    printf("Error in t_conorm_forward_cuda; the following ID is unknown: %d\n", t_conorm_id);
    return std::numeric_limits<scalar_t>::signaling_NaN();
}

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t t_conorm_backward_cuda(
        const int t_conorm_id,
        const scalar_t a_all,
        const scalar_t b_current,
        const int number_of_faces,
        const scalar_t t_conorm_p
    ) {
    if (t_conorm_id == MAX_TCN) {
        return (a_all == b_current) ? 1. : 0.;
    } else
    if (t_conorm_id == PROBABILISTIC_SUM_TCN) {
        return (1. - a_all) / max(1. - b_current, 1e-6);
    } else
    if (t_conorm_id == EINSTEIN_SUM_TCN) {
        return (1. - a_all * a_all) / max(1. - b_current * b_current, 1e-6);
    } else
    if (t_conorm_id == HAMACHER_TCN) {
        return (1.0 - a_all)*(-a_all - t_conorm_p*(1.0 - a_all) + t_conorm_p + 1.0) / max((1.0 - b_current)*(-b_current - t_conorm_p*(1.0 - b_current) + t_conorm_p + 1.0), 1e-6);
    } else
    if (t_conorm_id == FRANK_TCN) {
        const scalar_t d = pow(t_conorm_p, 1.0 - b_current) - 1.0;
        return pow(t_conorm_p, a_all - b_current) * (pow(t_conorm_p, 1.0 - a_all) - 1.0) / (d + copysign(1e-6, d));
    } else
    if (t_conorm_id == YAGER_TCN) {
        if (a_all == 1.) return 0.;
        return pow(b_current, t_conorm_p - 1.) * pow(a_all, 1. - t_conorm_p);
    } else
    if (t_conorm_id == ACZEL_ALSINA_TCN) {
        return (1. - a_all)
               * pow(-log1p(max(- b_current, -1.+1e-6)), t_conorm_p - 1.)
               * pow(-log1p(max(- a_all    , -1.+1e-6)), 1. - t_conorm_p)
               / max(1. - b_current, 1e-6);
    } else
    if (t_conorm_id == DOMBI_TCN) {
        return (1. - a_all) * (1. - a_all)
               * pow(b_current / max(1. - b_current, 1e-6), t_conorm_p - 1.)
               * pow(a_all / max(1. - a_all, 1e-6), 1. - t_conorm_p)
               / max(1. - b_current, 1e-6) / max(1. - b_current, 1e-6);
    } else
    if (t_conorm_id == SCHWEIZER_SKLAR_TCN) {
        const scalar_t a = max(1. - a_all, 1e-6);
        const scalar_t b = max(1. - b_current, 1e-6);
        const scalar_t c = pow(b, t_conorm_p - 1.) * pow(pow(b, t_conorm_p) + pow(pow(-pow(b, t_conorm_p) + pow(a, t_conorm_p) + 1., 1./t_conorm_p), t_conorm_p) - 1., (1. - t_conorm_p)/t_conorm_p);
        return c;
    }
    printf("Error in t_conorm_backward_cuda; the following ID is unknown: %d\n", t_conorm_id);
    return std::numeric_limits<scalar_t>::signaling_NaN();
}


// triangle preprocessing
template <typename scalar_t>
__global__ void forward_render_inv_cuda_kernel(
        const scalar_t* __restrict__ faces,
        scalar_t* faces_info,
        int batch_size,
        int num_faces,
        int image_size) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    // const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t* face_inv = &faces_info[i * 27];
    scalar_t* face_sym = &faces_info[i * 27+9];
    scalar_t* face_obt = &faces_info[i * 27+18];

    /* p[num][xy]: x, y is (-1, 1). */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = face[3 * num + dim]; // no normalize
        }
    }
    /* compute face_inv */
    scalar_t face_inv_star[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_determinant = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    face_inv_determinant = face_inv_determinant > 0 ? max(face_inv_determinant, 1e-10) : min(face_inv_determinant, -1e-10);
    /* set to global memory */
    for (int k = 0; k < 9; k++) {
        face_inv[k] = face_inv_star[k] / face_inv_determinant;
    }
    /* F * F.T */
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            face_sym[j * 3 + k] = face[j * 3 + 0] * face[k * 3 + 0] +
                                  face[j * 3 + 1] * face[k * 3 + 1] + 
                                  1;
        }
    }
    /* check if one arc is obt arc */
    for (int k = 0; k < 3; k++) {
        const int k0 = k;
        const int k1 = (k + 1) % 3;
        const int k2 = (k + 2) % 3;
        if ((p[k1][0] - p[k0][0]) * (p[k2][0] - p[k0][0]) + (p[k1][1] - p[k0][1]) * (p[k2][1] - p[k0][1]) < 0) {
            face_obt[k0] = 1;
            break;
        }
    }
}

template <typename scalar_t>
__global__ void forward_render_cuda_kernel(
        const scalar_t* __restrict__ faces,
        const scalar_t* __restrict__ textures,
        const scalar_t* __restrict__ faces_info,
        scalar_t* aggrs_info,
        scalar_t* soft_colors,
        int batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        int texture_res,
        int dist_func,
        scalar_t* dist_scale,
        scalar_t* dist_scale_rgb,
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
        scalar_t* w1,
        scalar_t* w2,
        scalar_t* w3,
        scalar_t* w4,
        scalar_t* w5,
        scalar_t* wrgb1,
        scalar_t* wrgb2,
        scalar_t* wrgb3,
        scalar_t* wrgb4,
        scalar_t* wrgb5
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = is - 1 - (pn / is);
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1. - is) / is;
    const scalar_t xp = (2. * xi + 1. - is) / is;

    const scalar_t *face = &faces[bn * nf * 9] - 9;
    const scalar_t *texture = &textures[bn * nf * texture_size * 3] - texture_size * 3;
    const scalar_t *face_info = &faces_info[bn * nf * 27] - 27;

    const scalar_t threshold = dist_eps * dist_scale[0];

    // Initialize pixel color
    scalar_t soft_color[4] = {1., 1., 1., 0.};
    scalar_t softmax_sum = exp(aggr_rgb_eps / aggr_rgb_gamma);
    scalar_t softmax_max = aggr_rgb_eps;
    for (int k = 0; k < 3; k++) {
        if (aggr_rgb_func == 0) { // hard assign, set to background
            soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn];
        } else
        if (aggr_rgb_func == 1) {
            soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn] * softmax_sum; // initialize background color
        } else {
            soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn] * softmax_sum; // initialize background color
        }
    }
    scalar_t depth_min = 10000000;
    int face_index_min = -1;

    for (int fn = 0; fn < nf; fn++) {
        face += 9;
        texture += texture_size * 3;
        face_info += 27;

        if (check_border(xp, yp, face, sqrt(threshold))) continue; // triangle too far away from pixel

        scalar_t dis;
        scalar_t dis_x;
        scalar_t dis_y;
        scalar_t t[3];
        scalar_t w[3];
        scalar_t w_clip[3];
        scalar_t sign;
        scalar_t soft_fragment;

        // compute barycentric coordinate w as a preprocessing step for check_pixel_inside / euclidean_p2f_distance
        barycentric_coordinate(w, xp, yp, face_info);

        // compute probability map based on distance functions
        if (dist_func == HEAVISIDE_ID) {
            // hard / heaviside: faster by directly checking.
            soft_fragment = check_pixel_inside(w) ? 1. : 0.;
        } else {
            euclidean_p2f_distance(sign, dis_x, dis_y, w, t, face, face_info, xp, yp);
            dis = dis_x * dis_x + dis_y * dis_y;
            if (sign < 0 && dis >= threshold) continue;
            if (!dist_squared) {
                dis = sqrt(dis);
            }
            if (dist_func == MLP_ID){
                scalar_t signed_x[1];
                signed_x[0] = sign*dis;
                soft_fragment = mlp_forward_cuda(signed_x,w1,w2,w3,w4,w5);
            } else {
                soft_fragment = sigmoid_forward_cuda(
                                    dist_func,
                                    sign,
                                    dis,
                                    (scalar_t) dist_scale[0],
                                    (scalar_t) dist_shape,
                                    (scalar_t) dist_shift
                            );
            }
        }

        if (soft_fragment <= PROBABILITY_THRESHOLD) {
            continue;
        }

    /**********************************************************************************************************************/

        // aggragate for alpha channel
        if (aggr_alpha_func == 0) { // hard assign
            if (soft_fragment > 0.5) soft_color[3] = 1.;
        } else {

            soft_color[3] = t_conorm_forward_cuda(
                    aggr_alpha_func,
                    soft_color[3],
                    soft_fragment,
                    fn,
                    (scalar_t) aggr_alpha_t_conorm_p
            );

        }

    /**********************************************************************************************************************/

        for (int k = 0; k < 3; k++) w_clip[k] = w[k];
        barycentric_clip(w_clip);
        const scalar_t zp = 1. / (w_clip[0] / face[2] + w_clip[1] / face[5] + w_clip[2] / face[8]);
        if (zp < near || zp > far) continue; // triangle out of screen, pass

    /**********************************************************************************************************************/

        // aggregate for rgb channels
        if (aggr_rgb_func == 0) { // Hard assign
            if (zp < depth_min && check_pixel_inside(w) && (double_side || check_face_frontside(face))) {
                depth_min = zp;
                face_index_min = fn;
                for (int k = 0; k < 3; k++) {
                    soft_color[k] = forward_sample_texture(texture, w_clip, texture_res, k, texture_type);
                }
            }
        } else
        if (aggr_rgb_func == 1) { // D * Softmax (Z)
            if (check_face_frontside(face) || double_side) {
                const scalar_t zp_norm =  (far - zp) / (far - near);
                scalar_t exp_delta_zp = 1.;
                if (zp_norm > softmax_max) {
                    exp_delta_zp = exp((softmax_max - zp_norm) / aggr_rgb_gamma);
                    softmax_max = zp_norm;
                }
                const scalar_t exp_z = exp((zp_norm - softmax_max) / aggr_rgb_gamma);
                softmax_sum = exp_delta_zp * softmax_sum + exp_z * soft_fragment;
                for (int k = 0; k < 3; k++) {
                    const scalar_t color_k = forward_sample_texture(texture, w_clip, texture_res, k, texture_type);
                    soft_color[k] = exp_delta_zp * soft_color[k] + exp_z * soft_fragment * color_k;// * soft_fragment;
                }
            }
        } else 
        if (aggr_rgb_func == 3) { // D * MLP (Z)
            if (check_face_frontside(face) || double_side) {
                scalar_t zp_norm[1];
                scalar_t exp_delta_zp;
                scalar_t exp_z_;

                zp_norm[0] = (far - zp) / (far - near);
                scalar_t mlp_depth = mlp_forward_cuda(zp_norm,wrgb1,wrgb2,wrgb3,wrgb4,wrgb5,'z');
                
                if (zp_norm[0] > softmax_max) {
                    exp_delta_zp = mlp_depth;
                    exp_z_ = 1;
                    softmax_max = zp_norm[0];
                }
                else {
                    exp_delta_zp = 1;
                    exp_z_ = mlp_depth;
                }

                const scalar_t exp_z = exp_z_; //this line is added to ensure that exp_z remains constant
                softmax_sum = exp_delta_zp * softmax_sum + exp_z * soft_fragment;
                for (int k = 0; k < 3; k++) {
                    const scalar_t color_k = forward_sample_texture(texture, w_clip, texture_res, k, texture_type);
                    soft_color[k] = exp_delta_zp * soft_color[k] + exp_z * soft_fragment * color_k;// * soft_fragment;
                }
            }
        } else
        if (aggr_rgb_func == 2) { // Softmax (D * Z)
            if (check_face_frontside(face) || double_side) {
                const scalar_t zp_norm =  (far - zp) / (far - near);
                scalar_t exp_delta_zp = 1.;
                if (zp_norm > softmax_max) {
                    exp_delta_zp = exp((softmax_max - zp_norm) / aggr_rgb_gamma);
                    softmax_max = zp_norm;
                }
                const scalar_t exp_z = exp((zp_norm - softmax_max) / aggr_rgb_gamma);
                softmax_sum = exp_delta_zp * softmax_sum + exp_z;
                for (int k = 0; k < 3; k++) {
                    const scalar_t color_k = forward_sample_texture(texture, w_clip, texture_res, k, texture_type);
                    soft_color[k] = exp_delta_zp * soft_color[k] + exp_z * color_k;// * soft_fragment;
                }
            }
        } else {
            if (check_face_frontside(face) || double_side) {
                scalar_t zp_norm[1];
                scalar_t exp_delta_zp;
                scalar_t exp_z_;

                zp_norm[0] = (far - zp) / (far - near);
                scalar_t sign = (zp_norm[0] >= 0) ? 1 : -1; 
                scalar_t value = (zp_norm[0] >= 0) ? zp_norm[0] : zp_norm[0] * -1;

                scalar_t soft_depth = sigmoid_forward_cuda(
                    aggr_rgb_func,
                    sign,
                    value,
                    (scalar_t) dist_scale_rgb[0],
                    (scalar_t) dist_shape,
                    (scalar_t) dist_shift
                );
                
                if (zp_norm[0] > softmax_max) {
                    exp_delta_zp = soft_depth;
                    exp_z_ = 1;
                    softmax_max = zp_norm[0];
                }
                else {
                    exp_delta_zp = 1;
                    exp_z_ = soft_depth;
                }

                const scalar_t exp_z = exp_z_; //this line is added to ensure that exp_z remains constant
                softmax_sum = exp_delta_zp * softmax_sum + exp_z * soft_fragment;
                for (int k = 0; k < 3; k++) {
                    const scalar_t color_k = forward_sample_texture(texture, w_clip, texture_res, k, texture_type);
                    soft_color[k] = exp_delta_zp * soft_color[k] + exp_z * soft_fragment * color_k;// * soft_fragment;
                }
            }
        }
    }

    /**********************************************************************************************************************/

    // finalize aggregation
    soft_colors[(bn * 4 + 3) * (is * is) + pn] = soft_color[3];

    if (aggr_rgb_func == 0) {
        if (face_index_min != -1)
            for (int k = 0; k < 3; k++) {
                soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k];
            }
        aggrs_info[(bn * 2 + 0) * (is * is) + pn] = depth_min;
        aggrs_info[(bn * 2 + 1) * (is * is) + pn] = face_index_min;
    } else
    if (aggr_rgb_func == 1) {
        for (int k = 0; k < 3; k++) {
            soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k] / softmax_sum;
        }
        aggrs_info[(bn * 2 + 0) * (is * is) + pn] = softmax_sum;
        aggrs_info[(bn * 2 + 1) * (is * is) + pn] = softmax_max;
    } else { // MLP and all other dists
        for (int k = 0; k < 3; k++) {
            soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k] / softmax_sum;
        }
        aggrs_info[(bn * 2 + 0) * (is * is) + pn] = softmax_sum;
        aggrs_info[(bn * 2 + 1) * (is * is) + pn] = softmax_max;
    }
}

template <typename scalar_t>
__global__ void backward_render_cuda_kernel(
        const scalar_t* __restrict__ faces,
        const scalar_t* __restrict__ textures,
        const scalar_t* __restrict__ soft_colors,
        const scalar_t* __restrict__ faces_info,
        const scalar_t* __restrict__ aggrs_info, // 0: sum, 1: max z*D
        scalar_t* grad_faces,
        scalar_t* grad_textures,
        scalar_t* grad_soft_colors,
        int batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        int texture_res,
        int dist_func,
        scalar_t* dist_scale,
        scalar_t* dist_scale_rgb,
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
        scalar_t* w1,
        scalar_t* w2,
        scalar_t* w3,
        scalar_t* w4,
        scalar_t* w5,
        scalar_t* wrgb1,
        scalar_t* wrgb2,
        scalar_t* wrgb3,
        scalar_t* wrgb4,
        scalar_t* wrgb5,
        scalar_t* grad_w1,
        scalar_t* grad_w2,
        scalar_t* grad_w3,
        scalar_t* grad_w4,
        scalar_t* grad_w5,
        scalar_t* grad_wrgb1,
        scalar_t* grad_wrgb2,
        scalar_t* grad_wrgb3,
        scalar_t* grad_wrgb4,
        scalar_t* grad_wrgb5,
        scalar_t* grad_scale,
        scalar_t* grad_scale_rgb
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = is - 1 - (pn / is);
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1 - is) / is;
    const scalar_t xp = (2. * xi + 1 - is) / is;

    const scalar_t* face = &faces[bn * nf * 9] - 9;
    const scalar_t* texture = &textures[bn * nf * texture_size * 3] - texture_size * 3;
    const scalar_t* face_info = &faces_info[bn * nf * 27] - 27;

    const scalar_t threshold = dist_eps * dist_scale[0];

    const scalar_t softmax_sum = aggrs_info[(bn * 2 + 0) * (is * is) + pn];
    const scalar_t softmax_max = aggrs_info[(bn * 2 + 1) * (is * is) + pn];

    for (int fn = 0; fn < nf; fn++) {
        face += 9;
        texture += texture_size * 3;
        face_info += 27;

        if (check_border(xp, yp, face, sqrt(threshold))) continue;

        scalar_t dis;
        scalar_t dis_x;
        scalar_t dis_y;
        scalar_t t[3];
        scalar_t w[3];
        scalar_t w0[3];
        scalar_t sign;
        scalar_t soft_fragment;

        barycentric_coordinate(w, xp, yp, face_info);

        // compute probability map based on distance functions
        if (dist_func == HEAVISIDE_ID) {
            // hard / heaviside: faster by directly checking.
            soft_fragment = check_pixel_inside(w) ? 1. : 0.;
        } else {

            euclidean_p2f_distance(sign, dis_x, dis_y, w, t, face, face_info, xp, yp);
            dis = dis_x * dis_x + dis_y * dis_y;
            if (sign < 0 && dis >= threshold) continue;
            if (!dist_squared) {
                dis = sqrt(dis);
            }
            if (dist_func == MLP_ID){
                scalar_t signed_x[1];
                signed_x[0] = sign*dis;
                soft_fragment = mlp_forward_cuda(signed_x,w1,w2,w3,w4,w5);
            } else {
                soft_fragment = sigmoid_forward_cuda(
                    dist_func,
                    sign,
                    dis,
                    (scalar_t) dist_scale[0],
                    (scalar_t) dist_shape,
                    (scalar_t) dist_shift
            );
            }
            
        }

        if (soft_fragment <= PROBABILITY_THRESHOLD) {
            continue;
        }

        scalar_t* grad_face = &grad_faces[(bn * nf + fn) * 9];
        scalar_t* grad_texture = &grad_textures[(bn * nf + fn) * texture_size * 3];
        scalar_t grad_v[3][3] = {0};
        scalar_t C_grad_xy = 0;
        scalar_t grad_x = 0;
    /**********************************************************************************************************************/

        // aggragate for alpha channel
        scalar_t C_grad_xy_alpha = grad_soft_colors[(bn * 4 + 3) * (is * is) + pn];

        if (aggr_alpha_func == 0) { // hard assign
            // hard assign alpha channels does not have gradient
        } else {
            C_grad_xy_alpha *= t_conorm_backward_cuda(
                    aggr_alpha_func,
                    soft_colors[(bn * 4 + 3) * (is * is) + pn],
                    soft_fragment,
                    nf,
                    (scalar_t) aggr_alpha_t_conorm_p
            );
        }

        C_grad_xy += C_grad_xy_alpha;

    /**********************************************************************************************************************/

        for (int k = 0; k < 3; k++) w0[k] = w[k];
        barycentric_clip(w);
        const scalar_t zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);
        if (zp < near || zp > far) continue; // triangle out of screen, pass

        // aggregate for rgb channels 
        if (aggr_rgb_func == 0) { // Hard assign, no gradient to xyz
            if (fn == softmax_max) {
                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < texture_size; j++) {
                        atomicAdd(&grad_texture[3 * j + k], backward_sample_texture(grad_soft_colors[(bn * 4 + k) * (is * is) + pn], w, texture_res, j, texture_type));
                    }
                }
            }
        } else
        if (aggr_rgb_func == 1 && (check_face_frontside(face) || double_side)) { // Softmax (Z * D)
            scalar_t C_grad_xyz_rgb = 0.;

            const scalar_t zp_norm = (far - zp) / (far - near);
            const scalar_t zp_softmax = soft_fragment * exp((zp_norm - softmax_max) / aggr_rgb_gamma) / softmax_sum;

            for (int k = 0; k < 3; k++) {
                const scalar_t grad_soft_color_k = grad_soft_colors[(bn * 4 + k) * (is * is) + pn];

                for (int j = 0; j < texture_size; j++) {
                    const scalar_t grad_t = backward_sample_texture(grad_soft_color_k, w, texture_res, j, texture_type);
                    atomicAdd(&grad_texture[3 * j + k], zp_softmax * grad_t);
                }

                const scalar_t color_k = forward_sample_texture(texture, w, texture_res, k, texture_type);
                C_grad_xyz_rgb += grad_soft_color_k * (color_k - soft_colors[(bn * 4 + k) * (is * is) + pn]);
            }
            C_grad_xyz_rgb *= zp_softmax;
            C_grad_xy += C_grad_xyz_rgb / soft_fragment;

            const scalar_t C_grad_z_rgb = C_grad_xyz_rgb / aggr_rgb_gamma / (near - far) * zp * zp;
            grad_v[0][2] = C_grad_z_rgb * w[0] / face[2] / face[2];
            grad_v[1][2] = C_grad_z_rgb * w[1] / face[5] / face[5];
            grad_v[2][2] = C_grad_z_rgb * w[2] / face[8] / face[8];
        } else
        if (aggr_rgb_func == 3 && (check_face_frontside(face) || double_side)) { // MLP (Z * D)
            scalar_t C_grad_xyz_rgb = 0.;

            scalar_t zp_norm[1];
            zp_norm[0] = (far - zp) / (far - near);
            scalar_t grad = mlp_backward_cuda(zp_norm,wrgb1,wrgb2,wrgb3,wrgb4,wrgb5,grad_wrgb1,grad_wrgb2,grad_wrgb3,grad_wrgb4,grad_wrgb5,'z');
            scalar_t zp_softmax = soft_fragment * grad / softmax_sum;

            for (int k = 0; k < 3; k++) {
                const scalar_t grad_soft_color_k = grad_soft_colors[(bn * 4 + k) * (is * is) + pn];

                for (int j = 0; j < texture_size; j++) {
                    const scalar_t grad_t = backward_sample_texture(grad_soft_color_k, w, texture_res, j, texture_type);
                    atomicAdd(&grad_texture[3 * j + k], zp_softmax * grad_t);
                }

                const scalar_t color_k = forward_sample_texture(texture, w, texture_res, k, texture_type);
                C_grad_xyz_rgb += grad_soft_color_k * (color_k - soft_colors[(bn * 4 + k) * (is * is) + pn]);
            }
            C_grad_xyz_rgb *= zp_softmax;
            C_grad_xy += C_grad_xyz_rgb / soft_fragment;

            const scalar_t C_grad_z_rgb = C_grad_xyz_rgb / aggr_rgb_gamma / (near - far) * zp * zp;
            grad_v[0][2] = C_grad_z_rgb * w[0] / face[2] / face[2];
            grad_v[1][2] = C_grad_z_rgb * w[1] / face[5] / face[5];
            grad_v[2][2] = C_grad_z_rgb * w[2] / face[8] / face[8];
        } else
        if (check_face_frontside(face) || double_side) {
            scalar_t C_grad_xyz_rgb = 0.;

            scalar_t zp_norm[1];
            zp_norm[0] = (far - zp) / (far - near);
            scalar_t sign = (zp_norm[0] >= 0) ? 1 : -1; 
            scalar_t value = (zp_norm[0] >= 0) ? zp_norm[0] : zp_norm[0] * -1;

            scalar_t grad = sigmoid_backward_cuda(
                aggr_rgb_func,
                sign,
                value,
                (scalar_t) dist_scale_rgb[0],
                (scalar_t) dist_shape,
                (scalar_t) dist_shift,
                grad_scale_rgb
            );
            if (isnan(grad))
                printf("grad is nan\n");

            scalar_t zp_softmax = soft_fragment * grad / softmax_sum;
            if (isnan(zp_softmax))
                printf("zp_softmax is nan\n");

            for (int k = 0; k < 3; k++) {
                const scalar_t grad_soft_color_k = grad_soft_colors[(bn * 4 + k) * (is * is) + pn];

                for (int j = 0; j < texture_size; j++) {
                    const scalar_t grad_t = backward_sample_texture(grad_soft_color_k, w, texture_res, j, texture_type);
                    atomicAdd(&grad_texture[3 * j + k], zp_softmax * grad_t);
                }

                const scalar_t color_k = forward_sample_texture(texture, w, texture_res, k, texture_type);
                C_grad_xyz_rgb += grad_soft_color_k * (color_k - soft_colors[(bn * 4 + k) * (is * is) + pn]);
            }
            C_grad_xyz_rgb *= zp_softmax;
            if (isnan(C_grad_xyz_rgb))
                printf("C_grad_xyz_rgb is nan\n");

            C_grad_xy += C_grad_xyz_rgb / soft_fragment;

            const scalar_t C_grad_z_rgb = C_grad_xyz_rgb / aggr_rgb_gamma / (near - far) * zp * zp;
            if (isnan(C_grad_z_rgb))
                printf("C_grad_z_rgb is nan\n");
            grad_v[0][2] = C_grad_z_rgb * w[0] / face[2] / face[2];
            grad_v[1][2] = C_grad_z_rgb * w[1] / face[5] / face[5];
            grad_v[2][2] = C_grad_z_rgb * w[2] / face[8] / face[8];
        }

    /**********************************************************************************************************************/
        if (dist_func == MLP_ID){
            scalar_t signed_x[1];
            signed_x[0] = sign*dis;
            grad_x = mlp_backward_cuda(signed_x,w1,w2,w3,w4,w5,grad_w1,grad_w2,grad_w3,grad_w4,grad_w5);
            C_grad_xy *= grad_x;
        } else {
            grad_x = sigmoid_backward_cuda(
                dist_func,
                sign,
                dis,
                (scalar_t) dist_scale[0],
                (scalar_t) dist_shape,
                (scalar_t) dist_shift,
                grad_scale
                );
            C_grad_xy *= grad_x;
        }
        

        // compute probability map gradient based on distance functions
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 2; l++) {
                if (dist_squared) {
                    grad_v[k][l] = 2 * sign * C_grad_xy * (t[k] + w0[k]) * (l == 0 ? dis_x : dis_y);
                } else {
                    grad_v[k][l] = sign * C_grad_xy * (t[k] + w0[k]) * (l == 0 ? dis_x : dis_y) / max(sqrt(dis_x * dis_x + dis_y * dis_y), 1e-6);
                }
            }
        }

        atomicAdd(&grad_face[0], grad_v[0][0]);
        atomicAdd(&grad_face[1], grad_v[0][1]);
        atomicAdd(&grad_face[3], grad_v[1][0]);
        atomicAdd(&grad_face[4], grad_v[1][1]);
        atomicAdd(&grad_face[6], grad_v[2][0]);
        atomicAdd(&grad_face[7], grad_v[2][1]);

        atomicAdd(&grad_face[2], grad_v[0][2]);
        atomicAdd(&grad_face[5], grad_v[1][2]);
        atomicAdd(&grad_face[8], grad_v[2][2]);
    }
}

}  // namespace

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
        ) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(2);
    const auto texture_res = int(sqrt(texture_size));
    const int threads = NUM_THREADS;
    const dim3 blocks_1 ((batch_size * num_faces - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.scalar_type(), "forward_render_inv_cuda", ([&] {
      forward_render_inv_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
          faces.data_ptr<scalar_t>(),
          faces_info.data_ptr<scalar_t>(),
          batch_size,
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_transform_inv_triangle: %s\n", cudaGetErrorString(err));

    const dim3 blocks_2 ((batch_size * image_size * image_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.scalar_type(), "forward_eff_render_cuda", ([&] {
      forward_render_cuda_kernel<scalar_t><<<blocks_2, threads>>>(
          faces.data_ptr<scalar_t>(),
          textures.data_ptr<scalar_t>(),
          faces_info.data_ptr<scalar_t>(),
          aggrs_info.data_ptr<scalar_t>(),
          soft_colors.data_ptr<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          texture_size,
          texture_res,
          dist_func,
          dist_scale.data_ptr<scalar_t>(),
          dist_scale_rgb.data_ptr<scalar_t>(),
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
          w1.data_ptr<scalar_t>(),
          w2.data_ptr<scalar_t>(),
          w3.data_ptr<scalar_t>(),
          w4.data_ptr<scalar_t>(),
          w5.data_ptr<scalar_t>(),
          wrgb1.data_ptr<scalar_t>(),
          wrgb2.data_ptr<scalar_t>(),
          wrgb3.data_ptr<scalar_t>(),
          wrgb4.data_ptr<scalar_t>(),
          wrgb5.data_ptr<scalar_t>()
          );
      }));

    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward_render: %s\n", cudaGetErrorString(err));

    return {faces_info, aggrs_info, soft_colors};
}

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
        ) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(2);
    const auto texture_res = int(sqrt(texture_size));
    const int threads = NUM_THREADS;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.scalar_type(), "backward_render_cuda", ([&] {
      backward_render_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data_ptr<scalar_t>(),
          textures.data_ptr<scalar_t>(),
          soft_colors.data_ptr<scalar_t>(),
          faces_info.data_ptr<scalar_t>(),
          aggrs_info.data_ptr<scalar_t>(),
          grad_faces.data_ptr<scalar_t>(),
          grad_textures.data_ptr<scalar_t>(),
          grad_soft_colors.data_ptr<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          texture_size,
          texture_res,
          dist_func,
          dist_scale.data_ptr<scalar_t>(),
          dist_scale_rgb.data_ptr<scalar_t>(),
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
          w1.data_ptr<scalar_t>(),
          w2.data_ptr<scalar_t>(),
          w3.data_ptr<scalar_t>(),
          w4.data_ptr<scalar_t>(),
          w5.data_ptr<scalar_t>(),
          wrgb1.data_ptr<scalar_t>(),
          wrgb2.data_ptr<scalar_t>(),
          wrgb3.data_ptr<scalar_t>(),
          wrgb4.data_ptr<scalar_t>(),
          wrgb5.data_ptr<scalar_t>(),
          grad_w1.data_ptr<scalar_t>(),
          grad_w2.data_ptr<scalar_t>(),
          grad_w3.data_ptr<scalar_t>(),
          grad_w4.data_ptr<scalar_t>(),
          grad_w5.data_ptr<scalar_t>(),
          grad_wrgb1.data_ptr<scalar_t>(),
          grad_wrgb2.data_ptr<scalar_t>(),
          grad_wrgb3.data_ptr<scalar_t>(),
          grad_wrgb4.data_ptr<scalar_t>(),
          grad_wrgb5.data_ptr<scalar_t>(),
          grad_scale.data_ptr<scalar_t>(),
          grad_scale_rgb.data_ptr<scalar_t>()
          );
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in backward_render: %s\n", cudaGetErrorString(err));

    return {grad_faces, grad_textures, grad_w1, grad_w2, grad_w3, grad_w4, grad_w5, grad_wrgb1, grad_wrgb2, grad_wrgb3, grad_wrgb4, grad_wrgb5, grad_scale, grad_scale_rgb};
}

float mlp_forward_float(
    float* signed_x,
    float* w1,
    float* w2,
    float* w3,
    float* w4,
    float* w5
    ){
    return mlp_forward_cuda<float>(
        signed_x,
        w1,
        w2,
        w3,
        w4,
        w5
        );
}

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
    ){
    float grad_x = mlp_backward_cuda<float>(
        signed_x,
        w1,
        w2,
        w3,
        w4,
        w5,
        grad_w1,
        grad_w2,
        grad_w3,
        grad_w4,
        grad_w5
        );
    return {grad_x,grad_w1,grad_w2,grad_w3,grad_w4,grad_w5};
}

float sigmoid_forward_float(
        int function_id,
        float sign,
        float x,
        float scale,
        float dist_shape,
        float dist_shift
    ) {
    return sigmoid_forward_cuda<float>(function_id, sign, x, scale, dist_shape, dist_shift);
}

std::tuple<float,float*> sigmoid_backward_float(
        int function_id,
        float sign,
        float x,
        float scale,
        float dist_shape,
        float dist_shift,
        float* grad_scale
    ) {
    float grad_x = sigmoid_backward_cuda<float>(function_id, sign, x, scale, dist_shape, dist_shift, grad_scale);
    return {grad_x, grad_scale};
}

float t_conorm_forward_float(
        int t_conorm_id,
        float a_existing,
        float b_new,
        int face_id,
        float t_conorm_p
    ) {
    return t_conorm_forward_cuda<float>(t_conorm_id, a_existing, b_new, face_id, t_conorm_p);
}

float t_conorm_backward_float(
        int t_conorm_id,
        float a_all,
        float b_current,
        int number_of_faces,
        float t_conorm_p
    ) {
    return t_conorm_backward_cuda<float>(t_conorm_id, a_all, b_current, number_of_faces, t_conorm_p);
}

