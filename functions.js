// sigmoid-functions.js
// 辅助函数
function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t) * Math.exp(-x * x);
  return sign * y;
}

function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const res = t * Math.exp(
    -z * z - 1.26551223 +
    t * (1.00002368 +
    t * (0.37409196 +
    t * (0.09678418 +
    t * (-0.18628806 +
    t * (0.27886807 +
    t * (-1.13520398 +
    t * (1.48851587 +
    t * (-0.82215223 +
    t * 0.17087277)))))))))
  return x >= 0 ? res : 2 - res;
}
// 所有函数
export const sigmoidFunctions = {
  heaviside: (x, scale, sign) => (sign > 0 ? 1.0 : 0.0),

  logistic: (x, scale, sign) =>
    1.0 / (1.0 + Math.exp(-sign * x / scale)),

  cauchy: (x, scale, sign) =>
    Math.atan(sign * x / scale) / Math.PI + 0.5,

  reciprocal: (x, scale, sign) =>
    sign * x / scale / (1 + x / scale) / 2.0 + 0.5,

  laplace: (x, scale, sign) => {
    if (sign < 0) return 0.5 * Math.exp(-x / scale);
    else return 1.0 - 0.5 * Math.exp(-x / scale);
  },

  uniform: (x, scale, sign) => {
    const val = sign * x / scale;
    if (val < -1) return 0.0;
    else if (val < 1) return (sign * x) * 0.5 / scale + 0.5;
    else return 1.0;
  },

  gudermannian: (x, scale, sign) =>
    Math.atan(Math.tanh(sign * x / scale / 2.0)) * 2.0 / Math.PI + 0.5,

  cubic_hermite: (x, scale, sign) => {
    const val = sign * x / scale;
    if (val < -1) return 0.0;
    else if (val < 1) {
      const y = (sign * x) * 0.5 / scale + 0.5;
      return 3 * y * y - 2 * y * y * y;
    } else return 1.0;
  },

  gaussian: (x, scale, sign) => {
    const z = sign * x / scale;
    return 0.5 * (1 + erf(z / Math.sqrt(2)));
  },

  wigner_semicircle: (x, scale, sign) => {
    const val = sign * x / scale;
    if (val < -1) return 0.0;
    else if (val < 1) {
      return 0.5 +
        (sign * x * Math.sqrt(scale * scale - x * x)) / (Math.PI * scale * scale) +
        Math.asin(val) / Math.PI;
    } else return 1.0;
  },

  gumbel_max: (x, scale, sign) =>
    Math.exp(-Math.exp(-sign * x / scale)),

  gumbel_min: (x, scale, sign) =>
    1.0 - Math.exp(-Math.exp(sign * x / scale)),

  exponential: (x, scale, sign) => {
    const val = sign * x / scale;
    if (val < 0) return 0.0;
    else return 1.0 - Math.exp(-val);
  },

  gamma: (x, scale, sign) => {
    const val = sign * x / scale;
    if (val < 0) return 0.0;
    else return 1.0 - Math.exp(-Math.pow(val, 2.0));
  },

  levy: (x, scale, sign) => {
    const xs = sign * x + 0.0 * scale;  // dist_shift 默认 0
    if (xs <= 1e-6) return 0.0;
    return erfc(Math.sqrt(scale / (2 * xs)));
  },

  neg_levy: (x, scale, sign) => {
    const xs = -(sign * x - 0.0 * scale);  // dist_shift 默认 0
    if (sign * x - 0.0 * scale >= -1e-6) return 1.0;
    return 1.0 - erfc(Math.sqrt(scale / (2 * xs)));
  }
};
