// Author: Immanuel Bayer
// License: BSD 3 clause

#include "fast_fm.h"

ffm_rng *ffm_rng_seed(int seed) { return kr_srand(seed); }

double ffm_rand_uniform(ffm_rng *kr) { return kr_rand(kr) / (double)ULONG_MAX; }

void ffm_rng_free(ffm_rng *rng) { free(rng); }

// Box-Muller transform
double ffm_rand_normal(ffm_rng *kr, double mean, double stddev) {
  double x, y, r;
  do {
    x = 2.0 * ffm_rand_uniform(kr) - 1;
    y = 2.0 * ffm_rand_uniform(kr) - 1;
    r = x * x + y * y;
  } while (r == 0.0 || r > 1.0);
  double d = sqrt(-2.0 * log(r) / r);
  return x * d * stddev + mean;
}

double ffm_rand_exp(ffm_rng *kr, double rate) {
  return -log(ffm_rand_uniform(kr)) / rate;
}

// A Simple Method for Generation Gamma Variables (Section 5)
double ffm_rand_gamma(ffm_rng *kr, double shape, double scale) {
  assert(scale > 0);

  if (shape < 1.) {
    double r = ffm_rand_uniform(kr);
    return ffm_rand_gamma(kr, 1.0 + shape, scale) * pow(r, 1.0 / shape);
  }

  double d, c, x, v, u;
  d = shape - 1. / 3.;
  c = 1. / sqrt(9. * d);
  while (true) {
    do {
      x = ffm_rand_normal(kr, 0, 1);
      v = 1. + c * x;
    } while (v <= 0.);
    v = v * v * v;
    u = ffm_rand_uniform(kr);
    if (u < 1. - .0331 * (x * x) * (x * x)) return scale * d * v;
    if (log(u) < 0.5 * x * x + d * (1. - v + log(v))) return scale * d * v;
  }
}

// normal truncated left
// Robert: Simulation of truncated normal variables
double ffm_rand_left_trunc_normal(ffm_rng *kr, double trunc) {
  /*
  if (trunc <= 0.)
      while (true)
      {
          double r = ffm_rand_normal(kr, trunc, 1);
          if (r > trunc) return r;
      }
      */

  double alpha_star = 0.5 * (trunc + sqrt(trunc * trunc + 4.0));
  while (true) {
    double z = trunc + ffm_rand_exp(kr, alpha_star);
    double tmp = z - alpha_star;
    double g = exp(-(tmp * tmp) / 2.0);
    double u = ffm_rand_uniform(kr);
    if (u <= g) return z;
  }
}

double ffm_rand_right_trunc_normal(ffm_rng *kr, double trunc) {
  return ffm_rand_left_trunc_normal(kr, trunc);
}
