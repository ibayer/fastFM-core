// Author: Immanuel Bayer
// License: BSD 3 clause

#ifndef FFM_RANDOM_H
#define FFM_RANDOM_H

#include "fast_fm.h"

typedef krand_t ffm_rng;

ffm_rng *ffm_rng_seed(int seed);
void ffm_rng_free(ffm_rng *rng);
double ffm_rand_normal(ffm_rng *kr, double mean, double stddev);
double ffm_rand_uniform(ffm_rng *kr);
double ffm_rand_gamma(ffm_rng *kr, double shape, double scale);
double ffm_rand_exp(ffm_rng *kr, double rate);  // lambda (rate) fixed at 1
double ffm_rand_left_trunc_normal(ffm_rng *kr, double mean);
double ffm_rand_left_trunc_normal(ffm_rng *kr, double trunc);
double ffm_rand_right_trunc_normal(ffm_rng *kr, double trunc);

#endif /* FFM_RANDOM_H */
