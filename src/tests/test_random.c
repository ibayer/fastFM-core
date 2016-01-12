#include "fast_fm.h"
#include <glib.h>

void test_rng_seed(void) {
  ffm_rng *rng1 = ffm_rng_seed(123);
  ffm_rng *rng2 = ffm_rng_seed(123);
  ffm_rng *rng3 = ffm_rng_seed(123 + 10);

  g_assert_cmpfloat(ffm_rand_normal(rng1, 0, 1), ==,
                    ffm_rand_normal(rng2, 0, 1));
  g_assert_cmpfloat(ffm_rand_normal(rng1, 0, 1), !=,
                    ffm_rand_normal(rng3, 0, 1));
}

void test_uniform_mean(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1200);

  double normal_mean = 0.5;
  double normal_sigma = sqrt((1. / 12));
  int n = x->size;

  for (int i = 0; i < x->size; i++) ffm_vector_set(x, i, ffm_rand_uniform(kr));
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_uniform_var(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  int n = x->size;
  ffm_rng *kr;
  kr = ffm_rng_seed(1200);

  double uniform_sigma = sqrt((1. / 12));
  double uniform_var = (1. / 12);

  double normal_mean = uniform_var;
  double tmp = uniform_sigma;
  double normal_sigma = sqrt(2 * (tmp * tmp * tmp * tmp) / (n - 1));

  for (int i = 0; i < n; i++) ffm_vector_set(x, i, ffm_rand_uniform(kr));
  double var = ffm_vector_variance(x);
  int n_var = 1;

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n_var), var,
          normal_mean + (3. * normal_sigma) / sqrt(n_var), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n_var), <, var);
  g_assert_cmpfloat(var, <, normal_mean + (3. * normal_sigma) / sqrt(n_var));
}

void test_normal_mean(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1200);

  double normal_mean = 20.;
  double normal_sigma = 4.;
  int n = x->size;

  for (int i = 0; i < x->size; i++)
    ffm_vector_set(x, i, ffm_rand_normal(kr, normal_mean, normal_sigma));
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_normal_var(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  int n = x->size;
  ffm_rng *kr;
  kr = ffm_rng_seed(123);

  double org_normal_sigma = sqrt(4.);
  double org_normal_var = org_normal_sigma * org_normal_sigma;

  double normal_mean = org_normal_var;
  double tmp = org_normal_sigma;
  double normal_sigma = sqrt(2. * (tmp * tmp * tmp * tmp) / (n - 1.0));

  for (int i = 0; i < n; i++)
    ffm_vector_set(x, i, ffm_rand_normal(kr, 10, org_normal_sigma));
  double var = ffm_vector_variance(x);
  int n_var = 1;

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n_var), var,
          normal_mean + (3. * normal_sigma) / sqrt(n_var), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n_var), <, var);
  g_assert_cmpfloat(var, <, normal_mean + (3. * normal_sigma) / sqrt(n_var));
}

void test_gamma_mean(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1200);

  double shape = 1;
  double scale = 5;
  double normal_mean = scale * shape;
  double normal_sigma = sqrt(shape * scale * scale);
  int n = x->size;

  for (int i = 0; i < x->size; i++)
    ffm_vector_set(x, i, ffm_rand_gamma(kr, shape, scale));
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_gamma_mean_small_scale(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1200);

  double shape = 1;
  double scale = .5;
  double normal_mean = scale * shape;
  double normal_sigma = sqrt(shape * scale * scale);
  int n = x->size;

  for (int i = 0; i < x->size; i++)
    ffm_vector_set(x, i, ffm_rand_gamma(kr, shape, scale));
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_gamma_var(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  int n = x->size;
  ffm_rng *kr;
  kr = ffm_rng_seed(123);

  double shape = 1;
  double scale = 5;
  double org_normal_sigma = sqrt(shape * scale * scale);
  double org_normal_var = shape * scale * scale;

  double normal_mean = org_normal_var;
  double tmp = org_normal_sigma;
  double normal_sigma = sqrt(2. * (tmp * tmp * tmp * tmp) / (n - 1.0));

  for (int i = 0; i < n; i++)
    ffm_vector_set(x, i, ffm_rand_gamma(kr, shape, scale));
  double var = ffm_vector_variance(x);
  int n_var = 1;

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n_var), var,
          normal_mean + (3. * normal_sigma) / sqrt(n_var), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n_var), <, var);
  g_assert_cmpfloat(var, <, normal_mean + (3. * normal_sigma) / sqrt(n_var));
}

void test_exp_mean(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1200);

  double scale = 5;
  double normal_mean = 1.0 / scale;
  double normal_sigma = sqrt(1.0 / (scale * scale));
  int n = x->size;

  for (int i = 0; i < x->size; i++)
    ffm_vector_set(x, i, ffm_rand_exp(kr, scale));
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_exp_var(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  int n = x->size;
  ffm_rng *kr;
  kr = ffm_rng_seed(1234);

  double scale = 5;
  double org_normal_sigma = sqrt(1.0 / (scale * scale));
  double org_normal_var = 1.0 / (scale * scale);

  double normal_mean = org_normal_var;
  double tmp = org_normal_sigma;
  double normal_sigma = sqrt(2. * (tmp * tmp * tmp * tmp) / (n - 1.0));

  for (int i = 0; i < n; i++) ffm_vector_set(x, i, ffm_rand_exp(kr, scale));
  double var = ffm_vector_variance(x);
  int n_var = 1;

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n_var), var,
          normal_mean + (3. * normal_sigma) / sqrt(n_var), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n_var), <, var);
  g_assert_cmpfloat(var, <, normal_mean + (3. * normal_sigma) / sqrt(n_var));
}

void test_left_trunc_normal_mean(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1202);

  double trunc = 5;
  double trunc_mean = 0;
  double trunc_sigma = 3;

  // formulas from http://en.wikipedia.org/wiki/Truncated_normal_distribution
  double alpha = (trunc - trunc_mean) / trunc_sigma;
  double lambda_alpha = ffm_normal_pdf(alpha) / (1.0 - ffm_normal_cdf(alpha));
  double delta = lambda_alpha * (lambda_alpha - alpha);

  double normal_mean = trunc_mean + trunc_sigma * lambda_alpha;
  double normal_sigma = sqrt(trunc_sigma * trunc_sigma * (1.0 - delta));
  int n = x->size;

  // for transformation to Nonstandart Normal Population (Seq. 3.1 & 3.2)
  // Barr & Sherrill: Mean and Variance of Truncated Normal Distributions
  trunc = (trunc + trunc_mean) / trunc_sigma;
  for (int i = 0; i < x->size; i++)
    ffm_vector_set(
        x, i, trunc_mean + ffm_rand_left_trunc_normal(kr, trunc) * trunc_sigma);
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_left_trunc_normal_mean_neg_trunc(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  ffm_rng *kr;
  kr = ffm_rng_seed(1202);

  double trunc = -2;
  double trunc_mean = 0;
  double trunc_sigma = 1;

  // formulas from http://en.wikipedia.org/wiki/Truncated_normal_distribution
  double alpha = (trunc - trunc_mean) / trunc_sigma;
  double lambda_alpha = ffm_normal_pdf(alpha) / (1.0 - ffm_normal_cdf(alpha));
  double delta = lambda_alpha * (lambda_alpha - alpha);

  double normal_mean = trunc_mean + trunc_sigma * lambda_alpha;
  double normal_sigma = sqrt(trunc_sigma * trunc_sigma * (1.0 - delta));
  int n = x->size;

  // for transformation to Nonstandart Normal Population (Seq. 3.1 & 3.2)
  // Barr & Sherrill: Mean and Variance of Truncated Normal Distributions
  trunc = (trunc + trunc_mean) / trunc_sigma;
  for (int i = 0; i < x->size; i++)
    ffm_vector_set(
        x, i, trunc_mean + ffm_rand_left_trunc_normal(kr, trunc) * trunc_sigma);
  double mean = ffm_vector_mean(x);

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n), mean,
          normal_mean + (3. * normal_sigma) / sqrt(n), n); */
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n), <, mean);
  g_assert_cmpfloat(mean, <, normal_mean + (3. * normal_sigma) / sqrt(n));
}

void test_left_trunc_normal_var(void) {
  ffm_vector *x = ffm_vector_alloc(100000);
  int n = x->size;
  ffm_rng *kr;
  kr = ffm_rng_seed(123);

  double trunc = 5;
  double trunc_mean = 0;
  double trunc_sigma = 3;

  double alpha = (trunc - trunc_mean) / trunc_sigma;
  double lambda_alpha = ffm_normal_pdf(alpha) / (1.0 - ffm_normal_cdf(alpha));
  double delta = lambda_alpha * (lambda_alpha - alpha);

  double org_normal_sigma = sqrt(trunc_sigma * trunc_sigma * (1.0 - delta));
  double org_normal_var = org_normal_sigma * org_normal_sigma;

  double normal_mean = org_normal_var;
  double tmp = org_normal_sigma;
  double normal_sigma = sqrt(2. * (tmp * tmp * tmp * tmp) / (n - 1.0));

  // for transformation to Nonstandart Normal Population (Seq. 3.1 & 3.2)
  // Barr & Sherrill: Mean and Variance of Truncated Normal Distributions
  trunc = (trunc + trunc_mean) / trunc_sigma;
  for (int i = 0; i < x->size; i++)
    ffm_vector_set(
        x, i, trunc_mean + ffm_rand_left_trunc_normal(kr, trunc) * trunc_sigma);
  double var = ffm_vector_variance(x);
  int n_var = 1;

  /*printf("test %f < %f < %f (n=%d)\n",
          normal_mean - (3. * normal_sigma) / sqrt(n_var), var,
          normal_mean + (3. * normal_sigma) / sqrt(n_var), n);*/
  g_assert_cmpfloat(normal_mean - (3. * normal_sigma) / sqrt(n_var), <, var);
  g_assert_cmpfloat(var, <, normal_mean + (3. * normal_sigma) / sqrt(n_var));
}

int main(int argc, char **argv) {
  g_test_init(&argc, &argv, NULL);
  g_test_add_func("/random/rng/ seed", test_rng_seed);
  g_test_add_func("/random/uniform/ mean-test", test_uniform_mean);
  g_test_add_func("/random/uniform/ var-test", test_uniform_var);

  g_test_add_func("/random/normal/ mean-test", test_normal_mean);
  g_test_add_func("/random/normal/ var-test", test_normal_var);

  g_test_add_func("/random/gamma/ mean-test", test_gamma_mean);
  g_test_add_func("/random/gamma/ mean-test (scale < 1)",
                  test_gamma_mean_small_scale);
  g_test_add_func("/random/gamma/ var-test", test_gamma_var);

  g_test_add_func("/random/exp/ mean-test", test_exp_mean);
  g_test_add_func("/random/exp/ var-test", test_exp_var);

  g_test_add_func("/random/left trunc normal/ mean-test",
                  test_left_trunc_normal_mean);
  g_test_add_func("/random/left trunc normal/ mean-test (neg trunc)",
                  test_left_trunc_normal_mean_neg_trunc);
  g_test_add_func("/random/left trunc normal/ var-test",
                  test_left_trunc_normal_var);

  return g_test_run();
}
