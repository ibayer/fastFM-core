#ifndef TESTFIXTURES_H
#define TESTFIXTURES_H
#include "fast_fm.h"
#include "TestFixtures.h"

void TestFixtureContructorSimple(TestFixture_T *pFixture, gconstpointer pg) {
  int n_features = 2;
  int n_samples = 3;
  int k = 1;
  // setup data
  int m = n_samples;
  // X = | 1 2|
  //     | 3 4|
  //     | 5 6|
  int n = n_features;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 1);
  cs_entry(X, 0, 1, 2);
  cs_entry(X, 1, 0, 3);
  cs_entry(X, 1, 1, 4);
  cs_entry(X, 2, 0, 5);
  cs_entry(X, 2, 1, 6);
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs *X_t = cs_transpose(X_csc, 1);
  cs_spfree(X);

  pFixture->X_t = X_t;
  pFixture->X = X_csc;
  pFixture->y = ffm_vector_calloc(n_samples);
  // y = [600, 2800, 10000]
  pFixture->y->data[0] = 600;
  pFixture->y->data[1] = 2800;
  pFixture->y->data[2] = 10000;

  // setup coefs
  pFixture->coef = alloc_fm_coef(n_features, k, false);

  // V = |300, 400|
  ffm_matrix_set(pFixture->coef->V, 0, 0, 300);
  ffm_matrix_set(pFixture->coef->V, 0, 1, 400);

  // w = [10 20]
  ffm_vector_set(pFixture->coef->w, 0, 10);
  ffm_vector_set(pFixture->coef->w, 1, 20);

  pFixture->coef->w_0 = 2;
  // hyperparameter
  pFixture->coef->lambda_w = 0;
  pFixture->coef->lambda_V = 0;
}

void TestFixtureContructorWide(TestFixture_T *pFixture, gconstpointer pg) {
  int n_features = 3;
  int n_samples = 2;
  int k = 2;
  // setup data
  int m = n_samples;
  // X = | 1 2 3|
  //     | 4 0 6|
  int n = n_features;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 1);
  cs_entry(X, 0, 1, 2);
  cs_entry(X, 0, 2, 3);
  cs_entry(X, 1, 0, 4);
  cs_entry(X, 1, 2, 6);
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs *X_t = cs_transpose(X_csc, 1);
  cs_spfree(X);

  pFixture->X_t = X_t;
  pFixture->X = X_csc;
  pFixture->y = ffm_vector_calloc(n_samples);
  // y = [1 20]
  pFixture->y->data[0] = 1;
  pFixture->y->data[1] = 20;

  // setup coefs
  pFixture->coef = alloc_fm_coef(n_features, k, false);

  // V = |1 2 3|
  //     |3 4 5|
  ffm_matrix_set(pFixture->coef->V, 0, 0, 1);
  ffm_matrix_set(pFixture->coef->V, 0, 1, 2);
  ffm_matrix_set(pFixture->coef->V, 0, 2, 3);
  ffm_matrix_set(pFixture->coef->V, 1, 0, 4);
  ffm_matrix_set(pFixture->coef->V, 1, 1, 5);
  ffm_matrix_set(pFixture->coef->V, 1, 2, 6);

  // w = [1 2 3]
  ffm_vector_set(pFixture->coef->w, 0, 1);
  ffm_vector_set(pFixture->coef->w, 1, 2);
  ffm_vector_set(pFixture->coef->w, 2, 3);

  pFixture->coef->w_0 = 2;
  // hyperparameter
  pFixture->coef->lambda_w = 1;
  ffm_vector_set_all(pFixture->coef->lambda_V, 1);
}

void TestFixtureContructorLong(TestFixture_T *pFixture, gconstpointer pg) {
  int n_features = 2;
  int n_samples = 5;
  int k = 2;
  // setup data
  int m = n_samples;
  // X = | 6 1 |
  //     | 2 3 |
  //     | 3 0 |
  //     | 6 1 |
  //     | 4 5 |
  int n = n_features;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 6);
  cs_entry(X, 0, 1, 1);
  cs_entry(X, 1, 0, 2);
  cs_entry(X, 1, 1, 3);
  cs_entry(X, 2, 0, 3);
  cs_entry(X, 3, 0, 6);
  cs_entry(X, 3, 1, 1);
  cs_entry(X, 4, 0, 4);
  cs_entry(X, 4, 1, 5);
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs *X_t = cs_transpose(X_csc, 1);
  cs_spfree(X);

  pFixture->X_t = X_t;
  pFixture->X = X_csc;
  pFixture->y = ffm_vector_calloc(n_samples);
  // y [ 298 266 29 298 848 ]
  pFixture->y->data[0] = 298;
  pFixture->y->data[1] = 266;
  pFixture->y->data[2] = 29;
  pFixture->y->data[3] = 298;
  pFixture->y->data[4] = 848;

  // setup coefs
  pFixture->coef = alloc_fm_coef(n_features, k, false);

  // V = |6 0|
  //     |5 8|
  ffm_matrix_set(pFixture->coef->V, 0, 0, 6);
  ffm_matrix_set(pFixture->coef->V, 0, 1, 0);
  ffm_matrix_set(pFixture->coef->V, 1, 0, 5);
  ffm_matrix_set(pFixture->coef->V, 1, 1, 8);

  // w = [9 2]
  ffm_vector_set(pFixture->coef->w, 0, 9);
  ffm_vector_set(pFixture->coef->w, 1, 2);

  // w_0 = 2
  pFixture->coef->w_0 = 2;
}

void TestFixtureDestructor(TestFixture_T *pFixture, gconstpointer pg) {
  cs_spfree(pFixture->X_t);
  cs_spfree(pFixture->X);

  ffm_vector_free(pFixture->y);
  free_ffm_coef(pFixture->coef);
}

cs *Cs_rand_spalloc(ffm_rng *rng, int n_samples, int n_features) {
  cs *X_trip = cs_spalloc(n_samples, n_features, n_samples * n_features, 1, 1);

  int i, j;
  for (i = 0; i < n_samples; i++)
    for (j = 0; j < n_features; j++)
      cs_entry(X_trip, i, j, ffm_rand_uniform(rng) * 2.0 - 1);

  cs_dropzeros(X_trip);
  cs *X_csc = cs_compress(X_trip);
  cs_spfree(X_trip);
  return X_csc;
}

TestFixture_T *makeTestFixture(int seed, int n_samples, int n_features, int k) {
  ffm_rng *rng = ffm_rng_seed(seed);

  // allocate coef
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  // draw hyperparameter
  coef->alpha = ffm_rand_gamma(rng, .5 * (1 + n_samples), 1.0);
  coef->lambda_w = ffm_rand_gamma(rng, .5 * (2 + n_features), 1.0 / 100.);
  double lambda_V_all = ffm_rand_gamma(rng, .5 * (2 + n_features), 1.0 / 100.);
  if (k > 0) ffm_vector_set_all(coef->lambda_V, lambda_V_all);
  double sigma2_w = 1.0 / (1 + n_samples);
  coef->mu_w = ffm_rand_normal(rng, sigma2_w * coef->lambda_w, sigma2_w);
  double sigma2_V = 1.0 / (1 + n_features);
  double mu_V_all = ffm_rand_normal(rng, sigma2_V * lambda_V_all, sigma2_V);
  if (k > 0) ffm_vector_set_all(coef->mu_V, mu_V_all);

  // generate w_0
  coef->w_0 = ffm_rand_normal(rng, 0, 1);

  // init w
  for (int i = 0; i < n_features; i++)
    ffm_vector_set(
        coef->w, i,
        coef->mu_w + ffm_rand_normal(rng, coef->mu_w, 1.0 / coef->lambda_w));

  // init V
  if (k > 0) {
    for (int i = 0; i < coef->V->size0; i++)
      for (int j = 0; j < coef->V->size1; j++) {
        double tmp = ffm_rand_normal(rng, mu_V_all, 1.0 / lambda_V_all);
        ffm_matrix_set(coef->V, i, j, tmp);
      }
  }

  // generate uniform X
  cs *X = Cs_rand_spalloc(rng, n_samples, n_features);
  cs *X_t = cs_transpose(X, 1);

  // generate y using second-order FM
  ffm_vector *y = ffm_vector_calloc(n_samples);
  sparse_predict(coef, X, y);

  // put everything into a TestFixture
  struct TestFixture_T *pFix = malloc(sizeof *pFix);
  pFix->X = X;
  pFix->X_t = X_t;
  pFix->y = y;
  pFix->coef = coef;
  ffm_rng_free(rng);
  return pFix;
}
#endif /* TESTFIXTURES_H */
