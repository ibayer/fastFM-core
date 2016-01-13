#include "fast_fm.h"
#include <fenv.h>
#include "TestFixtures.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

void test_eval_second_order_term(TestFixture_T *pFix, gconstpointer pg) {
  ffm_vector *y_pred = ffm_vector_calloc(5);
  eval_second_order_term(pFix->coef->V, pFix->X_t, y_pred);

  g_assert_cmpfloat(240.0, ==, ffm_vector_get(y_pred, 0));
  g_assert_cmpfloat(240.0, ==, ffm_vector_get(y_pred, 1));
  g_assert_cmpfloat(.0, ==, ffm_vector_get(y_pred, 2));
  g_assert_cmpfloat(240.0, ==, ffm_vector_get(y_pred, 3));
  g_assert_cmpfloat(800.0, ==, ffm_vector_get(y_pred, 4));

  ffm_vector_free(y_pred);
}

void test_update_second_order_error(TestFixture_T *pFix, gconstpointer pg) {
  ffm_vector *a_theta_v = ffm_vector_calloc(5);
  ffm_vector_set(a_theta_v, 2, 1);
  ffm_vector_set(a_theta_v, 3, 2);

  ffm_vector *error = ffm_vector_calloc(5);
  ffm_vector_set_all(error, 1.5);

  double delta = 0.5;
  int column = 1;
  update_second_order_error(column, pFix->X, a_theta_v, delta, error);

  g_assert_cmpfloat(1.5, ==, ffm_vector_get(error, 0));
  g_assert_cmpfloat(1.5, ==, ffm_vector_get(error, 1));
  g_assert_cmpfloat(1.5, ==, ffm_vector_get(error, 2));
  g_assert_cmpfloat(2.5, ==, ffm_vector_get(error, 3));
  g_assert_cmpfloat(1.5, ==, ffm_vector_get(error, 4));

  ffm_vector_free_all(error, a_theta_v);
}

void test_sparse_update_v_ij(TestFixture_T *pFix, gconstpointer pg) {
  double old_v_lf = 0.5;
  double l2_reg = 2;
  int n_samples = pFix->X->m;

  ffm_vector *err = ffm_vector_calloc(n_samples);
  int i;
  int column_index = 0;
  for (i = 0; i < n_samples; i++) ffm_vector_set(err, i, i);
  ffm_vector_scale(err, 0.1);

  ffm_vector *cache = ffm_vector_calloc(n_samples);
  ffm_vector *a_theta = ffm_vector_calloc(n_samples);
  ffm_vector_memcpy(cache, err);
  ffm_vector_scale(cache, 20);
  ffm_vector_scale(err,
                   -1);  // account for sign change in error due to refactoring
  // nominator 658.82
  // denominator 1286.2500
  // 0.51220602526724979
  double sum_denominator, sum_nominator;
  sum_nominator = sum_denominator = 0;
  sparse_v_lf_frac(&sum_denominator, &sum_nominator, pFix->X, column_index, err,
                   cache, a_theta, old_v_lf);
  double new_v_lf = sum_nominator / (sum_denominator + l2_reg);

  g_assert_cmpfloat(0.51220602526724979, ==, new_v_lf);
  ffm_vector_free_all(err, cache, a_theta);
}

void test_sparse_predict(TestFixture_T *pFix, gconstpointer pg) {
  ffm_vector *y_pred = ffm_vector_calloc(5);

  sparse_predict(pFix->coef, pFix->X, y_pred);

  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y_pred, 0));
  g_assert_cmpfloat(266.0, ==, ffm_vector_get(y_pred, 1));
  g_assert_cmpfloat(29.0, ==, ffm_vector_get(y_pred, 2));
  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y_pred, 3));
  g_assert_cmpfloat(848.0, ==, ffm_vector_get(y_pred, 4));

  ffm_vector_free(y_pred);
}

void test_row_predict(TestFixture_T *pFix, gconstpointer pg) {
  ffm_vector *y_pred = ffm_vector_calloc(5);

  row_predict(pFix->coef, pFix->X_t, y_pred);

  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y_pred, 0));
  g_assert_cmpfloat(266.0, ==, ffm_vector_get(y_pred, 1));
  g_assert_cmpfloat(29.0, ==, ffm_vector_get(y_pred, 2));
  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y_pred, 3));
  g_assert_cmpfloat(848.0, ==, ffm_vector_get(y_pred, 4));

  ffm_vector_free(y_pred);
}

void test_col_predict(TestFixture_T *pFix, gconstpointer pg) {
  ffm_vector *y_pred = ffm_vector_calloc(5);

  col_predict(pFix->coef, pFix->X, y_pred);

  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y_pred, 0));
  g_assert_cmpfloat(266.0, ==, ffm_vector_get(y_pred, 1));
  g_assert_cmpfloat(29.0, ==, ffm_vector_get(y_pred, 2));
  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y_pred, 3));
  g_assert_cmpfloat(848.0, ==, ffm_vector_get(y_pred, 4));

  ffm_vector_free(y_pred);
}

void test_sparse_als_zero_order_only(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 0;
  ffm_param param = {.n_iter = 1,
                     .warm_start = true,
                     .ignore_w = true,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION};

  ffm_coef *coef = alloc_fm_coef(n_features, k, true);
  param.init_lambda_w = 0;

  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  // g_assert_cmpfloat(4466.666666, ==, coef->w_0);
  g_assert_cmpfloat(fabs(4466.666666 - coef->w_0), <, 1e-6);

  free_ffm_coef(coef);
}

void test_sparse_als_first_order_only(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 0;
  ffm_param param = {.n_iter = 1,
                     .warm_start = true,
                     .ignore_w_0 = true,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION};

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  coef->w_0 = 0;
  param.init_lambda_w = 0;

  ffm_vector_set(coef->w, 0, 10);
  ffm_vector_set(coef->w, 1, 20);

  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  // hand calculated results 1660.57142857   -11.87755102
  g_assert_cmpfloat(fabs(1660.57142857 - ffm_vector_get(coef->w, 0)), <, 1e-8);
  g_assert_cmpfloat(fabs(-11.87755102 - ffm_vector_get(coef->w, 1)), <, 1e-8);

  free_ffm_coef(coef);
}

void test_sparse_als_second_order_only(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 1;
  ffm_param param = {.n_iter = 1,
                     .warm_start = true,
                     .ignore_w_0 = true,
                     .ignore_w = true,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION};

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  coef->w_0 = 0;

  param.init_lambda_w = 0;
  param.init_lambda_V = 0;

  ffm_matrix_set(coef->V, 0, 0, 300);
  ffm_matrix_set(coef->V, 0, 1, 400);

  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  // hand calculated results  0.79866412  400.
  g_assert_cmpfloat(fabs(0.79866412 - ffm_matrix_get(coef->V, 0, 0)), <, 1e-8);
  g_assert_cmpfloat(fabs(400 - ffm_matrix_get(coef->V, 0, 1)), <, 1e-8);

  free_ffm_coef(coef);
}

void test_sparse_als_all_interactions(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 1;
  ffm_param param = {.n_iter = 1,
                     .warm_start = true,
                     .ignore_w_0 = false,
                     .ignore_w = false,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION};

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  coef->w_0 = 0;

  ffm_vector_set(coef->w, 0, 10);
  ffm_vector_set(coef->w, 1, 20);

  ffm_matrix_set(coef->V, 0, 0, 300);
  ffm_matrix_set(coef->V, 0, 1, 400);

  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  // hand calculated results checked with libfm
  g_assert_cmpfloat(fabs(-1755643.33333 - coef->w_0), <, 1e-5);
  g_assert_cmpfloat(fabs(-191459.71428571 - ffm_vector_get(coef->w, 0)), <,
                    1e-6);
  g_assert_cmpfloat(fabs(30791.91836735 - ffm_vector_get(coef->w, 1)), <, 1e-6);
  g_assert_cmpfloat(fabs(253.89744249 - ffm_matrix_get(coef->V, 0, 0)), <,
                    1e-6);
  g_assert_cmpfloat(fabs(400 - ffm_matrix_get(coef->V, 0, 1)), <, 1e-6);

  param.n_iter = 99;
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);

  g_assert_cmpfloat(fabs(210911.940403 - coef->w_0), <, 1e-7);
  g_assert_cmpfloat(fabs(-322970.68313639 - ffm_vector_get(coef->w, 0)), <,
                    1e-6);
  g_assert_cmpfloat(fabs(51927.60978978 - ffm_vector_get(coef->w, 1)), <, 1e-6);
  g_assert_cmpfloat(fabs(94.76612018 - ffm_matrix_get(coef->V, 0, 0)), <, 1e-6);
  g_assert_cmpfloat(fabs(400 - ffm_matrix_get(coef->V, 0, 1)), <, 1e-6);

  free_ffm_coef(coef);
}

void test_sparse_als_first_order_interactions(TestFixture_T *pFix,
                                              gconstpointer pg) {
  ffm_vector *y_pred = ffm_vector_calloc(5);

  int n_features = pFix->X->n;
  int k = 0;
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 500,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION};
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, pFix->X, y_pred);

  /* reference values from sklearn LinearRegression
  y_pred:  [ 321.05084746  346.6779661   -40.15254237  321.05084746
  790.37288136]
  coef: [  69.6779661   152.16949153]
  mse: 3134.91525424 */
  g_assert_cmpfloat(fabs(321.05084746 - ffm_vector_get(y_pred, 0)), <, 1e-6);
  g_assert_cmpfloat(fabs(346.6779661 - ffm_vector_get(y_pred, 1)), <, 1e-6);
  g_assert_cmpfloat(fabs(-40.15254237 - ffm_vector_get(y_pred, 2)), <, 1e-6);
  g_assert_cmpfloat(fabs(321.05084746 - ffm_vector_get(y_pred, 3)), <, 1e-6);
  g_assert_cmpfloat(fabs(790.37288136 - ffm_vector_get(y_pred, 4)), <, 1e-6);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
}

void test_sparse_als_second_interactions(TestFixture_T *pFix,
                                         gconstpointer pg) {
  ffm_vector *y_pred = ffm_vector_calloc(5);

  int n_features = pFix->X->n;
  int k = 2;
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 1000, .init_sigma = 0.1, .SOLVER = SOLVER_ALS};
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, pFix->X, y_pred);

  /* reference values from sklearn LinearRegression
  y_pred: [ 298.  266.   29.  298.  848.]
  coeff: [  9.   2.  40.]
  mse: 4.53374139449e-27 */
  g_assert_cmpfloat(fabs(298 - ffm_vector_get(y_pred, 0)), <, 1e-4);
  g_assert_cmpfloat(fabs(266 - ffm_vector_get(y_pred, 1)), <, 1e-4);
  g_assert_cmpfloat(fabs(29 - ffm_vector_get(y_pred, 2)), <, 1e-3);
  g_assert_cmpfloat(fabs(298 - ffm_vector_get(y_pred, 3)), <, 1e-4);
  g_assert_cmpfloat(fabs(848.0 - ffm_vector_get(y_pred, 4)), <, 1e-4);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
}

void test_sparse_mcmc_second_interactions(TestFixture_T *pFix,
                                          gconstpointer pg) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 2;
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_vector *y_pred = ffm_vector_calloc(n_samples);
  ffm_param param = {.n_iter = 100,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_MCMC,
                     .TASK = TASK_REGRESSION,
                     .rng_seed = 1234};
  sparse_fit(coef, pFix->X, pFix->X, pFix->y, y_pred, param);

  g_assert_cmpfloat(ffm_r2_score(pFix->y, y_pred), >, .98);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
}

void test_sparse_mcmc_second_interactions_classification(TestFixture_T *pFix,
                                                         gconstpointer pg) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 2;
  ffm_vector_make_labels(pFix->y);
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_vector *y_pred = ffm_vector_calloc(n_samples);
  ffm_param param = {.n_iter = 10,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_MCMC,
                     .TASK = TASK_CLASSIFICATION};
  sparse_fit(coef, pFix->X, pFix->X, pFix->y, y_pred, param);

  g_assert_cmpfloat(ffm_vector_accuracy(pFix->y, y_pred), >=, .98);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
}

void test_train_test_of_different_size(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 2;

  int n_samples_short = 3;
  int m = n_samples_short;
  int n = n_features;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 6);
  cs_entry(X, 0, 1, 1);
  cs_entry(X, 1, 0, 2);
  cs_entry(X, 1, 1, 3);
  cs_entry(X, 2, 0, 3);
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs *X_t = cs_transpose(X_csc, 1);
  cs_spfree(X);

  ffm_vector *y = ffm_vector_calloc(n_samples_short);
  // y [ 298 266 29 298 848 ]
  y->data[0] = 298;
  y->data[1] = 266;
  y->data[2] = 29;

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_vector *y_pred = ffm_vector_calloc(n_samples_short);

  ffm_param param = {.n_iter = 20, .init_sigma = 0.01};
  // test: train > test

  param.SOLVER = SOLVER_ALS;
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, X_csc, y_pred);
  param.TASK = TASK_CLASSIFICATION;
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, X_csc, y_pred);

  param.SOLVER = SOLVER_MCMC;
  param.TASK = TASK_CLASSIFICATION;
  sparse_fit(coef, pFix->X, X_csc, pFix->y, y_pred, param);
  param.TASK = TASK_REGRESSION;
  sparse_fit(coef, pFix->X, X_csc, pFix->y, y_pred, param);

  // test: train < test
  param.SOLVER = SOLVER_MCMC;
  param.TASK = TASK_CLASSIFICATION;
  sparse_fit(coef, X_csc, pFix->X, y_pred, pFix->y, param);
  param.TASK = TASK_REGRESSION;
  sparse_fit(coef, X_csc, pFix->X, y_pred, pFix->y, param);

  param.SOLVER = SOLVER_ALS;
  sparse_fit(coef, X_csc, NULL, y_pred, NULL, param);
  sparse_predict(coef, pFix->X, pFix->y);
  param.TASK = TASK_CLASSIFICATION;
  sparse_fit(coef, X_csc, NULL, y_pred, NULL, param);
  sparse_predict(coef, pFix->X, pFix->y);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  cs_spfree(X_t);
  cs_spfree(X_csc);
}

void test_sparse_als_generated_data(void) {
  int n_features = 10;
  int n_samples = 100;
  int k = 2;

  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 50, .init_sigma = 0.01, .SOLVER = SOLVER_ALS};
  param.init_lambda_w = 23.5;
  param.init_lambda_V = 23.5;
  sparse_fit(coef, data->X, NULL, data->y, NULL, param);
  sparse_predict(coef, data->X, y_pred);

  g_assert_cmpfloat(ffm_r2_score(data->y, y_pred), >, 0.85);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_hyerparameter_sampling(void) {
  ffm_rng *rng = ffm_rng_seed(12345);

  int n_features = 20;
  int n_samples = 150;
  int k = 1;  // don't just change k, the rank is hard coded in the test
              // (ffm_vector_get(coef->lambda_V, 0);)

  int n_replication = 40;
  int n_draws = 1000;
  ffm_vector *alpha_rep = ffm_vector_calloc(n_replication);
  ffm_vector *lambda_w_rep = ffm_vector_calloc(n_replication);
  ffm_vector *lambda_V_rep = ffm_vector_calloc(n_replication);
  ffm_vector *mu_w_rep = ffm_vector_calloc(n_replication);
  ffm_vector *mu_V_rep = ffm_vector_calloc(n_replication);
  ffm_vector *err = ffm_vector_alloc(n_samples);

  for (int j = 0; j < n_replication; j++) {
    TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);
    ffm_coef *coef = data->coef;

    sparse_predict(coef, data->X, err);
    ffm_vector_scale(err, -1);
    ffm_vector_add(err, data->y);

    // make sure that distribution is converged bevore selecting
    // reference / init values
    for (int l = 0; l < 50; l++) sample_hyper_parameter(coef, err, rng);

    double alpha_init = coef->alpha;
    double lambda_w_init = coef->lambda_w;
    double lambda_V_init = ffm_vector_get(coef->lambda_V, 0);
    double mu_w_init = coef->mu_w;
    double mu_V_init = ffm_vector_get(coef->mu_V, 0);

    double alpha_count = 0;
    double lambda_w_count = 0, lambda_V_count = 0;
    double mu_w_count = 0, mu_V_count = 0;

    for (int l = 0; l < n_draws; l++) {
      sample_hyper_parameter(coef, err, rng);
      if (alpha_init > coef->alpha) alpha_count++;
      if (lambda_w_init > coef->lambda_w) lambda_w_count++;
      if (lambda_V_init > ffm_vector_get(coef->lambda_V, 0)) lambda_V_count++;
      if (mu_w_init > coef->mu_w) mu_w_count++;
      if (mu_V_init > ffm_vector_get(coef->mu_V, 0)) mu_V_count++;
    }
    ffm_vector_set(alpha_rep, j, alpha_count / (n_draws + 1));
    ffm_vector_set(lambda_w_rep, j, lambda_w_count / (n_draws + 1));
    ffm_vector_set(lambda_V_rep, j, lambda_V_count / (n_draws + 1));
    ffm_vector_set(mu_w_rep, j, mu_w_count / (n_draws + 1));
    ffm_vector_set(mu_V_rep, j, mu_V_count / (n_draws + 1));

    TestFixtureDestructor(data, NULL);
  }
  double chi_alpha = 0;
  for (int i = 0; i < n_replication; i++)
    chi_alpha +=
        ffm_pow_2(gsl_cdf_ugaussian_Qinv(ffm_vector_get(alpha_rep, i)));
  g_assert_cmpfloat(gsl_ran_chisq_pdf(chi_alpha, n_replication), <, .05);

  double chi_lambda_w = 0;
  for (int i = 0; i < n_replication; i++)
    chi_lambda_w +=
        ffm_pow_2(gsl_cdf_ugaussian_Qinv(ffm_vector_get(lambda_w_rep, i)));
  g_assert_cmpfloat(gsl_ran_chisq_pdf(chi_lambda_w, n_replication), <, .05);

  double chi_lambda_V = 0;
  for (int i = 0; i < n_replication; i++)
    chi_lambda_V +=
        ffm_pow_2(gsl_cdf_ugaussian_Qinv(ffm_vector_get(lambda_V_rep, i)));
  g_assert_cmpfloat(gsl_ran_chisq_pdf(chi_lambda_V, n_replication), <, .05);

  double chi_mu_w = 0;
  for (int i = 0; i < n_replication; i++)
    chi_mu_w += ffm_pow_2(gsl_cdf_ugaussian_Qinv(ffm_vector_get(mu_w_rep, i)));
  g_assert_cmpfloat(gsl_ran_chisq_pdf(chi_mu_w, n_replication), <, .05);

  double chi_mu_V = 0;
  for (int i = 0; i < n_replication; i++)
    chi_mu_V += ffm_pow_2(gsl_cdf_ugaussian_Qinv(ffm_vector_get(mu_V_rep, i)));
  g_assert_cmpfloat(gsl_ran_chisq_pdf(chi_mu_V, n_replication), <, .05);

  ffm_vector_free_all(alpha_rep, lambda_w_rep, lambda_V_rep, mu_w_rep, mu_V_rep,
                      err);
  ffm_rng_free(rng);
}

void test_sparse_map_gibbs_first_order_interactions(void) {
  int n_features = 10;
  int n_samples = 100;
  int k = 0;

  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 200, .init_sigma = 0.1, .SOLVER = SOLVER_MCMC};
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param);

  g_assert_cmpfloat(ffm_r2_score(data->y, y_pred), >, .99);
  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_train_test_data(void) {
  // test if training and test data a propertly handeled
  // no check of prediction quality
  int n_features = 10;
  int n_samples_train = 100;
  int n_samples_test = 30;
  int k = 3;

  TestFixture_T *data_train =
      makeTestFixture(124, n_samples_train, n_features, k);
  TestFixture_T *data_test =
      makeTestFixture(124, n_samples_test, n_features, k);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples_test);
  // gibts
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 200, .init_sigma = 0.1, .SOLVER = SOLVER_MCMC};
  sparse_fit(coef, data_train->X, data_test->X, data_train->y, y_pred, param);
  free_ffm_coef(coef);
  // als
  coef = alloc_fm_coef(n_features, k, false);
  ffm_param param_als = {
      .n_iter = 200, .init_sigma = 0.1, .SOLVER = SOLVER_ALS};
  sparse_fit(coef, data_train->X, data_test->X, data_train->y, y_pred,
             param_als);
  sparse_predict(coef, data_test->X, y_pred);

  free_ffm_coef(coef);
  TestFixtureDestructor(data_train, NULL);
  TestFixtureDestructor(data_test, NULL);
}

void test_sparse_map_gibbs_second_interactions(void) {
  int n_features = 10;
  int n_samples = 1000;
  int k = 2;
  double init_sigma = 0.1;

  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {
      .n_iter = 5, .init_sigma = init_sigma, .SOLVER = SOLVER_MCMC};
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param);
  double score_5_samples = ffm_r2_score(data->y, y_pred);

  free_ffm_coef(coef);
  ffm_vector_set_all(y_pred, 0);
  coef = alloc_fm_coef(n_features, 0, false);
  ffm_param param_50 = {
      .n_iter = 50, .init_sigma = init_sigma, .SOLVER = SOLVER_MCMC};
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param_50);
  double score_50_samples_first_order = ffm_r2_score(data->y, y_pred);

  free_ffm_coef(coef);
  ffm_vector_set_all(y_pred, 0);
  coef = alloc_fm_coef(n_features, k + 5, false);
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param_50);
  double score_50_samples = ffm_r2_score(data->y, y_pred);

  g_assert_cmpfloat(score_50_samples, >, score_50_samples_first_order);
  g_assert_cmpfloat(score_50_samples, >, score_5_samples);
  g_assert_cmpfloat(score_50_samples, >, .72);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_sparse_als_classification(void) {
  int n_features = 10;
  int n_samples = 100;
  int k = 2;
  double init_sigma = 0.01;

  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);
  // map to classification problem
  ffm_vector_make_labels(data->y);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 50,
                     .init_sigma = init_sigma,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_CLASSIFICATION};
  param.init_lambda_w = 5.5;
  param.init_lambda_V = 5.5;
  sparse_fit(coef, data->X, NULL, data->y, NULL, param);
  sparse_predict(coef, data->X, y_pred);
  ffm_vector_normal_cdf(y_pred);

  g_assert_cmpfloat(ffm_vector_accuracy(data->y, y_pred), >=, .8);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_sparse_als_classification_path(void) {
  int n_features = 10;
  int n_samples = 200;
  int k = 4;
  double init_sigma = 0.1;

  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);
  // map to classification problem
  ffm_vector_make_labels(data->y);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 0,
                     .init_sigma = init_sigma,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_CLASSIFICATION};
  param.init_lambda_w = 5.5;
  param.init_lambda_V = 5.5;

  double acc = 0;
  // objective does not decline strigtly monotonic because of latend target
  // but should still decrease on average (at least till convergence)
  for (int i = 1; i < 9; i = i * 2) {
    param.n_iter = i;
    sparse_fit(coef, data->X, NULL, data->y, NULL, param);
    sparse_predict(coef, data->X, y_pred);
    ffm_vector_normal_cdf(y_pred);
    double tmp_acc = ffm_vector_accuracy(data->y, y_pred);
    // training error should (almost) always decrease
    // printf("iter %d, last acc %f\n", i, acc);
    g_assert_cmpfloat(tmp_acc, >=, acc);
    acc = tmp_acc;
  }

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_sparse_mcmc_classification(void) {
  int n_features = 10;
  int n_samples = 100;
  int k = 2;
  double init_sigma = 0.1;

  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);
  // map to classification problem
  ffm_vector_make_labels(data->y);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 50,
                     .init_sigma = init_sigma,
                     .SOLVER = SOLVER_MCMC,
                     .TASK = TASK_CLASSIFICATION};
  param.init_lambda_w = 5.5;
  param.init_lambda_V = 5.5;
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param);
  sparse_predict(coef, data->X, y_pred);

  g_assert_cmpfloat(ffm_vector_accuracy(data->y, y_pred), >=, .84);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}
void test_numerical_stability(void) {
  int n_features = 10;
  int n_samples = 10000;
  int k = 2;

  TestFixture_T *data = makeTestFixture(15, n_samples, n_features, k);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_param param = {.n_iter = 7, .init_sigma = 0.01, .SOLVER = SOLVER_ALS};
  param.init_lambda_w = 400;
  param.init_lambda_V = 400;
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param);
  sparse_predict(coef, data->X, y_pred);
  double score_als = ffm_r2_score(data->y, y_pred);
  g_assert_cmpfloat(score_als, >, .98);

  free_ffm_coef(coef);
  ffm_vector_set_all(y_pred, 0);

  coef = alloc_fm_coef(n_features, k, false);
  ffm_param param_mcmc = {
      .n_iter = 50, .init_sigma = 0.01, .SOLVER = SOLVER_MCMC};
  sparse_fit(coef, data->X, data->X, data->y, y_pred, param_mcmc);
  double score_gibbs = ffm_r2_score(data->y, y_pred);
  g_assert_cmpfloat(score_gibbs, >, .99);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_map_update_target(void) {
  double pred[] = {0.5, 0.2, -0.2, -0.5, -0.1, 0.8};
  double true_[] = {1, 1, -1, -1, 1, -1};
  double z[] = {0.509160434,  0.6750731798, -0.6750731798,
                -0.509160434, 0.8626174715, -1.3674022692};
  ffm_vector y_pred = {.data = pred, .size = 6};
  ffm_vector y_true = {.data = true_, .size = 6};
  ffm_vector *z_target = ffm_vector_alloc(6);
  map_update_target(&y_pred, z_target, &y_true);
  for (int i = 0; i < 6; i++)
    g_assert_cmpfloat(fabs(z_target->data[i] - z[i]), <=, 1e-9);
  ffm_vector_free(z_target);
}

void test_als_warm_start(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 4;

  ffm_vector *y_10_iter = ffm_vector_calloc(n_samples);
  ffm_vector *y_15_iter = ffm_vector_calloc(n_samples);
  ffm_vector *y_5_plus_5_iter = ffm_vector_calloc(n_samples);

  ffm_param param = {.warm_start = false,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION,
                     .rng_seed = 123};

  param.n_iter = 10;
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, pFix->X, y_10_iter);

  param.n_iter = 15;
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, pFix->X, y_15_iter);

  param.n_iter = 5;
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  param.warm_start = true;
  sparse_fit(coef, pFix->X, NULL, pFix->y, NULL, param);
  sparse_predict(coef, pFix->X, y_5_plus_5_iter);

  // check that the results are equal
  double mse = ffm_vector_mean_squared_error(y_10_iter, y_5_plus_5_iter);
  double mse_diff = ffm_vector_mean_squared_error(y_15_iter, y_5_plus_5_iter);

  g_assert_cmpfloat(mse, <=, 1e-8);
  g_assert_cmpfloat(mse, <, mse_diff);

  free_ffm_coef(coef);
  ffm_vector_free_all(y_10_iter, y_5_plus_5_iter);
}

void test_mcmc_warm_start(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 4;

  ffm_vector *y_10_iter = ffm_vector_calloc(n_samples);
  ffm_vector *y_15_iter = ffm_vector_calloc(n_samples);
  ffm_vector *y_5_plus_5_iter = ffm_vector_calloc(n_samples);

  ffm_param param = {.warm_start = false,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_MCMC,
                     .TASK = TASK_REGRESSION,
                     .rng_seed = 125};

  param.n_iter = 100;
  // printf("n_iter %d\n", param.n_iter);
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  sparse_fit(coef, pFix->X, pFix->X, pFix->y, y_10_iter, param);

  param.n_iter = 150;
  sparse_fit(coef, pFix->X, pFix->X, pFix->y, y_15_iter, param);

  param.n_iter = 50;
  sparse_fit(coef, pFix->X, pFix->X, pFix->y, y_5_plus_5_iter, param);
  param.warm_start = true;
  param.iter_count = param.n_iter;
  param.n_iter += 50;  // add more iterations
  sparse_fit(coef, pFix->X, pFix->X, pFix->y, y_5_plus_5_iter, param);

  // check that the results are equal
  // double mse10 = ffm_vector_mean_squared_error(pFix->y, y_5_plus_5_iter);
  double mse10_55 = ffm_vector_mean_squared_error(y_10_iter, y_5_plus_5_iter);
  double mse15_55 = ffm_vector_mean_squared_error(y_15_iter, y_5_plus_5_iter);

  g_assert_cmpfloat(mse10_55, <, mse15_55);

  free_ffm_coef(coef);
  ffm_vector_free_all(y_10_iter, y_5_plus_5_iter);
}

int main(int argc, char **argv) {
  /*
  feenableexcept(FE_INVALID   |
                 FE_DIVBYZERO |
                 FE_OVERFLOW  |
                 FE_UNDERFLOW);
  */

  g_test_init(&argc, &argv, NULL);

  TestFixture_T Fixture;

  g_test_add("/als/update second-order error", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_update_second_order_error,
             TestFixtureDestructor);

  g_test_add("/als/eval second-order term", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_eval_second_order_term,
             TestFixtureDestructor);

  g_test_add("/als/update v_ij", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_sparse_update_v_ij,
             TestFixtureDestructor);

  g_test_add("/general/predict", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_sparse_predict,
             TestFixtureDestructor);

  g_test_add("/general/row_predict", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_row_predict,
             TestFixtureDestructor);

  g_test_add("/general/col_predict", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_col_predict,
             TestFixtureDestructor);

  g_test_add("/als/zero order only", TestFixture_T, &Fixture,
             TestFixtureContructorSimple, test_sparse_als_zero_order_only,
             TestFixtureDestructor);

  g_test_add("/als/first order only", TestFixture_T, &Fixture,
             TestFixtureContructorSimple, test_sparse_als_first_order_only,
             TestFixtureDestructor);

  g_test_add("/als/second order only", TestFixture_T, &Fixture,
             TestFixtureContructorSimple, test_sparse_als_second_order_only,
             TestFixtureDestructor);

  g_test_add("/als/all interactions", TestFixture_T, &Fixture,
             TestFixtureContructorSimple, test_sparse_als_all_interactions,
             TestFixtureDestructor);

  g_test_add("/als/first order", TestFixture_T, &Fixture,
             TestFixtureContructorLong,
             test_sparse_als_first_order_interactions, TestFixtureDestructor);

  g_test_add("/als/second order", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_sparse_als_second_interactions,
             TestFixtureDestructor);

  g_test_add("/mcmc/second order", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_sparse_mcmc_second_interactions,
             TestFixtureDestructor);

  g_test_add("/mcmc/second order classification", TestFixture_T, &Fixture,
             TestFixtureContructorLong,
             test_sparse_mcmc_second_interactions_classification,
             TestFixtureDestructor);

  g_test_add("/general/train test different size", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_train_test_of_different_size,
             TestFixtureDestructor);

  g_test_add_func("/als/generated data", test_sparse_als_generated_data);

  g_test_add_func("/mcmc/MAP gibbs first order",
                  test_sparse_map_gibbs_first_order_interactions);

  g_test_add_func("/mcmc/hyperparameter sampling", test_hyerparameter_sampling);

  g_test_add_func("/mcmc/MAP gibbs second order",
                  test_sparse_map_gibbs_second_interactions);

  g_test_add_func("/general/numerical stability", test_numerical_stability);

  g_test_add_func("/mcmc/map update target", test_map_update_target);

  g_test_add_func("/als/classification", test_sparse_als_classification);
  g_test_add_func("/als/classification path",
                  test_sparse_als_classification_path);
  g_test_add_func("/mcmc/classification", test_sparse_mcmc_classification);

  g_test_add("/als/warm_start", TestFixture_T, &Fixture,
             TestFixtureContructorSimple, test_als_warm_start,
             TestFixtureDestructor);

  g_test_add("/mcmc/warm_start", TestFixture_T, &Fixture,
             TestFixtureContructorSimple, test_mcmc_warm_start,
             TestFixtureDestructor);

  return g_test_run();
}
