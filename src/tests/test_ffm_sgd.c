#include "fast_fm.h"
#include <fenv.h>
#include "TestFixtures.h"

void test_sgd_predict_sample(TestFixture_T *pFixtureInput, gconstpointer pg) {
  int sample_row = 1;

  double y_pred =
      ffm_predict_sample(pFixtureInput->coef, pFixtureInput->X_t, sample_row);
  // only first order == 24
  g_assert_cmpfloat(y_pred, ==, 672);
}

void test_first_order_sgd(TestFixture_T *pFix, gconstpointer pg) {
  // int k = pFix->coef->V->size0;
  int k = 0;
  int n_features = pFix->X->n;
  int n_iter = 50;
  double init_sigma = .1;
  double step_size = .002;

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_vector *y_pred = ffm_vector_calloc(5);
  ffm_param param = {.n_iter = n_iter * 100,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_REGRESSION};
  param.init_lambda_w = 0.5;
  ffm_fit_sgd(coef, pFix->X_t, pFix->y, &param);
  row_predict(coef, pFix->X_t, y_pred);

  g_assert_cmpfloat(ffm_r2_score(y_pred, pFix->y), >, .85);

  ffm_vector *y_pred_als = ffm_vector_calloc(5);
  ffm_coef *coef_als = alloc_fm_coef(n_features, k, false);
  ffm_param param_als = {.n_iter = 50,
                         .init_sigma = 0.1,
                         .SOLVER = SOLVER_ALS,
                         .TASK = TASK_REGRESSION};
  param_als.init_lambda_w = 3.5;
  sparse_fit(coef_als, pFix->X, pFix->X, pFix->y, y_pred_als, param_als);
  row_predict(coef_als, pFix->X_t, y_pred_als);

  // compare fit of als and sgd
  g_assert_cmpfloat(ffm_r2_score(y_pred, y_pred_als), >, .98);
  // compare coef of als and sgd
  g_assert_cmpfloat(ffm_r2_score(coef->w, coef_als->w), >, .98);

  ffm_vector_free_all(y_pred, y_pred_als);
  free_ffm_coef(coef);
  free_ffm_coef(coef_als);
}

void test_second_order_sgd(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 3;
  int n_iter = 10;
  double init_sigma = .01;
  double step_size = .0002;

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_vector *y_pred = ffm_vector_calloc(5);
  ffm_param param = {.n_iter = n_iter * 100,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_REGRESSION};
  param.init_lambda_w = 0.5;
  param.init_lambda_V = 50.5;
  ffm_fit_sgd(coef, pFix->X_t, pFix->y, &param);
  row_predict(coef, pFix->X_t, y_pred);

  g_assert_cmpfloat(ffm_r2_score(y_pred, pFix->y), >, .98);

  ffm_vector *y_pred_als = ffm_vector_calloc(5);
  ffm_coef *coef_als = alloc_fm_coef(n_features, k, false);

  ffm_param param_als = {
      .n_iter = 10, .init_sigma = 0.01, .SOLVER = SOLVER_ALS};
  param_als.init_lambda_w = 3.5;
  param_als.init_lambda_V = 50.5;
  sparse_fit(coef_als, pFix->X, pFix->X, pFix->y, y_pred_als, param_als);
  sparse_predict(coef_als, pFix->X, y_pred_als);

  // compare fit of als and sgd
  g_assert_cmpfloat(ffm_r2_score(y_pred, y_pred_als), >, .98);

  ffm_vector_free_all(y_pred, y_pred_als);
  free_ffm_coef(coef);
  free_ffm_coef(coef_als);
}

void test_sgd_classification(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int k = 2;
  int n_iter = 10;
  double init_sigma = .01;
  double step_size = .0002;

  // map to classification problem
  ffm_vector_make_labels(pFix->y);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_vector *y_pred = ffm_vector_calloc(5);
  ffm_param param = {.n_iter = n_iter * 100,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_CLASSIFICATION};
  param.init_lambda_w = 0.5;
  param.init_lambda_V = 0.5;
  ffm_fit_sgd(coef, pFix->X_t, pFix->y, &param);
  row_predict(coef, pFix->X_t, y_pred);
  for (int i = 0; i < y_pred->size; i++)
    ffm_vector_set(y_pred, i, ffm_sigmoid(ffm_vector_get(y_pred, i)));

  g_assert_cmpfloat(ffm_vector_accuracy(pFix->y, y_pred), >=, .8);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
}

void test_first_order_bpr(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 0;
  int n_iter = 50;
  double init_sigma = .01;
  double step_size = .002;

  ffm_matrix *compares = ffm_vector_to_rank_comparision(pFix->y);
  ffm_vector *true_order = ffm_vector_get_order(pFix->y);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  for (int i = 0; i < 2; i++) coef->w->data[i] = 0.1;

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);
  ffm_param param = {.n_iter = n_iter,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_RANKING};
  param.init_lambda_w = 0.0;
  ffm_fit_sgd_bpr(coef, pFix->X_t, compares, param);
  row_predict(coef, pFix->X_t, y_pred);
  ffm_vector *pred_order = ffm_vector_get_order(y_pred);
  double kendall_tau = ffm_vector_kendall_tau(true_order, pred_order);
  g_assert_cmpfloat(kendall_tau, ==, 1);

  ffm_vector_free_all(y_pred, true_order, pred_order);
  free_ffm_coef(coef);
}

void test_update_second_order_bpr(TestFixture_T *pFix, gconstpointer pg) {
  double cache_p = 1.1;
  double cache_n = 2.2;
  double y_err = -1;
  double step_size = 0.1;
  double lambda_V = 4;

  int sample_row_p = 1;
  int sample_row_n = 0;
  int V_col = 0;
  update_second_order_bpr(pFix->X_t, pFix->coef->V, cache_n, cache_p, y_err,
                          step_size, lambda_V, sample_row_p, sample_row_n,
                          V_col);

  // 1 - 0.1*(-1 * (4*1.1 - 4^2 - (1*2.2 - 1^2*1)) + 4 *1) = -0.68
  g_assert_cmpfloat(fabs(ffm_matrix_get(pFix->coef->V, 0, 0) - (-0.68)), <,
                    1e-10);

  //> 2 - 0.1*(-1 * (0*1.1 - 0^2*2 - (2*2.2 - 2^2*2)) + 4 *2)
  //[1] 1.56
  g_assert_cmpfloat(ffm_matrix_get(pFix->coef->V, 0, 1), ==, 1.56);
}

void test_second_order_bpr(TestFixture_T *pFix, gconstpointer pg) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 2;
  int n_iter = 200;
  double init_sigma = .01;
  double step_size = .02;

  ffm_matrix *compares = ffm_vector_to_rank_comparision(pFix->y);
  ffm_vector *true_order = ffm_vector_get_order(pFix->y);

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);
  ffm_param param = {.n_iter = n_iter,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_RANKING};
  param.init_lambda_w = 0.5;
  param.init_lambda_V = 0.5;
  ffm_fit_sgd_bpr(coef, pFix->X_t, compares, param);

  sparse_predict(coef, pFix->X, y_pred);
  ffm_vector *pred_order = ffm_vector_get_order(y_pred);
  double kendall_tau = ffm_vector_kendall_tau(true_order, pred_order);
  g_assert_cmpfloat(kendall_tau, ==, 1);

  ffm_vector_free_all(y_pred, true_order, pred_order);
  free_ffm_coef(coef);
}

void test_sgd_generated_data(void) {
  int n_features = 10;
  int n_samples = 100;
  int k = 0;
  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);
  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  int n_iter = 40;
  double init_sigma = 0.1;
  double step_size = .05;

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_param param = {.n_iter = n_iter * 100,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_REGRESSION};
  param.init_lambda_w = 0.05;
  ffm_fit_sgd(coef, data->X_t, data->y, &param);
  sparse_predict(coef, data->X, y_pred);

  g_assert_cmpfloat(ffm_r2_score(y_pred, data->y), >, 0.95);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_sgd_classification_generated_data(void) {
  int n_features = 10;
  int n_samples = 100;
  int k = 2;
  TestFixture_T *data = makeTestFixture(124, n_samples, n_features, k);
  ffm_vector_make_labels(data->y);
  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  int n_iter = 200;
  double init_sigma = 0.1;
  double step_size = .2;

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_param param = {.n_iter = n_iter,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_CLASSIFICATION};
  param.init_lambda_w = 0.05;
  param.init_lambda_V = 0.05;

  ffm_fit_sgd(coef, data->X_t, data->y, &param);
  sparse_predict(coef, data->X, y_pred);
  for (int i = 0; i < y_pred->size; i++)
    ffm_vector_set(y_pred, i, ffm_sigmoid(ffm_vector_get(y_pred, i)));

  g_assert_cmpfloat(ffm_vector_accuracy(data->y, y_pred), >=, .81);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_sgd_bpr_generated_data(void) {
  int n_features = 15;
  int n_samples = 10;
  int k = 4;
  TestFixture_T *data = makeTestFixture(1245, n_samples, n_features, k);
  ffm_matrix *compares = ffm_vector_to_rank_comparision(data->y);
  ffm_vector *true_order = ffm_vector_get_order(data->y);

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);

  int n_iter = 100;
  double init_sigma = 0.1;
  double step_size = .1;

  ffm_coef *coef = alloc_fm_coef(n_features, k, false);

  ffm_param param = {.n_iter = n_iter,
                     .init_sigma = init_sigma,
                     .stepsize = step_size,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_RANKING};
  param.init_lambda_w = 0.05;
  param.init_lambda_V = 0.05;
  ffm_fit_sgd_bpr(coef, data->X_t, compares, param);

  sparse_predict(coef, data->X, y_pred);
  ffm_vector *pred_order = ffm_vector_get_order(y_pred);

  double kendall_tau = ffm_vector_kendall_tau(true_order, pred_order);
  g_assert_cmpfloat(kendall_tau, >, .45);

  ffm_vector_free(y_pred);
  free_ffm_coef(coef);
  TestFixtureDestructor(data, NULL);
}

void test_extract_gradient() {
  int n_features = 3;
  int k = 2;
  double stepsize = .5;

  ffm_coef *coef_t0 = alloc_fm_coef(n_features, k, false);
  coef_t0->w_0 = 0.5;
  ffm_vector_set(coef_t0->w, 0, 1);
  ffm_vector_set(coef_t0->w, 1, 2);
  ffm_vector_set(coef_t0->w, 2, 3);
  ffm_matrix_set(coef_t0->V, 0, 0, 4);
  ffm_matrix_set(coef_t0->V, 1, 0, 5);
  ffm_matrix_set(coef_t0->V, 0, 1, 6);
  ffm_matrix_set(coef_t0->V, 1, 1, 7);
  ffm_matrix_set(coef_t0->V, 0, 2, 8);
  ffm_matrix_set(coef_t0->V, 1, 2, 9);

  ffm_coef *coef_t1 = alloc_fm_coef(n_features, k, false);

  ffm_coef *grad = extract_gradient(coef_t0, coef_t1, stepsize);

  g_assert_cmpfloat(coef_t0->w_0, ==, grad->w_0 * -stepsize);
  // check w grad
  for (int i = 0; i < n_features; i++)
    g_assert_cmpfloat(ffm_vector_get(coef_t0->w, i), ==,
                      ffm_vector_get(grad->w, i) * stepsize);
  // check V grad
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n_features; j++)
      g_assert_cmpfloat(ffm_matrix_get(coef_t0->V, i, j), ==,
                        ffm_matrix_get(grad->V, i, j) * stepsize);

  free_ffm_coef(coef_t0);
  free_ffm_coef(coef_t1);
  free_ffm_coef(grad);
}

void test_l2_penalty() {
  int n_features = 2;
  int k = 1;
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  ffm_vector_set(coef->w, 0, 1);
  ffm_vector_set(coef->w, 1, 2);
  ffm_matrix_set(coef->V, 0, 0, 3);
  ffm_matrix_set(coef->V, 0, 1, 4);

  coef->lambda_w = 0.5;
  double lambda_V_all = 0.5;
  ffm_vector_set_all(coef->lambda_V, lambda_V_all);

  double true_loss = coef->lambda_w * 5 + lambda_V_all * 25;
  double loss = l2_penalty(coef);
  g_assert_cmpfloat(true_loss, ==, loss);
  free_ffm_coef(coef);
}

void test_gradient_check_reg(TestFixture_T *pFix, gconstpointer pg) {
  cs *X_crs = pFix->X_t;
  ffm_vector *y = pFix->y;
  int test_sample_row = 0;
  double y_true = ffm_vector_get(y, test_sample_row);
  int n_features = pFix->coef->w->size;

  double eps = 0.0001;

  ffm_param param = {.n_iter = 1,
                     .stepsize = .001,
                     .init_sigma = .1,
                     .k = 2,
                     .init_lambda_w = 0.5,
                     .init_lambda_V = 1.5,
                     .warm_start = 1,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_REGRESSION,
                     .rng_seed = 44};

  ffm_coef *coef_t0 = alloc_fm_coef(n_features, param.k, false);
  init_ffm_coef(coef_t0, param);

  ffm_coef *coef_t1 = alloc_fm_coef(n_features, param.k, false);
  init_ffm_coef(coef_t1, param);

  ffm_fit_sgd(coef_t1, X_crs, y, &param);
  ffm_coef *grad = extract_gradient(coef_t0, coef_t1, param.stepsize);

  // check w gradient updates
  for (int i = 0; i < n_features; i++) {
    // keep copy
    double tmp = ffm_vector_get(coef_t0->w, i);
    // x + eps
    ffm_vector_set(coef_t0->w, i, tmp + eps);
    double y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
    double sq_loss = 0.5 * pow(y_true - y_pred, 2);
    double l_plus = sq_loss + 0.5 * l2_penalty(coef_t0);
    // x - eps
    ffm_vector_set(coef_t0->w, i, tmp - eps);
    y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
    sq_loss = 0.5 * pow(y_true - y_pred, 2);
    double l_minus = sq_loss + 0.5 * l2_penalty(coef_t0);
    // restore
    ffm_vector_set(coef_t0->w, i, tmp);
    double grad_i = (l_plus - l_minus) / (2 * eps);

    g_assert_cmpfloat(fabs(grad_i - ffm_vector_get(grad->w, i)), <, 1e-10);
  }

  // check V gradient updates
  for (int f = 0; f < param.k; f++)
    for (int i = 0; i < n_features; i++) {
      // keep copy
      double tmp = ffm_matrix_get(coef_t0->V, f, i);
      // x + eps
      ffm_matrix_set(coef_t0->V, f, i, tmp + eps);
      double y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
      double sq_loss = 0.5 * pow(y_true - y_pred, 2);
      double l_plus = sq_loss + 0.5 * l2_penalty(coef_t0);
      // x - eps
      ffm_matrix_set(coef_t0->V, f, i, tmp - eps);
      y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
      sq_loss = 0.5 * pow(y_true - y_pred, 2);
      double l_minus = sq_loss + 0.5 * l2_penalty(coef_t0);
      // restore
      ffm_matrix_set(coef_t0->V, f, i, tmp);
      double grad_i = (l_plus - l_minus) / (2 * eps);

      g_assert_cmpfloat(fabs(grad_i - ffm_matrix_get(grad->V, f, i)), <, 1e-10);
    }

  free_ffm_coef(coef_t0);
  free_ffm_coef(coef_t1);
  free_ffm_coef(grad);
}

void test_gradient_check_class(TestFixture_T *pFix, gconstpointer pg) {
  cs *X_crs = pFix->X_t;
  ffm_vector *y = pFix->y;
  int test_sample_row = 0;
  double y_true = ffm_vector_get(y, test_sample_row);
  int n_features = pFix->coef->w->size;

  double eps = 0.0001;

  ffm_param param = {.n_iter = 1,
                     .stepsize = .01,
                     .init_sigma = .01,
                     .k = 2,
                     .init_lambda_w = 1.5,
                     .init_lambda_V = 2.0,
                     .warm_start = 1,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_CLASSIFICATION,
                     .rng_seed = 44};

  ffm_coef *coef_t0 = alloc_fm_coef(n_features, param.k, false);
  init_ffm_coef(coef_t0, param);

  ffm_coef *coef_t1 = alloc_fm_coef(n_features, param.k, false);
  init_ffm_coef(coef_t1, param);

  ffm_fit_sgd(coef_t1, X_crs, y, &param);
  ffm_coef *grad = extract_gradient(coef_t0, coef_t1, param.stepsize);

  // check w gradient updates
  for (int i = 0; i < n_features; i++) {
    // keep copy
    double tmp = ffm_vector_get(coef_t0->w, i);
    // x + eps
    ffm_vector_set(coef_t0->w, i, tmp + eps);
    double y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
    double log_loss = -log(ffm_sigmoid(y_true * y_pred));
    double l_plus = log_loss + 0.5 * l2_penalty(coef_t0);
    // x - eps
    ffm_vector_set(coef_t0->w, i, tmp - eps);
    y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
    log_loss = -log(ffm_sigmoid(y_true * y_pred));
    double l_minus = log_loss + 0.5 * l2_penalty(coef_t0);
    // restore
    ffm_vector_set(coef_t0->w, i, tmp);
    // finite central differences
    double grad_i = (l_plus - l_minus) / (2 * eps);

    // g_assert_cmpfloat(grad_i, ==, ffm_vector_get(grad->w, i));
    g_assert_cmpfloat(fabs(grad_i - ffm_vector_get(grad->w, i)), <, 1e-9);
  }

  // check V gradient updates
  for (int f = 0; f < param.k; f++)
    for (int i = 0; i < n_features; i++) {
      // keep copy
      double tmp = ffm_matrix_get(coef_t0->V, f, i);
      // x + eps
      ffm_matrix_set(coef_t0->V, f, i, tmp + eps);
      double y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
      double log_loss = -log(ffm_sigmoid(y_true * y_pred));
      double l_plus = log_loss + 0.5 * l2_penalty(coef_t0);
      // x - eps
      ffm_matrix_set(coef_t0->V, f, i, tmp - eps);
      y_pred = ffm_predict_sample(coef_t0, X_crs, test_sample_row);
      log_loss = -log(ffm_sigmoid(y_true * y_pred));
      double l_minus = log_loss + 0.5 * l2_penalty(coef_t0);
      // restore
      ffm_matrix_set(coef_t0->V, f, i, tmp);
      // finite central differences
      double grad_i = (l_plus - l_minus) / (2 * eps);

      g_assert_cmpfloat(fabs(grad_i - ffm_matrix_get(grad->V, f, i)), <, 1e-10);
    }

  free_ffm_coef(coef_t0);
  free_ffm_coef(coef_t1);
  free_ffm_coef(grad);
}

void test_gradient_check_bpr(TestFixture_T *pFix, gconstpointer pg) {
  cs *X_crs = pFix->X_t;
  ffm_matrix *pairs = ffm_matrix_calloc(1, 2);
  int pos_row = 0;
  ffm_matrix_set(pairs, 0, 0, pos_row);
  int neg_row = 1;
  ffm_matrix_set(pairs, 0, 1, neg_row);

  int n_features = pFix->coef->w->size;

  double eps = 0.0001;

  ffm_param param = {.n_iter = 1,
                     .stepsize = .01,
                     .init_sigma = .01,
                     .k = 2,
                     .init_lambda_w = 0.0,
                     .init_lambda_V = 0.0,
                     .warm_start = 1,
                     .SOLVER = SOLVER_SGD,
                     .TASK = TASK_RANKING,
                     .rng_seed = 44};

  ffm_coef *coef_t0 = alloc_fm_coef(n_features, param.k, false);
  init_ffm_coef(coef_t0, param);

  ffm_coef *coef_t1 = alloc_fm_coef(n_features, param.k, false);
  init_ffm_coef(coef_t1, param);

  ffm_fit_sgd_bpr(coef_t1, X_crs, pairs, param);
  ffm_coef *grad = extract_gradient(coef_t0, coef_t1, param.stepsize);

  double y_pos, y_neg, bpr_loss, l_plus, l_minus, grad_i, tmp;
  // check w gradient updates
  for (int i = 0; i < n_features; i++) {
    // keep copy
    tmp = ffm_vector_get(coef_t0->w, i);
    // x + eps
    ffm_vector_set(coef_t0->w, i, tmp + eps);
    y_pos = ffm_predict_sample(coef_t0, X_crs, pos_row);
    y_neg = ffm_predict_sample(coef_t0, X_crs, neg_row);
    bpr_loss = -log(ffm_sigmoid(y_pos - y_neg));
    l_plus = bpr_loss + 0.5 * l2_penalty(coef_t0);
    // x - eps
    ffm_vector_set(coef_t0->w, i, tmp - eps);
    y_pos = ffm_predict_sample(coef_t0, X_crs, pos_row);
    y_neg = ffm_predict_sample(coef_t0, X_crs, neg_row);
    bpr_loss = -log(ffm_sigmoid(y_pos - y_neg));
    l_minus = bpr_loss + 0.5 * l2_penalty(coef_t0);
    // restore
    ffm_vector_set(coef_t0->w, i, tmp);
    // finite central differences
    grad_i = (l_plus - l_minus) / (2 * eps);

    // g_assert_cmpfloat(grad_i, ==, ffm_vector_get(grad->w, i));
    g_assert_cmpfloat(fabs(grad_i - ffm_vector_get(grad->w, i)), <, 1e-9);
  }

  // check V gradient updates
  for (int f = 0; f < param.k; f++)
    for (int i = 0; i < n_features; i++) {
      // keep copy
      tmp = ffm_matrix_get(coef_t0->V, f, i);
      // x + eps
      ffm_matrix_set(coef_t0->V, f, i, tmp + eps);
      y_pos = ffm_predict_sample(coef_t0, X_crs, pos_row);
      y_neg = ffm_predict_sample(coef_t0, X_crs, neg_row);
      bpr_loss = -log(ffm_sigmoid(y_pos - y_neg));
      l_plus = bpr_loss + 0.5 * l2_penalty(coef_t0);
      // x - eps
      ffm_matrix_set(coef_t0->V, f, i, tmp - eps);
      y_pos = ffm_predict_sample(coef_t0, X_crs, pos_row);
      y_neg = ffm_predict_sample(coef_t0, X_crs, neg_row);
      bpr_loss = -log(ffm_sigmoid(y_pos - y_neg));
      l_minus = bpr_loss + 0.5 * l2_penalty(coef_t0);
      // restore
      ffm_matrix_set(coef_t0->V, f, i, tmp);
      // finite central differences
      grad_i = (l_plus - l_minus) / (2 * eps);

      // g_assert_cmpfloat(grad_i, ==, ffm_matrix_get(grad->V, f, i));
      g_assert_cmpfloat(fabs(grad_i - ffm_matrix_get(grad->V, f, i)), <, 1e-10);
    }

  free_ffm_coef(coef_t0);
  free_ffm_coef(coef_t1);
  free_ffm_coef(grad);
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
  g_test_add("/sgd/util/predict sample", TestFixture_T, &Fixture,
             TestFixtureContructorWide, test_sgd_predict_sample,
             TestFixtureDestructor);
  g_test_add("/sgd/reg/first order", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_first_order_sgd,
             TestFixtureDestructor);
  g_test_add("/sgd/reg/second order", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_second_order_sgd,
             TestFixtureDestructor);
  g_test_add("/sgd/class/full", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_sgd_classification,
             TestFixtureDestructor);
  g_test_add("/sgd/bpr/update second order", TestFixture_T, &Fixture,
             TestFixtureContructorWide, test_update_second_order_bpr,
             TestFixtureDestructor);
  g_test_add("/sgd/bpr/first order", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_first_order_bpr,
             TestFixtureDestructor);
  g_test_add("/sgd/bpr/second order", TestFixture_T, &Fixture,
             TestFixtureContructorLong, test_second_order_bpr,
             TestFixtureDestructor);
  g_test_add_func("/sgd/class/generated data",
                  test_sgd_classification_generated_data);
  g_test_add_func("/sgd/reg/generated data", test_sgd_generated_data);
  g_test_add_func("/sgd/bpr/generated data", test_sgd_bpr_generated_data);

  g_test_add_func("/sgd/util/extract_gradient", test_extract_gradient);
  g_test_add_func("/sgd/util/l2_penalty", test_l2_penalty);
  g_test_add("/sgd/reg/gradient check", TestFixture_T, &Fixture,
             TestFixtureContructorWide, test_gradient_check_reg,
             TestFixtureDestructor);
  g_test_add("/sgd/class/gradient check", TestFixture_T, &Fixture,
             TestFixtureContructorWide, test_gradient_check_class,
             TestFixtureDestructor);
  g_test_add("/sgd/bpr/gradient check", TestFixture_T, &Fixture,
             TestFixtureContructorWide, test_gradient_check_bpr,
             TestFixtureDestructor);
  return g_test_run();
}
