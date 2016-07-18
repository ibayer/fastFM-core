// Author: Immanuel Bayer
// License: BSD 3 clause

#include "fast_fm.h"

void ffm_fit_sgd(ffm_coef *coef, cs *A, ffm_vector *y, ffm_param *param) {
  // X is transpose of design matrix. Samples are stored in columns.
  double step_size = param->stepsize;
  int n_samples = A->n;
  int k;
  if (coef->V)
    k = coef->V->size0;
  else
    k = 0;
  if (!param->warm_start) init_ffm_coef(coef, *param);

  for (int i = 0; i < param->n_iter; i++) {
    int sample_row = i % n_samples;
    int y_true = y->data[sample_row];
    double y_err = 0;

    // predict y(x)
    if (param->TASK == TASK_REGRESSION){
      y_err = ffm_predict_sample(coef, A, sample_row) - y_true;
    }
    else{
      y_err =
          (ffm_sigmoid(ffm_predict_sample(coef, A, sample_row) * y_true) - 1) *
          y_true;
    }
    coef->w_0 = coef->w_0 - step_size * y_err;

    int p, *Ap, *Ai;
    double *Ax;
    Ap = A->p;
    Ai = A->i;
    Ax = A->x;

    // update w
    // p starts add the current sample position and ends befor the next
    // samples starts or the last column ends.
    // The last entry in Ap is the number of nz which is always greater
    // then the start of the last sample.
    for (p = Ap[sample_row]; p < Ap[sample_row + 1]; p++) {
      double theta_w = ffm_vector_get(coef->w, Ai[p]);
      ffm_vector_set(
          coef->w, Ai[p],
          theta_w - step_size * (y_err * Ax[p] + theta_w * coef->lambda_w));
    }
    if (k > 0)
      // update V
      for (int f = 0; f < k; f++) {
        // calc cache
        double cache = 0;
        for (p = Ap[sample_row]; p < Ap[sample_row + 1]; p++)
          cache += Ax[p] * ffm_matrix_get(coef->V, f, Ai[p]);
        // update V_i_j
        for (p = Ap[sample_row]; p < Ap[sample_row + 1]; p++) {
          double v = ffm_matrix_get(coef->V, f, Ai[p]);
          double grad = Ax[p] * cache - (Ax[p] * Ax[p]) * v;
          double lambda_V_f = ffm_vector_get(coef->lambda_V, f);
          ffm_matrix_set(coef->V, f, Ai[p],
                         v - step_size * (y_err * grad + v * lambda_V_f));
        }
      }
  }
}

void ffm_fit_sgd_bpr(ffm_coef *coef, cs *A, ffm_matrix *pairs,
                     ffm_param param) {
  // A is transpose of design matrix. Samples are stored in columns.
  int p, *Ap, *Ai;
  double *Ax;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;

  double step_size = param.stepsize;
  int n_comparisons = pairs->size0;
  int k;
  if (coef->V)
    k = coef->V->size0;
  else
    k = 0;
  if (!param.warm_start) init_ffm_coef(coef, param);
  coef->w_0 = 0;  // should always be zero

  for (int i = 0; i < param.n_iter; i++) {
    int comparison_row = i % n_comparisons;
    int sample_row_p = ffm_matrix_get(pairs, comparison_row, 0);
    int sample_row_n = ffm_matrix_get(pairs, comparison_row, 1);

    // predict y(x)
    double pairs_err =
        -1 + ffm_sigmoid(ffm_predict_sample(coef, A, sample_row_p) -
                         ffm_predict_sample(coef, A, sample_row_n));

    int p_n = Ap[sample_row_n]; // Start of positive sample in value array.
    int p_n_end = Ap[sample_row_n + 1]; // End of positive sample in value array.

    int p_p = Ap[sample_row_p]; // Start of negative sample in value array.
    int p_p_end = Ap[sample_row_p +1 ]; // Ende of negative sample in value array.

    // update w
    // Iterate over nnz positive (p) and negative (n) features vectors in
    // parallel. Stop if both vectors reached there last entry.
    while (p_p < p_p_end || p_n < p_n_end){

      // Check if p should be added to gradient.
      // 1) p has any features left.
      // 2) p's next feature commes bevore n's next feature or n has no
      // features left.
      bool add_p = false;
      if (p_p < p_p_end && (Ai[p_p] <= Ai[p_n] || !(p_n < p_n_end))) {
        add_p = true;
      }

      // this is symmetrical to above.
      bool add_n = false;
      if (p_n < p_n_end && (Ai[p_n] <= Ai[p_p] || !(p_p < p_p_end))) {
        add_n = true;
      }

      // Add feature value to gradient and increment feature position.
      double grad = 0;
      int feature_to_update = - 1;
      if (add_p) {
        feature_to_update = Ai[p_p];
        grad = Ax[p_p++];
      }
      if (add_n) {
        feature_to_update = Ai[p_n];
        grad -= Ax[p_n++];
      }
      assert (feature_to_update >= 0);

      double theta_w = coef->w->data[feature_to_update];
      ffm_vector_set(
          coef->w, feature_to_update,
          theta_w - step_size * (pairs_err * grad + theta_w * coef->lambda_w));
    }
    if (k > 0)
      // update V
      for (int f = 0; f < k; f++) {
        // calc cache
        double cache_p = 0;
        for (p = Ap[sample_row_p]; p < Ap[sample_row_p + 1]; p++)
          cache_p += Ax[p] * ffm_matrix_get(coef->V, f, Ai[p]);
        double cache_n = 0;
        for (p = Ap[sample_row_n]; p < Ap[sample_row_n + 1]; p++)
          cache_n += Ax[p] * ffm_matrix_get(coef->V, f, Ai[p]);
        // update V_i
        double lambda_V_f = ffm_vector_get(coef->lambda_V, f);
        update_second_order_bpr(A, coef->V, cache_n, cache_p, pairs_err,
                                step_size, lambda_V_f, sample_row_p,
                                sample_row_n, f);
      }
  }
}

// TODO: Bedder code reuse between first and second order updates.
void update_second_order_bpr(cs *A, ffm_matrix *V, double cache_n,
                             double cache_p, double pairs_err, double step_size,
                             double lambda_V, int sample_row_p,
                             int sample_row_n, int f) {
  int *Ap, *Ai;
  double *Ax;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  int p_n = Ap[sample_row_n]; // Start of positive sample in value array.
  int p_n_end = Ap[sample_row_n + 1]; // End of positive sample in value array.

  int p_p = Ap[sample_row_p]; // Start of negative sample in value array.
  int p_p_end = Ap[sample_row_p +1 ]; // Ende of negative sample in value array.

  // Iterate over nnz positive (p) and negative (n) features vectors in
  // parallel. Stop if both vectors reached there last entry.
  while (p_p < p_p_end || p_n < p_n_end){

    // Check if p should be added to gradient.
    // 1) p has any features left.
    // 2) p's next feature commes bevore n's next feature or n has no
    // features left.
    bool add_p = false;
    int feature_to_update = - 1;
    if (p_p < p_p_end && (Ai[p_p] <= Ai[p_n] || !(p_n < p_n_end))) {
      add_p = true;
      feature_to_update = Ai[p_p];
    }

    // this is symmetrical to above.
    bool add_n = false;
    if (p_n < p_n_end && (Ai[p_n] <= Ai[p_p] || !(p_p < p_p_end))) {
      add_n = true;
      feature_to_update = Ai[p_n];
    }
    assert (feature_to_update >= 0);

    // Add feature value to gradient and increment feature position.
    double grad = 0;
    double v = ffm_matrix_get(V, f, feature_to_update);
    if (add_p) {
      grad = Ax[p_p] * cache_p - (Ax[p_p] * Ax[p_p]) * v;
      p_p++;
    }
    if (add_n) {
      feature_to_update = Ai[p_n];
      grad -= Ax[p_n] * cache_n - (Ax[p_n] * Ax[p_n]) * v;
      p_n++;
    }
    ffm_matrix_set(V, f, feature_to_update,
                 v - step_size * (pairs_err * grad + v * lambda_V));
  }
}

double ffm_predict_sample(ffm_coef *coef, cs *A, int sample_row) {
  double result = coef->w_0;
  int k;
  if (coef->V)
    k = coef->V->size0;
  else
    k = 0;

  // add first order contributions
  int p, *Ap, *Ai;
  double *Ax;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  for (p = Ap[sample_row]; p < Ap[sample_row + 1]; p++)
    result += Ax[p] * coef->w->data[Ai[p]];

  // add second order contributions
  double sec_order = 0;
  for (int f = 0; f < k; f++) {
    double late_square = 0;  // <V[f,:], x>^2
    double square = 0;       // <V[f,:]^2, x^2>
    for (p = Ap[sample_row]; p < Ap[sample_row + 1]; p++) {
      double x_l = Ax[p];
      double v = ffm_matrix_get(coef->V, f, Ai[p]);
      late_square += v * x_l;
      square += (v * v) * (x_l * x_l);
    }
    sec_order += (late_square * late_square) - square;
  }
  result += .5 * sec_order;
  return result;
}

ffm_coef *extract_gradient(ffm_coef *coef_t0, ffm_coef *coef_t1,
                           double stepsize) {
  int n_features = coef_t0->w->size;
  int k;
  if (coef_t0->V)
    k = coef_t0->V->size0;
  else
    k = 0;

  ffm_coef *grad = alloc_fm_coef(n_features, k, 0);

  grad->w_0 = coef_t1->w_0 / stepsize - coef_t0->w_0 / stepsize;

  for (int i = 0; i < n_features; i++)
    ffm_vector_set(grad->w, i, -(ffm_vector_get(coef_t1->w, i) / stepsize -
                                 ffm_vector_get(coef_t0->w, i) / stepsize));

  for (int i = 0; i < k; i++)
    for (int j = 0; j < n_features; j++)
      ffm_matrix_set(grad->V, i, j,
                     -(ffm_matrix_get(coef_t1->V, i, j) / stepsize -
                       ffm_matrix_get(coef_t0->V, i, j) / stepsize));
  return grad;
}

double l2_penalty(ffm_coef *coef) {
  double loss = 0;
  // w l2 penalty
  double l2_norm = ffm_blas_dnrm2(coef->w);
  loss += coef->lambda_w * (l2_norm * l2_norm);

  if (!coef->V) return loss;
  // V l2 penalty
  int k = coef->V->size0;
  int n_features = coef->V->size1;
  for (int i = 0; i < k; i++) {
    double lambda_V_i = ffm_vector_get(coef->lambda_V, i);
    for (int j = 0; j < n_features; j++) {
      double V_ij = ffm_matrix_get(coef->V, i, j);
      loss += lambda_V_i * (V_ij * V_ij);
    }
  }
  return loss;
}
