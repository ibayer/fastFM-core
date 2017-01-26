// Author: Immanuel Bayer
// License: BSD 3 clause

#ifndef FAST_MF_H
#define FAST_MF_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "../externals/CXSparse/Include/cs.h"
#include "../externals/OpenBLAS/cblas.h"

#include "kmath.h"
#include "../include/ffm.h"

#define SOLVER_ALS 1
#define SOLVER_MCMC 2
#define SOLVER_SGD 3

typedef struct ffm_vector {
  int size;
  double *data;
  int owner;
} ffm_vector;

typedef struct ffm_matrix {
  int size0;  // number of row
  int size1;  // number of columns
  double *data;
  int owner;
} ffm_matrix;

#define Fn_apply(type, fn, ...)                                         \
  {                                                                     \
    void *stopper_for_apply = (int[]){0};                               \
    type **list_for_apply = (type *[]){__VA_ARGS__, stopper_for_apply}; \
    for (int i = 0; list_for_apply[i] != stopper_for_apply; i++)        \
      fn(list_for_apply[i]);                                            \
  }
#define ffm_vector_free_all(...) \
  Fn_apply(ffm_vector, ffm_vector_free, __VA_ARGS__);
#define ffm_matrix_free_all(...) \
  Fn_apply(ffm_matrix, ffm_matrix_free, __VA_ARGS__);

typedef struct ffm_coef {
  double w_0;
  ffm_vector *w;
  ffm_matrix *V;
  // hyperparameter
  double alpha;
  double lambda_w, mu_w;
  ffm_vector *lambda_V;
  ffm_vector *mu_V;
} ffm_coef;

typedef struct fm_data {
  ffm_vector *y;
  cs *X;
} fm_data;

#include "ffm_utils.h"
#include "ffm_random.h"

void sparse_fit(ffm_coef *coef, cs *X, cs *X_test, ffm_vector *y,
                ffm_vector *y_pred, ffm_param param);

void sparse_predict(ffm_coef *coef, cs *X, ffm_vector *y_pred);

void row_predict(ffm_coef *coef, cs *A, ffm_vector *y_pred);

void col_predict(ffm_coef *coef, cs *A, ffm_vector *y_pred);

// ############### internal functions for ALS / MCMC ####################

int eval_second_order_term(ffm_matrix *V, cs *X, ffm_vector *result);

void update_second_order_error(int column, cs *X, ffm_vector *a_theta_v,
                               double delta, ffm_vector *error);

void sparse_v_lf_frac(double *sum_denominator, double *sum_nominator, cs *A,
                      int col_index, ffm_vector *err, ffm_vector *cache,
                      ffm_vector *a_theta_v, double v_lf);

void sample_hyper_parameter(ffm_coef *coef, ffm_vector *err, ffm_rng *rng);

void map_update_target(ffm_vector *y_pred, ffm_vector *sample_target,
                       ffm_vector *y_train);

void sample_target(ffm_rng *r, ffm_vector *y_pred, ffm_vector *z_target,
                   ffm_vector *y_true);

// ############### internal functions for SGD  ####################

void ffm_fit_sgd(ffm_coef *coef, cs *X, ffm_vector *y, ffm_param *param);

double ffm_predict_sample(ffm_coef *coef, cs *X, int sample_row);

void ffm_fit_sgd_bpr(ffm_coef *coef, cs *A, ffm_matrix *pairs, ffm_param param);

void update_second_order_bpr(cs *A, ffm_matrix *V, double cache_n,
                             double cache_p, double pairs_err, double step_size,
                             double lambda_V, int sample_row_p,
                             int sample_row_n, int f);

ffm_coef *extract_gradient(ffm_coef *coef_t0, ffm_coef *coef_t1,
                           double stepsize);

double l2_penalty(ffm_coef *coef);
#endif /* FAST_MF_H */
