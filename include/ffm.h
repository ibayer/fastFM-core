// Author: Immanuel Bayer
// License: BSD 3 clause

#ifndef FFM_H
#define FFM_H
#include "../externals/CXSparse/Include/cs.h"

#define TASK_CLASSIFICATION 10
#define TASK_REGRESSION 20
#define TASK_RANKING 30

// ############### library interface ####################

typedef struct ffm_param {
  int n_iter;
  int k;
  double init_sigma;
  double init_lambda_w;
  double init_lambda_V;
  int TASK;
  int SOLVER;
  double stepsize;
  int rng_seed;

  int iter_count;
  int ignore_w_0;
  int ignore_w;
  int warm_start;

  int n_hyper_param;
  double *hyper_param;
} ffm_param;

void ffm_predict(double *w_0, double *w, double *V, cs *X, double *y_pred,
                 int k);

void ffm_als_fit(double *w_0, double *w, double *V, cs *X, double *y,
                 ffm_param *param);

void ffm_mcmc_fit_predict(double *w_0, double *w, double *V, cs *X_train,
                          cs *X_test, double *y_train, double *y_pred,
                          ffm_param *param);

void ffm_sgd_fit(double *w_0, double *w, double *V, cs *X, double *y,
                 ffm_param *param);

void ffm_sgd_bpr_fit(double *w_0, double *w, double *V, cs *X, double *pairs,
                     int n_pairs, ffm_param *param);

#endif /* FFM_H */
