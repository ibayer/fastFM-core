// Author: Immanuel Bayer
// License: BSD 3 clause

#include <stdio.h>
#include "../include/ffm.h"

cs *create_design_matrix() {
  // X = | 6 1 |
  //     | 2 3 |
  //     | 3 0 |
  //     | 6 1 |
  //     | 4 5 |
  int m = 5;
  int n = 2;
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
  cs_spfree(X_t);
  return X_csc;
}

void sgd_bpr_example() {
  printf("### SGD BPR example ###\n");

  // int n_features = 2;
  int n_samples = 5;
  int k = 2;  // # of hiden variables per feature

  // double y[] = {298, 266, 29, 298, 848};
  double y[] = {2, 1, 0, 3, 4};
  double pairs[10][2] = {{0, 1},
                         {0, 2},
                         {3, 0},
                         {4, 0},
                         {1, 2},
                         {3, 1},
                         {4, 1},
                         {3, 2},
                         {4, 2},
                         {4, 3}};
  int n_pairs = 10;
  cs *X = create_design_matrix();
  cs *X_t = cs_transpose(X, 1);

  ffm_param param = {.n_iter = 2000,
                     .k = k,
                     .init_sigma = .01,
                     .init_lambda_w = .5,
                     .init_lambda_V = .5,
                     .stepsize = 0.002,
                     .TASK = TASK_RANKING};
  // allocate fm parameter
  double w_0 = 0;
  double w[2];
  double V[2 * 2];  // k * n_features
  // regularizatio
  ffm_sgd_bpr_fit(&w_0, &w[0], &V[0], X_t, &pairs[0][0], n_pairs, &param);

  double y_pred[5];  // allocate space for the predictions
  ffm_predict(&w_0, &w[0], &V[0], X, y_pred, k);

  printf("y_true: [");
  for (int i = 0; i < n_samples; i++) printf(" %.2f,", y[i]);
  printf("]\n");
  printf("y_pred: [");
  for (int i = 0; i < n_samples; i++) printf(" %.2f,", y_pred[i]);
  printf("]\n\n");
}

int main(void) { sgd_bpr_example(); }
