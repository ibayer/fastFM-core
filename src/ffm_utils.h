// Author: Immanuel Bayer
// License: BSD 3 clause

#ifndef FFM_UTILS_H
#define FFM_UTILS_H

#include "fast_fm.h"

// ########################### ffm scalar ###################################
double ffm_sigmoid(double x);
double ffm_normal_cdf(double x);
double ffm_normal_pdf(double x);
double ffm_pow_2(double x);

// ########################### ffm_vector ####################################
double ffm_vector_variance(ffm_vector *x);
double ffm_vector_mean(ffm_vector *x);
void ffm_vector_normal_cdf(ffm_vector *x);
int ffm_vector_free(ffm_vector *a);
ffm_vector *ffm_vector_alloc(int size);
ffm_vector *ffm_vector_calloc(int size);
int ffm_vector_memcpy(ffm_vector *a, ffm_vector const *b);
int ffm_vector_add(ffm_vector *a, ffm_vector const *b);
int ffm_vector_sub(ffm_vector *a, ffm_vector const *b);
int ffm_vector_scale(ffm_vector *a, double b);
int ffm_vector_mul(ffm_vector *a, ffm_vector const *b);
double ffm_vector_sum(ffm_vector *a);
void ffm_vector_set_all(ffm_vector *a, double b);
void ffm_vector_set(ffm_vector *a, int i, double alpha);
double ffm_vector_get(ffm_vector *a, int i);
void ffm_vector_add_constant(ffm_vector *a, double alpha);
void ffm_vector_printf(ffm_vector *a);
void ffm_vector_sort(ffm_vector *y);
void ffm_vector_make_labels(ffm_vector *y);  // use media as class boundary
double ffm_vector_median(ffm_vector *y);
double ffm_vector_accuracy(ffm_vector *y_true, ffm_vector *y_pred);
ffm_matrix *ffm_vector_to_rank_comparision(ffm_vector *y);
bool ffm_vector_contains(ffm_vector *y, double value, int cutoff);
ffm_vector *ffm_vector_get_order(ffm_vector *y);
double ffm_vector_kendall_tau(ffm_vector *a, ffm_vector *b);
// index starts at 0
void ffm_vector_update_mean(ffm_vector *mean, int index, ffm_vector *x);
double ffm_vector_mean_squared_error(ffm_vector *y_true, ffm_vector *y_pred);
// ########################### ffm_matrix ####################################
ffm_matrix *ffm_matrix_from_file(char *path);
void ffm_matrix_printf(ffm_matrix *X);
int ffm_matrix_free(ffm_matrix *a);
ffm_matrix *ffm_matrix_alloc(int size0, int size1);
ffm_matrix *ffm_matrix_calloc(int size0, int size1);
double *ffm_matrix_get_row_ptr(ffm_matrix *X, int i);
void ffm_matrix_set(ffm_matrix *X, int i, int j, double a);
double ffm_matrix_get(ffm_matrix *X, int i, int j);
// ###########################  cblas helper #################################

double ffm_blas_ddot(ffm_vector *x, ffm_vector const *y);
// y  <--  alpha*x + y
void ffm_blas_daxpy(double alpha, ffm_vector *x, ffm_vector const *y);
// ||x||_2 / euclidean norm
double ffm_blas_dnrm2(ffm_vector *x);
// dgemv y := alpha*A*x + beta*y,
// --------------- utils ---------------------------------

// read svm_light file, if target is omitted creates
// dummy target vector of zeros
fm_data read_svm_light_file(char *path);

int Cs_write(FILE *f, const cs *A);

void free_fm_data(fm_data *data);

void init_ffm_coef(ffm_coef *coef, ffm_param param);
ffm_coef *alloc_fm_coef(int n_features, int k, int ignore_w);

void free_ffm_coef(ffm_coef *coef);

double ffm_r2_score(ffm_vector *y_true, ffm_vector *y_pred);

double ffm_average_precision_at_cutoff(ffm_vector *y_true, ffm_vector *y_pred,
                                       int cutoff);

// y = A*x+y
// with A in RowMajor format.
int Cs_row_gaxpy(const cs *A, const double *x, double *y);

/* y = alpha*A[:,j]*x+y */
int Cs_daxpy(const cs *A, int col_index, double alpha, const double *x,
             double *y);

/* y = alpha*A[:,j]+y */
int Cs_scal_apy(const cs *A, int col_index, double alpha, double *y);

/* y = alpha*A[:,j]^2+y */
int Cs_scal_a2py(const cs *A, int col_index, double alpha, double *y);

/* y = X^2.sum(axis=0) */
int Cs_col_norm(const cs *A, ffm_vector *y);

/* y = <A[:,j],y> */
double Cs_ddot(const cs *A, int col_index, double *y);

#endif /* FFM_UTILS_H */
