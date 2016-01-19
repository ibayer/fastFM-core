#include "fast_fm.h"
#include <glib.h>

void test_ffm_vector_mean(void) {
  double data[] = {1, 2, -3, 4, 5};
  ffm_vector x = {.data = data, .size = 5};
  double mean = ffm_vector_mean(&x);
  double ref_mean = 1.8;
  g_assert_cmpfloat(fabs(ref_mean - mean), <=, 1e-15);
}

void test_ffm_vector_variance(void) {
  double data[] = {-1, 2, 3, 4, 5};
  ffm_vector x = {.data = data, .size = 5};
  double var = ffm_vector_variance(&x);
  double ref_var = 4.24;
  g_assert_cmpfloat(fabs(ref_var - var), <=, 1e-15);
}

void test_ffm_normal_cdf(void) {
  double x[] = {0.1, 1.2, 2.3, -1.1, -3.3, -7.7};
  double Phi[] = {.539827837277028981,  .884930329778291731,
                  .989275889978324194,  .1356660609463826751,
                  .0004834241423837772, .0000000000000068033};
  for (int i = 0; i < 6; i++)
    g_assert_cmpfloat(fabs(ffm_normal_cdf(x[i]) - Phi[i]), <=, 1e-15);
}
void test_ffm_normal_pdf(void) {
  double x[] = {0.1, 1.2, 2.3, -1.1, -3.3, -7.7};
  double phi[] = {3.96952547477011808e-01, 1.94186054983212952e-01,
                  2.83270377416011861e-02, 2.17852177032550526e-01,
                  1.72256893905368123e-03, 5.32414837225295172e-14};
  for (int i = 0; i < 6; i++)
    g_assert_cmpfloat(fabs(ffm_normal_pdf(x[i]) - phi[i]), <=, 1e-15);
}
void test_ffm_vector_update_mean(void) {
  double true_mean[] = {5, -100, 1};
  double update0[] = {5, 100, 10};
  double update1[] = {7, 0, -7};
  double update2[] = {3, -400, 0};
  ffm_vector y_true_mean = {.data = true_mean, .size = 3};
  ffm_vector y_update0 = {.data = update0, .size = 3};
  ffm_vector y_update1 = {.data = update1, .size = 3};
  ffm_vector y_update2 = {.data = update2, .size = 3};
  ffm_vector *y_running_mean = ffm_vector_calloc(3);

  ffm_vector_update_mean(y_running_mean, 0, &y_update0);
  ffm_vector_update_mean(y_running_mean, 1, &y_update1);
  ffm_vector_update_mean(y_running_mean, 2, &y_update2);

  for (int i = 0; i < 3; i++)
    g_assert_cmpfloat(ffm_vector_get(y_running_mean, i), ==,
                      ffm_vector_get(&y_true_mean, i));
  ffm_vector_free(y_running_mean);
}

void test_ffm_vector_kendall_tau(void) {
  double order[] = {1, 2, 3, 4, 5};
  double order_wrong[] = {5, 3, 4, 2, 1};
  double order_inv[] = {5, 4, 3, 2, 1};
  ffm_vector y_order = {.data = order, .size = 5};
  ffm_vector y_inv = {.data = order_inv, .size = 5};
  ffm_vector y_wrong = {.data = order_wrong, .size = 5};

  g_assert_cmpfloat(ffm_vector_kendall_tau(&y_order, &y_order), ==, 1);
  g_assert_cmpfloat(ffm_vector_kendall_tau(&y_order, &y_inv), ==, -1);
  g_assert_cmpfloat(ffm_vector_kendall_tau(&y_order, &y_wrong), !=, -1);
}

void test_ffm_vector_get_order(void) {
  double values[] = {1, 2, 5.5, 20, 3};
  double order[] = {0, 1, 4, 2, 3};
  ffm_vector y_values = {.data = values, .size = 5};
  ffm_vector *y_order = ffm_vector_get_order(&y_values);
  for (int i = 0; i < y_order->size; i++)
    assert(ffm_vector_get(y_order, i) == order[i]);
}

void test_ffm_vector_to_rank_comparision(void) {
  double y_inc[] = {1, 2, 3};
  ffm_vector y = {.data = y_inc, .size = 3};
  ffm_matrix *comparison_inc = ffm_vector_to_rank_comparision(&y);
  for (int i = 0; i < comparison_inc->size0; i++)
    assert(ffm_matrix_get(comparison_inc, i, 0) >
           ffm_matrix_get(comparison_inc, i, 1));

  double y_dec[] = {3, 2, 1};
  y.data = y_dec;
  ffm_matrix *comparison_dec = ffm_vector_to_rank_comparision(&y);
  for (int i = 0; i < comparison_dec->size0; i++)
    assert(ffm_matrix_get(comparison_dec, i, 0) <
           ffm_matrix_get(comparison_dec, i, 1));
  ffm_matrix_free(comparison_dec);
  ffm_matrix_free(comparison_inc);
}

void test_ffm_average_precision_at_cutoff(void) {
  double org_d[] = {1, 2, 3, 4, 5};
  ffm_vector org = {.data = org_d, .size = 5};
  double pred_d[] = {6, 4, 7, 1, 2};
  ffm_vector pred = {.data = pred_d, .size = 5};
  g_assert_cmpfloat(ffm_average_precision_at_cutoff(&org, &pred, 2), ==, 0.25);
  double pred_d2[] = {1, 1, 1, 1, 1, 1};
  pred.data = pred_d2;
  g_assert_cmpfloat(ffm_average_precision_at_cutoff(&org, &pred, 5), ==, 0.2);
  double pred_d3[] = {1, 2, 3, 1, 1, 1};
  pred.data = pred_d3;
  g_assert_cmpfloat(ffm_average_precision_at_cutoff(&org, &pred, 3), ==, 1.0);
}

void test_ffm_sigmoid(void) {
  g_assert_cmpfloat(ffm_sigmoid(-100000), <, 1e-16);
  g_assert_cmpfloat(fabs(ffm_sigmoid(100000) - 1), <, 1e-16);
  g_assert_cmpfloat(fabs(ffm_sigmoid(0) - .5), <, 1e-16);
}

void test_ffm_vector_accuracy(void) {
  double labels[] = {1, 1, -1, -1};
  double labels_wrong[] = {-1, -1, 1, 1};
  double labels_half[] = {-1, 1, 1, -1};
  double labels_probas[] = {0.55, 0.9, .01, .48};
  ffm_vector org = {.data = labels, .size = 4};
  ffm_vector wrong = {.data = labels_wrong, .size = 4};
  ffm_vector half = {.data = labels_half, .size = 4};
  ffm_vector probas = {.data = labels_probas, .size = 4};

  g_assert_cmpfloat(ffm_vector_accuracy(&org, &wrong), ==, 0);
  g_assert_cmpfloat(ffm_vector_accuracy(&org, &org), ==, 1);
  g_assert_cmpfloat(ffm_vector_accuracy(&org, &half), ==, 0.5);
  g_assert_cmpfloat(ffm_vector_accuracy(&org, &probas), ==, 1);
}

void test_ffm_vector_median(void) {
  double values[] = {5.1, 10.0, 1.1, 2.2, -2};
  ffm_vector v = {.data = values, .size = 5};
  double median = ffm_vector_median(&v);
  assert(median == 2.2);

  double values_even[] = {5.1, 10.0, 1.1, -2};
  ffm_vector v_even = {.data = values_even, .size = 4};
  double median_even = ffm_vector_median(&v_even);
  g_assert_cmpfloat(fabs(median_even - 3.1), <, 1e-9);
}
void test_ffm_vector_make_labels(void) {
  double values[] = {5.1, 10.0, 1.1, -2};
  ffm_vector v = {.data = values, .size = 4};
  double labels[] = {1, 1, -1, -1};
  ffm_vector_make_labels(&v);
  for (int i = 0; i < v.size; i++)
    g_assert_cmpfloat(labels[i], ==, ffm_vector_get(&v, i));
}
void test_ffm_vector_sort(void) {
  double values[] = {5.1, 10.0, 1.1, 2.2, -2};
  ffm_vector v = {.data = values, .size = 5};
  double values_sorted[] = {-2, 1.1, 2.2, 5.1, 10.0};
  ffm_vector_sort(&v);
  for (int i = 0; i < 5; i++) assert(ffm_vector_get(&v, i) == values_sorted[i]);
}

void test_ffm_r2_score(void) {
  double y_true_d[] = {3, -.5, 2, 7};
  ffm_vector y_true = {.data = y_true_d, .size = 4};
  double y_pred_d[] = {2.5, 0, 2, 8};
  ffm_vector y_pred = {.data = y_pred_d, .size = 4};

  g_assert_cmpfloat(fabs(0.94860 - ffm_r2_score(&y_true, &y_pred)), <, 1e-3);
}

void test_ffm_vector_sum(void) {
  ffm_vector *v = ffm_vector_alloc(5);
  ffm_vector_set_all(v, 2);

  g_assert_cmpfloat(10, ==, ffm_vector_sum(v));
}

void test_Cs_daxpy(void) {
  // init X
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
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */

  ffm_vector *v = ffm_vector_alloc(5);
  ffm_vector_set_all(v, 10);
  ffm_vector_set(v, 4, 0);

  ffm_vector *res = ffm_vector_calloc(5);

  // test multiplying second column
  Cs_daxpy(X_csc, 1, .5, v->data, res->data);
  g_assert_cmpfloat(5, ==, ffm_vector_get(res, 0));
  g_assert_cmpfloat(0, ==, ffm_vector_get(res, 4));

  // test multiplying first column
  ffm_vector_set_all(res, 0);
  Cs_daxpy(X_csc, 0, 3, v->data, res->data);
  g_assert_cmpfloat(180, ==, ffm_vector_get(res, 0));
  g_assert_cmpfloat(0, ==, ffm_vector_get(res, 4));
}

void test_Cs_row_gaxpy(void) {
  // init X
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
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs *X_csr = cs_transpose(X_csc, 1);

  ffm_vector *v = ffm_vector_alloc(2);
  ffm_vector_set(v, 0, 2);
  ffm_vector_set(v, 1, 3);

  ffm_vector *res = ffm_vector_calloc(5);
  ffm_vector_set(res, 1, 3);

  ffm_vector *res_row = ffm_vector_calloc(5);
  ffm_vector_set(res_row, 1, 3);

  Cs_row_gaxpy(X_csr, v->data, res->data);
  cs_gaxpy(X_csc, v->data, res_row->data);

  g_assert_cmpfloat(ffm_vector_get(res_row, 0), ==, ffm_vector_get(res, 0));
  g_assert_cmpfloat(ffm_vector_get(res_row, 1), ==, ffm_vector_get(res, 1));

  cs_spfree(X_csc);
  cs_spfree(X_csr);
}

void test_Cs_scal_apy(void) {
  // init X
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
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */

  ffm_vector *res = ffm_vector_alloc(5);
  ffm_vector_set_all(res, 1);

  // test multiplying second column
  Cs_scal_apy(X_csc, 0, 2.0, res->data);
  g_assert_cmpfloat(13, ==, ffm_vector_get(res, 0));
  g_assert_cmpfloat(9, ==, ffm_vector_get(res, 4));
}

void test_Cs_scal_a2py(void) {
  // init X
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
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */

  ffm_vector *res = ffm_vector_alloc(5);
  ffm_vector_set_all(res, 1);

  // test multiplying second column
  Cs_scal_a2py(X_csc, 0, 2.0, res->data);
  g_assert_cmpfloat(73, ==, ffm_vector_get(res, 0));
  g_assert_cmpfloat(33, ==, ffm_vector_get(res, 4));
}

void test_Cs_col_norm(void) {
  // init X
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
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */

  ffm_vector *res = ffm_vector_calloc(n);

  // test multiplying second column
  Cs_col_norm(X_csc, res);
  g_assert_cmpfloat(101, ==, ffm_vector_get(res, 0));
  g_assert_cmpfloat(36, ==, ffm_vector_get(res, 1));
}

void test_Cs_ddot(void) {
  // init X
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
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */

  ffm_vector *y = ffm_vector_calloc(m);
  ffm_vector_set_all(y, 2);

  g_assert_cmpfloat(42, ==, Cs_ddot(X_csc, 0, y->data));
  g_assert_cmpfloat(20, ==, Cs_ddot(X_csc, 1, y->data));
}

void test_Cs_write(void) {
  // init X
  int m = 5;
  int n = 2;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 6);
  cs_entry(X, 0, 1, 1);
  cs_entry(X, 1, 0, 2.22);
  cs_entry(X, 1, 1, 3);
  cs_entry(X, 2, 0, 3.333);
  cs_entry(X, 3, 0, 6);
  cs_entry(X, 3, 1, 1);
  cs_entry(X, 4, 0, 4.4444);
  cs_entry(X, 4, 1, 5);
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */

  FILE *f = fopen("data/x_dummy.txt", "r");
  Cs_write(f, X);
  fclose(f);

  f = fopen("data/x_dummy.txt", "r");
  cs *X_from_file = cs_load(f);
  fclose(f);

  // test that matrix is correct
  for (int j = 0; j < X->nz; j++)
    g_assert_cmpfloat(X_from_file->x[j], ==, X->x[j]);
}

void test_read_ffm_matrix_from_file(void) {
  ffm_matrix *X = ffm_matrix_from_file("data/matrix");
  assert(ffm_matrix_get(X, 0, 0) == 1);
  assert(ffm_matrix_get(X, 1, 0) == 2);
  assert(ffm_matrix_get(X, 2, 0) == 4);

  assert(ffm_matrix_get(X, 0, 1) == 2);
  assert(ffm_matrix_get(X, 1, 1) == 3);
  assert(ffm_matrix_get(X, 2, 1) == 5);
  assert(X->size0 == 3);
  assert(X->size1 == 2);
}

void test_read_svm_light_file(void) {
  int m = 5;
  int n = 2;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 6);
  cs_entry(X, 0, 1, 1);
  cs_entry(X, 1, 0, 2);
  cs_entry(X, 1, 1, 3);
  //cs_entry(X, 2, 0, 3);
  cs_entry(X, 3, 0, 6);
  cs_entry(X, 3, 1, 1);
  cs_entry(X, 4, 0, 4);
  cs_entry(X, 4, 1, 5);
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs_spfree(X);

  fm_data data = read_svm_light_file("data/svm_light_dummy");
  ffm_vector *y = data.y;

  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y, 0));
  g_assert_cmpfloat(266.0, ==, ffm_vector_get(y, 1));
  g_assert_cmpfloat(29.0, ==, ffm_vector_get(y, 2));
  g_assert_cmpfloat(298.0, ==, ffm_vector_get(y, 3));
  g_assert_cmpfloat(848.0, ==, ffm_vector_get(y, 4));

  int size = sizeof(double) * X_csc->nzmax;
  assert(!memcmp(X_csc->x, data.X->x, size));
}

void test_read_svm_light_file_without_target(void) {
  int m = 5;
  int n = 2;
  cs *X = cs_spalloc(m, n, m * n, 1, 1); /* create triplet identity matrix */
  cs_entry(X, 0, 0, 6);
  cs_entry(X, 0, 1, 1);
  cs_entry(X, 1, 0, 2);
  cs_entry(X, 1, 1, 3);
  cs_entry(X, 2, 0, 3);
  //cs_entry(X, 3, 0, 6);
  //cs_entry(X, 3, 1, 1);
  cs_entry(X, 4, 0, 4);
  cs_entry(X, 4, 1, 5);
  // printf ("X:\n") ; cs_print (X, 0) ; /* print A */
  cs *X_csc = cs_compress(X); /* A = compressed-column form of T */
  cs_spfree(X);

  fm_data data = read_svm_light_file("data/svm_light_dummy_witout_target");

  // check if dummy zero target exists
  for (int i = 0; i < data.y->size; i++) assert(data.y->data[i] == 0);

  int size = sizeof(double) * X_csc->nzmax;
  assert(!memcmp(X_csc->x, data.X->x, size));
}

void test_ffm_vector_mean_square_error(void) {
  int size = 10;
  ffm_vector *a = ffm_vector_alloc(size);
  for (int i = 0; i < size; i++) a->data[i] = i;
  ffm_vector *b = ffm_vector_alloc(size);
  ffm_vector_memcpy(b, a);
  g_assert_cmpfloat(ffm_vector_mean_squared_error(a, b), ==, 0);
  ffm_vector_free_all(a, b);
}

void test_ffm_vector_functions(void) {
  int size = 5;
  ffm_vector *a = ffm_vector_alloc(size);
  ffm_vector *b = ffm_vector_alloc(size);

  ffm_vector_set_all(a, 1.0);
  ffm_vector_scale(a, 2.0);
  ffm_vector_memcpy(b, a);
  for (int i = 0; i < size; i++) g_assert_cmpfloat(a->data[i], ==, b->data[i]);

  ffm_vector_add(a, b);
  for (int i = 0; i < size; i++)
    g_assert_cmpfloat(a->data[i], ==, 2.0 * b->data[i]);
  ffm_vector_sub(a, b);
  for (int i = 0; i < size; i++) g_assert_cmpfloat(a->data[i], ==, b->data[i]);

  ffm_vector_set_all(b, 2.0);
  ffm_vector_mul(a, b);
  for (int i = 0; i < size; i++)
    g_assert_cmpfloat(a->data[i], ==, 2.0 * b->data[i]);
  ffm_vector_free_all(a, b);
}

void test_ffm_matrix_functions(void) {
  ffm_matrix *X = ffm_matrix_calloc(3, 4);
  for (int i = 0; i < X->size0 * X->size1; i++) X->data[i] = i;
  g_assert_cmpfloat(ffm_matrix_get(X, 0, 2), ==, 2);
  g_assert_cmpfloat(ffm_matrix_get(X, 2, 2), ==, 10);

  ffm_matrix *Y = ffm_matrix_calloc(3, 4);
  double count = 0;
  for (int i = 0; i < Y->size0; i++)
    for (int j = 0; j < Y->size1; j++) {
      ffm_matrix_set(Y, i, j, count);
      count++;
    }

  for (int i = 0; i < Y->size0; i++)
    for (int j = 0; j < Y->size1; j++)
      g_assert_cmpfloat(ffm_matrix_get(Y, i, j), ==, ffm_matrix_get(X, i, j));

  g_assert_cmpfloat(*ffm_matrix_get_row_ptr(X, 0), ==, 0);
  g_assert_cmpfloat(*ffm_matrix_get_row_ptr(X, 1), ==, 4);
  g_assert_cmpfloat(*ffm_matrix_get_row_ptr(X, 2), ==, 8);

  ffm_matrix_set(X, 1, 1, 3.21);
  g_assert_cmpfloat(ffm_matrix_get(X, 1, 1), ==, 3.21);
  ffm_matrix_set(X, 2, 3, 43.21);
  g_assert_cmpfloat(ffm_matrix_get(X, 2, 3), ==, 43.21);
}

void test_ffm_blas(void) {
  int size = 6;
  ffm_vector *a = ffm_vector_calloc(size);
  ffm_vector *b = ffm_vector_calloc(size);

  ffm_vector_set_all(a, 1.5);
  ffm_vector_set_all(b, 2.0);

  g_assert_cmpfloat(ffm_blas_ddot(a, b), ==, 18);
  ffm_blas_daxpy(2.0, a, b);
  for (int i = 0; i < size; i++) g_assert_cmpfloat(b->data[i], ==, 5);

  ffm_vector_set_all(a, 1.0);
  g_assert_cmpfloat(ffm_blas_dnrm2(a), ==, sqrt(6.0));
}

int main(int argc, char **argv) {
  g_test_init(&argc, &argv, NULL);
  g_test_add_func("/utils/ffm_vector/ mean", test_ffm_vector_mean);
  g_test_add_func("/utils/ffm_vector/ var", test_ffm_vector_variance);
  g_test_add_func("/utils/ffm/ normal cdf", test_ffm_normal_cdf);
  g_test_add_func("/utils/ffm/ normal pdf", test_ffm_normal_pdf);
  g_test_add_func("/utils/ffm_matrix/ operations", test_ffm_matrix_functions);
  g_test_add_func("/utils/ffm_vector/ blas", test_ffm_blas);
  g_test_add_func("/utils/ffm_vector/ sort", test_ffm_vector_sort);
  g_test_add_func("/utils/ffm_vector/ median", test_ffm_vector_median);
  g_test_add_func("/utils/ffm_vector/ make labels",
                  test_ffm_vector_make_labels);
  g_test_add_func("/utils/ffm_vector/ get order", test_ffm_vector_get_order);
  g_test_add_func("/utils/ffm_vector/ rank comparison",
                  test_ffm_vector_to_rank_comparision);
  g_test_add_func("/utils/ffm_vector/ operations", test_ffm_vector_functions);
  g_test_add_func("/utils/vector sum", test_ffm_vector_sum);
  g_test_add_func("/utils/ffm_vector/ mean_square_error",
                  test_ffm_vector_mean_square_error);
  g_test_add_func("/utils/ffm_vector/ accuracy", test_ffm_vector_accuracy);
  g_test_add_func("/utils/ffm_vector/ average precision at cutoff",
                  test_ffm_average_precision_at_cutoff);
  g_test_add_func("/utils/ffm_vector/ kendall tau",
                  test_ffm_vector_kendall_tau);
  g_test_add_func("/utils/ffm_vector/ update mean",
                  test_ffm_vector_update_mean);
  g_test_add_func("/utils/ffm_sigmoid ", test_ffm_sigmoid);
  g_test_add_func("/utils/cs daxpy", test_Cs_daxpy);
  g_test_add_func("/utils/cs gaxpy row", test_Cs_row_gaxpy);
  g_test_add_func("/utils/cs scal_apy", test_Cs_scal_apy);
  g_test_add_func("/utils/cs scal_a2py", test_Cs_scal_a2py);
  g_test_add_func("/utils/cs col_norm", test_Cs_col_norm);
  g_test_add_func("/utils/cs ddot", test_Cs_ddot);
  g_test_add_func("/utils/ffm_r2_score", test_ffm_r2_score);
  g_test_add_func("/utils/Cs_write", test_Cs_write);
  g_test_add_func("/utils/read ffm_matrix_from_file",
                  test_read_ffm_matrix_from_file);
  g_test_add_func("/utils/read svm_light file", test_read_svm_light_file);
  g_test_add_func("/utils/read svm_light file w.o. target",
                  test_read_svm_light_file_without_target);

  return g_test_run();
}
