// Author: Immanuel Bayer
// License: BSD 3 clause

#include "fast_fm.h"
#include "kvec.h"
#include <unistd.h>

// ########################### ffm scalar ###################################
double ffm_sigmoid(double x) {
  if (fabs(x) > 36) return x > 0 ? 1 : 0;
  return 1.0 / (1.0 + exp(-x));
}

double ffm_pow_2(double x) { return x * x; }

double ffm_normal_pdf(double x) { return exp(-(x * x) / 2.0) / sqrt(M_PI * 2); }
// source: Evaluating the Normal Distribution - Marsaglia
double ffm_normal_cdf(double x) {
  if (x > 8) return 1;
  if (x < -8) return 0;
  long double s = x, t = 0, b = x, q = x * x, i = 1;
  while (s != t) s = (t = s) + (b *= q / (i += 2));
  return .5 + s * exp(-.5 * q - .91893853320467274178L);
}
// ########################### ffm_vector ##################################

// Algorithms for Computing the Sample Variance: Analysis and Recommendations
// corrected two-pass algoritm (eq. 1.7)
double ffm_vector_variance(ffm_vector *x) {
  double mean = ffm_vector_mean(x);
  double var = 0;
  double correction = 0;
  for (int j = 0; j < x->size; j++) {
    correction += x->data[j] - mean;
    var += (x->data[j] - mean) * (x->data[j] - mean);
  }
  var = var * (1.0 / (double)x->size);
  var -= (1.0 / (double)x->size) * (correction * correction);
  return var;
}

// Algorithms for Computing the Sample Variance: Analysis and Recommendations
// equation 1.3a
double ffm_vector_mean(ffm_vector *x) {
  double mean = x->data[0];
  for (int j = 1; j < x->size; j++)
    mean = mean + (1.0 / (j + 1)) * (x->data[j] - mean);
  return mean;
}

void ffm_vector_normal_cdf(ffm_vector *x) {
  for (int i = 0; i < x->size; i++) x->data[i] = ffm_normal_cdf(x->data[i]);
}
void ffm_vector_update_mean(ffm_vector *mean, int index, ffm_vector *x) {
  assert(mean->size == x->size && "vectors have different length");
  if (index == 0) {
    ffm_vector_memcpy(mean, x);
    return;
  }
  double weight = 1.0 / (index + 1.0);
  int N = mean->size;
  for (int i = 0; i < N; i++)
    mean->data[i] = mean->data[i] + weight * (x->data[i] - mean->data[i]);
}

double ffm_vector_kendall_tau(ffm_vector *a, ffm_vector *b) {
  assert(a->size == b->size && "vectors have different length");
  double N = b->size;
  double n_concordant = 0;
  double n_disconcordant = 0;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++) {
      // concordant
      if (ffm_vector_get(a, i) > ffm_vector_get(a, j) &&
          ffm_vector_get(b, i) > ffm_vector_get(b, j))
        n_concordant++;
      if (ffm_vector_get(a, i) < ffm_vector_get(a, j) &&
          ffm_vector_get(b, i) < ffm_vector_get(b, j))
        n_concordant++;
      // discordant
      if (ffm_vector_get(a, i) > ffm_vector_get(a, j) &&
          ffm_vector_get(b, i) < ffm_vector_get(b, j))
        n_disconcordant++;
      if (ffm_vector_get(a, i) < ffm_vector_get(a, j) &&
          ffm_vector_get(b, i) > ffm_vector_get(b, j))
        n_disconcordant++;
    }
  return (n_concordant - n_disconcordant) / (.5 * N * (N - 1));
}

ffm_vector *ffm_vector_get_order(ffm_vector *y) {
  int N = y->size;
  ffm_vector *indices = ffm_vector_calloc(N);
  ffm_vector *a = ffm_vector_calloc(N);
  ffm_vector_memcpy(a, y);
  for (int i = 0; i < N; i++) ffm_vector_set(indices, i, i);
  // use selection sort on tmp value array and indices array
  for (int i = 0; i < N; i++) {
    int min = i;
    for (int j = i + 1; j < N; j++)
      if (ffm_vector_get(a, j) < ffm_vector_get(a, min)) min = j;
    // exchange values
    double tmp_a = ffm_vector_get(a, i);
    ffm_vector_set(a, i, ffm_vector_get(a, min));
    ffm_vector_set(a, min, tmp_a);
    // exchange indices array
    double tmp_int = ffm_vector_get(indices, i);
    ffm_vector_set(indices, i, ffm_vector_get(indices, min));
    ffm_vector_set(indices, min, tmp_int);
  }
  ffm_vector_free(a);
  return indices;
}

ffm_matrix *ffm_vector_to_rank_comparision(ffm_vector *y) {
  int n_compares = 0;
  for (int i = 0; i < y->size; i++)
    for (int j = i + 1; j < y->size; j++) n_compares++;
  ffm_matrix *compars = ffm_matrix_calloc(n_compares, 2);
  int comp_row = 0;
  for (int i = 0; i < y->size; i++)
    for (int j = i + 1; j < y->size; j++) {
      if (ffm_vector_get(y, i) > ffm_vector_get(y, j)) {
        ffm_matrix_set(compars, comp_row, 0, i);
        ffm_matrix_set(compars, comp_row, 1, j);
      } else {
        ffm_matrix_set(compars, comp_row, 0, j);
        ffm_matrix_set(compars, comp_row, 1, i);
      }
      comp_row++;
    }
  return compars;
}
// if cutoff =-1 ignore cutoff
double ffm_average_precision_at_cutoff(ffm_vector *y_true, ffm_vector *y_pred,
                                       int cutoff) {
  double score = 0;
  double num_hits = 0.0;
  if (cutoff == -1) cutoff = y_true->size;
  for (int i = 0; i < y_pred->size; i++) {
    if (i >= cutoff) break;
    double p = ffm_vector_get(y_pred, i);
    bool in_true = ffm_vector_contains(y_true, p, -1);
    bool already_found = ffm_vector_contains(y_pred, p, i);
    if (in_true && !already_found) {
      num_hits += 1.0;
      score += num_hits / (i + 1.0);
    }
  }
  double dev = y_true->size < cutoff ? y_true->size : cutoff;
  return score / dev;
}

// if cutoff =-1 ignore cutoff
bool ffm_vector_contains(ffm_vector *y, double value, int cutoff) {
  int stop = y->size < cutoff ? y->size : cutoff;
  if (cutoff == -1) stop = y->size;
  for (int i = 0; i < stop; i++)
    if (ffm_vector_get(y, i) == value) return true;
  return false;
}

double ffm_vector_accuracy(ffm_vector *y_true, ffm_vector *y_pred) {
  assert(y_true->size == y_pred->size && "vectors have different length");
  double acc = 0;
  for (int i = 0; i < y_true->size; i++) {
    if (ffm_vector_get(y_true, i) >= .0 && ffm_vector_get(y_pred, i) >= 0.5)
      acc++;
    else if (ffm_vector_get(y_true, i) < .0 && ffm_vector_get(y_pred, i) < 0.5)
      acc++;
  }
  if (acc == 0) return 0;
  return acc / (double)(y_true->size);
}
double ffm_vector_median(ffm_vector *y) {
  ffm_vector *cp = ffm_vector_alloc(y->size);
  ffm_vector_memcpy(cp, y);
  ffm_vector_sort(cp);
  double median = NAN;

  if (y->size % 2 == 0)
    median = (ffm_vector_get(cp, y->size / 2) +
              ffm_vector_get(cp, (y->size / 2) - 1)) /
             2.0;
  else
    median = ffm_vector_get(cp, y->size / 2);
  ffm_vector_free(cp);
  return median;
}
void ffm_vector_make_labels(ffm_vector *y) {
  double median = ffm_vector_median(y);
  for (int i = 0; i < y->size; i++)
    if (ffm_vector_get(y, i) > median)
      ffm_vector_set(y, i, 1);
    else
      ffm_vector_set(y, i, -1);
}
int __cmpfunc_for_ffm_vector_sort(const void *a, const void *b) {
  return (*(double *)a - *(double *)b);
}
void ffm_vector_sort(ffm_vector *y) {
  qsort(y->data, y->size, sizeof(double), __cmpfunc_for_ffm_vector_sort);
}

double ffm_vector_mean_squared_error(ffm_vector *a, ffm_vector *b) {
  assert(a->size == b->size && "vectors have different length");
  double sum = 0;
  for (int i = 0; i < a->size; i++) {
    double tmp = (a->data[i] - b->data[i]);
    sum += tmp * tmp;
  }
  if (sum != 0) return sqrt(sum / a->size);
  return 0;
}
int ffm_vector_free(ffm_vector *a) {
  if (a->owner) {
    free(a->data);
    free(a);
    return 0;
  } else
    return 1;
}
ffm_vector *ffm_vector_alloc(int size) {
  assert(size > 0 && "can't allocate vector with size <= 0");
  struct ffm_vector *x = malloc(sizeof *x);
  double *ptr;
  ptr = malloc(size * sizeof(double));
  x->data = ptr;
  x->owner = 1;
  x->size = size;
  return x;
}

ffm_vector *ffm_vector_calloc(int size) {
  struct ffm_vector *x = malloc(sizeof *x);
  double *ptr;
  ptr = calloc(size, sizeof(double));
  x->data = ptr;
  x->owner = 1;
  x->size = size;
  return x;
}
// copy values from b to a
int ffm_vector_memcpy(ffm_vector *a, ffm_vector const *b) {
  assert(a->size == b->size && "vectors have different length");
  memcpy(a->data, b->data, a->size * sizeof(double));
  return 1;
}
// a = a +b
int ffm_vector_add(ffm_vector *a, ffm_vector const *b) {
  assert(a->size == b->size && "vectors have different length");
  for (int i = 0; i < a->size; i++) a->data[i] = a->data[i] + b->data[i];
  return 1;
}
// a = a - b
int ffm_vector_sub(ffm_vector *a, ffm_vector const *b) {
  assert(a->size == b->size && "vectors have different length");
  for (int i = 0; i < a->size; i++) a->data[i] = a->data[i] - b->data[i];
  return 1;
}
// a = a * alpha
int ffm_vector_scale(ffm_vector *a, double b) {
  for (int i = 0; i < a->size; i++) a->data[i] = a->data[i] * b;
  return 1;
}
// a = a * b
int ffm_vector_mul(ffm_vector *a, ffm_vector const *b) {
  assert(a->size == b->size && "vectors have different length");
  for (int i = 0; i < a->size; i++) a->data[i] = a->data[i] * b->data[i];
  return 1;
}
// a = alpha
void ffm_vector_set_all(ffm_vector *a, double b) {
  for (int i = 0; i < a->size; i++) a->data[i] = b;
}
double ffm_vector_sum(ffm_vector *a) {
  double tmp = 0;
  for (int i = 0; i < a->size; i++) tmp += a->data[i];
  return tmp;
}
void ffm_vector_set(ffm_vector *a, int i, double alpha) { a->data[i] = alpha; }
double ffm_vector_get(ffm_vector *a, int i) { return a->data[i]; }
void ffm_vector_add_constant(ffm_vector *a, double alpha) {
  for (int i = 0; i < a->size; i++) a->data[i] = a->data[i] + alpha;
}

void ffm_vector_printf(ffm_vector *a) {
  for (int i = 0; i < a->size; i++) printf("%f, ", a->data[i]);
  printf("\n");
}

// ########################### ffm_matrix ##################################
void ffm_matrix_printf(ffm_matrix *X) {
  for (int i = 0; i < X->size0; i++) {
    for (int j = 0; j < X->size1; j++)
      printf("%f, ", X->data[i * X->size1 + j]);
    printf("\n");
  }
}
int ffm_matrix_free(ffm_matrix *a) {
  if (a->owner) free(a->data);
  return 1;
}
ffm_matrix *ffm_matrix_alloc(int size0, int size1) {
  struct ffm_matrix *x = malloc(sizeof *x);
  double *ptr;
  ptr = malloc(size0 * size1 * sizeof(double));
  x->data = ptr;
  x->owner = 1;
  x->size0 = size0;
  x->size1 = size1;
  return x;
}

ffm_matrix *ffm_matrix_calloc(int size0, int size1) {
  assert(size0 > 0 && "can't allocate matrix with size0 <= 0");
  assert(size1 > 0 && "can't allocate matrix with size1 <= 0");
  struct ffm_matrix *x = malloc(sizeof *x);
  double *ptr;
  ptr = calloc(size0 * size1, sizeof(double));
  x->data = ptr;
  x->owner = 1;
  x->size0 = size0;
  x->size1 = size1;
  return x;
}
double *ffm_matrix_get_row_ptr(ffm_matrix *X, int i) {
  return X->data + i * X->size1;
}
void ffm_matrix_set(ffm_matrix *X, int i, int j, double a) {
  assert(i < X->size0 && "index out of range");
  assert(j < X->size1 && "index out of range");
  X->data[i * X->size1 + j] = a;
}
double ffm_matrix_get(ffm_matrix *X, int i, int j) {
  assert(i < X->size0 && "index out of range");
  assert(j < X->size1 && "index out of range");
  return X->data[i * X->size1 + j];
}
// ###########################  cblas helper #################################
double ffm_blas_ddot(ffm_vector *x, ffm_vector const *y) {
  assert(x->size == y->size && "vectors have different length");
  return cblas_ddot(x->size, x->data, 1, y->data, 1);
}
void ffm_blas_daxpy(double alpha, ffm_vector *x, ffm_vector const *y) {
  assert(x->size == y->size && "vectors have different length");
  return cblas_daxpy(x->size, alpha, x->data, 1, y->data, 1);
}
double ffm_blas_dnrm2(ffm_vector *x) {
  return cblas_dnrm2(x->size, x->data, 1);
}

ffm_matrix *ffm_matrix_from_file(char *path) {
  assert(access(path, F_OK) != -1 && "file doesn't exist");
  FILE *fp = fopen(path, "r");

  // get number of rows
  size_t len = 1;
  char *line = NULL;
  ssize_t read;
  unsigned long row_count = 0;
  while ((read = getline(&line, &len, fp)) != -1) row_count++;
  rewind(fp);

  // get number of columns
  unsigned long column_count = 1;
  char c = fgetc(fp);
  while (c != '\n') {
    if (c == ' ') column_count++;
    c = fgetc(fp);
  }
  rewind(fp);

  ffm_matrix *X = ffm_matrix_calloc(row_count, column_count);

  int current_row = 0;
  while ((read = getline(&line, &len, fp)) != -1) {
    char *end_str;
    char *token = strtok_r(line, " ", &end_str);
    int current_col = 0;
    // loop over features in current row
    while (token != NULL) {
      ffm_matrix_set(X, current_row, current_col, atof(token));
      current_col++;
      token = strtok_r(NULL, " ", &end_str);
    }
    current_row++;
  }
  return X;
}
// ######## fm helper ############

fm_data read_svm_light_file(char *path) {
  assert(access(path, F_OK) != -1 && "file doesn't exist");
  FILE *fp = fopen(path, "r");

  cs *T = cs_spalloc(0, 0, 1, 1, 1); /* allocate result */

  char *line = NULL;
  size_t len = 1;
  ssize_t read;
  int line_nr = 0;

  // check if file contains target
  bool hasTarget = true;
  char c = fgetc(fp);
  while (c != ' ') {
    if (c == ':') hasTarget = false;
    c = fgetc(fp);
  }
  rewind(fp);

  /* We create a new array to store double values.
     We don't want it zero-terminated or cleared to 0's. */
  double target;
  double dummy_target = 0;

  kvec_t(double)array;
  kv_init(array);

  // read svm_light file line by line
  while ((read = getline(&line, &len, fp)) != -1) {
    char *end_str;
    char *token = strtok_r(line, " ", &end_str);
    //printf("linr nr: %i \n", line_nr);

    if (hasTarget) {
      target = atof(token);
      kv_push(double, array, target);  // append
    } else {
      kv_push(double, array, dummy_target);  // append
    }

    if (hasTarget) token = strtok_r(NULL, " ", &end_str);

    // loop over features in current row
    while (token != NULL) {
      char *end_token;
      char *token2 = strtok_r(token, ":", &end_token);
      double col_nr = atoi(token2);

      token2 = strtok_r(NULL, ":", &end_token);
      if (token2 != NULL) {
        double value = atof(token2);
        assert(cs_entry(T, (int)line_nr, (int)col_nr, value) &&
                "cs_entry failed, out of memory?");
      }
      token = strtok_r(NULL, " ", &end_str);
    }
    line_nr++;
  }

  ffm_vector *y = ffm_vector_alloc(line_nr);

  // copy from kvec to ffm_vector
  for (int k = 0; k < line_nr; k++)
    ffm_vector_set(y, k, kv_a(double, array, k));

  kv_destroy(array);
  cs *X = cs_compress(T);
  cs_spfree(T);

  return (fm_data){.y = y, .X = X};
}

int Cs_write(FILE *f, const cs *A) {
  if (!CS_TRIPLET(A)) return (0); /* check inputs */

  int p, nz, *Ap, *Ai;
  double *Ax;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  nz = A->nz;

  for (p = 0; p < nz; p++) {
    fprintf(f, "%g %g %g\n", (double)(Ai[p]), (double)(Ap[p]), Ax ? Ax[p] : 1);
  }
  return (1);
}

ffm_coef *alloc_fm_coef(int n_features, int k, int ignore_w) {
  struct ffm_coef *coef = malloc(sizeof *coef);

  if (ignore_w)
    coef->w = NULL;
  else
    coef->w = ffm_vector_calloc(n_features);

  if (k > 0) {
    coef->V = ffm_matrix_calloc(k, n_features);
    coef->mu_V = ffm_vector_calloc(k);
    coef->lambda_V = ffm_vector_calloc(k);
  } else {
    coef->V = NULL;
    coef->mu_V = NULL;
    coef->lambda_V = NULL;
  }

  coef->alpha = 0;
  coef->mu_w = 0;
  coef->w_0 = 0;
  coef->lambda_w = 0;
  return coef;
}

void free_ffm_coef(ffm_coef *coef) {
  if (coef->w) {
    ffm_vector_free(coef->w);
    coef->w = NULL;
  }
  if (coef->mu_V) {
    ffm_vector_free(coef->mu_V);
    coef->mu_V = NULL;
  }
  if (coef->lambda_V) {
    ffm_vector_free(coef->lambda_V);
    coef->lambda_V = NULL;
  }
  if (!coef->V) return;
  ffm_matrix_free(coef->V);
  coef->V = NULL;
}

void init_ffm_coef(ffm_coef *coef, ffm_param param) {
  int k = coef->V ? coef->V->size0 : 0;

  ffm_rng *rng = ffm_rng_seed(param.rng_seed);

  coef->w_0 = 0;

  if (!param.ignore_w) {
    double sum = 0;
    for (int i = 0; i < coef->w->size; i++) {
      double tmp = ffm_rand_normal(rng, 0, param.init_sigma);
      ffm_vector_set(coef->w, i, tmp);
      sum += tmp;
    }
    coef->mu_w = sum / (coef->w->size);
  }
  // init V
  if (k > 0) {
    for (int i = 0; i < coef->V->size0; i++) {
      double sum = 0;
      for (int j = 0; j < coef->V->size1; j++) {
        double tmp = ffm_rand_normal(rng, 0, param.init_sigma);
        ffm_matrix_set(coef->V, i, j, tmp);
        sum += tmp;
      }
      ffm_vector_set(coef->mu_V, i, sum / coef->V->size1);
    }
    ffm_vector_set_all(coef->lambda_V, param.init_lambda_V);
  }
  coef->lambda_w = param.init_lambda_w;
  // use default hyperparameter settings if not set
  if (param.SOLVER == SOLVER_MCMC) {
    if (coef->lambda_w == 0) coef->lambda_w = 1;
  }

  ffm_rng_free(rng);
}

void free_fm_data(fm_data *data) {
  ffm_vector_free(data->y);
  cs_spfree(data->X);
}

double ffm_r2_score(ffm_vector *y_true, ffm_vector *y_pred) {
  double ss_tot = ffm_vector_variance(y_true);

  double n_samples = y_true->size;
  ss_tot *= n_samples;
  double ss_res = 0;
  for (int i = 0; i < y_pred->size; i++)
    ss_res += ffm_pow_2(ffm_vector_get(y_true, i) - ffm_vector_get(y_pred, i));

  return 1.0 - (ss_res / ss_tot);
}

/* y = alpha*A[:,j]*x+y */
// A sparse, x, y dense
// modification of cs_gaxpy
int Cs_daxpy(const cs *A, int col_index, double alpha, const double *x,
             double *y) {
  int p, j, *Ap, *Ai;
  double *Ax;
  if (!CS_CSC(A) || !x || !y) return (0); /* check inputs */
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  j = col_index;
  // for (j = 0 ; j < n ; j++)
  //{
  for (p = Ap[j]; p < Ap[j + 1]; p++) {
    y[Ai[p]] += Ax[p] * x[Ai[p]] * alpha;
  }
  // }
  return (1);
}

// y = A*x+y
// with A in RowMajor format.
int Cs_row_gaxpy(const cs *A, const double *x, double *y) {
  CS_INT p, j, n, *Ap, *Ai;
  CS_ENTRY *Ax;
  // if (!CS_CSC (A) || !x || !y) return (0) ;       /* check inputs */
  n = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  for (j = 0; j < n; j++) {
    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      y[j] += Ax[p] * x[Ai[p]];
    }
  }
  return (1);
}

/* y = alpha*A[:,j]+y */
// A sparse, x, y dense
// modification of cs_gaxpy
int Cs_scal_apy(const cs *A, int col_index, double alpha, double *y) {
  int p, j, *Ap, *Ai;
  double *Ax;
  if (!CS_CSC(A) || !y) return (0); /* check inputs */
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  j = col_index;
  // for (j = 0 ; j < n ; j++)
  //{
  for (p = Ap[j]; p < Ap[j + 1]; p++) {
    y[Ai[p]] += Ax[p] * alpha;
  }
  // }
  return (1);
}

/* y = <A[:,j],y> */
// A sparse, y dense
// modification of cs_gaxpy
double Cs_ddot(const cs *A, int col_index, double *y) {
  int p, j, *Ap, *Ai;
  double *Ax;
  if (!CS_CSC(A) || !y) return (0); /* check inputs */
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  j = col_index;
  double sum = 0;
  // for (j = 0 ; j < n ; j++)
  //{
  for (p = Ap[j]; p < Ap[j + 1]; p++) {
    sum += Ax[p] * y[Ai[p]];
  }
  // }
  return sum;
}

/* y = alpha*A[:,j]^2+y */
// A sparse, x, y dense
// modification of cs_gaxpy
int Cs_scal_a2py(const cs *A, int col_index, double alpha, double *y) {
  int p, j, *Ap, *Ai;
  double *Ax;
  if (!CS_CSC(A) || !y) return (0); /* check inputs */
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  j = col_index;
  // for (j = 0 ; j < n ; j++)
  //{
  for (p = Ap[j]; p < Ap[j + 1]; p++) {
    y[Ai[p]] += Ax[p] * Ax[p] * alpha;
  }
  // }
  return (1);
}

/* y = X^2.sum(axis=0) */
// A sparse, x, y dense
// modification of cs_gaxpy
int Cs_col_norm(const cs *A, ffm_vector *y) {
  int p, n, j, *Ap;  // *Ai ;
  double *Ax;
  if (!CS_CSC(A) || !y) return (0); /* check inputs */
  Ap = A->p;                        /* Ai = A->i ;*/
  n = A->n, Ax = A->x;
  for (j = 0; j < n; j++) {
    double norm = 0;
    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      norm += Ax[p] * Ax[p];
    }
    ffm_vector_set(y, j, norm);
  }
  return (1);
}
