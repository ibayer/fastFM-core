// Author: Immanuel Bayer
// License: BSD 3 clause

#include "fast_fm.h"
#include <time.h>
#include <stdlib.h>
#include <argp.h>
#include <fenv.h>

const char *argp_program_version = "fastFM dev";
const char *argp_program_bug_address = "<immanuel.bayer@uni-konstanz.de>";

/* Program documentation. */
static char doc[] =
    "fastFM -- Provides a range of solvers and loss functions for the "
    "FACTORICATION MACHINE model.";

/* A description of the arguments we accept. */
static char args_doc[] = "TRAIN_FILE TEST_FILE";

/* The options we understand. */
static struct argp_option options[] = {
    {"rng-seed", 556, "NUM", 0,
     "Seed for random number generator (default current time(NULL))"},
    {"task", 't', "S", 0,
     "The tasks: 'classification', 'regression' or 'ranking' are supported "
     "(default 'regression'). Ranking uses the Bayesian Pairwise Ranking (BPR) "
     "loss and needs an additional file (see '--train-pairs')"},
    {"train-pairs", 555, "FILE", 0,
     "Ranking only! Required training pairs for bpr training."},
    {0, 0, 0, 0, "Solver:", 1},
    {"solver", 's', "S", 0,
     "The solvers: 'als', 'mcmc' and 'sgd' are available for 'regression' and "
     "'classification (default 'mcmc'). Ranking is only supported by 'sgd'."},
    {"n-iter", 'n', "NUM", 0, "Number of iterations (default 50)"},
    {"step-size", 444, "NUM", 0, "Step-size for 'sgd' updates (default 0.01)"},
    {"init-var", 'i', "NUM", 0,
     "N(0, var) is used to initialize the coefficients of matrix V (default "
     "0.1)"},
    {0, 0, 0, 0, "Model Complexity and Regularization:", 3},
    {"rank", 'k', "NUM", 0, "Rank of the factorization, Matrix V (default 8)."},
    {"l2-reg", 'r', "NUM", 0,
     "l2 regularization, set equal penalty for all coefficients (default 1)"},
    {"l2-reg-w", 222, "NUM", 0,
     "l2 regularization for the linear coefficients (w)"},
    {"l2-reg-V", 333, "NUM", 0,
     "l2 regularization for the latent representation (V) of the pairwise "
     "coefficients"},
    {0, 0, 0, 0, "I/O options:", -5},
    {"test-predict", 55, "FILE", 0, "Save prediction from TEST_FILE to FILE."},
    {0, 0, 0, 0, "Informational Options:", -3},
    {"verbose", 'v', 0, 0, "Produce verbose output"},
    {"quiet", 'q', 0, 0, "Don't produce any output"},
    {"silent", 's', 0, OPTION_ALIAS},
    {0}};

/* Used by main to communicate with parse_opt. */
struct arguments {
  char *args[2]; /* arg1 & arg2 */
  int silent, verbose;

  char *test_file;
  char *train_pairs;
  char *train_file;
  char *test_predict_file;

  int *arg_count;

  // fm parameters
  int k;
  int n_iter;
  double init_var;
  double step_size;
  double l2_reg;
  double l2_reg_w;
  double l2_reg_V;
  char *solver;
  char *task;
  int rng_seed;
};

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  /* Get the input argument from argp_parse, which we
      know is a pointer to our arguments structure. */
  struct arguments *arguments = state->input;
  int *arg_count = arguments->arg_count;

  switch (key) {
    case 't':
      arguments->task = arg;
      break;

    case 'k':
      arguments->k = atoi(arg);
      break;

    case 'n':
      arguments->n_iter = atoi(arg);
      break;

    case 's':
      arguments->solver = arg;
      break;

    case 'r':
      arguments->l2_reg = atof(arg);
      break;

    case 222:
      arguments->l2_reg_w = atof(arg);
      break;

    case 333:
      arguments->l2_reg_V = atof(arg);
      break;

    case 444:
      arguments->step_size = atof(arg);
      break;

    case 556:
      arguments->rng_seed = atof(arg);
      break;

    case 555:
      arguments->train_pairs = arg;
      break;

    case 55:
      arguments->test_predict_file = arg;
      break;

    case 'i':
      arguments->init_var = atof(arg);
      break;

    case 'q':
      arguments->silent = 1;
      break;

    case 'v':
      arguments->verbose = 1;
      break;

    case ARGP_KEY_ARG: {
      (*arg_count)--;

      if (state->arg_num == 0) {
        arguments->train_file = arg;
      }

      if (state->arg_num == 1) {
        arguments->test_file = arg;
      }

      arguments->args[state->arg_num] = arg;
    } break;

    case ARGP_KEY_END: {
      if (*arg_count > 0)
        argp_failure(state, 1, 0, "too few arguments");
      else if (*arg_count < 0)
        argp_failure(state, 1, 0, "too many arguments");
    } break;

    default:
      return ARGP_ERR_UNKNOWN;
  }

  return 0;
}

/* Our argp parser. */
static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char **argv) {
  /*
  feenableexcept(FE_INVALID   |
                 FE_DIVBYZERO |
                 FE_OVERFLOW  |
                 FE_UNDERFLOW);
  */

  struct arguments arguments;

  /* Default values. */
  arguments.silent = 0;
  arguments.verbose = 0;

  // file paths
  arguments.test_file = NULL;
  arguments.train_file = NULL;
  arguments.test_predict_file = NULL;
  arguments.train_pairs = NULL;

  // fm default parameters
  arguments.k = 8;
  arguments.n_iter = 50;
  arguments.init_var = 0.01;
  arguments.step_size = 0.01;
  arguments.l2_reg = 1;
  arguments.l2_reg_w = 0;
  arguments.l2_reg_V = 0;
  arguments.solver = "mcmc";
  arguments.task = "regression";
  arguments.rng_seed = time(NULL);

  int arg_count = 2;
  arguments.arg_count = &arg_count;

  /* Parse our arguments; every option seen by parse_opt will
     be reflected in arguments. */
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  ffm_param param = {.n_iter = arguments.n_iter,
                     .init_sigma = arguments.init_var,
                     .k = arguments.k,
                     .stepsize = arguments.step_size,
                     .rng_seed = arguments.rng_seed};

  // parse solver
  if (strcmp(arguments.solver, "mcmc") == 0)
    param.SOLVER = SOLVER_MCMC;
  else if (strcmp(arguments.solver, "als") == 0)
    param.SOLVER = SOLVER_ALS;
  else if (strcmp(arguments.solver, "sgd") == 0)
    param.SOLVER = SOLVER_SGD;
  else
    assert(0 && "unknown solver");

  // parse task
  if (strcmp(arguments.task, "regression") == 0)
    param.TASK = TASK_REGRESSION;
  else if (strcmp(arguments.task, "classification") == 0)
    param.TASK = TASK_CLASSIFICATION;
  else if (strcmp(arguments.task, "ranking") == 0)
    param.TASK = TASK_RANKING;
  else
    assert(0 && "unknown task");

  printf(
      "TRAIN_FILE = %s\nTEST_FILE = %s\n"
      "VERBOSE = %s\nSILENT = %s\n",
      arguments.args[0], arguments.args[1], arguments.verbose ? "yes" : "no",
      arguments.silent ? "yes" : "no");

  printf("task=%s", arguments.task);
  printf(", init-var=%f", param.init_sigma);
  printf(", n-iter=%i", param.n_iter);
  if (param.TASK == TASK_RANKING) printf(", step-size=%f", param.stepsize);
  printf(", solver=%s", arguments.solver);
  printf(", k=%i", param.k);

  // default if no l2_reg_w specified
  param.init_lambda_w = arguments.l2_reg;
  param.init_lambda_V = arguments.l2_reg;

  if (arguments.l2_reg_w != 0.0) param.init_lambda_w = arguments.l2_reg_w;
  if (arguments.l2_reg_V != 0.0) param.init_lambda_V = arguments.l2_reg_V;

  if (strcmp(arguments.solver, "mcmc") != 0) {
    printf(", l2-reg-w=%f", param.init_lambda_w);
    if (arguments.k > 0) printf(", l2-reg-V=%f", param.init_lambda_V);
    printf("\n");
  }

  printf("\nload data\n");
  fm_data train_data = read_svm_light_file(arguments.args[0]);
  fm_data test_data = read_svm_light_file(arguments.args[1]);

  int n_features = train_data.X->n;
  ffm_vector *y_test_predict = ffm_vector_calloc(test_data.y->size);
  ffm_coef *coef = alloc_fm_coef(n_features, arguments.k, false);

  printf("fit model\n");
  if (param.TASK == TASK_RANKING) {
    assert(arguments.train_pairs != NULL &&
           "Ranking requires the option '--train-pairs'");
    ffm_matrix *train_pairs = ffm_matrix_from_file(arguments.train_pairs);
    cs *X_t = cs_transpose(train_data.X, 1);
    cs_spfree(train_data.X);
    train_data.X = X_t;
    ffm_fit_sgd_bpr(coef, train_data.X, train_pairs, param);
    // printf("c%", arguments.train_pairs);
  } else
    sparse_fit(coef, train_data.X, test_data.X, train_data.y, y_test_predict,
               param);

  // the predictions are calculated during the training phase for mcmc
  if (param.SOLVER == SOLVER_ALS) {
    sparse_predict(coef, test_data.X, y_test_predict);
    if (param.TASK == TASK_CLASSIFICATION)
      ffm_vector_normal_cdf(y_test_predict);
  }

  if (param.SOLVER == SOLVER_SGD) {
    sparse_predict(coef, test_data.X, y_test_predict);
    if (param.TASK == TASK_CLASSIFICATION)
      ffm_vector_normal_cdf(y_test_predict);
  }

  // save predictions
  if (arguments.test_predict_file) {
    FILE *f = fopen(arguments.test_predict_file, "w");
    for (int i = 0; i < y_test_predict->size; i++)
      fprintf(f, "%f\n", y_test_predict->data[i]);
    fclose(f);
  }

  if (param.TASK == TASK_REGRESSION)
    printf("\nr2 score: %f \n", ffm_r2_score(test_data.y, y_test_predict));
  if (param.TASK == TASK_CLASSIFICATION)
    printf("\nacc score: %f \n",
           ffm_vector_accuracy(test_data.y, y_test_predict));

  /*
  printf("calculate kendall tau\n");
  if (param.TASK == TASK_RANKING)
  {
      ffm_vector * true_order = ffm_vector_get_order(test_data.y);
      ffm_vector * pred_order = ffm_vector_get_order(y_test_predict);
      double kendall_tau = \
              ffm_vector_kendall_tau(true_order, pred_order);
      printf("\nkendall tau: %f \n", kendall_tau);
  }
  */

  exit(0);
}
