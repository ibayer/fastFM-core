#include "fast_fm.h"
/*
 * Interface for external call
 * used by the python wrapper and cli interface
 */

void ffm_predict(double *w_0, double *w, double *V, cs *X, double *y_pred, int k){
    int n_samples = X->m;
    int n_features = X->n;
    ffm_vector ffm_w = {.size=n_features, .data=w, .owner=0};
    ffm_matrix ffm_V = {.size0=k, .size1=n_features, .data=V, .owner=0};
    ffm_coef coef = {.w_0=*w_0, .w=&ffm_w, .V=&ffm_V};

    ffm_vector ffm_y_pred = {.size=n_samples, .data=y_pred, .owner=0};
    sparse_predict(&coef, X, &ffm_y_pred);
}

void ffm_als_fit(double *w_0, double *w, double *V, cs *X, double *y,
    ffm_param *param){
    param->SOLVER = SOLVER_ALS;
    int n_samples = X->m;
    int n_features = X->n;

    ffm_vector ffm_w = {.size=n_features, .data=w, .owner=0};
    ffm_matrix ffm_V = {.size0=param->k, .size1=n_features, .data=V, .owner=0};
    ffm_coef coef = {.w_0=*w_0, .w=&ffm_w, .V=&ffm_V,
                    .lambda_w=param->init_lambda_w};
    if (param->k > 0)
    {
        coef.lambda_V = ffm_vector_alloc(param->k);
        coef.mu_V = ffm_vector_alloc(param->k);
        ffm_vector_set_all(coef.lambda_V, param->init_lambda_V);
    } else
    {
        coef.lambda_V = NULL;
        coef.mu_V = NULL;
    }

    ffm_vector ffm_y = {.size=n_samples, .data=y, .owner=0};
    sparse_fit(&coef, X, NULL, &ffm_y, NULL, *param);

    // copy the last coef values back into the python memory
    *w_0 = coef.w_0;
    ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}


void ffm_mcmc_fit_predict(double *w_0, double *w, double *V,
        cs *X_train, cs *X_test, double *y_train, double *y_pred,
        ffm_param *param){
    param->SOLVER = SOLVER_MCMC;
    int k = param->k;
    double * hyper_param = param->hyper_param;
    int n_test_samples = X_test->m;
    int n_train_samples = X_train->m;
    int n_features = X_train->n;
    ffm_vector ffm_w = {.size=n_features, .data=w, .owner=0};
    ffm_matrix ffm_V = {.size0=param->k, .size1=n_features, .data=V, .owner=0};
    ffm_coef coef = {.w_0=*w_0, .w=&ffm_w, .V=&ffm_V,
                .lambda_w=param->init_lambda_w, .alpha=1, .mu_w=0};
    if (k > 0)
    {
        coef.lambda_V = ffm_vector_alloc(param->k);
        coef.mu_V = ffm_vector_alloc(param->k);
    }
    else
    {
        coef.lambda_V = NULL;
        coef.mu_V = NULL;
    }

    // set inital values for hyperparameter
    int w_groups = 1;
    assert(param->n_hyper_param == 1 + 2 * k + 2 * w_groups &&
            "hyper_parameter vector has wrong size");
    if (param->warm_start)
    {
       coef.alpha = hyper_param[0];
       coef.lambda_w = hyper_param[1];
       // copy V lambda's over
       for (int i=0; i<k; i++) ffm_vector_set(coef.lambda_V, i,
               hyper_param[i + 1 + w_groups]);
       coef.mu_w = hyper_param[k + 1 + w_groups];
       // copy V mu's over
       for (int i=0; i<k; i++) ffm_vector_set(coef.mu_V, i,
               hyper_param[i + 1 + (2 * w_groups) + k]);
    }

    ffm_vector ffm_y_train = {.size=n_train_samples, .data=y_train, .owner=0};
    ffm_vector ffm_y_pred = {.size=n_test_samples, .data=y_pred, .owner=0};
    sparse_fit(&coef, X_train, X_test, &ffm_y_train, &ffm_y_pred, *param);
    // copy the last coef values back into the python memory
    *w_0 = coef.w_0;

    // copy current hyperparameter back
    hyper_param[0] = coef.alpha;
    hyper_param[1] = coef.lambda_w;
    // copy V lambda's back
    for (int i=0; i<k; i++) hyper_param[i + 1 + w_groups] =
        ffm_vector_get(coef.lambda_V, i);
    hyper_param[k + 1 + w_groups] = coef.mu_w;
    // copy mu's back
    for (int i=0; i<k; i++) hyper_param[i + 1 + (2 * w_groups) + k] =
        ffm_vector_get(coef.mu_V, i);

    if ( k > 0 )
        ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}


void ffm_sgd_bpr_fit(double *w_0, double *w, double *V,
        cs *X, double *pairs, int n_pairs, ffm_param *param){

    int n_features = X->m;
    ffm_vector ffm_w = {.size=n_features, .data=w, .owner=0};
    ffm_matrix ffm_V = {.size0=param->k, .size1=n_features, .data=V, .owner=0};
    ffm_coef coef = {.w_0=*w_0, .w=&ffm_w, .V=&ffm_V,
                .lambda_w=param->init_lambda_w};
    if (param->k > 0)
    {
        coef.lambda_V = ffm_vector_alloc(param->k);
        coef.mu_V = ffm_vector_alloc(param->k);
    }
    else
    {
        coef.lambda_V = NULL;
        coef.mu_V = NULL;
    }

    ffm_matrix ffm_y = {.size0=n_pairs, .size1=2, .data=pairs, .owner=0};
    ffm_fit_sgd_bpr(&coef, X, &ffm_y, *param);

    // copy the last coef values back into the python memory
    *w_0 = coef.w_0;
    if ( param->k > 0 )
        ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}

void ffm_sgd_fit(double *w_0, double *w, double *V,
        cs *X, double *y, ffm_param *param){
    int n_samples = X->n;
    int n_features = X->m;

    ffm_vector ffm_w = {.size=n_features, .data=w, .owner=0};
    ffm_matrix ffm_V = {.size0=param->k, .size1=n_features, .data=V, .owner=0};
    ffm_coef coef = {.w_0=*w_0, .w=&ffm_w, .V=&ffm_V,
                    .lambda_w=param->init_lambda_w};
    if (param->k > 0)
    {
        coef.lambda_V = ffm_vector_alloc(param->k);
        coef.mu_V = ffm_vector_alloc(param->k);
    }
    else
    {
        coef.lambda_V = NULL;
        coef.mu_V = NULL;
    }

    ffm_vector ffm_y = {.size=n_samples, .data=y, .owner=0};
    ffm_fit_sgd(&coef, X, &ffm_y, param);

    // copy the last coef values back into the python memory
    *w_0 = coef.w_0;
    if ( param->k > 0 )
        ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}
