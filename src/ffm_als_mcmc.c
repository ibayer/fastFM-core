#include "fast_fm.h"


void sparse_predict(ffm_coef * coef, cs * X, ffm_vector *y_pred){

    // y[:] = w_0
    ffm_vector_set_all(y_pred, coef->w_0);
    // y += Xw
    if (coef->w) cs_gaxpy (X, coef->w->data, y_pred->data);

    // check if second order interactions are used
    if (!coef->V) return;
    eval_second_order_term(coef->V, X, y_pred);
}


void sparse_fit(ffm_coef *coef, cs *X_train, cs *X_test, ffm_vector *y,
        ffm_vector *y_pred_test, ffm_param param){

    int n_features = X_train->n;
    int n_samples = X_train->m;
    int k = coef->V ? coef->V->size0: 0;
    int n_iter = param.n_iter;
    ffm_vector *w = coef->w;
    ffm_matrix *V = coef->V;
    double *w_0 = &coef->w_0;

    ffm_rng *rng = ffm_rng_seed(param.rng_seed);

    if (!param.warm_start)
        init_ffm_coef(coef, param);

    // init err = predict - y
    ffm_vector *err = ffm_vector_alloc(n_samples);
    sparse_predict(coef, X_train, err);

    ffm_vector *z_target = NULL;
    if (param.TASK == TASK_CLASSIFICATION){
        z_target = ffm_vector_calloc(n_samples);

        //ffm_vector_normal_cdf(err);

        if (param.SOLVER == SOLVER_MCMC)
            sample_target(rng, z_target, z_target, y);
        else
            map_update_target(z_target, z_target, y);

    // update class err
        ffm_blas_daxpy(-1, z_target, err);
    }
    else
        ffm_blas_daxpy(-1, y, err);


    // allocate memory for caches
    ffm_vector *column_norms = ffm_vector_alloc(n_features);
    Cs_col_norm(X_train, column_norms);
    ffm_vector *a_theta_v = ffm_vector_calloc(n_samples);
    ffm_vector *XV_f = ffm_vector_calloc(n_samples);
    ffm_vector *V_f = ffm_vector_calloc(n_features);
    // caches that are not always needed
    ffm_vector *tmp_predict_test = NULL;
    if (param.SOLVER == SOLVER_MCMC){
        tmp_predict_test = ffm_vector_calloc(y_pred_test->size);
        if (!param.warm_start)
            ffm_vector_set_all(y_pred_test, 0);
    }

    int n;
    for (n=param.iter_count; n<n_iter; n++){
        if (param.SOLVER == SOLVER_MCMC)
            sample_hyper_parameter(coef, err,rng);

        double tmp_sigma2 = 0;
        double tmp_mu = 0;
        // learn bias
        if (!param.ignore_w_0)
        {
            double w_0_old = coef->w_0;
            if (param.SOLVER == SOLVER_MCMC){
                double tmp_sigma2 = 1. / (coef->alpha * n_samples);
                double tmp_mu = tmp_sigma2 * (coef->alpha * (-ffm_vector_sum(err) + *w_0 * n_samples));
                *w_0 = ffm_rand_normal(rng, tmp_mu, sqrt(tmp_sigma2));
            }
            else
                *w_0 = (-ffm_vector_sum(err) + *w_0 * n_samples) / ((double) n_samples);
            assert( isfinite(*w_0) && "w_0 not finite");
            ffm_vector_add_constant(err, + (*w_0 - w_0_old)); // update error
        }

        // first order interactions
        if (!param.ignore_w)
            for (int f=0; f<n_features; f++){
                double w_f = ffm_vector_get(w, f);
                // w[f] = (err.dot(X_f) + w[f] * norm_rows_X[f]) / (norm_rows_X[f] + lambda_)
                double tmp = Cs_ddot (X_train, f, err->data);
                double c_norm = ffm_vector_get(column_norms, f);
                double new_w = 0;
                if (param.SOLVER == SOLVER_MCMC){
                    tmp_sigma2 = 1. / (coef->alpha * c_norm + coef->lambda_w);
                    tmp_mu = tmp_sigma2 * (coef->alpha * (w_f * c_norm - tmp) + coef->mu_w * coef->lambda_w);
                    new_w = ffm_rand_normal(rng, tmp_mu, sqrt(tmp_sigma2));
                }
                else
                    new_w = (-tmp + w_f * c_norm) / (c_norm + coef->lambda_w);
                assert( isfinite(new_w) && "w not finite");
                ffm_vector_set(w, f, new_w);
                Cs_scal_apy (X_train, f, ffm_vector_get(w, f) - w_f, err->data); // update error
            }

        // second order interactions
        if (k>0){
            for (int f=0; f<k; f++){
                // XV_f = X.dot(V[:,f])
                ffm_vector_set_all(XV_f, 0);
                //ffm_matrix_get_row(V_f, V, f);
                //cs_gaxpy(X_train, V_f->data, XV_f->data);
                double *V_f_ptr =  ffm_matrix_get_row_ptr(V, f);
                // cache
                cs_gaxpy(X_train, V_f_ptr, XV_f->data);
                double lambda_V_k = ffm_vector_get(coef->lambda_V, f);
                double mu_V_k = ffm_vector_get(coef->mu_V, f);

                for (int l=0; l<n_features; l++){

                    double V_fl = ffm_matrix_get(V, f, l);
                    double sum_denominator, sum_nominator;
                    sum_nominator = sum_denominator = 0;
                    sparse_v_lf_frac(&sum_denominator, &sum_nominator, X_train, l, err,
                            XV_f, a_theta_v, V_fl);
                    double new_V_fl = 0;
                    if (param.SOLVER == SOLVER_MCMC){
                        tmp_sigma2 = 1. / (coef->alpha * sum_denominator + lambda_V_k);
                        tmp_mu = tmp_sigma2 * (coef->alpha * sum_nominator +\
                                                            mu_V_k * lambda_V_k);
                        new_V_fl = ffm_rand_normal(rng, tmp_mu, sqrt(tmp_sigma2));
                    }
                    else
                         new_V_fl = sum_nominator / (sum_denominator + lambda_V_k);
                    assert( isfinite(new_V_fl) && "V not finite");
                    ffm_matrix_set(V, f, l, new_V_fl);
                    //err = err - a_theta * (V[l, f] - V_fl) # update residual
                    update_second_order_error(l, X_train, a_theta_v,
                            new_V_fl - V_fl, err);
                    // update cache
                    /* y = alpha*A[:,j]*x+y */
                    Cs_scal_apy (X_train, l, new_V_fl - V_fl, XV_f->data);
                }
            }
        }

        // recalculate error in order to stop error amplification
        // from numerical inexact error and cache updates

        sparse_predict(coef, X_train, err);
        if (param.TASK == TASK_CLASSIFICATION){
        //printf("pred\n");
        //ffm_vector_printf(err);
            // approximate target
            if (param.SOLVER == SOLVER_MCMC)
                sample_target(rng, err, z_target, y);
            else
                map_update_target(err, z_target, y);
            //ffm_vector_normal_cdf(err);
            ffm_blas_daxpy(-1, z_target, err);
        //printf("z_target\n");
        //ffm_vector_printf(z_target);
        }
        else
            ffm_blas_daxpy(-1, y, err);

        // save test predictions for posterior mean
        if (param.SOLVER == SOLVER_MCMC){
            sparse_predict(coef, X_test, tmp_predict_test);

            if (param.TASK == TASK_CLASSIFICATION)
                ffm_vector_normal_cdf(tmp_predict_test);
            ffm_vector_update_mean(y_pred_test, n, tmp_predict_test);
        }
    }
    if (param.TASK == TASK_CLASSIFICATION)
        ffm_vector_free(z_target);
    param.iter_count = n; //TODO this gets lost when returning param
    ffm_vector_free_all(column_norms, err, a_theta_v, XV_f, V_f);
    ffm_rng_free(rng);
}

void sample_target(ffm_rng * rng, ffm_vector *y_pred, ffm_vector *z_target,
        ffm_vector *y_true){
    assert((y_pred->size == z_target->size && y_pred->size == y_true->size) && \
                                            "vectors have different length");
    for (int i=0; i<y_pred->size; i++){
        double mu = fabs(y_pred->data[i]);
        double t_gaussian = ffm_rand_left_trunc_normal(rng, mu);
        if( y_true->data[i] > 0) // left truncated
            z_target->data[i] = + t_gaussian;
        else // right truncated
            z_target->data[i] = - t_gaussian;
    }
}

void map_update_target(ffm_vector *y_pred, ffm_vector *z_target,
        ffm_vector *y_true){
    assert((y_pred->size == z_target->size && y_pred->size == y_true->size) && \
                                            "vectors have different length");
    for (int i=0; i<y_pred->size; i++){
        double mu = y_pred->data[i];
        if( y_true->data[i] > 0) // left truncated
            z_target->data[i] = ffm_normal_pdf(-mu) / (1.0 - ffm_normal_cdf(-mu));
        else // right truncated
            z_target->data[i] = - (ffm_normal_pdf(-mu) / ffm_normal_cdf(-mu));
    }
}

int eval_second_order_term(ffm_matrix *V, cs * X, ffm_vector *y){
    // operate on X.T
    cs * A = cs_transpose (X, 1);
    int k = V->size0;

    int p, j, f, n, *Ap, *Ai ;
    double *Ax ;
    if (!CS_CSC (A)) return (0) ;       /* check inputs */
    n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    // over all k
    for (f = 0; f < k; f++){
        // over all rows
        for (j = 0 ; j < n ; j++)
        {

            double tmp_sum = 0;
            // all nz in this column
            // Ai[p] is the (col) position in the original matrix
            // Ax[p] is the value at position Ai[p]
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                // V
                double tmp_v = ffm_matrix_get(V, f, Ai [p]);
                double tmp_x = Ax[p];
                tmp_sum += tmp_x * tmp_v;
                y->data[j] -= 0.5 * (tmp_x * tmp_x) * (tmp_v * tmp_v);
            }
            y->data[j] += 0.5 * (tmp_sum * tmp_sum);
        }
    }
    cs_spfree (A) ;
    return (1) ;
}


void sample_hyper_parameter(ffm_coef *coef, ffm_vector * err, ffm_rng *rng){

    int n_features = coef->w->size;
    int n_samples = err->size;
    int k = coef->V ? coef->V->size0: 0;

    /*
    printf("alpah%f, lambda_w%f, mu_w%f",
            coef->alpha, coef->lambda_w, coef->mu_w);
    if (k> 0)
    {
        ffm_vector_printf(coef->mu_V);
        ffm_vector_printf(coef->lambda_V);
    }
    */

    ffm_vector *w = coef->w;
    ffm_matrix *V = coef->V;

    // sample alpha
    double alpha_n = .5 *(1. + n_samples);
    double l2_norm = ffm_blas_dnrm2(err);
    double beta_n = .5 * (l2_norm * l2_norm + 1.);
    coef->alpha = ffm_rand_gamma(rng, alpha_n, 1. / beta_n);

    // sample lambda's
    double alpha_w = 0.5 * (1 + n_features + 1);
    double beta_w  = 0;
    for (int i=0; i<n_features; i++)
        beta_w += + ffm_pow_2(ffm_vector_get(w, i) - coef->mu_w);
    beta_w += ffm_pow_2(coef->mu_w) + 1;
    beta_w *= 0.5;
    coef->lambda_w = ffm_rand_gamma(rng, alpha_w, 1. / beta_w);

    double alpha_V = 0.5 * (1 + n_features + 1);
    for (int j=0; j<k; j++)
    {
        double beta_V_fl  = 0;
        double mu_V_j = ffm_vector_get(coef->mu_V, j);
        for (int i=0; i<n_features; i++)
                beta_V_fl += + ffm_pow_2(ffm_matrix_get(V, j, i) - mu_V_j);
        beta_V_fl += ffm_pow_2(mu_V_j) + 1;
        beta_V_fl *= 0.5;
        ffm_vector_set(coef->lambda_V, j,
                ffm_rand_gamma(rng, alpha_V, 1. / beta_V_fl));
    }

    // sample mu's
    double sigma2_mu_w = 1.0 / ((n_features +1) * coef->lambda_w);
    double w_sum = 0;
    for (int i=0; i<n_features; i++)
        w_sum += ffm_vector_get(w, i);
    double mu_mu_w = w_sum / (n_features + 1);
    coef->mu_w = ffm_rand_normal(rng, mu_mu_w, sqrt( sigma2_mu_w));


    for (int j=0; j<k; j++)
    {
        double sigma2_mu_v = 1.0 / ((n_features + 1) *
                ffm_vector_get(coef->lambda_V, j));
        double v_sum = 0;
        for (int i=0; i<n_features; i++)
                v_sum += ffm_matrix_get(V, j, i);
        double mu_mu_v = v_sum / (n_features + 1);
        ffm_vector_set(coef->mu_V, j,
                 ffm_rand_normal(rng, mu_mu_v, sqrt(sigma2_mu_v)));
    }
}

void update_second_order_error(int j_column, cs *A, ffm_vector * a_theta_v,
        double delta, ffm_vector *error){

    int p, *Ap, *Ai ;
    Ap = A->p ; Ai = A->i ;

    // iterate over all nz elements of column j
    // Ai [p] original row pos
    for (p = Ap [j_column] ; p < Ap [j_column+1] ; p++)
        error->data[Ai [p]] += delta * a_theta_v->data[Ai [p]];
}

void sparse_v_lf_frac(double *sum_denominator, double * sum_nominator,
        cs *A, int col_index, ffm_vector *err, ffm_vector *cache,
        ffm_vector *a_theta_v, double v_lf){

    int p, j, *Ap, *Ai ;
    double *Ax ;
    //if (!CS_CSC (A)) return (0) ;       /* check inputs */
    Ap = A->p ; Ai = A->i ; Ax = A->x ;
    j = col_index;
    //for (j = 0 ; j < n ; j++)
    //{
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            double A_jp = Ax[p];
            double a_theta = A_jp * cache->data[Ai [p]] - (v_lf * A_jp * A_jp);
            a_theta_v->data[Ai [p]] = a_theta;
            *sum_denominator +=  a_theta * a_theta;
            *sum_nominator += (v_lf * a_theta - err->data[Ai [p]]) * a_theta;
        }
   // }

}

