#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>

typedef float real;

int main(int argc, char **argv)
{
    assert(argc == 4);

    int neg = strtol(argv[1], 0, 0);
    int nout = strtol(argv[2], 0, 0);
    int nhid = strtol(argv[3], 0, 0);
    double lr = 0.01;
    gsl_rng * rng = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(rng, 234);


    gsl_matrix * x = gsl_matrix_alloc(neg, nout);
    gsl_matrix * w = gsl_matrix_alloc(nout, nhid);
    gsl_vector * a = gsl_vector_alloc(nhid);
    gsl_vector * b = gsl_vector_alloc(nout);
    gsl_matrix * xw = gsl_matrix_alloc(neg, nhid);
    gsl_matrix * hid = gsl_matrix_alloc(neg, nhid);
    gsl_matrix * hidwt = gsl_matrix_alloc(neg, nout);
    gsl_matrix * g_hidwt = gsl_matrix_alloc(neg, nout);
    gsl_matrix * g_hid = gsl_matrix_alloc(neg, nhid);
    gsl_matrix * g_w = gsl_matrix_alloc(nout, nhid);
    gsl_vector * g_b = gsl_vector_alloc(nout);

    for (int i = 0; i < neg*nout; ++i) x->data[i] = (gsl_rng_uniform(rng) -0.5)*1.5;
    for (int i = 0; i < nout*nhid; ++i) w->data[i] = gsl_rng_uniform(rng);
    for (int i = 0; i < nhid; ++i) a->data[i] = 0.0;
    for (int i = 0; i < nout; ++i) b->data[i] = 0.0;

//  
//
//
//

    double err = 0.0;
    for (int iter = 0; iter < 1000; ++iter)
    {
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, x, w, 0.0, xw);

        for (int i = 0; i < neg; ++i)
            for (int j = 0; j < nhid; ++j)
            {
                double act = xw->data[i*nhid+j] + a->data[j];
                hid->data[i*nhid+j] = tanh(act);
            }

        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, hid, w, 0.0, hidwt);

        for (int i = 0; i < nout; ++i) g_b->data[i] = 0.0;
        err = 0.0;
        for (int i = 0; i < neg; ++i)
            for (int j = 0; j < nout; ++j)
            {
                double act = hidwt->data[i*nout+j] + b->data[j];
                double out = tanh(act);
                double g_out = out - x->data[i*nout+j];
                err += g_out * g_out;
                g_hidwt->data[i*nout+j] = g_out * (1.0 - out*out);
                g_b->data[j] += g_hidwt->data[i*nout+j];
            }
        for (int i = 0; i < nout; ++i) b->data[i] -= lr * g_b->data[i];

        if (1)
        {
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, g_hidwt, w, 0.0, g_hid);
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, g_hidwt, hid, 0.0, g_w);
            

            for (int i = 0; i < neg; ++i)
                for (int j = 0; j < nhid; ++j)
                {
                    g_hid->data[i*nhid+j] *= (1.0 - hid->data[i*nhid+j] * hid->data[i*nhid+j]);
                    a->data[j] -= lr * g_hid->data[i*nhid+j];
                }

            gsl_blas_dgemm(CblasTrans, CblasNoTrans, -lr, x, g_hid, 1.0, w);
            for (int i = 0; i < nout*nhid; ++i) w->data[i] -= lr * g_w->data[i];
        }

    }

    fprintf(stdout, "%lf\n", 0.5 * err);
    //skip freeing
    return 0;
}

