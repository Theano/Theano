import core 
import scipy.weave as weave

"""
File: omega/blas.py

This file is in omega's core because it consists mostly of optimizations of the
graphs that can be constructed from omega/core.py.  The optimizations provided
by this file are aimed at the goal of inserting gemm Ops in place of more
fine-grained motifs of iadd, isub, scale, and dot.
"""

_gemm_support_code = """
    template< typename T >
    struct TMat_t
    {
        T *  __restrict__ d;/**< pointer to element (0,0) */
        size_t    M;    /**< number of rows */
        size_t    N;    /**< number of columns */
        size_t    m;    /**< row stride */
        size_t    n;    /**< column stride */
        bool invalid;

        /** null  */
        TMat_t(const PyArrayObject *o) : 
            d((double*) o->data),
            M((o->nd==2) ? o->dimensions[0] : 0),
            N((o->nd==2) ? o->dimensions[1] : 0),
            m((o->nd==2) ? o->strides[0] / sizeof(double) : 0),
            n((o->nd==2) ? o->strides[1] / sizeof(double) : 0),
            invalid((o->nd !=2) 
                || (o->descr->elsize != sizeof(T)))
        {
        }
        /** unsafe element access */
        const T & operator()(size_t i, size_t j) const
        {
            return d[ i * m + j*n];
        }
        /** unsafe element access */
        T & operator()(size_t i, size_t j)
        {
            return d[ i * m + j*n];
        }
        /** safe element access */
        const T & at(size_t i, size_t j) const
        {
            return d[ assert((i < M) && (j < N)), i * m + j*n];
        }
        /** safe element access */
        T & at(size_t i, size_t j)
        {
            return d[ assert((i < M) && (j < N)), i * m + j*n];
        }
    };
    typedef TMat_t<double> mat_t;

    static int mat_gemm_general(double a, const mat_t &A, const mat_t &B, double b, mat_t &C)
    {
        fprintf(stderr, "INFO: running mat_gemm_general\\n");
        for (size_t i = 0; i < C.M; ++i)
        {
            for (size_t j = 0; j < C.N; ++j)
            {
                C(i,j) *= b;
                C(i,j) += a * cblas_ddot(A.N, &A(i,0), A.n, &B(0,j), B.m );
            }
        }
        return 0;
    }

    static int mat_gemm(double a, const mat_t &A, const mat_t &B, double b, mat_t &C)
    {
        if ((A.M != C.M) || (A.N != B.M) || (B.N != C.N))
        {
            PyErr_SetString(PyExc_ValueError, "mat_gemm input array size mismatch");
            return 1;
        }
        if ((A.m < 1) || (A.n < 1)
           ||(B.m < 1) || (B.n < 1)
           ||(C.m < 1) || (C.n < 1))
        {
           return mat_gemm_general(a, A, B, b, C);
        }

        //TODO: OPTIMIZE for many special cases:
        //- gemv
        //- ger
        //- ddot
        //- others?

        int unit = 0;
        unit |= ((A.n == 1) ? 0x0 : (A.m == 1) ? 0x1 : 0x2) << 0;
        unit |= ((B.n == 1) ? 0x0 : (B.m == 1) ? 0x1 : 0x2) << 4;
        unit |= ((C.n == 1) ? 0x0 : (C.m == 1) ? 0x1 : 0x2) << 8;

        /*
        fprintf(stderr, "M N   %zu %zu   %zu %zu   %zu %zu\n", A.M, A.N, B.M, B.N, C.M, C.N);
        fprintf(stderr, "m n   %zu %zu   %zu %zu   %zu %zu\n", A.m, A.n, B.m, B.n, C.m, C.n);
        fprintf(stderr, "unit  %i\n", unit);
        */

        size_t A_m = (A.M > 1) ? A.m : A.N;
        size_t A_n = (A.N > 1) ? A.n : A.M;
        size_t B_m = (B.M > 1) ? B.m : B.N;
        size_t B_n = (B.N > 1) ? B.n : B.M;
        size_t C_m = (C.M > 1) ? C.m : C.N;
        size_t C_n = (C.N > 1) ? C.n : C.M;

        switch(unit)
        {
            case 0x000: cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.M, C.N, A.N, a, A.d, A_m, B.d, B_m, b, C.d, C_m); break;
            case 0x001: cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, C.M, C.N, A.N, a, A.d, A_n, B.d, B_m, b, C.d, C_m); break;
            case 0x010: cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   C.M, C.N, A.N, a, A.d, A_m, B.d, B_n, b, C.d, C_m); break;
            case 0x011: cblas_dgemm(CblasRowMajor, CblasTrans,   CblasTrans,   C.M, C.N, A.N, a, A.d, A_n, B.d, B_n, b, C.d, C_m); break;
            case 0x100: cblas_dgemm(CblasColMajor, CblasTrans,   CblasTrans,   C.M, C.N, A.N, a, A.d, A_m, B.d, B_m, b, C.d, C_n); break;
            case 0x101: cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,   C.M, C.N, A.N, a, A.d, A_n, B.d, B_m, b, C.d, C_n); break;
            case 0x110: cblas_dgemm(CblasColMajor, CblasTrans,   CblasNoTrans, C.M, C.N, A.N, a, A.d, A_m, B.d, B_n, b, C.d, C_n); break;
            case 0x111: cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C.M, C.N, A.N, a, A.d, A_n, B.d, B_n, b, C.d, C_n); break;
            default: mat_gemm_general(a, A, B, b, C); break;
        };
        return 0;
    }
    """
_gemm_code = """
    mat_t mx(x_array), my(y_array), mz(z_array);
    if (mx.invalid || my.invalid || mz.invalid)
    {
        fprintf(stderr, "error in train_classifier_new.py, _gemm_code\\n");
    }
    else
    {
        mat_gemm(a[0], mx, my, b[0], mz);
    }

    """
def _gemm(a, x, y, b, z):
    weave.inline(_gemm_code, 
            ['a', 'x', 'y', 'b', 'z'], 
            support_code = _gemm_support_code,
            headers=['<gsl/gsl_cblas.h>'],
            libraries=['cblas','atlas', 'g2c'])

#TODO: modify gemm to work with vectors and tensors too!
# (trac ticket 18)
class gemm(core.omega_op, core.inplace):
    def impl_unused(z, a,x,y,b):
        if b == 0.0:
            if a == 1.0:
                z = numpy.dot(x,y)
            elif a == -1.0:
                z = -numpy.dot(x,y)
            else:
                z = a * numpy.dot(x,y)
        elif b == 1.0:
            if a == 1.0:
                z += numpy.dot(x,y)
            elif a == -1.0:
                z -= numpy.dot(x,y)
            else:
                z += a * numpy.dot(x,y)
        else:
            z *= b
            z += a * numpy.dot(x,y)
        return z[:]
    def impl(z, a, x, y, b):
        _gemm(a, x, y, b, z)
        return z[:]

    def grad(x,gz):
        raise NotImplemented


#TODO: put more optimizations in here
optimizations = [
        pattern_opt(
            (C.isub_elemwise, 'z', (C.dot,'x','y')), 
            (gemm, 'z', -1.0, 'x', 'y', 1.0))
    ]
