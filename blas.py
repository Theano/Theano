import numpy
import core 
from gof import PatternOptimizer as pattern_opt, OpSubOptimizer as op_sub
import scipy.weave as weave

"""
File: omega/blas.py

This file is in omega's core because it consists mostly of optimizations of the
graphs that can be constructed from omega/core.py.  The optimizations provided
by this file are aimed at the goal of inserting gemm Ops in place of more
fine-grained motifs of iadd, isub, scale, and dot.
"""

_general_gemm_code = """

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
"""

_gemm_code_template = """
    if ((Nx[0] != Nz[0]) || (Nx[1] != Ny[0]) || (Ny[1] != Nz[1]))
    {
        PyErr_SetString(PyExc_ValueError, "mat_gemm input array size mismatch");
       fprintf(stderr, "Should be calling mat_gemm_general, but quitting instead\\n");
       exit(1);
    }
    if ((Sx[0] < 1) || (Sx[1] < 1)
       || (Sy[0] < 1) || (Sy[1] < 1)
       || (Sz[0] < 1) || (Sz[1] < 1))
    {
       fprintf(stderr, "Should be calling mat_gemm_general, but quitting instead\\n");
       exit(1);
       //return mat_gemm_general(a, A, B, b, C);
    }

    //TODO: OPTIMIZE for many special cases:
    //- gemv
    //- ger
    //- ddot
    //- others?

    int unit = 0;
    unit |= ((Sx[1] == sizeof(%(dtype)s)) ? 0x0 : (Sx[0] == sizeof(%(dtype)s)) ? 0x1 : 0x2) << 0;
    unit |= ((Sy[1] == sizeof(%(dtype)s)) ? 0x0 : (Sy[0] == sizeof(%(dtype)s)) ? 0x1 : 0x2) << 4;
    unit |= ((Sz[1] == sizeof(%(dtype)s)) ? 0x0 : (Sz[0] == sizeof(%(dtype)s)) ? 0x1 : 0x2) << 8;

    /* create appropriate strides for malformed matrices that are row or column
     * vectors 
     */
    size_t sx_0 = (Nx[0] > 1) ? Sx[0]/sizeof(%(dtype)s) : Nx[1];
    size_t sx_1 = (Nx[1] > 1) ? Sx[1]/sizeof(%(dtype)s) : Nx[0];
    size_t sy_0 = (Ny[0] > 1) ? Sy[0]/sizeof(%(dtype)s) : Ny[1];
    size_t sy_1 = (Ny[1] > 1) ? Sy[1]/sizeof(%(dtype)s) : Ny[0];
    size_t sz_0 = (Nz[0] > 1) ? Sz[0]/sizeof(%(dtype)s) : Nz[1];
    size_t sz_1 = (Nz[1] > 1) ? Sz[1]/sizeof(%(dtype)s) : Nz[0];

    switch(unit)
    {
        case 0x000: %(gemm)s(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a[0], x, sx_0, y, sy_0, b[0], z, sz_0); break;
        case 0x001: %(gemm)s(CblasRowMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a[0], x, sx_1, y, sy_0, b[0], z, sz_0); break;
        case 0x010: %(gemm)s(CblasRowMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a[0], x, sx_0, y, sy_1, b[0], z, sz_0); break;
        case 0x011: %(gemm)s(CblasRowMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a[0], x, sx_1, y, sy_1, b[0], z, sz_0); break;
        case 0x100: %(gemm)s(CblasColMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a[0], x, sx_0, y, sy_0, b[0], z, sz_1); break;
        case 0x101: %(gemm)s(CblasColMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a[0], x, sx_1, y, sy_0, b[0], z, sz_1); break;
        case 0x110: %(gemm)s(CblasColMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a[0], x, sx_0, y, sy_1, b[0], z, sz_1); break;
        case 0x111: %(gemm)s(CblasColMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a[0], x, sx_1, y, sy_1, b[0], z, sz_1); break;
        default: 
           fprintf(stderr, "Should be calling mat_gemm_general, but quitting instead\\n");
           exit(1);
    };
    /* v 1 */
"""

_gemm_code = { 'f': _gemm_code_template % { 'gemm':'cblas_sgemm', 'dtype':'float'},
                'd': _gemm_code_template % { 'gemm':'cblas_dgemm', 'dtype':'double'}}

def _gemm_rank2(a, x, y, b, z):
    weave.inline(_gemm_code[z.dtype.char],
            ['a', 'x', 'y', 'b', 'z'],
            headers=['"/home/bergstra/cvs/lgcm/omega/cblas.h"'],
            libraries=['mkl', 'm'])

def _gemm(a, x, y, b, z):
    if len(x.shape) == 2 and len(y.shape) == 2:
        _gemm_rank2(a, x, y, b, z)
    else:
        if b == 0.0:
            if a == 1.0:
                z[:] = numpy.dot(x,y)
            elif a == -1.0:
                z[:] = -numpy.dot(x,y)
            else:
                z[:] = a * numpy.dot(x,y)
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

_gdot_coefs = { 'f':
    (numpy.ones((),dtype='float32'),numpy.zeros((),dtype='float32')),
                'd': (numpy.ones(()),numpy.zeros(()))}

def _gdot(x,y):
    a,b = _gdot_coefs[x.dtype.char]
    z = numpy.ndarray((x.shape[0],y.shape[1]),dtype=x.dtype)
    _gemm(a, x, y, b, z)
    return z

class gemm(core.omega_op, core.inplace):

    def impl(z, a, x, y, b):
        _gemm(a, x, y, b, z)
        return z[:]

    def grad(x,gz):
        raise NotImplemented

class gdot(core.omega_op):

    impl = _gdot

    def grad(x,gz):
        raise NotImplemented

#TODO: put more optimizations in here (Trac # 18)
optimizations = [
        pattern_opt(
            (core.isub_elemwise, 'z', (core.dot,'x','y')), 
            (gemm, 'z', -1.0, 'x', 'y', 1.0)),
        pattern_opt(
            (core.dot,'x', 'y'), 
            (gdot, 'x', 'y'))
    ]

