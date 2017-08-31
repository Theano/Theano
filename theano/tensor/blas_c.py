from __future__ import absolute_import, print_function, division

from theano import config
from theano.gof.params_type import ParamsType
from theano.scalar import bool as bool_t
from theano.tensor.opt import in2out
from theano.tensor.blas import ldflags, blas_header_text, blas_header_version
from theano.tensor.blas import blas_optdb, optdb, local_optimizer
from theano.tensor.blas import Ger, ger, ger_destructive
from theano.tensor.blas import Gemv, gemv_inplace, gemv_no_inplace
from theano.tensor import basic as T


class BaseBLAS(object):
    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_support_code(self):
        return blas_header_text()


# ##### ####### #######
# GER
# ##### ####### #######

def ger_c_code(A, a, x, y, Z, fail, params):
    return """

    int elemsize ;

    if (PyArray_NDIM(%(A)s) != 2)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(A) != 2"); %(fail)s;}
    if (PyArray_NDIM(%(x)s) != 1)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 1"); %(fail)s;}
    if (PyArray_NDIM(%(y)s) != 1)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 1"); %(fail)s;}
    if (PyArray_NDIM(%(a)s) != 0)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 0"); %(fail)s;}

    if (PyArray_DESCR(%(A)s)->type_num != PyArray_DESCR(%(x)s)->type_num)
    { PyErr_SetString(PyExc_TypeError, "A vs. x"); %(fail)s; }
    if (PyArray_DESCR(%(A)s)->type_num != PyArray_DESCR(%(y)s)->type_num)
    { PyErr_SetString(PyExc_TypeError, "A vs. y"); %(fail)s; }

    if (PyArray_DIMS(%(A)s)[0] != PyArray_DIMS(%(x)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[0] != x.shape[0]");
        %(fail)s;
    }
    if (PyArray_DIMS(%(A)s)[1] != PyArray_DIMS(%(y)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[1] != y.shape[0]");
        %(fail)s;
    }

    if  (PyArray_DESCR(%(A)s)->type_num == NPY_DOUBLE) { elemsize = 8; }
    else if (PyArray_DESCR(%(A)s)->type_num == NPY_FLOAT) { elemsize = 4;}
    else
    {
        PyErr_SetString(PyExc_NotImplementedError, "complex CGer");
        %(fail)s;
    }

    // copy A if !self.destructive or A is fully strided
    if (!%(params)s->destructive
        || (PyArray_STRIDES(%(A)s)[0] < 0)
        || (PyArray_STRIDES(%(A)s)[1] < 0)
        || ((PyArray_STRIDES(%(A)s)[0] != elemsize)
            && (PyArray_STRIDES(%(A)s)[1] != elemsize)))
    {
        npy_intp dims[2];
        dims[0] = PyArray_DIMS(%(A)s)[0];
        dims[1] = PyArray_DIMS(%(A)s)[1];

        if ((NULL == %(Z)s)
            || (PyArray_DIMS(%(Z)s)[0] != PyArray_DIMS(%(A)s)[0])
            || (PyArray_DIMS(%(Z)s)[1] != PyArray_DIMS(%(A)s)[1])
            || (PyArray_STRIDES(%(Z)s)[0] < 0)
            || (PyArray_STRIDES(%(Z)s)[1] < 0)
            || ((PyArray_STRIDES(%(Z)s)[0] != elemsize)
                && (PyArray_STRIDES(%(Z)s)[1] != elemsize)))
        {
            Py_XDECREF(%(Z)s);
            %(Z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims,
                                                       PyArray_TYPE(%(A)s));
            if(!%(Z)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc ger output");
                %(fail)s
            }
        }
        if (%(Z)s == %(A)s)
        {
            PyErr_SetString(PyExc_AssertionError, "%(Z)s != %(A)s");
            %(fail)s
        }
        if (PyArray_DESCR(%(Z)s)->type_num == NPY_FLOAT)
        {
            float * zoutdata = (float*)PyArray_DATA(%(Z)s);
            const float * zdata = (float*)PyArray_DATA(%(A)s);
            const float * xdata = (float*)PyArray_DATA(%(x)s);
            const float * ydata = (float*)PyArray_DATA(%(y)s);
            const float * adata = (float*)PyArray_DATA(%(a)s);
            const float alpha = adata[0];
            float tmp, xx;
            int Ai = PyArray_STRIDES(%(A)s)[0]/sizeof(float);
            int Aj = PyArray_STRIDES(%(A)s)[1]/sizeof(float);
            int Zi = PyArray_STRIDES(%(Z)s)[0]/sizeof(float);
            int Zj = PyArray_STRIDES(%(Z)s)[1]/sizeof(float);
            int xi = PyArray_STRIDES(%(x)s)[0]/sizeof(float);
            int yj = PyArray_STRIDES(%(y)s)[0]/sizeof(float);
            for (int i = 0; i < dims[0]; ++i)
            {
                xx = alpha * xdata[xi * i];
                for (int j = 0; j < dims[1]; ++j)
                {
                    tmp = zdata[Ai*i+Aj*j];
                    tmp += xx * ydata[yj * j];
                    zoutdata[Zi*i+Zj*j] = tmp;
                }
            }
        }
        else if (PyArray_DESCR(%(Z)s)->type_num == NPY_DOUBLE)
        {
            double * zoutdata = (double*) PyArray_DATA(%(Z)s);
            const double * zdata = (double*)PyArray_DATA(%(A)s);
            const double * xdata = (double*)PyArray_DATA(%(x)s);
            const double * ydata = (double*)PyArray_DATA(%(y)s);
            const double * adata = (double*)PyArray_DATA(%(a)s);
            const double alpha = adata[0];
            double tmp, xx;

            int Ai = PyArray_STRIDES(%(A)s)[0]/sizeof(double);
            int Aj = PyArray_STRIDES(%(A)s)[1]/sizeof(double);
            int Zi = PyArray_STRIDES(%(Z)s)[0]/sizeof(double);
            int Zj = PyArray_STRIDES(%(Z)s)[1]/sizeof(double);
            int xi = PyArray_STRIDES(%(x)s)[0]/sizeof(double);
            int yj = PyArray_STRIDES(%(y)s)[0]/sizeof(double);
            for (int i = 0; i < dims[0]; ++i)
            {
                xx = alpha * xdata[xi * i];
                for (int j = 0; j < dims[1]; ++j)
                {
                    tmp = zdata[Ai*i+Aj*j];
                    tmp += xx * ydata[yj * j];
                    zoutdata[Zi*i+Zj*j] = tmp;
                }
            }
        }
        else
        {
            PyErr_SetString(PyExc_AssertionError,
                            "neither float nor double dtype");
            %(fail)s
        }
    }
    else
    {
        if (%(Z)s != %(A)s)
        {
            if (%(Z)s) { Py_DECREF(%(Z)s); }
            %(Z)s = %(A)s;
            Py_INCREF(%(Z)s);
        }
        npy_intp dims[2];
        dims[0] = PyArray_DIMS(%(A)s)[0];
        dims[1] = PyArray_DIMS(%(A)s)[1];
        if ((dims[0] * dims[1]) < 100000)
        {
            if (PyArray_DESCR(%(Z)s)->type_num == NPY_FLOAT)
            {
                float * zoutdata = (float*)PyArray_DATA(%(Z)s);
                const float * xdata = (float*)PyArray_DATA(%(x)s);
                const float * ydata = (float*)PyArray_DATA(%(y)s);
                const float * adata = (float*)PyArray_DATA(%(a)s);
                const float alpha = adata[0];
                float tmp, axi;
                int Zi = PyArray_STRIDES(%(Z)s)[0]/sizeof(float);
                int Zj = PyArray_STRIDES(%(Z)s)[1]/sizeof(float);
                int xi = PyArray_STRIDES(%(x)s)[0]/sizeof(float);
                int yj = PyArray_STRIDES(%(y)s)[0]/sizeof(float);
                for (int i = 0; i < dims[0]; ++i)
                {
                    axi = alpha * xdata[xi * i];
                    for (int j = 0; j < dims[1]; ++j)
                    {
                        zoutdata[Zi*i+Zj*j] += axi * ydata[yj * j];
                    }
                }
            }
            else if (PyArray_DESCR(%(Z)s)->type_num == NPY_DOUBLE)
            {
                double * zoutdata = (double*) PyArray_DATA(%(Z)s);
                const double * xdata = (double*)PyArray_DATA(%(x)s);
                const double * ydata = (double*)PyArray_DATA(%(y)s);
                const double * adata = (double*)PyArray_DATA(%(a)s);
                const double alpha = adata[0];
                double tmp, axi;

                int Zi = PyArray_STRIDES(%(Z)s)[0]/sizeof(double);
                int Zj = PyArray_STRIDES(%(Z)s)[1]/sizeof(double);
                int xi = PyArray_STRIDES(%(x)s)[0]/sizeof(double);
                int yj = PyArray_STRIDES(%(y)s)[0]/sizeof(double);
                for (int i = 0; i < dims[0]; ++i)
                {
                    axi = alpha * xdata[xi * i];
                    for (int j = 0; j < dims[1]; ++j)
                    {
                        zoutdata[Zi*i+Zj*j] += axi * ydata[yj * j];
                    }
                }
            }
        }
        else
        {
            int Nz0 = PyArray_DIMS(%(Z)s)[0];
            int Nz1 = PyArray_DIMS(%(Z)s)[1];
            int Sx = PyArray_STRIDES(%(x)s)[0] / elemsize;
            int Sy = PyArray_STRIDES(%(y)s)[0] / elemsize;

            /* create appropriate strides for Z, if it is a row or column matrix.
             * In that case, the value of the stride does not really matter, but
             * some versions of BLAS insist that:
             *  - they are not smaller than the number of elements in the array,
             *  - they are not 0.
             */
            int Sz0 = (Nz0 > 1) ? (PyArray_STRIDES(%(Z)s)[0] / elemsize) : (Nz1 + 1);
            int Sz1 = (Nz1 > 1) ? (PyArray_STRIDES(%(Z)s)[1] / elemsize) : (Nz0 + 1);

            dtype_%(x)s* x_data = (dtype_%(x)s*) PyArray_DATA(%(x)s);
            dtype_%(y)s* y_data = (dtype_%(y)s*) PyArray_DATA(%(y)s);
            // gemv expects pointers to the beginning of memory arrays,
            // but numpy provides provides a pointer to the first element,
            // so when the stride is negative, we need to get the last one.
            if (Sx < 0)
                x_data += (Nz0 - 1) * Sx;
            if (Sy < 0)
                y_data += (Nz1 - 1) * Sy;

            if (PyArray_STRIDES(%(Z)s)[0] == elemsize)
            {
                if (PyArray_DESCR(%(Z)s)->type_num == NPY_FLOAT)
                {
                    float alpha = ((dtype_%(a)s*)PyArray_DATA(%(a)s))[0];
                    sger_(&Nz0, &Nz1, &alpha,
                        (float*)x_data, &Sx,
                        (float*)y_data, &Sy,
                        (float*)(PyArray_DATA(%(Z)s)), &Sz1);
                }
                else if (PyArray_DESCR(%(Z)s)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_%(a)s*)PyArray_DATA(%(a)s))[0];
                    dger_(&Nz0, &Nz1, &alpha,
                        (double*)x_data, &Sx,
                        (double*)y_data, &Sy,
                        (double*)(PyArray_DATA(%(Z)s)), &Sz1);


                }
                else {
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "not float nor double");
                    %(fail)s
                }
            }
            else if (PyArray_STRIDES(%(Z)s)[1] == elemsize)
            {
                if (PyArray_DESCR(%(Z)s)->type_num == NPY_FLOAT)
                {
                    float alpha = ((dtype_%(a)s*)(PyArray_DATA(%(a)s)))[0];
                    sger_(&Nz1, &Nz0, &alpha,
                        (float*)y_data, &Sy,
                        (float*)x_data, &Sx,
                        (float*)(PyArray_DATA(%(Z)s)), &Sz0);
                }
                else if (PyArray_DESCR(%(Z)s)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_%(a)s*)PyArray_DATA(%(a)s))[0];
                    dger_(&Nz1, &Nz0, &alpha,
                        (double*)y_data, &Sy,
                        (double*)x_data, &Sx,
                        (double*)(PyArray_DATA(%(Z)s)), &Sz0);
                }
                else
                {
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "not float nor double");
                    %(fail)s
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError,
                    "A is a double-strided matrix, and should have been copied "
                    "into a memory-contiguous one.");
                %(fail)s
            }
        }
    }

    """ % locals()


class CGer(BaseBLAS, Ger):
    params_type = ParamsType(destructive=bool_t,)

    def c_code(self, node, name, inp, out, sub):
        A, a, x, y = inp
        Z, = out
        code = ger_c_code(A, a, x, y, Z,
                          fail=sub['fail'],
                          params=sub['params'])
        return code

    def c_code_cache_version(self):
        return (11, blas_header_version())
cger_inplace = CGer(True)
cger_no_inplace = CGer(False)


@local_optimizer([ger, ger_destructive])
def use_c_ger(node):
    if not config.blas.ldflags:
        return
    # Only float32 and float64 are supported for now.
    if (node.op == ger and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGer(False)(*node.inputs)]
    if (node.op == ger_destructive and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGer(True)(*node.inputs)]


@local_optimizer([CGer(False)])
def make_c_ger_destructive(node):
    if isinstance(node.op, CGer) and not node.op.destructive:
        return [cger_inplace(*node.inputs)]


# ##### ####### #######
# GEMV
# ##### ####### #######


def gemv_c_code(y, A, x, z, alpha, beta, fail,
                force_init_beta=False, params=None):
    """
    z <- beta * y + alpha * dot(A, x)

    where A is a matrix, y and x are vectors (ergo z is vector)
    """
    code = """

    int elemsize;
    float fbeta;
    double dbeta;

    if (PyArray_DIMS(%(A)s)[0] != PyArray_DIMS(%(y)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[0] != y.shape[0]");
        %(fail)s;
    }
    if (PyArray_DIMS(%(A)s)[1] != PyArray_DIMS(%(x)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[1] != x.shape[0]");
        %(fail)s;
    }

    if  (PyArray_DESCR(%(y)s)->type_num == NPY_DOUBLE) { elemsize = 8; }
    else if (PyArray_DESCR(%(y)s)->type_num == NPY_FLOAT) { elemsize = 4;}
    else {
        PyErr_SetString(PyExc_NotImplementedError, "complex Gemv");
        %(fail)s;
    }

    fbeta = dbeta = ((dtype_%(beta)s*)PyArray_DATA(%(beta)s))[0];

    // copy y if not destructive
    if (!%(params)s->inplace)
    {
        if ((NULL == %(z)s)
            || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(y)s)[0]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(%(y)s), PyArray_TYPE(%(y)s));
            if(!%(z)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemv output");
                %(fail)s
            }
        }
        if (dbeta != 0)
        {
            if (PyArray_CopyInto(%(z)s, %(y)s) != 0) {
                %(fail)s
            }
        }
        else if (%(force_init_beta)d)
        {
            PyObject *zero = PyFloat_FromDouble(0.);
            if (zero == NULL) %(fail)s;
            if (PyArray_FillWithScalar(%(z)s, zero) != 0) %(fail)s;
            Py_DECREF(zero);
        }
    }
    else
    {
        if (%(z)s != %(y)s)
        {
            Py_XDECREF(%(z)s);
            %(z)s = %(y)s;
            Py_INCREF(%(z)s);
        }
    }
    {
        char TRANS = 'T';
        char NOTRANS = 'N';
        int NA0 = PyArray_DIMS(%(A)s)[0];
        int NA1 = PyArray_DIMS(%(A)s)[1];
        /* This formula is needed in the case where A is actually a row or
         * column matrix, because BLAS sometimes insists that the strides:
         *  - are not smaller than the number of elements in the array
         *  - are not 0.
         */
        int SA0 = (NA0 > 1) ? (PyArray_STRIDES(%(A)s)[0] / elemsize) : (NA1 + 1);
        int SA1 = (NA1 > 1) ? (PyArray_STRIDES(%(A)s)[1] / elemsize) : (NA0 + 1);
        int Sz = PyArray_STRIDES(%(z)s)[0] / elemsize;
        int Sx = PyArray_STRIDES(%(x)s)[0] / elemsize;

        dtype_%(x)s* x_data = (dtype_%(x)s*) PyArray_DATA(%(x)s);
        dtype_%(z)s* z_data = (dtype_%(z)s*) PyArray_DATA(%(z)s);
        // gemv expects pointers to the beginning of memory arrays,
        // but numpy provides a pointer to the first element,
        // so when the stride is negative, we need to get the last one.
        if (Sx < 0)
            x_data += (NA1 - 1) * Sx;
        if (Sz < 0)
            z_data += (NA0 - 1) * Sz;

        if (NA0 * NA1)
        {
            // If A is neither C- nor F-contiguous, we make a copy.
            // TODO:
            // - if one stride is equal to "- elemsize", we can still call
            //   gemv on reversed matrix and vectors
            // - if the copy is too long, maybe call vector/vector dot on
            //   each row instead
            if ((PyArray_STRIDES(%(A)s)[0] < 0)
                || (PyArray_STRIDES(%(A)s)[1] < 0)
                || ((PyArray_STRIDES(%(A)s)[0] != elemsize)
                    && (PyArray_STRIDES(%(A)s)[1] != elemsize)))
            {
                npy_intp dims[2];
                dims[0] = NA0;
                dims[1] = NA1;

                PyArrayObject * A_copy = (PyArrayObject *) PyArray_Copy(
                                                                   %(A)s);
                if (!A_copy)
                    %(fail)s
                Py_XDECREF(%(A)s);
                %(A)s = A_copy;
                SA0 = (NA0 > 1) ? (PyArray_STRIDES(%(A)s)[0] / elemsize) : (NA1 + 1);
                SA1 = (NA1 > 1) ? (PyArray_STRIDES(%(A)s)[1] / elemsize) : (NA0 + 1);
            }

            if (PyArray_STRIDES(%(A)s)[0] == elemsize)
            {
                if (PyArray_DESCR(%(A)s)->type_num == NPY_FLOAT)
                {
                    float alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    sgemv_(&NOTRANS, &NA0, &NA1,
                        &alpha,
                        (float*)(PyArray_DATA(%(A)s)), &SA1,
                        (float*)x_data, &Sx,
                        &fbeta,
                        (float*)z_data, &Sz);
                }
                else if (PyArray_DESCR(%(A)s)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    dgemv_(&NOTRANS, &NA0, &NA1,
                        &alpha,
                        (double*)(PyArray_DATA(%(A)s)), &SA1,
                        (double*)x_data, &Sx,
                        &dbeta,
                        (double*)z_data, &Sz);
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    %(fail)s
                }
            }
            else if (PyArray_STRIDES(%(A)s)[1] == elemsize)
            {
                if (PyArray_DESCR(%(A)s)->type_num == NPY_FLOAT)
                {
                    float alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];

                    // Check for vector-vector dot (NA0 == 1). The code may work
                    // for SA1 != 1 as well, but has not been tested for this case,
                    // so SA1 == 1 is required for safety.
                    if (NA0 == 1 && SA1 == 1)
                    {
                        if (fbeta != 0.f) {
                          z_data[0] = fbeta*z_data[0];
                        } else {
                          z_data[0] = 0.f;
                        }
                        z_data[0] += alpha*sdot_(&NA1,
                              (float*)(PyArray_DATA(%(A)s)), &SA1,
                              (float*)x_data, &Sx);
                    }
                    else
                    {
                        sgemv_(&TRANS, &NA1, &NA0,
                            &alpha,
                            (float*)(PyArray_DATA(%(A)s)), &SA0,
                            (float*)x_data, &Sx,
                            &fbeta,
                            (float*)z_data, &Sz);
                    }
                }
                else if (PyArray_DESCR(%(A)s)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];

                    // Check for vector-vector dot (NA0 == 1). The code may work
                    // for SA1 != 1 as well, but has not been tested for this case,
                    // so SA1 == 1 is required for safety.
                    if (NA0 == 1 && SA1 == 1)
                    {
                        if (dbeta != 0.) {
                          z_data[0] = dbeta*z_data[0];
                        } else {
                          z_data[0] = 0.;
                        }
                        z_data[0] += alpha*ddot_(&NA1,
                              (double*)(PyArray_DATA(%(A)s)), &SA1,
                              (double*)x_data, &Sx);
                    }
                    else
                    {
                        dgemv_(&TRANS, &NA1, &NA0,
                            &alpha,
                            (double*)(PyArray_DATA(%(A)s)), &SA0,
                            (double*)x_data, &Sx,
                            &dbeta,
                            (double*)z_data, &Sz);
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    %(fail)s
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError,
                    "xx is a double-strided matrix, and should have been "
                    "copied into a memory-contiguous one.");
                %(fail)s
            }
        }
        else if (dbeta != 1.0)
        {
            // the matrix has at least one dim of length 0
            // so we do this loop, which either iterates over 0 elements
            // or else it does the right thing for length-0 A.
            dtype_%(z)s * zptr = (dtype_%(z)s*)(PyArray_DATA(%(z)s));
            for (int i = 0; i < NA0; ++i)
            {
                zptr[i * Sz] = (dbeta == 0.0 ? 0.0 : zptr[i * Sz] * dbeta);
            }
        }
    }
    """
    return code % locals()


class CGemv(BaseBLAS, Gemv):
    params_type = ParamsType(inplace=bool_t,)

    def __init__(self, inplace):
        super(CGemv, self).__init__(inplace)

    def c_code(self, node, name, inp, out, sub):
        y, alpha, A, x, beta = inp
        z, = out
        code = gemv_c_code(
            y, A, x, z, alpha, beta,
            fail=sub['fail'],
            force_init_beta=check_force_gemv_init(),
            params=sub['params'],
        )
        return code

    def c_code_cache_version(self):
        return (14, blas_header_version(), check_force_gemv_init())

cgemv_inplace = CGemv(inplace=True)
cgemv_no_inplace = CGemv(inplace=False)


def check_force_gemv_init():
    if check_force_gemv_init._force_init_beta is None:
        from theano.gof.cmodule import GCC_compiler
        """
        Test issue 1569.
        Namely when evaluating

            beta*y + alpha*dot(A, x)

        where we set y * beta = zeros of the correct dimensions we
        do not actually set y = zeros and instead let the BLAS
        perform beta*y with uninitialized memory for
        speed. Occasionally the memory contains values that are
        equivalent to NaN in which case the product beta*y contains
        NaN's for correctly implemented BLAS libraries. In this
        situation, since we are introducing the NaN's, we need to test
        whether the BLAS performs correctly. If it *does*, i.e. it
        actually performs the multiplication beta*y which will result
        in NaN's in the result, then we need intialize the memory to
        zeros.
        """
        test_code = """
#include <math.h>
extern "C" void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int *);
int main() {
  double A[2][2] = {{1., 1.}, {1., 1.}};
  double x[2] = {1., 1.};
  double y[2] = {NAN, NAN};
  const int s = 2;
  const int inc = 1;
  const double alpha = 1.0;
  const double beta = 0.0;

  dgemv_("T", &s, &s, &alpha, A, &s, x, &inc, &beta, &y, &inc);

  return (isnan(y[0]) || isnan(y[1]) ? 1 : 0;
}
"""
        res = GCC_compiler.try_compile_tmp(test_code, tmp_prefix='check_beta_',
                                           flags=ldflags(libs=True, flags=True,
                                                         libs_dir=True),
                                           try_run=True)
        if res:
            if res[0]:
                check_force_gemv_init._force_init_beta = res[1]
            else:
                check_force_gemv_init._force_init_beta = False
        else:
            check_force_gemv_init._force_init_beta = False

    return check_force_gemv_init._force_init_beta

check_force_gemv_init._force_init_beta = None


@local_optimizer([gemv_inplace, gemv_no_inplace])
def use_c_gemv(node):
    if not config.blas.ldflags:
        return
    # Only float32 and float64 are supported for now.
    if (node.op == gemv_no_inplace and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [cgemv_no_inplace(*node.inputs)]
    if (node.op == gemv_inplace and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [cgemv_inplace(*node.inputs)]


@local_optimizer([CGemv(inplace=False)])
def make_c_gemv_destructive(node):
    if isinstance(node.op, CGemv) and not node.op.inplace:
        inputs = list(node.inputs)
        dest = inputs[0]
        if (dest.owner and
                isinstance(dest.owner.op, T.AllocEmpty) and
                len(dest.clients) > 1):
            inputs[0] = T.AllocEmpty(dest.dtype)(*dest.owner.inputs)

        return [cgemv_inplace(*inputs)]


# ##### ####### #######
# Optimizers
# ##### ####### #######

blas_optdb.register('use_c_blas',
                    in2out(use_c_ger, use_c_gemv),
                    20, 'fast_run', 'c_blas')

# this matches the InplaceBlasOpt defined in blas.py
optdb.register('c_blas_destructive',
               in2out(make_c_ger_destructive,
                      make_c_gemv_destructive,
                      name="c_blas_destructive"),
               70.0, 'fast_run', 'inplace', 'c_blas')
