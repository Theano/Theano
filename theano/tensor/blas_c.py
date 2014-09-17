import numpy

from theano import config

from theano.tensor.opt import in2out
from theano.tensor.blas import ldflags, blas_header_text, blas_header_version
from theano.tensor.blas import (
    blas_optdb, optdb, local_optimizer, EquilibriumOptimizer)
from theano.tensor.blas import Ger, ger, ger_destructive
from theano.tensor.blas import Gemv, gemv_inplace, gemv_no_inplace
from theano.tensor import basic as T
import theano.compile


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

def ger_c_code(A, a, x, y, Z, destructive, fail):
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
    if (!%(destructive)s
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
            if (%(Z)s) Py_XDECREF(%(Z)s);
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
            int Ai = PyArray_STRIDES(%(A)s)[0]/sizeof(float);
            int Aj = PyArray_STRIDES(%(A)s)[1]/sizeof(float);
            int Zi = PyArray_STRIDES(%(Z)s)[0]/sizeof(float);
            int Zj = PyArray_STRIDES(%(Z)s)[1]/sizeof(float);
            for (int i = 0; i < dims[0]; ++i)
            {
                for (int j = 0; j < dims[1]; ++j)
                {
                    zoutdata[Zi*i+Zj*j] = zdata[Ai*i+Aj*j];
                }
            }
        }
        else if (PyArray_DESCR(%(Z)s)->type_num == NPY_DOUBLE)
        {
            double * zoutdata = (double*) PyArray_DATA(%(Z)s);
            const double * zdata = (double*)PyArray_DATA(%(A)s);
            int Ai = PyArray_STRIDES(%(A)s)[0]/sizeof(double);
            int Aj = PyArray_STRIDES(%(A)s)[1]/sizeof(double);
            int Zi = PyArray_STRIDES(%(Z)s)[0]/sizeof(double);
            int Zj = PyArray_STRIDES(%(Z)s)[1]/sizeof(double);
            for (int i = 0; i < dims[0]; ++i)
            {
                for (int j = 0; j < dims[1]; ++j)
                {
                    zoutdata[Zi*i+Zj*j] = zdata[Ai*i+Aj*j];
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
        //fprintf(stderr, "USING A\\n");
        if (%(Z)s != %(A)s)
        {
            if (%(Z)s) { Py_DECREF(%(Z)s); }
            %(Z)s = %(A)s;
            Py_INCREF(%(Z)s);
        }
    }

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
                //fprintf(stderr, "A\\n");
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
                //fprintf(stderr, "B %%i %%i %%i %%i\\n", Nz0, Nz1, Sz0, Sz1);
                float alpha = ((dtype_%(a)s*)(PyArray_DATA(%(a)s)))[0];
                //fprintf(stderr, "alpha=%%f\\n", alpha);
                //fprintf(stderr, "sx  sy %%i %%i\\n", Sx, Sy);
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

    """ % locals()


class CGer(BaseBLAS, Ger):
    def c_code(self, node, name, inp, out, sub):
        A, a, x, y = inp
        Z, = out
        code = ger_c_code(A, a, x, y, Z,
                          destructive=int(self.destructive),
                          fail=sub['fail'])
        return code

    def c_code_cache_version(self):
        return (8, blas_header_version())
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
    if node.op == cger_no_inplace:
        return [cger_inplace(*node.inputs)]


# ##### ####### #######
# GEMV
# ##### ####### #######


def gemv_c_code(aa, xx, yy, zz, alpha, beta, destructive, fail,
                force_init_beta=False):
    """
    zz <- beta * aa + alpha * dot(xx, yy)

    where xx is a matrix, yy and aa are vectors (ergo zz is vector)
    """
    code = """

    int elemsize ;
    float fbeta;
    double dbeta;

    if (PyArray_NDIM(%(aa)s) != 1)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(aa) != 1");
        %(fail)s;
    }
    if (PyArray_NDIM(%(xx)s) != 2)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(xx) != 2");
        %(fail)s;
    }
    if (PyArray_NDIM(%(yy)s) != 1)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(yy) != 1");
        %(fail)s;
    }
    if (PyArray_NDIM(%(alpha)s) != 0)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(alpha) != 0");
        %(fail)s;
    }
    if (PyArray_NDIM(%(beta)s) != 0)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(beta) != 0");
        %(fail)s;
    }

    if (PyArray_DESCR(%(aa)s)->type_num != PyArray_DESCR(%(xx)s)->type_num)
    { PyErr_SetString(PyExc_TypeError, "Gemv: aa vs. xx"); %(fail)s; }
    if (PyArray_DESCR(%(aa)s)->type_num != PyArray_DESCR(%(yy)s)->type_num)
    { PyErr_SetString(PyExc_TypeError, "Gemv: aa vs. yy"); %(fail)s; }

    if (PyArray_DIMS(%(xx)s)[0] != PyArray_DIMS(%(aa)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[0] != x.shape[0]");
        %(fail)s;
    }
    if (PyArray_DIMS(%(xx)s)[1] != PyArray_DIMS(%(yy)s)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[1] != y.shape[0]");
        %(fail)s;
    }

    if  (PyArray_DESCR(%(aa)s)->type_num == NPY_DOUBLE) { elemsize = 8; }
    else if (PyArray_DESCR(%(aa)s)->type_num == NPY_FLOAT) { elemsize = 4;}
    else {
        PyErr_SetString(PyExc_NotImplementedError, "complex Gemv");
        %(fail)s;
    }

    fbeta = dbeta = ((dtype_%(beta)s*)PyArray_DATA(%(beta)s))[0];

    // copy aa if not destructive
    if (!%(destructive)s)
    {
        if ((NULL == %(zz)s)
            || (PyArray_DIMS(%(zz)s)[0] != PyArray_DIMS(%(aa)s)[0]))
        {
            if (%(zz)s) Py_XDECREF(%(zz)s);
            %(zz)s = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(%(aa)s), PyArray_TYPE((PyArrayObject*) py_%(aa)s));
            if(!%(zz)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemv output");
                %(fail)s
            }
        }
        if (%(zz)s == %(aa)s)
        {
            PyErr_SetString(PyExc_AssertionError, "%(zz)s != %(aa)s");
            %(fail)s
        }
        if (dbeta != 0)
        {
            if (PyArray_DESCR(%(zz)s)->type_num == NPY_FLOAT)
            {
                float * zoutdata = (float*)PyArray_DATA(%(zz)s);
                const float * zdata = (float*)PyArray_DATA(%(aa)s);
                int Ai = PyArray_STRIDES(%(aa)s)[0]/sizeof(float);
                int Zi = PyArray_STRIDES(%(zz)s)[0]/sizeof(float);
                for (int i = 0; i < PyArray_DIMS(%(aa)s)[0]; ++i)
                {
                    zoutdata[Zi*i] = fbeta * zdata[Ai*i];
                }
            }
            else if (PyArray_DESCR(%(zz)s)->type_num == NPY_DOUBLE)
            {
                double * zoutdata = (double*) PyArray_DATA(%(zz)s);
                const double * zdata = (double*)PyArray_DATA(%(aa)s);
                int Ai = PyArray_STRIDES(%(aa)s)[0]/sizeof(double);
                int Zi = PyArray_STRIDES(%(zz)s)[0]/sizeof(double);
                for (int i = 0; i < PyArray_DIMS(%(aa)s)[0]; ++i)
                {
                    zoutdata[Zi*i] = dbeta * zdata[Ai*i];
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError,
                                "neither float nor double dtype");
                %(fail)s
            }
            fbeta = dbeta = 1.0;
        }
        else if (%(force_init_beta)d)
        {
            if (PyArray_CHKFLAGS(%(zz)s, NPY_ARRAY_C_CONTIGUOUS))
            {
                memset((void *)PyArray_DATA(%(zz)s), 0, PyArray_SIZE(%(zz)s)*PyArray_ITEMSIZE(%(zz)s));
            }
            else
            {
                if (PyArray_DESCR(%(zz)s)->type_num == NPY_FLOAT)
                {
                    float *zoutdata = (float *)PyArray_DATA(%(zz)s);
                    int Zi = PyArray_STRIDES(%(zz)s)[0]/sizeof(float);
                    for (int i = 0; i < PyArray_DIMS(%(aa)s)[0]; ++i)
                    {
                        zoutdata[Zi*i] = 0.0f;
                    }
                }
                else if (PyArray_DESCR(%(zz)s)->type_num == NPY_DOUBLE)
                {
                    double *zoutdata = (double *)PyArray_DATA(%(zz)s);
                    int Zi = PyArray_STRIDES(%(zz)s)[0]/sizeof(double);
                    for (int i = 0; i < PyArray_DIMS(%(aa)s)[0]; ++i)
                    {
                        zoutdata[Zi*i] = 0.0;
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    %(fail)s
                }
            }
        }
    }
    else
    {
        //fprintf(stderr, "Gemv working in-place \\n");
        if (%(zz)s != %(aa)s)
        {
            if (%(zz)s) { Py_DECREF(%(zz)s); }
            %(zz)s = %(aa)s;
            Py_INCREF(%(zz)s);
        }
    }
    {
        char TRANS = 'T';
        char NOTRANS = 'N';
        int Nx0 = PyArray_DIMS(%(xx)s)[0];
        int Nx1 = PyArray_DIMS(%(xx)s)[1];
        /* This formula is needed in the case where xx is actually a row or
         * column matrix, because BLAS sometimes insists that the strides:
         *  - are not smaller than the number of elements in the array
         *  - are not 0.
         */
        int Sx0 = (Nx0 > 1) ? (PyArray_STRIDES(%(xx)s)[0] / elemsize) : (Nx1 + 1);
        int Sx1 = (Nx1 > 1) ? (PyArray_STRIDES(%(xx)s)[1] / elemsize) : (Nx0 + 1);
        int Sz = PyArray_STRIDES(%(zz)s)[0] / elemsize;
        int Sy = PyArray_STRIDES(%(yy)s)[0] / elemsize;

        dtype_%(yy)s* yy_data = (dtype_%(yy)s*) PyArray_DATA(%(yy)s);
        dtype_%(zz)s* zz_data = (dtype_%(zz)s*) PyArray_DATA(%(zz)s);
        // gemv expects pointers to the beginning of memory arrays,
        // but numpy provides provides a pointer to the first element,
        // so when the stride is negative, we need to get the last one.
        if (Sy < 0)
            yy_data += (Nx1 - 1) * Sy;
        if (Sz < 0)
            zz_data += (Nx0 - 1) * Sz;

        if (Nx0 * Nx1)
        {
            // If xx is neither C- nor F-contiguous, we make a copy.
            // TODO:
            // - if one stride is equal to "- elemsize", we can still call
            //   gemv on reversed matrix and vectors
            // - if the copy is too long, maybe call vector/vector dot on
            //   each row instead
            if ((PyArray_STRIDES(%(xx)s)[0] < 0)
                || (PyArray_STRIDES(%(xx)s)[1] < 0)
                || ((PyArray_STRIDES(%(xx)s)[0] != elemsize)
                    && (PyArray_STRIDES(%(xx)s)[1] != elemsize)))
            {
                npy_intp dims[2];
                dims[0] = Nx0;
                dims[1] = Nx1;

                PyArrayObject * xx_copy = (PyArrayObject *) PyArray_Copy(
                                                                    %(xx)s);
                if (!xx_copy)
                    %(fail)s
                Py_XDECREF(%(xx)s);
                %(xx)s = xx_copy;
                Sx0 = (Nx0 > 1) ? (PyArray_STRIDES(%(xx)s)[0] / elemsize) : (Nx1 + 1);
                Sx1 = (Nx1 > 1) ? (PyArray_STRIDES(%(xx)s)[1] / elemsize) : (Nx0 + 1);
            }

            if (PyArray_STRIDES(%(xx)s)[0] == elemsize)
            {
                if (PyArray_DESCR(%(xx)s)->type_num == NPY_FLOAT)
                {
                    //fprintf(stderr, "A\\n");
                    float alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    sgemv_(&NOTRANS, &Nx0, &Nx1,
                        &alpha,
                        (float*)(PyArray_DATA(%(xx)s)), &Sx1,
                        (float*)yy_data, &Sy,
                        &fbeta,
                        (float*)zz_data, &Sz);
                }
                else if (PyArray_DESCR(%(xx)s)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    dgemv_(&NOTRANS, &Nx0, &Nx1,
                        &alpha,
                        (double*)(PyArray_DATA(%(xx)s)), &Sx1,
                        (double*)yy_data, &Sy,
                        &dbeta,
                        (double*)zz_data, &Sz);
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    %(fail)s
                }
            }
            else if (PyArray_STRIDES(%(xx)s)[1] == elemsize)
            {
                if (PyArray_DESCR(%(xx)s)->type_num == NPY_FLOAT)
                {
                    float alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];

                    // Check for vector-vector dot (Nx0 == 1). The code may work
                    // for Sx1 != 1 as well, but has not been tested for this case,
                    // so Sx1 == 1 is required for safety.
                    if (Nx0 == 1 && Sx1 == 1)
                    {
                        zz_data[0] = fbeta*zz_data[0] + alpha*sdot_(&Nx1, 
                            (float*)(PyArray_DATA(%(xx)s)), &Sx1,
                            (float*)yy_data, &Sy);
                    }
                    else
                    {
                        sgemv_(&TRANS, &Nx1, &Nx0,
                            &alpha,
                            (float*)(PyArray_DATA(%(xx)s)), &Sx0,
                            (float*)yy_data, &Sy,
                            &fbeta,
                            (float*)zz_data, &Sz);
                    }
                }
                else if (PyArray_DESCR(%(xx)s)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];

                    // Check for vector-vector dot (Nx0 == 1). The code may work
                    // for Sx1 != 1 as well, but has not been tested for this case,
                    // so Sx1 == 1 is required for safety.
                    if (Nx0 == 1 && Sx1 == 1)
                    {
                        zz_data[0] = dbeta*zz_data[0] + alpha*ddot_(&Nx1, 
                              (double*)(PyArray_DATA(%(xx)s)), &Sx1,
                              (double*)yy_data, &Sy);
                    }
                    else
                    {
                        dgemv_(&TRANS, &Nx1, &Nx0,
                            &alpha,
                            (double*)(PyArray_DATA(%(xx)s)), &Sx0,
                            (double*)yy_data, &Sy,
                            &dbeta,
                            (double*)zz_data, &Sz);
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
            // or else it does the right thing for length-0 x.
            dtype_%(zz)s * zptr = (dtype_%(zz)s*)(PyArray_DATA(%(zz)s));
            for (int i = 0; i < Nx0; ++i)
            {
                zptr[i * Sz] *= dbeta;
            }
        }
    }

    """
    return code % locals()


class CGemv(BaseBLAS, Gemv):
    def __init__(self, inplace, force_init_beta=False):
        super(CGemv, self).__init__(inplace)
        self.force_init_beta = force_init_beta

    def c_code(self, node, name, inp, out, sub):
        aa, alpha, xx, yy, beta = inp
        zz, = out
        code = gemv_c_code(
            aa, xx, yy, zz, alpha, beta,
            destructive=int(self.inplace),
            fail=sub['fail'],
            force_init_beta=self.force_init_beta
        )
        return code

    def c_code_cache_version(self):
        return (11, blas_header_version())
cgemv_inplace = CGemv(inplace=True)
cgemv_no_inplace = CGemv(inplace=False)


def check_force_gemv_init():
    if check_force_gemv_init._force_init_beta is None:
        """
        Test issue 1569.
        Namely when evaulating

            beta*aa + alpha*dot(xx, yy)

        where we set aa = betas = zeros of the correct dimensions we do not
        actually set aa = zeros and instead let the BLAS perform beta*aa with
        uninitialized memory for speed. Occasionally the memory contains values
        that are equivalent to NaN in which case the product beta*aa contains
        NaN's for correctly implemented BLAS libraries. In this situation, since
        we are introducing the NaN's, we need to test whether the BLAS performs
        correctly. If it *does*, i.e. it actually performs the multiplication
        beta*aa which will result in NaN's in the result, then we need intialize
        the memory to zeros.
        """
        aa = T.vector('aa')
        yy = T.vector('yy')
        xx = T.matrix('xx')
        f = theano.function(
            [aa, yy, xx],
            gemv_no_inplace(aa, 1., xx, yy, 0.),
            theano.compile.Mode(optimizer='fast_compile')
        )

        # Here we introduce NaNs into the data, if they are returned by the BLAS
        # then we want gemv_c_code to initiliaze the memory to 0 so that we
        # don't inadvertantly introduce NaNs to the users data.
        aa_data = numpy.array(
            float('NaN')*numpy.ones((2,)),
            dtype=theano.config.floatX
        )
        yy_data = numpy.array(
            numpy.ones((2,))*2,
            dtype=theano.config.floatX
        )
        xx_data = numpy.array(
            numpy.ones((2, 2)),
            dtype=theano.config.floatX
        )
        zz = f(aa_data, yy_data, xx_data)

        check_force_gemv_init._force_init_beta = numpy.isnan(zz).any()

    return check_force_gemv_init._force_init_beta

check_force_gemv_init._force_init_beta = None


@local_optimizer([gemv_inplace, gemv_no_inplace])
def use_c_gemv(node):
    if not config.blas.ldflags:
        return
    # Only float32 and float64 are supported for now.
    if (node.op == gemv_no_inplace and
            node.outputs[0].dtype in ['float32', 'float64']):

        """
        We want to maintain the behavoir of any operation that the user adds
        even if it results in NaNs. However we do not want optimizations to
        introduce NaNs.

        GEMV is not always implemented consistenly across BLAS libraries.
        Sometimes, when beta is 0, they do not perform the multiplication with
        beta. Other implmentations do. This can cause problems for the inplace
        GEMV implementation if NaNs happen to be in the newly allocated but
        uninitalized memory. When the multiplication is not done we do not need
        to initialize the output memory resulting in a speed up. Otherwise we
        must initialize the memory to avoid introducing NaN's in the output
        that weren't in the original graph.

        The following check determines whether the output memory needs to be
        initiliazed. It is done here, as opposed to in global scope, because
        the setup has not been completed at that time and therefore the check
        cannot be performed at that time.
        """
        force_init_beta = check_force_gemv_init()

        return [CGemv(inplace=False,
                      force_init_beta=force_init_beta)(*node.inputs)]
    if (node.op == gemv_inplace and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGemv(inplace=True)(*node.inputs)]


@local_optimizer([CGemv(inplace=False)])
def make_c_gemv_destructive(node):
    if node.op == cgemv_no_inplace:
        return [cgemv_inplace(*node.inputs)]


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
