from theano.gof import Op

from blas import ldflags, blas_header_text
from blas import blas_optdb, optdb, local_optimizer, EquilibriumOptimizer
from blas import Ger, ger, ger_destructive
from blas import Gemv, gemv_inplace, gemv_no_inplace


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


####### ####### #######
# GER
####### ####### #######

def ger_c_code(A, a, x, y, Z, destructive, fail):
    return """

    int elemsize ;

    if (%(A)s->nd != 2)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(A) != 2"); %(fail)s;}
    if (%(x)s->nd != 1)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 1"); %(fail)s;}
    if (%(y)s->nd != 1)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 1"); %(fail)s;}
    if (%(a)s->nd != 0)
    {PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 0"); %(fail)s;}

    if (%(A)s->descr->type_num != %(x)s->descr->type_num)
    { PyErr_SetString(PyExc_TypeError, "A vs. x"); %(fail)s; }
    if (%(A)s->descr->type_num != %(y)s->descr->type_num)
    { PyErr_SetString(PyExc_TypeError, "A vs. y"); %(fail)s; }

    if (%(A)s->dimensions[0] != %(x)s->dimensions[0])
    {PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[0] != x.shape[0]"); %(fail)s;}
    if (%(A)s->dimensions[1] != %(y)s->dimensions[0])
    {PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[1] != y.shape[0]"); %(fail)s;}

    if  (%(A)s->descr->type_num == PyArray_DOUBLE) { elemsize = 8; }
    else if (%(A)s->descr->type_num == PyArray_FLOAT) { elemsize = 4;}
    else {PyErr_SetString(PyExc_NotImplementedError, "complex CGer"); %(fail)s;}


    // copy A if !self.destructive or A is fully strided
    if (!%(destructive)s
        || ((%(A)s->strides[0] != elemsize)
            &&
            (%(A)s->strides[1] != elemsize)))
    {
        npy_intp dims[2];
        dims[0] = %(A)s->dimensions[0];
        dims[1] = %(A)s->dimensions[1];

        if ((NULL == %(Z)s)
            || (%(Z)s->dimensions[0] != %(A)s->dimensions[0])
            || (%(Z)s->dimensions[1] != %(A)s->dimensions[1]))
        {
            if (%(Z)s) Py_XDECREF(%(Z)s);
            %(Z)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(A)s));
            if(!%(Z)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc ger output");
                %(fail)s
            }
        }
        assert (%(Z)s != %(A)s);
        if (%(Z)s->descr->type_num == PyArray_FLOAT)
        {
            float * zoutdata = (float*)%(Z)s->data;
            const float * zdata = (float*)%(A)s->data;
            int Ai = %(A)s->strides[0]/sizeof(float);
            int Aj = %(A)s->strides[1]/sizeof(float);
            int Zi = %(Z)s->strides[0]/sizeof(float);
            int Zj = %(Z)s->strides[1]/sizeof(float);
            for (int i = 0; i < dims[0]; ++i)
            {
                for (int j = 0; j < dims[1]; ++j)
                {
                    zoutdata[Zi*i+Zj*j] = zdata[Ai*i+Aj*j];
                }
            }
        }
        else if (%(Z)s->descr->type_num == PyArray_DOUBLE)
        {
            double * zoutdata = (double*) %(Z)s->data;
            const double * zdata = (double*)%(A)s->data;
            int Ai = %(A)s->strides[0]/sizeof(double);
            int Aj = %(A)s->strides[1]/sizeof(double);
            int Zi = %(Z)s->strides[0]/sizeof(double);
            int Zj = %(Z)s->strides[1]/sizeof(double);
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
            PyErr_SetString(PyExc_AssertionError, "neither float nor double dtype");
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
        int Nz0 = %(Z)s->dimensions[0];
        int Nz1 = %(Z)s->dimensions[1];
        int Sx = %(x)s->strides[0] / elemsize;
        int Sy = %(y)s->strides[0] / elemsize;

        /* create appropriate strides for Z, if it is a row or column matrix.
         * In that case, the value of the stride does not really matter, but
         * some versions of BLAS insist that:
         *  - they are not smaller than the number of elements in the array,
         *  - they are not 0.
         */
        int Sz0 = (Nz0 > 1) ? (%(Z)s->strides[0] / elemsize) : (Nz1 + 1);
        int Sz1 = (Nz1 > 1) ? (%(Z)s->strides[1] / elemsize) : (Nz0 + 1);

        if (1)
        {
        if (%(Z)s->strides[0] == elemsize)
        {
            if (%(Z)s->descr->type_num == PyArray_FLOAT)
            {
                //fprintf(stderr, "A\\n");
                float alpha = ((dtype_%(a)s*)%(a)s->data)[0];
                sger_(&Nz0, &Nz1, &alpha,
                    (float*)(%(x)s->data), &Sx,
                    (float*)(%(y)s->data), &Sy,
                    (float*)(%(Z)s->data), &Sz1);
            }
            else if (%(Z)s->descr->type_num == PyArray_DOUBLE)
            {
                double alpha = ((dtype_%(a)s*)%(a)s->data)[0];
                dger_(&Nz0, &Nz1, &alpha,
                    (double*)(%(x)s->data), &Sx,
                    (double*)(%(y)s->data), &Sy,
                    (double*)(%(Z)s->data), &Sz1);
            }
            else { assert(0); }
        }
        else if (%(Z)s->strides[1] == elemsize)
        {
            if (%(Z)s->descr->type_num == PyArray_FLOAT)
            {
                //fprintf(stderr, "B %%i %%i %%i %%i\\n", Nz0, Nz1, Sz0, Sz1);
                float alpha = ((dtype_%(a)s*)(%(a)s->data))[0];
                //fprintf(stderr, "alpha=%%f\\n", alpha);
                //fprintf(stderr, "sx  sy %%i %%i\\n", Sx, Sy);
                sger_(&Nz1, &Nz0, &alpha,
                    (float*)(%(y)s->data), &Sy,
                    (float*)(%(x)s->data), &Sx,
                    (float*)(%(Z)s->data), &Sz0);
            }
            else if (%(Z)s->descr->type_num == PyArray_DOUBLE)
            {
                double alpha = ((dtype_%(a)s*)%(a)s->data)[0];
                dger_(&Nz1, &Nz0, &alpha,
                    (double*)(%(y)s->data), &Sy,
                    (double*)(%(x)s->data), &Sx,
                    (double*)(%(Z)s->data), &Sz0);
            }
            else { assert(0); }
        }
        else { assert(0); }
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
        return (4,)


@local_optimizer([ger, ger_destructive])
def use_c_ger(node):
    # Only float32 and float64 are supported for now.
    if (node.op == ger and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGer(False)(*node.inputs)]
    if (node.op == ger_destructive and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGer(True)(*node.inputs)]

@local_optimizer([CGer(False)])
def make_c_ger_destructive(node):
    if node.op == CGer(False):
        return [CGer(True)(*node.inputs)]



####### ####### #######
# GEMV
####### ####### #######


def gemv_c_code(aa, xx, yy, zz, alpha, beta, destructive, fail):
    """
    zz <- beta * aa + alpha * dot(xx, yy)

    where xx is a matrix, yy and aa are vectors (ergo zz is vector)
    """
    return """

    int elemsize ;
    float fbeta;
    double dbeta;

    if (%(aa)s->nd != 1)
    {PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(aa) != 1"); %(fail)s;}
    if (%(xx)s->nd != 2)
    {PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(xx) != 2"); %(fail)s;}
    if (%(yy)s->nd != 1)
    {PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(yy) != 1"); %(fail)s;}
    if (%(alpha)s->nd != 0)
    {PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(alpha) != 0"); %(fail)s;}
    if (%(beta)s->nd != 0)
    {PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(beta) != 0"); %(fail)s;}

    if (%(aa)s->descr->type_num != %(xx)s->descr->type_num)
    { PyErr_SetString(PyExc_TypeError, "Gemv: aa vs. xx"); %(fail)s; }
    if (%(aa)s->descr->type_num != %(yy)s->descr->type_num)
    { PyErr_SetString(PyExc_TypeError, "Gemv: aa vs. yy"); %(fail)s; }

    if (%(xx)s->dimensions[0] != %(aa)s->dimensions[0])
    {PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[0] != x.shape[0]"); %(fail)s;}
    if (%(xx)s->dimensions[1] != %(yy)s->dimensions[0])
    {PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[1] != y.shape[0]"); %(fail)s;}

    if  (%(aa)s->descr->type_num == PyArray_DOUBLE) { elemsize = 8; }
    else if (%(aa)s->descr->type_num == PyArray_FLOAT) { elemsize = 4;}
    else {PyErr_SetString(PyExc_NotImplementedError, "complex Gemv"); %(fail)s;}

    fbeta = dbeta = ((dtype_%(beta)s*)%(beta)s->data)[0];

    // copy aa if not destructive
    if (!%(destructive)s)
    {
        if ((NULL == %(zz)s)
            || (%(zz)s->dimensions[0] != %(aa)s->dimensions[0]))
        {
            if (%(zz)s) Py_XDECREF(%(zz)s);
            %(zz)s = (PyArrayObject*)PyArray_SimpleNew(1,
                %(aa)s->dimensions, type_num_%(aa)s);
            if(!%(zz)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc gemv output");
                %(fail)s
            }
        }
        assert (%(zz)s != %(aa)s);
        if (dbeta != 0)
        {
            if (%(zz)s->descr->type_num == PyArray_FLOAT)
            {
                float * zoutdata = (float*)%(zz)s->data;
                const float * zdata = (float*)%(aa)s->data;
                int Ai = %(aa)s->strides[0]/sizeof(float);
                int Zi = %(zz)s->strides[0]/sizeof(float);
                for (int i = 0; i < %(aa)s->dimensions[0]; ++i)
                {
                    zoutdata[Zi*i] = fbeta * zdata[Ai*i];
                }
            }
            else if (%(xx)s->descr->type_num == PyArray_DOUBLE)
            {
                double * zoutdata = (double*) %(zz)s->data;
                const double * zdata = (double*)%(aa)s->data;
                int Ai = %(aa)s->strides[0]/sizeof(double);
                int Zi = %(zz)s->strides[0]/sizeof(double);
                for (int i = 0; i < %(aa)s->dimensions[0]; ++i)
                {
                    zoutdata[Zi*i] = dbeta * zdata[Ai*i];
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError, "neither float nor double dtype");
                %(fail)s
            }
            fbeta = dbeta = 1.0;
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
        int Nx0 = %(xx)s->dimensions[0];
        int Nx1 = %(xx)s->dimensions[1];
        int Sx0 = %(xx)s->strides[0] / elemsize;
        int Sx1 = %(xx)s->strides[1] / elemsize;
        int Sz = %(zz)s->strides[0] / elemsize;
        int Sy = %(yy)s->strides[0] / elemsize;

        if (Nx0 * Nx1)
        {
            if (%(xx)s->strides[0] == elemsize)
            {
                if (%(xx)s->descr->type_num == PyArray_FLOAT)
                {
                    //fprintf(stderr, "A\\n");
                    float alpha = ((dtype_%(alpha)s*)%(alpha)s->data)[0];
                    sgemv_(&NOTRANS, &Nx0, &Nx1,
                        &alpha,
                        (float*)(%(xx)s->data), &Sx1,
                        (float*)(%(yy)s->data), &Sy,
                        &fbeta,
                        (float*)(%(zz)s->data), &Sz);
                }
                else if (%(xx)s->descr->type_num == PyArray_DOUBLE)
                {
                    double alpha = ((dtype_%(alpha)s*)%(alpha)s->data)[0];
                    dgemv_(&NOTRANS, &Nx0, &Nx1,
                        &alpha,
                        (double*)(%(xx)s->data), &Sx1,
                        (double*)(%(yy)s->data), &Sy,
                        &dbeta,
                        (double*)(%(zz)s->data), &Sz);
                }
                else
                {
                    assert(0);
                }
            }
            else if (%(xx)s->strides[1] == elemsize)
            {
                if (%(xx)s->descr->type_num == PyArray_FLOAT)
                {
                    //fprintf(stderr, "B %%i %%i %%i %%i\\n", Nz0, Nz1, Sz0, Sz1);
                    float alpha = ((dtype_%(alpha)s*)%(alpha)s->data)[0];
                    //fprintf(stderr, "alpha=%%f\\n", alpha);
                    //fprintf(stderr, "sx  sy %%i %%i\\n", Sx, Sy);
                    sgemv_(&TRANS, &Nx1, &Nx0,
                        &alpha,
                        (float*)(%(xx)s->data), &Sx0,
                        (float*)(%(yy)s->data), &Sy,
                        &fbeta,
                        (float*)(%(zz)s->data), &Sz);
                }
                else if (%(xx)s->descr->type_num == PyArray_DOUBLE)
                {
                    double alpha = ((dtype_%(alpha)s*)%(alpha)s->data)[0];
                    dgemv_(&TRANS, &Nx1, &Nx0,
                        &alpha,
                        (double*)(%(xx)s->data), &Sx0,
                        (double*)(%(yy)s->data), &Sy,
                        &dbeta,
                        (double*)(%(zz)s->data), &Sz);
                }
                else
                {
                    assert(0);
                }
            }
            else
            {
                // if xx is strided in both directions, then just do the gemv with a
                // pair of for loops.
                assert (0);
            }
        }
        else if (dbeta != 1.0)
        {
            // the matrix has at least one dim of length 0
            // so we do this loop, which either iterates over 0 elements
            // or else it does the right thing for length-0 x.
            dtype_%(zz)s * zptr = (dtype_%(zz)s*)(%(zz)s->data);
            for (int i = 0; i < Nx0; ++i)
            {
                zptr[i * Sz] *= dbeta;
            }
        }
    }

    """ % locals()


class CGemv(BaseBLAS, Gemv):
    def c_code(self, node, name, inp, out, sub):
        aa, alpha, xx, yy, beta = inp
        zz, = out
        code = gemv_c_code(
                aa, xx, yy, zz, alpha, beta,
                destructive=int(self.inplace),
                fail=sub['fail'])
        return code

    def c_code_cache_version(self):
        return (2,)


@local_optimizer([gemv_inplace, gemv_no_inplace])
def use_c_gemv(node):
    # Only float32 and float64 are supported for now.
    if (node.op == gemv_no_inplace and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGemv(inplace=False)(*node.inputs)]
    if (node.op == gemv_inplace and
            node.outputs[0].dtype in ['float32', 'float64']):
        return [CGemv(inplace=True)(*node.inputs)]


@local_optimizer([CGemv(inplace=False)])
def make_c_gemv_destructive(node):
    if node.op == CGemv(inplace=False):
        return [CGemv(inplace=True)(*node.inputs)]



####### ####### #######
# Optimizers
####### ####### #######

blas_optdb.register('use_c_blas',
    EquilibriumOptimizer([
        use_c_ger,
        use_c_gemv,
        ],
        max_use_ratio=5),
    20, 'fast_run', 'c_blas')
#print 'BLAS_OPTDB'
#print blas_optdb

# this matches the InplaceBlasOpt defined in blas.py
optdb.register('c_blas_destructive',
        EquilibriumOptimizer([
                make_c_ger_destructive,
                make_c_gemv_destructive,
            ],
            failure_callback=EquilibriumOptimizer.warn_inplace,
            max_use_ratio=5),
        70.0, 'fast_run', 'inplace', 'c_blas')
