from theano.tensor.opt import in2out
from theano.gof import Op
from blas import Ger, ger, ger_destructive
from blas import ldflags, blas_header_text
from blas import blas_optdb, optdb, local_optimizer

ger_c_code = """

int elemsize ;

if (%(A)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(A) != 2"); %(fail)s;}
if (%(x)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 1"); %(fail)s;}
if (%(y)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 1"); %(fail)s;}
if (%(a)s->nd != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 0"); %(fail)s;}

if (%(A)s->descr->type_num != %(x)s->descr->type_num)
{ PyErr_SetString(PyExc_TypeError, "A vs. x"); %(fail)s; }
if (%(A)s->descr->type_num != %(y)s->descr->type_num)
{ PyErr_SetString(PyExc_TypeError, "A vs. y"); %(fail)s; }

if (%(A)s->dimensions[0] != %(x)s->dimensions[0])
{PyErr_SetString(PyExc_ValueError, "A.shape[0] != x.shape[0]"); %(fail)s;}
if (%(A)s->dimensions[1] != %(y)s->dimensions[0])
{PyErr_SetString(PyExc_ValueError, "A.shape[1] != y.shape[0]"); %(fail)s;}

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
        %(Z)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, type_num_%(A)s);
        if(!%(Z)s) {
            PyErr_SetString(PyExc_MemoryError, "failed to alloc gemm_no_inplace output");
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
                zoutdata[Zi*i*+Zj*j] = zdata[Ai*i+Aj*j];
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
    int Sz0 = %(Z)s->strides[0] / elemsize;
    int Sz1 = %(Z)s->strides[1] / elemsize;
    int Sx = %(x)s->strides[0] / elemsize;
    int Sy = %(y)s->strides[0] / elemsize;

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

"""

class CGer(Ger):

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

    def c_code(self, node, name, inp, out, sub):
        print 'C_CODE'
        A, a, x, y = inp
        Z, = out
        destructive = int(self.destructive)
        fail = sub['fail']
        code = ger_c_code % locals()
        return code

    def c_code_cache_version(self):
        return ()

    def make_thunk(*args, **kwargs):
        # skip over Ger.make_thunk
        return Op.make_thunk(*args, **kwargs)

@local_optimizer([ger, ger_destructive])
def use_c_ger(node):
    if node.op == ger:
        print "inserting C_GER"
        return [CGer(False)(*node.inputs)]
    if node.op == ger_destructive:
        print "inserting dstruc C_GER"
        return [CGer(True)(*node.inputs)]

@local_optimizer([CGer(False)])
def make_c_ger_destructive(node):
    if node.op == CGer(False):
        print "inserting destructive C_GER"
        return [CGer(True)(*node.inputs)]

use_c_blas = in2out(use_c_ger)
make_c_blas_destructive = in2out(make_c_ger_destructive)

blas_optdb.register('c_blas',
    use_c_blas,
    90, 'fast_run')
print 'BLAS_OPTDB'
print blas_optdb

# this matches the InplaceBlasOpt defined in blas.py
optdb.register('make_c_blas_destructive',
        make_c_blas_destructive,
        70.0, 'fast_run', 'inplace')
