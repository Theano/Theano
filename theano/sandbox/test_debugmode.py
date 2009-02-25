import numpy

import scipy.sparse
from theano import gof
import theano.sparse
import theano
import theano.tensor
import debugmode
import theano.compile

def test0():
    x = theano.tensor.dvector()
    f = theano.function([x], (2.*x + 7) / 2., mode=debugmode.OptCheck())
    print f([1,2])

class BROKEN_ON_PURPOSE_StructuredDotCSC(gof.Op):
    def __init__(self, py_offset):
        gof.Op.__init__(self)
        self.py_offset = py_offset
    def __eq__(self, other):
        return type(self) == type(other) and (self.py_offset == other.py_offset)
    def __hash__(self):
        return 29834 ^ hash(type(self)) ^ hash(self.py_offset)
    def make_node(self, a_val, a_ind, a_ptr, a_nrows, b):
        a_nrows = theano.tensor.as_tensor(a_nrows)
        assert a_val.type.dtype == b.type.dtype
        r = gof.Apply(self, [a_val, a_ind, a_ptr, a_nrows, b], 
                [theano.tensor.tensor(a_val.type.dtype, (False, False))])
        return r

    def perform(self, node, (a_val, a_ind, a_ptr, a_nrows, b), (out,)):
        a = scipy.sparse.csc_matrix((a_val, a_ind, a_ptr), 
                (a_nrows, b.shape[0]),
                copy = False)
#       TODO: todense() is automatic in 0.7.0, just remove the following line:
#        out[0] = numpy.asarray(a.dot(b).todense())
        out[0] = a.dot(b)+0.5 if self.py_offset else a.dot(b) #ERROR TO ADD THIS CRAPPY OFFSET
        #assert _is_dense(out[0])

    def c_code(self, node, name, (a_val, a_ind, a_ptr, a_nrows, b), (z,), sub):
        return """
        if (%(a_val)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (%(a_ind)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (%(a_ptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (%(a_nrows)s->nd != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); %(fail)s;}
        if (%(b)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (%(a_val)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "a_val dtype not NPY_DOUBLE"); %(fail)s;}

        if (%(a_ind)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (%(a_ptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (%(a_nrows)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_nrows dtype not INT32"); %(fail)s;}

        if (%(b)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "b's dtype not NPY_DOUBLE"); %(fail)s;}

        if (%(a_val)s->dimensions[0] != %(a_ind)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if (%(a_ptr)s->dimensions[0] != %(b)s->dimensions[0]+1)
        {PyErr_SetString(PyExc_NotImplementedError, "a's number of columns doesn't match b's rows"); %(fail)s;}

        if ((!%(z)s)
            || (%(z)s->dimensions[0] != ((npy_int32 *)%(a_nrows)s->data)[0])
            || (%(z)s->dimensions[1] != %(b)s->dimensions[1])
            )
        {
            if (%(z)s) Py_DECREF(%(z)s);
            npy_intp dims[] = {0,0};
            dims[0] = ((npy_int32 *)%(a_nrows)s->data)[0];
            dims[1] = %(b)s->dimensions[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(b)s->descr->type_num);
        }

        {
            //the output array has size M x N
            npy_intp M = %(z)s->dimensions[0];
            npy_intp N = %(z)s->dimensions[1];
            npy_intp K = %(b)s->dimensions[0];
            npy_intp Szm = %(z)s->strides[0] / %(z)s->descr->elsize;
            npy_intp Szn = %(z)s->strides[1] / %(z)s->descr->elsize;
            //npy_intp Sbm = %(b)s->strides[0] / %(b)s->descr->elsize;
            npy_intp Sbn = %(b)s->strides[1] / %(b)s->descr->elsize;
            npy_intp Sval = %(a_val)s->strides[0] / %(a_val)s->descr->elsize;
            npy_intp Sind = %(a_ind)s->strides[0] / %(a_ind)s->descr->elsize;
            npy_intp Sptr = %(a_ptr)s->strides[0] / %(a_ptr)s->descr->elsize;

            npy_double * __restrict__ Dz = (npy_double*)%(z)s->data;
            //const npy_double * __restrict__ Db = (npy_double*)%(b)s->data;
            const npy_double * __restrict__ Dval = (npy_double*)%(a_val)s->data;
            const npy_int32 * __restrict__ Dind = (npy_int32*)%(a_ind)s->data;
            const npy_int32 * __restrict__ Dptr = (npy_int32*)%(a_ptr)s->data;

            //npy_intp nnz = %(a_ind)s->dimensions[0];

            //clear the output array
            for (npy_intp m = 0; m < M; ++m)
            {
                for (npy_intp n = 0; n < N; ++n)
                {
                    //Dz[m*Szm + n*Szn] = 0.0;
                    Dz[m*Szm + n*Szn] = 0.5;  //here is the py_offset amount
                }
            }

            //iterate over the sparse array, making the most of an entry wherever we find it.
            //
            // Normal matrix matrix multiply:
            // for m
            //   for n
            //     for k
            //        z[m,n] += a[m,k] * b[k,n]
            // Here instead:
            // for k
            //   for m (sparse)
            //     for n
            //        z[m,n] += a[m,k] * b[k,n]

            for (npy_int32 k = 0; k < K; ++k)
            {
                const npy_double * __restrict__ bk = (double *)(%(b)s->data + %(b)s->strides[0] * k);

                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1) * Sptr]; ++m_idx)
                {
                    npy_int32 m = Dind[m_idx * Sind];
                    const double Amk = Dval[m_idx * Sval];

                    npy_double * __restrict__ zm = (npy_double *)(%(z)s->data + %(z)s->strides[0] * m);

                    if (m >= %(z)s->dimensions[0]) 
                    {PyErr_SetString(PyExc_NotImplementedError, "illegal row index in a"); %(fail)s;}

                    for(npy_int32 n = 0; n < N; ++n)
                    {
                        zm[n*Szn] += Amk * bk[n*Sbn];
                    }
                }
            }
        }
        """% dict(locals(), **sub)
# inconsistent is a invalid op, whose perform and c_code do not match
inconsistent = BROKEN_ON_PURPOSE_StructuredDotCSC(False)
# off_by_half is a good op, that is different from theano.sparse.sd_csc
off_by_half = BROKEN_ON_PURPOSE_StructuredDotCSC(True) 


def test_badclinkeroutput():

    vals = theano.tensor.dvector()
    inds = theano.tensor.ivector()
    ptrs = theano.tensor.ivector()
    nrows = theano.tensor.iscalar()

    b = theano.tensor.dmatrix()

    f_good = theano.function([vals, inds, ptrs, nrows, b], 
            theano.sparse.StructuredDotCSC()(vals, inds, ptrs, nrows, b), 
            mode=debugmode.OptCheck(check_c_code=True))
    f_inconsistent = theano.function([vals, inds, ptrs, nrows, b], 
            inconsistent(vals, inds, ptrs, nrows, b), 
            mode=debugmode.OptCheck(check_c_code=True))

    #this should evaluate with no error
    rval_good = f_good([1.0, 2.0, 3.0],
            [0,1,2],
            [0,1,2,3],
            3,
            numpy.asarray([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]))
    try:
        rval = f_inconsistent([1.0, 2.0, 3.0],
                [0,1,2],
                [0,1,2,3],
                3,
                numpy.asarray([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]))
    except debugmode.BadClinkerOutput, e:
        print repr(e)
        assert e.r.owner.op is inconsistent
        return #TEST PASS

    assert False  #an error should have been detected
        

def test_badoptimization():
    @gof.local_optimizer([theano.sparse.sd_csc])
    def insert_broken_csc(node):
        if node.op == theano.sparse.sd_csc:
            return [off_by_half(*node.inputs)]
        return False
    edb = gof.EquilibriumDB()
    edb.register('insert_broken_csc', insert_broken_csc, 'all')
    opt = edb.query('+all')

    vals = theano.tensor.dvector()
    inds = theano.tensor.ivector()
    ptrs = theano.tensor.ivector()
    nrows = theano.tensor.iscalar()

    b = theano.tensor.dmatrix()

    f = theano.function([vals, inds, ptrs, nrows, b], 
            theano.sparse.sd_csc(vals, inds, ptrs, nrows, b), 
            mode=debugmode.OptCheck(optimizer=opt, check_c_code=True))

    try:
        rval = f([1.0, 2.0, 3.0],
                [0,1,2],
                [0,1,2,3],
                3,
                numpy.asarray([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]))
    except debugmode.BadOptimization, e:
        assert str(e.reasons[e.new_r][0][0]) == 'insert_broken_csc'
        return #TEST PASS

    assert False

