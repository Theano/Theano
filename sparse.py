import unittest

import numpy
from scipy import sparse

import gof.lib
import core
import grad

# Wrapper type

class SparseR(gof.PythonR):
    """
    Attribute:
    format - a subclass of sparse.spmatrix indicating self.data.__class__
    """
    def __init__(self, x = core.UNCOMPUTED, constant = False, 
            format = sparse.csr_matrix):
        gof.PythonR.__init__(self, x, constant)
        self.format = isinstance(x, sparse.spmatrix) and x.__class__ or format

    def set_value(self, value):
        """Extend base impl, assert value is sparse matrix"""
        gof.PythonR.set_value(self,value)
        if self.data is not core.UNCOMPUTED:
            if not isinstance(self.data, sparse.spmatrix):
                print self.data.__class__
                print self.owner.__class__
                raise TypeError(('hrm',value))

    def __add__(left, right): return add(left, right)
    def __radd__(right, left): return add(left, right)

    T = property(lambda self: transpose(self), doc = "Return aliased transpose")

# convenience base class
class op(gof.PythonOp, grad.update_gradient_via_grad):
    pass

#
# Conversion
#

# convert a sparse matrix to an ndarray
class sparse2dense(op):
    def gen_outputs(self): return [core.NumpyR()]
    def impl(x): return numpy.asarray(x.todense())
    def grad(self, x, gz): 
        if x.format is sparse.coo_matrix: return dense2coo(gz)
        if x.format is sparse.csc_matrix: return dense2csc(gz)
        if x.format is sparse.csr_matrix: return dense2csr(gz)
        if x.format is sparse.dok_matrix: return dense2dok(gz)
        if x.format is sparse.lil_matrix: return dense2lil(gz)

# convert an ndarray to various sorts of sparse matrices.
class _dense2sparse(op):
    def gen_outputs(self): return [SparseR()]
    def grad(self, x, gz): return sparse2dense(gz)
class dense2coo(_dense2sparse):
    def impl(x): return sparse.coo_matrix(x)
class dense2csc(_dense2sparse):
    def impl(x): return sparse.csc_matrix(x)
class dense2csr(_dense2sparse):
    def impl(x): return sparse.csr_matrix(x)
class dense2dok(_dense2sparse):
    def impl(x): return sparse.dok_matrix(x)
class dense2lil(_dense2sparse):
    def impl(x): return sparse.lil_matrix(x)


# Linear Algebra

class add(op):
    def gen_outputs(self): return [SparseR()]
    def impl(csr,y): return csr + y

class transpose(op):
    def gen_outputs(self): return [SparseR()]
    def impl(x): return x.transpose() 
    def grad(self, x, gz): return transpose(gz)

class _testCase_transpose(unittest.TestCase):
    def setUp(self):
        core.build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        core.pop_mode()
    def test_transpose(self):
        a = SparseR(sparse.csr_matrix(sparse.speye(5,3)))
        self.failUnless(a.data.shape == (5,3))
        ta = transpose(a)
        self.failUnless(ta.data.shape == (3,5))

class dot(op):
    """
    Attributes:
    grad_preserves_dense - an array of boolean flags (described below)


    grad_preserves_dense controls whether gradients with respect to inputs are
    converted to dense matrices when the corresponding inputs are not in a
    SparseR wrapper.  This can be a good idea when dot is in the middle of a
    larger graph, because the types of gx and gy will match those of x and y.
    This conversion might be annoying if the gradients are graph outputs though,
    hence this mask.
    """
    def __init__(self, *args, **kwargs):
        op.__init__(self, *args, **kwargs)
        self.grad_preserves_dense = [True, True]
    def gen_outputs(self): return [SparseR()]
    def impl(x,y):
        if hasattr(x, 'getnnz'):
            return x.dot(y)
        else:
            return y.transpose().dot(x.transpose()).transpose()

    def grad(self, x, y, gz):
        rval = [dot(gz, y.T), dot(x.T, gz)]
        for i in 0,1:
            if not isinstance(self.inputs[i], SparseR):
                #assume it is a dense matrix
                if self.grad_preserves_dense[i]:
                    rval[i] = sparse2dense(rval[i])
        return rval

class _testCase_dot(unittest.TestCase):
    def setUp(self):
        core.build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        core.pop_mode()
    def test_basic0(self):
        for mtype in [sparse.csc_matrix, sparse.csr_matrix]:
            x = SparseR(mtype(sparse.speye(5,3)))
            y = core.NumpyR(numpy.random.rand(3, 2))

            z = dot(x,y)
            self.failUnless(z.data.shape == (5,2))
            self.failUnless(type(z.data) is mtype)
    def test_basic1(self):
        """dot: sparse left"""
        a = numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0, 0]],
                dtype='float64')
        b = numpy.random.rand(5, 3)
        for mtype in [sparse.csr_matrix, sparse.csc_matrix, sparse.dok_matrix,
                sparse.lil_matrix]:#, sparse.coo_matrix]:
            #print type(a), mtype
            m = mtype(a)
            ab = m.dot(b)
            try:
                z = dot(SparseR(m),gof.lib.PythonR(b))
                self.failUnless(z.data.shape == ab.shape)
                self.failUnless(type(z.data) == type(ab))
            except Exception, e:
                print mtype, e, str(e)
                raise
    def test_basic2(self):
        """dot: sparse right"""
        a = numpy.random.rand(2, 5)
        b = numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0, 0]],
                dtype='float64').transpose()

        for mtype in [sparse.csr_matrix, sparse.csc_matrix, sparse.dok_matrix,
                sparse.lil_matrix]:#, sparse.coo_matrix]:
            m = mtype(b)
            ab = m.transpose().dot(a.transpose()).transpose()
            z = dot(gof.lib.PythonR(a),SparseR(mtype(b)))
            self.failUnless(z.data.shape == ab.shape)
            self.failUnless(type(z.data) == type(ab))
    def test_graph_bprop0(self):
        x = core.NumpyR(numpy.random.rand(10,2))
        w = SparseR(sparse.csr_matrix(numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0,
            0]],dtype='float64')))

        for epoch in xrange(50):
            xw = sparse2dense(dot(x, w))
            y = sparse2dense(dot(xw, transpose(w)))
            loss = core.sum(core.sqr(x-y))
            gy = y-x
            g = grad.Grad({y:gy})
            g.bprop()
            lr = 0.002
            g(w).data[1,0] = 0
            g(w).data[1,4] = 0
            w.data = -lr * g(w).data + w.data

        self.failUnless('3.08560636025' == str(loss))

    def test_graph_bprop1(self):
        x = core.NumpyR(numpy.random.rand(10,2))
        w = SparseR(sparse.csr_matrix(numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0,
            0]],dtype='float64')))

        for epoch in xrange(50):
            xw = sparse2dense(dot(x, w))
            y = sparse2dense(dot(xw, transpose(w)))
            loss = core.sum(core.sqr(x-y))
            g = grad.grad(loss)
            lr = 0.001

            g(w).data[1,0] = 0
            g(w).data[1,4] = 0
            w.data = -lr * g(w).data + w.data

        self.failUnless('3.08560636025' == str(loss))
        
if __name__ == '__main__':
    unittest.main()

