import unittest

import numpy

from theano import gof, tensor,compile

from theano.sparse.tests.test_basic import eval_outputs
from theano.sparse.basic import _is_sparse_variable, _is_dense_variable, as_sparse_variable, _is_sparse, _mtypes, _mtype_to_str
from theano.sparse import SparseType, dense_from_sparse, transpose

###############
#
# TrueDot
#
class TrueDot(gof.op.Op):
    """
    Attributes:
    grad_preserves_dense - a boolean flags [default: True].
    grad_preserves_dense controls whether gradients with respect to inputs
    are converted to dense matrices when the corresponding input y is
    dense (not in a L{SparseVariable} wrapper). This is generally a good idea
    when L{Dot} is in the middle of a larger graph, because the types
    of gy will match that of y. This conversion might be inefficient if
    the gradients are graph outputs though, hence this mask.

    @todo: Simplify code by splitting into DotSS and DotSD.
    """
    def __init__(self, grad_preserves_dense=True):
        self.grad_preserves_dense = grad_preserves_dense
    def __eq__(self, other):
        return type(self) == type(other) and self.grad_preserves_dense == other.grad_preserves_dense
    def __hash__(self):
        return hash(self.grad_preserves_dense)
    def __ne__(self, other):
        return not (self == other)
    def make_node(self, x, y):
        """
        :note: Because of trickiness of implementing, we assume that the left argument x is SparseVariable (not dense)
        """
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()

        if not _is_sparse_variable(x):
            raise TypeError(x)

        # These are the conversions performed by scipy.sparse.dot
        if x.type.format == "csc" or x.type.format == "coo":
            myformat = "csc"
        elif x.type.format == "csr":
            myformat = "csr"
        else:
            raise NotImplementedError()

        inputs = [x, y]    # Need to convert? e.g. assparse
        outputs = [SparseType(dtype = x.type.dtype, format = myformat).make_variable()]
        return gof.Apply(self, inputs, outputs)
    def perform(self, node, inp, out_):
        """
        @todo: Verify that output is sufficiently sparse, and raise a warning if it is not
        @todo: Also determine that we are storing the output in the best storage format?
        """
        x, y = inp
        out, = out_
        rval = x.dot(y)
        out[0] = rval
    def grad(self, inp, grads):
        x, y = inp
        gz, = grads
        assert _is_sparse_variable(gz)
        assert _is_sparse_variable(x)
        rval = [true_dot(gz, y.T), true_dot(x.T, gz)]
        if _is_dense_variable(y):
            if self.grad_preserves_dense:
                rval[1] = dense_from_sparse(rval[1])
        return rval

def true_dot(x, y, grad_preserves_dense=True):
    """
    @todo: Maybe the triple-transposition formulation (when x is dense)
    is slow. See if there is a direct way to do this.
    """
    if hasattr(x, 'getnnz'): x = as_sparse_variable(x)
    if hasattr(y, 'getnnz'): y = as_sparse_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)
    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError()
    if x_is_sparse_variable:
        return TrueDot(grad_preserves_dense)(x, y)
    else:
        assert y_is_sparse_variable
        return transpose(TrueDot(grad_preserves_dense)(y.T, x.T))


class test_true_dot(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    def test_basicSS(self):
        for mtype in _mtypes:
            x = as_sparse_variable(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.assertTrue(_is_sparse_variable(x))

            xT = x.T
            self.assertTrue(_is_sparse_variable(xT))

            zop = true_dot(x,xT)
            self.assertTrue(_is_sparse_variable(zop))
            z = eval_outputs([zop])
            self.assertTrue(_is_sparse(z))
            self.assertTrue(z.shape == (500,500))
            self.assertTrue(type(z) is mtype)

            w = mtype((500,500))
            w[(10, 10)] = 1
            w[(20, 20)] = 4
            self.assertTrue(z.shape == w.shape)
            self.assertTrue(type(z) == type(w))
            self.assertTrue(z.dtype == w.dtype)

            #self.assertTrue(z == w)
            self.assertTrue(abs(z-w).nnz == 0)

            z = z.todense()
            w = w.todense()
            self.assertTrue((z == w).all() == True)

    def test_basicSD(self):
        for mtype in _mtypes:
            x = as_sparse_variable(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.assertTrue(_is_sparse_variable(x))

            y = tensor.as_tensor_variable([[1., 2], [3, 4], [2, 1]])
            self.assertTrue(_is_dense_variable(y))

            zop = true_dot(x,y)
            self.assertTrue(_is_sparse_variable(zop))
            z = eval_outputs([zop])
            self.assertTrue(_is_sparse(z))
            self.assertTrue(z.shape == (500,2))
            self.assertTrue(type(z) is mtype)

            w = mtype((500,2))
            w[(10, 0)] = 3.
            w[(20, 0)] = 4
            w[(10, 1)] = 4
            w[(20, 1)] = 2
            self.assertTrue(z.shape == w.shape)
            self.assertTrue(type(z) == type(w))
            self.assertTrue(z.dtype == w.dtype)

            #self.assertTrue(z == w)
            self.assertTrue(abs(z-w).nnz == 0)

            z = z.todense()
            w = w.todense()
            self.assertTrue((z == w).all() == True)

    def test_basicDS(self):
        for mtype in _mtypes:
            x = as_sparse_variable(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.assertTrue(_is_sparse_variable(x))

            y = tensor.as_tensor_variable([[1., 2], [3, 4], [2, 1]])
            self.assertTrue(_is_dense_variable(y))

            x.data = x.data.T
            y.data = y.data.T

            zop = true_dot(y, x)
            zop = transpose(true_dot(y, x))
            self.assertTrue(_is_sparse_variable(zop))
            z = eval_outputs([zop])
            self.assertTrue(_is_sparse(z))
            self.assertTrue(z.shape == (500,2))
#            self.assertTrue(type(z) is mtype)

            w = mtype((500,2))
            w[(10, 0)] = 3.
            w[(20, 0)] = 4
            w[(10, 1)] = 4
            w[(20, 1)] = 2
            self.assertTrue(z.shape == w.shape)
            # Type should switch from csr to csc and vice-versa, so don't perform this test
            #self.assertTrue(type(z) == type(w))
            self.assertTrue(z.dtype == w.dtype)

            # Type should switch from csr to csc and vice-versa, so don't perform this test
            #self.assertTrue(z == w)
            self.assertTrue(abs(z-w).nnz == 0)

            z = z.todense()
            w = w.todense()
            self.assertTrue((z == w).all() == True)

    def test_graph_bprop0(self):
        for mtype in _mtypes:
            x = tensor.matrix('x') #TensorType('float64', broadcastable=[False,False], name='x')
            w = SparseType(dtype = 'float64', format = _mtype_to_str[mtype]).make_variable()
            xw = dense_from_sparse(true_dot(w, x))
            y = dense_from_sparse(true_dot(w.T, xw))
            diff = x-y
            loss = tensor.sum(tensor.sqr(diff))
            gw = tensor.grad(loss, w)
            trainfn = compile.function([x, w], [y, loss, gw])

            x = numpy.asarray([[1., 2], [3, 4], [2, 1]])
            w = mtype((500,3))
            w[(10, 1)] = 1
            w[(20, 2)] = 2
            lr = 0.001
            y, origloss, gw = trainfn(x, w)
            for epoch in xrange(50):
                y, loss, gw = trainfn(x, w)
                w = w - (lr * gw)
                print loss

            self.assertTrue(origloss > loss)
            self.assertTrue('1.05191241115' == str(loss))

    def test_graph_bprop_rand(self):
        for i in range(10):
            xorig = numpy.random.rand(3,2)
            for mtype in _mtypes:
                x = tensor.matrix('x')
                w = SparseType(dtype = 'float64', format = _mtype_to_str[mtype]).make_variable()
                xw = dense_from_sparse(true_dot(w, x))
                y = dense_from_sparse(true_dot(w.T, xw))
                diff = x-y
                loss = tensor.sum(tensor.sqr(diff))
                gw = tensor.grad(loss, w)
                trainfn = compile.function([x, w], [y, loss, gw])

                x = xorig
                w = mtype((500,3))
                w[(10, 1)] = 1
                w[(20, 2)] = 2
                lr = 0.001
                y, origloss, gw = trainfn(x, w)
                for epoch in xrange(50):
                    y, loss, gw = trainfn(x, w)
                    w = w - (lr * gw)

                self.assertTrue(origloss > loss)
