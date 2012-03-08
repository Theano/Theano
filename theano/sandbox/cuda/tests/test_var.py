import unittest
import numpy
from nose.plugins.skip import SkipTest

import theano
from theano import tensor

from theano.ifelse import ifelse
from theano import sparse
from theano.tensor import TensorType
from theano.tests import unittest_tools as utt
from theano.sandbox.cuda.var import float32_shared_constructor as f32sc
from theano.sandbox.cuda import CudaNdarrayType, cuda_available

# Skip test if cuda_ndarray is not available.
if cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

def test_float32_shared_constructor():

    npy_row = numpy.zeros((1,10), dtype='float32')

    def eq(a,b):
        return a==b

    # test that we can create a CudaNdarray
    assert (f32sc(npy_row).type == CudaNdarrayType((False, False)))

    # test that broadcastable arg is accepted, and that they
    # don't strictly have to be tuples
    assert eq(
            f32sc(npy_row, broadcastable=(True, False)).type,
            CudaNdarrayType((True, False)))
    assert eq(
            f32sc(npy_row, broadcastable=[True, False]).type,
            CudaNdarrayType((True, False)))
    assert eq(
            f32sc(npy_row, broadcastable=numpy.array([True, False])).type,
            CudaNdarrayType([True, False]))

    # test that we can make non-matrix shared vars
    assert eq(
            f32sc(numpy.zeros((2,3,4,5), dtype='float32')).type,
            CudaNdarrayType((False,)*4))

def test_givens():
    # Test that you can use a TensorType expression to replace a
    # CudaNdarrayType in the givens dictionary.
    # This test case uses code mentionned in #757
    data = numpy.float32([1,2,3,4])
    x = f32sc(data)
    y = x**2
    f = theano.function([], y, givens={x:x+1})

class T_updates(unittest.TestCase):
    # Test that you can use a TensorType expression to update a
    # CudaNdarrayType in the updates dictionary.

    def test_1(self):
        data = numpy.float32([1,2,3,4])
        x = f32sc(data)
        y = x**2
        f = theano.function([], y, updates={x:x+1})

    def test_2(self):
        # This test case uses code mentionned in #698
        data = numpy.random.rand(10,10).astype('float32')
        output_var = f32sc(name="output",
                value=numpy.zeros((10,10), 'float32'))

        x = tensor.fmatrix('x')
        output_updates = {output_var:x**2}
        output_givens = {x:data}
        output_func = theano.function(inputs=[], outputs=[],
                updates=output_updates, givens=output_givens)
        output_func()

    def test_3(self):
        # Test that broadcastable dimensions don't screw up
        # update expressions.
        data = numpy.random.rand(10,10).astype('float32')
        output_var = f32sc(name="output",
                value=numpy.zeros((10,10), 'float32'))

        # the update_var has type matrix, and the update expression
        # is a broadcasted scalar, and that should be allowed.
        output_func = theano.function(inputs=[], outputs=[],
                updates={output_var:output_var.sum().dimshuffle('x', 'x')})
        output_func()

class T_ifelse(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        self.rng = numpy.random.RandomState(seed=utt.fetch_seed())

    def test_cuda_tensor(self):
        data = self.rng.rand(4).astype('float32')
        x = f32sc(data)
        y = x + 1
        cond = theano.tensor.iscalar('cond')

        assert isinstance(x.type, CudaNdarrayType)
        assert isinstance(y.type, TensorType)

        out1 = ifelse(cond, x, y)
        out2 = ifelse(cond, y, x)

        assert isinstance(out1.type, TensorType)
        assert isinstance(out2.type, TensorType)

        f = theano.function([cond], out1)
        g = theano.function([cond], out2)

        assert numpy.all(f(0) == data+1)
        assert numpy.all(f(1) == data)
        assert numpy.all(g(0) == data)
        assert numpy.all(g(1) == data+1)

    def test_dtype_mismatch(self):
        data = self.rng.rand(5).astype('float32')
        x = f32sc(data)
        y = tensor.cast(x, 'float64')
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)

    def test_ndim_mismatch(self):
        data = self.rng.rand(5).astype('float32')
        x = f32sc(data)
        y = tensor.fcol('y')
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)

    def test_broadcast_mismatch(self):
        data = self.rng.rand(2,3).astype('float32')
        x = f32sc(data)
        print x.broadcastable
        y = tensor.frow('y')
        print y.broadcastable
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)

    def test_sparse_tensor_error(self):
        data = self.rng.rand(2,3).astype('float32')
        x = f32sc(data)
        y = sparse.matrix('csc', dtype='float32', name='y')
        z = sparse.matrix('csr', dtype='float32', name='z')
        cond = theano.tensor.iscalar('cond')

        # Right now (2012-01-19), a ValueError gets raised, but I thing
        # a TypeError (like in the other cases) would be fine.
        self.assertRaises((TypeError, ValueError), ifelse, cond, x, y)
        self.assertRaises((TypeError, ValueError), ifelse, cond, y, x)
        self.assertRaises((TypeError, ValueError), ifelse, cond, x, z)
        self.assertRaises((TypeError, ValueError), ifelse, cond, z, x)
        self.assertRaises((TypeError, ValueError), ifelse, cond, y, z)
        self.assertRaises((TypeError, ValueError), ifelse, cond, z, y)


