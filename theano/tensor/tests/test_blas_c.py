import sys
import numpy
import theano
import theano.tensor as tensor

from theano.tensor.blas_c import CGer
from theano.tensor.blas_scipy import ScipyGer
from theano.tensor.blas import Ger

from theano.tensor.blas_c import CGemv
from theano.tensor.blas_scipy import ScipyGer
from theano.tensor.blas import Gemv

from theano.tests import unittest_tools
from theano.tests.unittest_tools import TestOptimizationMixin

from test_blas import TestCase
from test_blas import BaseGemv

mode_blas_opt = theano.compile.get_default_mode().including(
    'BlasOpt', 'specialize', 'InplaceBlasOpt', 'c_blas')


class TestCGer(TestCase, TestOptimizationMixin):

    def setUp(self, dtype='float64'):
        self.dtype = dtype
        self.mode = theano.compile.get_default_mode().including('fast_run')
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor.tensor(dtype=dtype, broadcastable=())
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.Aval = numpy.ones((2,3), dtype=dtype)
        self.xval = numpy.asarray([1,2], dtype=dtype)
        self.yval = numpy.asarray([1.5,2.7,3.9], dtype=dtype)

    def function(self, inputs, outputs):
        return theano.function(inputs, outputs,
                mode=self.mode,
                #allow_inplace=True,
                )

    def run_f(self, f):
        return f(self.Aval, self.xval, self.yval)

    def b(self, bval):
        return tensor.as_tensor_variable(numpy.asarray(bval, dtype=self.dtype))

    def test_eq(self):
        self.assert_(CGer(True) == CGer(True))
        self.assert_(CGer(False) == CGer(False))
        self.assert_(CGer(False) != CGer(True))

        self.assert_(CGer(True) != ScipyGer(True))
        self.assert_(CGer(False) != ScipyGer(False))
        self.assert_(CGer(True) != Ger(True))
        self.assert_(CGer(False) != Ger(False))

        # assert that eq works for non-CGer instances
        self.assert_(CGer(False) != None)
        self.assert_(CGer(True) != None)

    def test_hash(self):
        self.assert_(hash(CGer(True)) == hash(CGer(True)))
        self.assert_(hash(CGer(False)) == hash(CGer(False)))
        self.assert_(hash(CGer(False)) != hash(CGer(True)))

    def test_optimization_pipeline(self):
        f = self.function([self.x, self.y], tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=True))
        f(self.xval, self.yval)  #DebugMode tests correctness

    def test_optimization_pipeline_float(self):
        self.setUp('float32')
        f = self.function([self.x, self.y], tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=True))
        f(self.xval, self.yval)  #DebugMode tests correctness

    def test_int_fails(self):
        self.setUp('int32')
        f = self.function([self.x, self.y], tensor.outer(self.x, self.y))
        self.assertFunctionContains0(f, CGer(destructive=True))
        self.assertFunctionContains0(f, CGer(destructive=False))

    def test_A_plus_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=False))
        self.run_f(f) #DebugMode tests correctness

    def test_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + 0.1 * tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=False))
        self.run_f(f) #DebugMode tests correctness


class TestCGemv(TestCase, TestOptimizationMixin):
    """
    Tests of CGemv specifically.

    Generic tests of Gemv-compatibility, including both dtypes are done below in
    TestCGemvFloat32 and TestCGemvFloat64
    """
    def setUp(self, dtype='float64'):
        self.dtype = dtype
        self.mode = theano.compile.get_default_mode().including('fast_run')
        # matrix
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.Aval = numpy.ones((2,3), dtype=dtype)

        # vector
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.xval = numpy.asarray([1,2], dtype=dtype)
        self.yval = numpy.asarray([1.5,2.7,3.9], dtype=dtype)

        # scalar
        self.a = tensor.tensor(dtype=dtype, broadcastable=())

    def test_optimizations_vm(self):
        ''' Test vector dot matrix '''
        f = theano.function([self.x, self.A],
                theano.dot(self.x, self.A),
                mode=self.mode)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, tensor.dot)
        self.assertFunctionContains1(f, CGemv(True))

        # Assert they produce the same output
        assert numpy.allclose(f(self.xval, self.Aval),
                numpy.dot(self.xval, self.Aval))

    def test_optimizations_mv(self):
        ''' Test matrix dot vector '''
        f = theano.function([self.A, self.y],
                theano.dot(self.A, self.y),
                mode=self.mode)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, tensor.dot)
        self.assertFunctionContains1(f, CGemv(True))

        # Assert they produce the same output
        assert numpy.allclose(f(self.Aval, self.yval),
                numpy.dot(self.Aval, self.yval))

    def t_gemv1(self, m_shp):
        ''' test vector2 + dot(matrix, vector1) '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v1 = theano.shared(numpy.array(rng.uniform(size=(m_shp[1],)), dtype='float32'))
        v2_orig = numpy.array(rng.uniform(size=(m_shp[0],)), dtype='float32')
        v2 = theano.shared(v2_orig)
        m  = theano.shared(numpy.array(rng.uniform(size=m_shp), dtype='float32'))

        f = theano.function([], v2 + tensor.dot(m, v1),
                mode=self.mode)

        # Assert they produce the same output
        assert numpy.allclose(f(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = [n.op for n in f.maker.env.toposort()]
        assert topo == [CGemv(inplace=False)], topo

        #test the inplace version
        f = theano.function([], [],
                updates={v2:v2+theano.dot(m,v1)},
                mode=self.mode)

        # Assert they produce the same output
        f()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = [n.op for n in f.maker.env.toposort()]
        assert topo == [CGemv(inplace=True)]

    def test_gemv1(self):
        self.t_gemv1((3,2))
        self.t_gemv1((0,2))
        self.t_gemv1((3,0))
        self.t_gemv1((0,0))

    def test_gemv_dimensions(self, dtype='float32'):
        alpha = theano.shared(theano._asarray(1.0, dtype=dtype),
                name='alpha')
        beta = theano.shared(theano._asarray(1.0, dtype=dtype),
                name='beta')

        z = beta * self.y + alpha * tensor.dot(self.A, self.x)
        f = theano.function([self.A, self.x, self.y], z,
                mode=self.mode)

        # Matrix value
        A_val = numpy.ones((5,3), dtype=dtype)
        # Different vector length
        ones_3 = numpy.ones(3, dtype=dtype)
        ones_4 = numpy.ones(4, dtype=dtype)
        ones_5 = numpy.ones(5, dtype=dtype)
        ones_6 = numpy.ones(6, dtype=dtype)

        f(A_val, ones_3, ones_5)
        self.assertRaises(ValueError, f, A_val, ones_4, ones_5)
        self.assertRaises(ValueError, f, A_val, ones_3, ones_6)
        self.assertRaises(ValueError, f, A_val, ones_4, ones_6)


class TestCGemvFloat32(TestCase, BaseGemv, TestOptimizationMixin):
    mode = mode_blas_opt
    dtype = 'float32'
    gemv = CGemv(inplace=False)
    gemv_inplace = CGemv(inplace=True)


class TestCGemvFloat64(TestCase, BaseGemv, TestOptimizationMixin):
    mode = mode_blas_opt
    dtype = 'float64'
    gemv = CGemv(inplace=False)
    gemv_inplace = CGemv(inplace=True)
