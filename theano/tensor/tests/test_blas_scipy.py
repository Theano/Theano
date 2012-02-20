import sys
import numpy
import theano
import theano.tensor as tensor
from theano.tensor.blas_scipy import ScipyGer

from test_blas import TestCase, gemm_no_inplace, TestBlasStrides
from theano.tests.unittest_tools import TestOptimizationMixin

class TestScipyGer(TestCase, TestOptimizationMixin):

    def setUp(self):
        self.mode = theano.compile.get_default_mode()
        self.mode = self.mode.including('fast_run')
        self.mode = self.mode.excluding('c_blas')  # c_blas trumps scipy Ops
        dtype = self.dtype = 'float64'  # optimization isn't dtype-dependent
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor.tensor(dtype=dtype, broadcastable=())
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.Aval = numpy.ones((2,3), dtype=dtype)
        self.xval = numpy.asarray([1,2], dtype=dtype)
        self.yval = numpy.asarray([1.5,2.7,3.9], dtype=dtype)
        if not theano.tensor.blas_scipy.have_fblas:
            self.SkipTest()


    def function(self, inputs, outputs):
        return theano.function(inputs, outputs, self.mode)

    def run_f(self, f):
        f(self.Aval, self.xval, self.yval)
        f(self.Aval[::-1, ::-1], self.xval[::-1], self.yval[::-1])

    def b(self, bval):
        return tensor.as_tensor_variable(numpy.asarray(bval, dtype=self.dtype))

    def test_outer(self):
        f = self.function([self.x, self.y], tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=True))

    def test_A_plus_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=False))
        self.run_f(f) #DebugMode tests correctness

    def test_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + 0.1 * tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=False))
        self.run_f(f) #DebugMode tests correctness

    def test_scaled_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                0.2 * self.A + 0.1 * tensor.outer(self.x, self.y))
        self.assertFunctionContains(f, gemm_no_inplace)
        self.run_f(f) #DebugMode tests correctness

class TestBlasStridesScipy(TestBlasStrides):
    mode = theano.compile.get_default_mode()
    mode = mode.including('fast_run').excluding('gpu', 'c_blas')
