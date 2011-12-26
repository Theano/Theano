import sys
import numpy
import theano
import theano.tensor as tensor
from theano.tensor.blas_c import CGer
from theano.tensor.blas_scipy import ScipyGer
from theano.tensor.blas import Ger

from test_blas import TestCase, TestOptimizationMixin, gemm_no_inplace

class TestCGer(TestCase, TestOptimizationMixin):

    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('fast_run')
        dtype = self.dtype = 'float64'  # optimization isn't dtype-dependent
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor.tensor(dtype=dtype, broadcastable=())
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.Aval = numpy.ones((2,3), dtype=dtype)
        self.xval = numpy.asarray([1,2], dtype=dtype)
        self.yval = numpy.asarray([1.5,2.7,3.9], dtype=dtype)
        if not theano.tensor.blas_scipy.optimizations_enabled:
            self.SkipTest()

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

