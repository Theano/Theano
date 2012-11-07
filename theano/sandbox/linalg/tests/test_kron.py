from nose.plugins.skip import SkipTest
import numpy

from theano import tensor, function
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg.kron import Kron, kron

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False

if not imported_scipy:
    raise SkipTest('Kron Op need the scipy package to be installed')


class TestKron(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestKron, self).setUp()
        self.op_class = Kron
        self.op = kron

    def test_perform(self):
        x = tensor.dmatrix()
        y = tensor.dmatrix()
        f = function([x, y], kron(x, y))

        for shp0 in [(8, 6), (5, 8)]:
            for shp1 in [(5, 7), (3, 3)]:
                a = numpy.random.rand(*shp0)
                b = numpy.random.rand(*shp1)
                out = f(a, b)
                assert numpy.allclose(out, scipy.linalg.kron(a, b))

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.dmatrix()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(8, 5),
                                 numpy.random.rand(3, 7)],
                                self.op_class)
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(2, 5),
                                 numpy.random.rand(6, 3)],
                                self.op_class)

if __name__ == "__main__":
    t = TestKron('setUp')
    t.setUp()
    t.test_perform()
    t.test_infer_shape()
