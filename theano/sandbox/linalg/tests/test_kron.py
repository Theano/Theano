from nose.plugins.skip import SkipTest
import numpy

import theano
from theano import tensor, function
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg.kron import kron

floatX = theano.config.floatX

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False


class TestKron(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestKron, self).setUp()
        self.op = kron

    def test_perform(self):
        if not imported_scipy:
            raise SkipTest('kron tests need the scipy package to be installed')

        for shp0 in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            for shp1 in [(6,), (6, 7), (6, 7, 8), (6, 7, 8, 9)]:
                if len(shp0) + len(shp1) == 2:
                    continue
                x = tensor.tensor(dtype='floatX',
                                  broadcastable=(False,) * len(shp0))
                y = tensor.tensor(dtype='floatX',
                                  broadcastable=(False,) * len(shp1))
                f = function([x, y], kron(x, y))
                a = numpy.asarray(self.rng.rand(*shp0)).astype(floatX)
                b = self.rng.rand(*shp1).astype(floatX)
                out = f(a, b)
                assert numpy.allclose(out, scipy.linalg.kron(a, b))

    def test_numpy_2d(self):
        for shp0 in [(2, 3)]:
            for shp1 in [(6, 7)]:
                if len(shp0) + len(shp1) == 2:
                    continue
                x = tensor.tensor(dtype='floatX',
                                  broadcastable=(False,) * len(shp0))
                y = tensor.tensor(dtype='floatX',
                                  broadcastable=(False,) * len(shp1))
                f = function([x, y], kron(x, y))
                a = numpy.asarray(self.rng.rand(*shp0)).astype(floatX)
                b = self.rng.rand(*shp1).astype(floatX)
                out = f(a, b)
                assert numpy.allclose(out, numpy.kron(a, b))


if __name__ == "__main__":
    t = TestKron('setUp')
    t.setUp()
    t.test_perform()
    t.test_infer_shape()
