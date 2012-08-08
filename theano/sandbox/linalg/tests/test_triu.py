import numpy
from theano.gof import Op
from theano import tensor, function
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg import triu, Triu


class TestTriu(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestTriu, self).setUp()
        self.op_class = Triu
        self.op = triu

    def test_perform(self):
        x = tensor.dmatrix()
        y = tensor.iscalar()
        f = function([x, y], self.op(x, y))
        f_def = function([x], self.op(x))

        for shp in [(2, 5), (1, 4), (4, 1)]:
            m = numpy.random.rand(*shp)
            out = f_def(m)
            assert numpy.allclose(out, numpy.triu(m))

            for k in [0, 1, 2, -1]:
                out = f(m, k)
                assert numpy.allclose(out, numpy.triu(m, k))

    def test_gradient(self):
        def helper(x):
            return self.op(x, k)

        for shp in [(5, 5), (1, 4), (4, 1)]:
            m = numpy.random.rand(*shp)
            utt.verify_grad(self.op, [m], n_tests=1, rng=TestTriu.rng)

            for k in [0, 1, 2, -1]:
                utt.verify_grad(helper, [m], n_tests=1, rng=TestTriu.rng)

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.iscalar()

        for shp in [(2, 5), (1, 4), (4, 1)]:
            m = numpy.random.rand(*shp)
            self._compile_and_check([x], [self.op(x)],
                                [m], self.op_class)
            for k in [0, 1, 2, -1]:
                self._compile_and_check([x, y], [self.op(x, y)],
                                [m, k], self.op_class)


if __name__ == "__main__":
    t = TestTriu('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
