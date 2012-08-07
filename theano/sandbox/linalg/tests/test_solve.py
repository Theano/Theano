import numpy
from theano.gof import Op, Apply
from theano import tensor, function
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg import matrix_inverse, kron, solve, Solve


try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False
      

class TestSolve(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):

        super(TestSolve, self).setUp()
        self.op_class = Solve
        self.op = solve

    def test_perform(self):

        x = tensor.dmatrix()
        y = tensor.dmatrix()
        f = function([x, y], self.op(x, y))
        a = numpy.random.rand(4, 4)
        for shp1 in [(4, 5), (4, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b))

        a = numpy.random.rand(1, 1)
        for shp1 in [(1, 5), (1, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b))

        y = tensor.dvector()
        f = function([x, y], self.op(x, y))
        a = numpy.random.rand(4, 4)
        b = numpy.random.rand(4)
        out = f(a, b)
        assert numpy.allclose(out, scipy.linalg.solve(a, b))

        x = tensor.dmatrix()
        y = tensor.dmatrix()
        f = function([x, y], self.op(x, y, True, True))
        a = numpy.random.rand(4, 4)
        a = numpy.dot(a, numpy.transpose(a))
        for shp1 in [(4, 5), (4, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b, True, True))

        a = numpy.random.rand(1, 1)
        a = numpy.dot(a, numpy.transpose(a))
        for shp1 in [(1, 5), (1, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b, True, True))

        y = tensor.dvector()
        f = function([x, y], self.op(x, y))
        a = numpy.random.rand(4, 4)
        a = numpy.dot(a, numpy.transpose(a))
        b = numpy.random.rand(4)
        out = f(a, b)
        assert numpy.allclose(out, scipy.linalg.solve(a, b, True, True))

    def test_gradient(self):

        utt.verify_grad(self.op, [numpy.random.rand(5, 5),
                                numpy.random.rand(5, 1)],
                        n_tests=1, rng=TestSolve.rng)

        utt.verify_grad(self.op, [numpy.random.rand(4, 4),
                                       numpy.random.rand(4, 3)],
                      n_tests=1, rng=TestSolve.rng)

        utt.verify_grad(self.op, [numpy.random.rand(4, 4),
                                         numpy.random.rand(4)],
                      n_tests=1, rng=TestSolve.rng)

    def test_infer_shape(self):

        x = tensor.dmatrix()
        y = tensor.dmatrix()

        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(5, 5),
                                 numpy.random.rand(5, 2)],
                                self.op_class)

        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(4, 4),
                                 numpy.random.rand(4, 1)],
                                self.op_class)
        y = tensor.dvector()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(4, 4),
                                 numpy.random.rand(4)],
                                self.op_class)


if __name__ == "__main__":
    
    t = TestSolve('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
   
