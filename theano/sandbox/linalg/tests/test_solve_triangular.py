import numpy
from theano.gof import Op
from theano import tensor, function
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg import (matrix_inverse, triu, tril,
    solve_triangular, SolveTriangular)

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False
assert imported_scipy, ("Scipy not available. Scipy is needed for the"
        " TestSolveTriangular class")

  
class TestSolveTriangular(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):

        super(TestSolveTriangular, self).setUp()
        self.op_class = SolveTriangular
        self.op = solve_triangular

    def test_perform(self):

        init = numpy.random.rand(4, 4)
        x = tensor.dmatrix()
        for val in [True, False]:
            if val:
                a = numpy.tril(init)
            else:
                a = numpy.triu(init)
                
            y = tensor.dmatrix()
            f = function([x, y], self.op(x, y, lower=val))
            for shp in [(4, 5), (4, 1)]:
                b = numpy.random.rand(*shp)
                out = f(a, b)
                assert numpy.allclose(out,
                        scipy.linalg.solve_triangular(a, b, lower=val))

            y = tensor.dvector()
            f = function([x, y], self.op(x, y, lower=val))
            b = numpy.random.rand(4)
            out = f(a, b)
            assert numpy.allclose(out,
                    scipy.linalg.solve_triangular(a, b, lower=val))
          
            y = tensor.dmatrix()
            f = function([x, y], self.op(x, y, lower=val,
                unit_diagonal=False, overwrite_b=True))
            for shp in [(4, 5), (4, 1)]:
                b = numpy.random.rand(*shp)
                out = f(a, b)
                assert numpy.allclose(out,
                        scipy.linalg.solve_triangular(a, b, lower=val,
                        unit_diagonal=False, overwrite_b=True))

            y = tensor.dvector()
            f = function([x, y], self.op(x, y, lower=val,
                unit_diagonal=False, overwrite_b=True))
            b = numpy.random.rand(4)
            out = f(a, b)
            assert numpy.allclose(out,
                    scipy.linalg.solve_triangular(a, b, lower=val,
                    unit_diagonal=False, overwrite_b=True))

    def test_gradient(self):

        utt.verify_grad(SolveTriangular(lower=True),
                        [numpy.tril(numpy.random.rand(5, 5)),
                                numpy.random.rand(5, 1)],
                        n_tests=1, rng=TestSolveTriangular.rng)
        
        utt.verify_grad(SolveTriangular(lower=True),
                        [numpy.tril(numpy.random.rand(4, 4)),
                                       numpy.random.rand(4, 3)],
                      n_tests=1, rng=TestSolveTriangular.rng)
        
        utt.verify_grad(SolveTriangular(lower=True),
                        [numpy.tril(numpy.random.rand(4, 4)),
                                         numpy.random.rand(4)],
                      n_tests=1, rng=TestSolveTriangular.rng)
        
        utt.verify_grad(SolveTriangular(lower=False),
                        [numpy.triu(numpy.random.rand(5, 5)),
                                numpy.random.rand(5, 1)],
                        n_tests=1, rng=TestSolveTriangular.rng)

        utt.verify_grad(SolveTriangular(lower=False),
                        [numpy.triu(numpy.random.rand(4, 4)),
                                       numpy.random.rand(4, 3)],
                      n_tests=1, rng=TestSolveTriangular.rng)

        utt.verify_grad(SolveTriangular(lower=False),
                        [numpy.triu(numpy.random.rand(4, 4)),
                                         numpy.random.rand(4)],
                      n_tests=1, rng=TestSolveTriangular.rng)

    def test_infer_shape(self):

        x = tensor.dmatrix()
        y = tensor.dmatrix()

        a55 = numpy.random.rand(5, 5)
        a44 = numpy.random.rand(4, 4)
        b52 = numpy.random.rand(5, 2)
        b41 = numpy.random.rand(4, 1)
        b4 = numpy.random.rand(4)

        self._compile_and_check([x, y], [self.op(x, y, lower=True)],
                                [numpy.tril(a55), b52], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=True)],
                                [numpy.tril(a44), b41], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=False)],
                                [numpy.triu(a55), b52], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=False)],
                                [numpy.triu(a44), b41], self.op_class)

        y = tensor.dvector()

        self._compile_and_check([x, y], [self.op(x, y, lower=True)],
                                [numpy.triu(a44), b4], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=False)],
                                [numpy.triu(a44), b4], self.op_class)


if __name__ == "__main__":
    t = TestSolveTriangular('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
