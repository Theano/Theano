import numpy as np
import numpy

import theano
from theano.tests import unittest_tools as utt
from theano.tensor.extra_ops import *
from theano import tensor as T
from theano import tensor, function, scalar


class TestBinCountOp(utt.InferShapeTester):
    def setUp(self):
        super(TestBinCountOp, self).setUp()
        self.op_class = BinCountOp
        self.op = BinCountOp()

    def test_bincountOp(self):
        x = T.lvector('x')
        w = T.dvector('w')
        a = np.random.random_integers(50, size=(25))
        weights = np.random.random((25,))

        f1 = theano.function([x], bincount(x))
        f2 = theano.function([x, w], bincount(x, weights=w))
        f3 = theano.function([x], bincount(x, minlength=23))
        f4 = theano.function([x], bincount(x, minlength=5))

        assert (np.bincount(a) == f1(a)).all()
        assert np.allclose(np.bincount(a, weights=weights), f2(a, weights))
        assert (np.bincount(a, minlength=23) == f3(a)).all()
        assert (np.bincount(a, minlength=5) == f3(a)).all()

    def test_infer_shape(self):
        x = T.lvector('x')

        self._compile_and_check([x],
                                [bincount(x)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)

        weights = np.random.random((25,))
        self._compile_and_check([x],
                                [bincount(x, weights=weights)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)

        self._compile_and_check([x],
                                [bincount(x, minlength=60)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)

        self._compile_and_check([x],
                                [bincount(x, minlength=5)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)


class TestDiffOp(utt.InferShapeTester):
    nb = 10  # Number of time iterating for n

    def setUp(self):
        super(TestDiffOp, self).setUp()
        self.op_class = DiffOp
        self.op = DiffOp()

    def test_diffOp(self):
        x = T.dmatrix('x')
        a = np.random.random((30, 50))

        f = theano.function([x], diff(x))
        assert np.allclose(np.diff(a), f(a))

        for axis in range(len(a.shape)):
            for k in range(TestDiffOp.nb):
                g = theano.function([x], diff(x, n=k, axis=axis))
                assert np.allclose(np.diff(a, n=k, axis=axis), g(a))

    def test_infer_shape(self):
        x = T.dmatrix('x')
        a = np.random.random((30, 50))

        self._compile_and_check([x],
                                [self.op(x)],
                                [a],
                                self.op_class)

        for axis in range(len(a.shape)):
            for k in range(TestDiffOp.nb):
                self._compile_and_check([x],
                                        [diff(x, n=k, axis=axis)],
                                        [a],
                                        self.op_class)

    def test_grad(self):
        x = T.vector('x')
        a = np.random.random(50)

        gf = theano.function([x], T.grad(T.sum(diff(x)), x))
        utt.verify_grad(self.op, [a])

        for k in range(TestDiffOp.nb):
            dg = theano.function([x], T.grad(T.sum(diff(x, n=k)), x))
            utt.verify_grad(DiffOp(n=k), [a])


class TestSqueezeOp(utt.InferShapeTester):
    def setUp(self):
        super(TestSqueezeOp, self).setUp()
        self.op_class = SqueezeOp
        self.op = SqueezeOp(out_nd=1)

    def test_squeezeOp(self):
        x = T.dmatrix('x')
        a = np.random.random((1, 50))

        f = theano.function([x], squeeze(x, out_nd=1))
        assert np.allclose(np.squeeze(a), f(a))

        x = T.dtensor4('x')
        f = theano.function([x], squeeze(x, out_nd=2))

        a = np.random.random((1, 1, 2, 3))
        assert np.allclose(np.squeeze(a), f(a))

        a = np.random.random((1, 2, 2, 1))
        assert np.allclose(np.squeeze(a), f(a))

        a = np.random.random((4, 1, 2, 1))
        assert np.allclose(np.squeeze(a), f(a))

    def test_grad(self):
        x = T.dtensor4('x')
        a = np.random.random((1, 1, 3, 4))

        gf = theano.function([x], T.grad(T.sum(squeeze(x, out_nd=1)), x))
        utt.verify_grad(SqueezeOp(out_nd=2), [a])


class TestRepeatOp(utt.InferShapeTester):
    nb = 5

    def setUp(self):
        super(TestRepeatOp, self).setUp()
        self.op_class = RepeatOp
        self.op = RepeatOp()

    def test_repeatOp(self):
        x = T.dmatrix('x')
        a = np.random.random((30, 50))

        for axis in [None] + range(len(a.shape)):
            for repeats in range(TestRepeatOp.nb):
                f = theano.function([x], repeat(x, repeats, axis=axis))
                assert np.allclose(np.repeat(a, repeats, axis=axis), f(a))

    def test_infer_shape(self):
        x = T.dvector('x')
        m = T.iscalars('m')
        a = np.random.random(50)

        self._compile_and_check([x, m],
                                [repeat(x, m)],
                                [a, 2],
                                self.op_class)

        x = T.dmatrix('x')
        a = np.random.random((40, 50))
        for axis in range(len(a.shape)):
            self._compile_and_check([x, m],
                                    [repeat(x, m, axis=axis)],
                                    [a, 2],
                                    self.op_class)

        m = T.lvector('m')
        repeats = np.random.random_integers(5, size=(40, ))
        self._compile_and_check([x, m],
                                [repeat(x, m, axis=0)],
                                [a, repeats],
                                self.op_class)

    def test_grad(self):
        for ndim in range(3)[1:]:
            x = T.TensorType('float64', [False] * ndim)
            a = np.random.random((10, ) * ndim)

            for axis in [None] + range(ndim):
                utt.verify_grad(lambda x: RepeatOp(axis=axis)(x, 3), [a])


class TestBartlett(utt.InferShapeTester):

    def setUp(self):
        super(TestBartlett, self).setUp()
        self.op_class = Bartlett
        self.op = bartlett

    def test_perform(self):
        x = tensor.lscalar()
        f = function([x], self.op(x))
        M = numpy.random.random_integers(3, 50, size=())
        assert numpy.allclose(f(M), numpy.bartlett(M))
        assert numpy.allclose(f(0), numpy.bartlett(0))
        assert numpy.allclose(f(-1), numpy.bartlett(-1))
        b = numpy.array([17], dtype='uint8')
        assert numpy.allclose(f(b[0]), numpy.bartlett(b[0]))

    def test_infer_shape(self):
        x = tensor.lscalar()
        self._compile_and_check([x], [self.op(x)],
                                [numpy.random.random_integers(3, 50, size=())],
                                self.op_class)
        self._compile_and_check([x], [self.op(x)], [0], self.op_class)
        self._compile_and_check([x], [self.op(x)], [1], self.op_class)


if __name__ == "__main__":
    t = TestBartlett('setUp')
    t.setUp()
    t.test_perform()
    t.test_infer_shape()


class TestFillDiagonal(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestFillDiagonal, self).setUp()
        self.op_class = FillDiagonal
        self.op = fill_diagonal

    def test_perform(self):
        x = tensor.dmatrix()
        y = tensor.dscalar()
        f = function([x, y], fill_diagonal(x, y))
        for shp in [(8, 8), (5, 8), (8, 5)]:
            a = numpy.random.rand(*shp)
            val = numpy.random.rand()
            out = f(a, val)
            # We can't use numpy.fill_diagonal as it is bugged.
            assert numpy.allclose(numpy.diag(out), val)
            assert (out == val).sum() == min(a.shape)

        # test for 3d tensor
        a = numpy.random.rand(3, 3, 3)
        x = tensor.dtensor3()
        y = tensor.dscalar()
        f = function([x, y], fill_diagonal(x, y))
        val = numpy.random.rand() + 10
        out = f(a, val)
        # We can't use numpy.fill_diagonal as it is bugged.
        assert out[0, 0, 0] == val
        assert out[1, 1, 1] == val
        assert out[2, 2, 2] == val
        assert (out == val).sum() == min(a.shape)

    def test_gradient(self):
        utt.verify_grad(fill_diagonal, [numpy.random.rand(5, 8),
                                        numpy.random.rand()],
                        n_tests=1, rng=TestFillDiagonal.rng)
        utt.verify_grad(fill_diagonal, [numpy.random.rand(8, 5),
                                        numpy.random.rand()],
                        n_tests=1, rng=TestFillDiagonal.rng)

    def test_infer_shape(self):
        z = tensor.dtensor3()
        x = tensor.dmatrix()
        y = tensor.dscalar()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(8, 5),
                                 numpy.random.rand()],
                                self.op_class)
        self._compile_and_check([z, y], [self.op(z, y)],
                                [numpy.random.rand(8, 8, 8),
                                 numpy.random.rand()],
                                self.op_class)

if __name__ == "__main__":
    t = TestFillDiagonal('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
