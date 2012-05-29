import theano
import numpy as np
from theano import tensor as T
from theano.tests import unittest_tools as utt

from theano.tensor.extra_ops import *


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
        a = np.random.random(500)

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
