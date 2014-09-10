"""
 Tests for the Op decorator
"""
import numpy as np

from theano.tests import unittest_tools as utt
from theano import function
import theano
from theano import tensor
from theano.tensor import dmatrix, dvector
from numpy import allclose
from theano.compile import as_op
import pickle


# This is for test_pickle, since the function still has to be
# reachable from pickle (as in it cannot be defined inline)
@as_op([dmatrix, dmatrix], dmatrix)
def mul(a, b):
    return a*b


class OpDecoratorTests(utt.InferShapeTester):
    def test_1arg(self):
        x = dmatrix('x')

        @as_op(dmatrix, dvector)
        def diag(x):
            return np.diag(x)

        fn = function([x], diag(x))
        r = fn([[1.5, 5], [2, 2]])
        r0 = np.array([1.5, 2])

        assert allclose(r, r0), (r, r0)

    def test_2arg(self):
        x = dmatrix('x')
        x.tag.test_value = np.zeros((2, 2))
        y = dvector('y')
        y.tag.test_value = [0, 0]

        @as_op([dmatrix, dvector], dvector)
        def diag_mult(x, y):
            return np.diag(x) * y

        fn = function([x, y], diag_mult(x, y))
        r = fn([[1.5, 5], [2, 2]], [1, 100])
        r0 = np.array([1.5, 200])

        assert allclose(r, r0), (r, r0)

    def test_infer_shape(self):
        x = dmatrix('x')
        x.tag.test_value = np.zeros((2, 2))
        y = dvector('y')
        y.tag.test_value = [0, 0]

        def infer_shape(node, shapes):
            x, y = shapes
            return [y]

        @as_op([dmatrix, dvector], dvector, infer_shape)
        def diag_mult(x, y):
            return np.diag(x) * y

        self._compile_and_check([x, y], [diag_mult(x, y)],
                                [[[1.5, 5], [2, 2]], [1, 100]],
                                diag_mult.__class__, warn=False)

    def test_pickle(self):
        x = dmatrix('x')
        y = dmatrix('y')

        m = mul(x, y)

        s = pickle.dumps(m)
        m2 = pickle.loads(s)

        assert m2.owner.op == m.owner.op


def test_shape_i_hash():
    assert isinstance(theano.tensor.opt.Shape_i(np.int64(1)).__hash__(),
                      int)
