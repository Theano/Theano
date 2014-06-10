import unittest
from theano.tests import unittest_tools as utt

import numpy as np

import theano
from theano import tensor

from theano.tensor.sort import sort, SortOp
from theano.tensor.sort import argsort, ArgSortOp


class test_sort(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(seed=utt.fetch_seed())
        self.m_val = self.rng.rand(3, 2)
        self.v_val = self.rng.rand(4)

    def test1(self):
        a = tensor.dmatrix()
        w = sort(a)
        f = theano.function([a], w)
        assert np.allclose(f(self.m_val), np.sort(self.m_val))

    def test2(self):
        a = tensor.dmatrix()
        axis = tensor.scalar()
        w = sort(a, axis)
        f = theano.function([a, axis], w)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            assert np.allclose(gv, gt)

    def test3(self):
        a = tensor.dvector()
        w2 = sort(a)
        f = theano.function([a], w2)
        gv = f(self.v_val)
        gt = np.sort(self.v_val)
        assert np.allclose(gv, gt)

    def test4(self):
        a = tensor.dmatrix()
        axis = tensor.scalar()
        l = sort(a, axis, "mergesort")
        f = theano.function([a, axis], l)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            assert np.allclose(gv, gt)

    def test5(self):
        a1 = SortOp("mergesort", [])
        a2 = SortOp("quicksort", [])

        #All the below should give true
        assert a1 != a2
        assert a1 == SortOp("mergesort", [])
        assert a2 == SortOp("quicksort", [])

    def test_None(self):
        a = tensor.dmatrix()
        l = sort(a, None)
        f = theano.function([a], l)
        gv = f(self.m_val)
        gt = np.sort(self.m_val, None)
        assert np.allclose(gv, gt)


class TensorInferShapeTester(utt.InferShapeTester):
    def test_sort(self):
        x = tensor.matrix()
        self._compile_and_check(
                [x],
                [sort(x)],
                [np.random.randn(10, 40).astype(theano.config.floatX)],
                SortOp)
        self._compile_and_check(
                [x],
                [sort(x, axis=None)],
                [np.random.randn(10, 40).astype(theano.config.floatX)],
                SortOp)


def test_argsort():
    #Set up
    rng = np.random.RandomState(seed=utt.fetch_seed())
    m_val = rng.rand(3, 2)
    v_val = rng.rand(4)

    #Example 1
    a = tensor.dmatrix()
    w = argsort(a)
    f = theano.function([a], w)
    gv = f(m_val)
    gt = np.argsort(m_val)
    assert np.allclose(gv, gt)

    #Example 2
    a = tensor.dmatrix()
    axis = tensor.scalar()
    w = argsort(a, axis)
    f = theano.function([a, axis], w)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        assert np.allclose(gv, gt)

    #Example 3
    a = tensor.dvector()
    w2 = argsort(a)
    f = theano.function([a], w2)
    gv = f(v_val)
    gt = np.argsort(v_val)
    assert np.allclose(gv, gt)

    #Example 4
    a = tensor.dmatrix()
    axis = tensor.scalar()
    l = argsort(a, axis, "mergesort")
    f = theano.function([a, axis], l)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        assert np.allclose(gv, gt)

    #Example 5
    a = tensor.dmatrix()
    axis = tensor.scalar()
    a1 = ArgSortOp("mergesort", [])
    a2 = ArgSortOp("quicksort", [])
    #All the below should give true
    assert a1 != a2
    assert a1 == ArgSortOp("mergesort", [])
    assert a2 == ArgSortOp("quicksort", [])

    #Example 6: Testing axis=None
    a = tensor.dmatrix()
    w2 = argsort(a, None)
    f = theano.function([a], w2)
    gv = f(m_val)
    gt = np.argsort(m_val, None)
    assert np.allclose(gv, gt)


