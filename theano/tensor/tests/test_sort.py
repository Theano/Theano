from __future__ import absolute_import, print_function, division
from itertools import product, chain
from functools import reduce
import unittest
from theano.tests import unittest_tools as utt

import numpy as np

import theano
from theano import tensor

from theano.tensor.sort import sort, SortOp
from theano.tensor.sort import argsort, ArgSortOp
from theano.tensor.sort import argtopk, ArgTopKOp

_dtypes = (
    'float32', 'float64',
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64')
_int_dtypes = (
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64')


def gen_unique_vector(size, dtype):
    # generate a randomized vector with unique elements
    retval = np.cumsum(np.random.uniform(1.01, 3.01, size))
    return (retval[np.random.permutation(size)] - size).astype(dtype)


class Test_sort(unittest.TestCase):

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

        # All the below should give true
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

    def test_grad_vector(self):
        data = np.random.rand(10).astype(theano.config.floatX)
        utt.verify_grad(sort, [data])

    def test_grad_none_axis(self):
        data = np.random.rand(10).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        utt.verify_grad(lambda x: sort(x, 0), [data])

        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])

    def test_grad_negative_axis_2d(self):
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])

    def test_grad_negative_axis_3d(self):
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])

    def test_grad_negative_axis_4d(self):
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, -4), [data])

    def test_grad_nonnegative_axis_2d(self):
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = np.random.rand(2, 3).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])

    def test_grad_nonnegative_axis_3d(self):
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = np.random.rand(2, 3, 4).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])

    def test_grad_nonnegative_axis_4d(self):
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])
        data = np.random.rand(2, 3, 4, 2).astype(theano.config.floatX)
        utt.verify_grad(lambda x: sort(x, 3), [data])


class SortInferShapeTester(utt.InferShapeTester):
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
    # Set up
    rng = np.random.RandomState(seed=utt.fetch_seed())
    m_val = rng.rand(3, 2)
    v_val = rng.rand(4)

    # Example 1
    a = tensor.dmatrix()
    w = argsort(a)
    f = theano.function([a], w)
    gv = f(m_val)
    gt = np.argsort(m_val)
    assert np.allclose(gv, gt)

    # Example 2
    a = tensor.dmatrix()
    axis = tensor.lscalar()
    w = argsort(a, axis)
    f = theano.function([a, axis], w)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        assert np.allclose(gv, gt)

    # Example 3
    a = tensor.dvector()
    w2 = argsort(a)
    f = theano.function([a], w2)
    gv = f(v_val)
    gt = np.argsort(v_val)
    assert np.allclose(gv, gt)

    # Example 4
    a = tensor.dmatrix()
    axis = tensor.lscalar()
    l = argsort(a, axis, "mergesort")
    f = theano.function([a, axis], l)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        assert np.allclose(gv, gt)

    # Example 5
    a = tensor.dmatrix()
    axis = tensor.lscalar()
    a1 = ArgSortOp("mergesort", [])
    a2 = ArgSortOp("quicksort", [])
    # All the below should give true
    assert a1 != a2
    assert a1 == ArgSortOp("mergesort", [])
    assert a2 == ArgSortOp("quicksort", [])

    # Example 6: Testing axis=None
    a = tensor.dmatrix()
    w2 = argsort(a, None)
    f = theano.function([a], w2)
    gv = f(m_val)
    gt = np.argsort(m_val, None)
    assert np.allclose(gv, gt)


def test_argsort_grad():
    # Testing grad of argsort
    data = np.random.rand(2, 3).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-1), [data])

    data = np.random.rand(2, 3, 4, 5).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-3), [data])

    data = np.random.rand(2, 3, 3).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=2), [data])


class Test_topk(unittest.TestCase):

    def setUp(self):
        pass

    @utt.parameterized.expand(product(
        _dtypes, _int_dtypes, [-1, 0, None]))
    def test_sanity(self, dtype, out_dtype, axis):
        x = tensor.vector(name='x', dtype=dtype)
        fn = theano.function([x], argtopk(x, 1, axis=axis, out_dtype=out_dtype))
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == np.asarray([0], dtype=out_dtype)

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, 10, -10, 'n//2', 'n-1', '-n', '1-n'),
            ('float32', 'int32'),
            ('int32', 'int64')),
        ((2049, 1337, 'float32', 'int32'),)))
    def test_1d(self, size, k, dtype, out_dtype):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = argtopk(x, k, out_dtype=out_dtype)
        fn = theano.function([x], y)
        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.argsort(xval)[idx].astype(out_dtype)
        print(yval)
        print(goal)
        print(np.argsort(xval))

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yval)] == xval[np.sort(goal)])

    @utt.parameterized.expand(chain(
        product(
            (18, 62, 258),
            (1, -1, 'n//2'),
            ('int32', 'float32')),
        ((2048, 1337, 'float32'),)))
    def test_1d_collision(self, size, k, dtype):
        # with non-unique kth max value
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = argtopk(x, k, out_dtype='int32')
        fn = theano.function([x], y)
        xval = np.repeat(np.random.uniform(-100., 100., size=size // 2).astype(dtype), 2)
        xval = xval[np.random.permutation(size)]
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.argsort(xval)[idx].astype('int32')
        print(goal)
        print(np.argsort(xval))
        assert np.allclose(np.sort(xval[yval]), np.sort(xval[goal]))

    @utt.parameterized.expand(product(
        ((1, 1), (2, 3), (17, 15), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2017, 5, 3)),
        (1, -1, '(1+n)//2', 'n-1', '-n', '1-n'),
        ('float32', 'int32'),
        ('int32', 'int64')))
    def test_nd(self, shp, k_, dtype, out_dtype):
        ndim = len(shp)
        for axis in range(-ndim, ndim):
            if isinstance(k_, str):
                k = eval(k_.replace('n', str(shp[axis])))
            else:
                k = k_

            if k == 0:
                continue

            x = theano.tensor.tensor(
                name='x', broadcastable=(False,) * len(shp), dtype=dtype)
            y = argtopk(x, k, axis=axis, out_dtype=out_dtype)
            fn = theano.function([x], y)
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, dtype).reshape(shp)
            yval = fn(xval)
            idx = slice(-k, None) if k > 0 else slice(-k)
            l = axis % ndim
            r = ndim - l
            idx = (slice(None),) * l + (idx,) + (slice(None),) * (r - 1)
            goal = np.argsort(xval, axis=axis)[idx].astype(out_dtype)

            print(dict(k=k, axis=axis, shp=shp))
            print(np.sort(yval, axis=axis))
            print(np.sort(goal, axis=axis))
            # print(np.argsort(xval))
            assert np.all(np.sort(yval, axis=axis) == np.sort(goal, axis=axis))


class ArgTopKInferShapeTester(utt.InferShapeTester):
    @utt.parameterized.expand(product(
        ((2, 3), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2, 4, 3, 1)),
        (1, -1, '(1+n)//2', 'n-1', '-n', '1-n')))
    def test_infer_shape(self, shp, k_):
        ndim = len(shp)
        for axis in range(-ndim, ndim):
            if isinstance(k_, str):
                k = eval(k_.replace('n', str(shp[axis])))
            else:
                k = k_

            if k == 0:
                continue

            x = theano.tensor.tensor(
                name='x', broadcastable=(False,) * len(shp),
                dtype=theano.config.floatX)
            y = argtopk(x, k, axis=axis, out_dtype='int32')
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, theano.config.floatX).reshape(shp)
            self._compile_and_check(
                [x], [y], [xval], ArgTopKOp)
