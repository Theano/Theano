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
from theano.tensor.sort import topk, argtopk, topk_and_argtopk, TopKOp

_dtypes = (
    'float32', 'float64',
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64')
_int_dtypes = (
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64')


def gen_unique_vector(size, dtype):
    # generate a randomized vector with unique elements
    retval = np.arange(size) * 3. + np.random.uniform(-1., 1.)
    return (retval[np.random.permutation(size)] - size * 1.5).astype(dtype)


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


class Test_TopK(unittest.TestCase):

    def setUp(self):
        pass

    @utt.parameterized.expand(product(
        _dtypes, _int_dtypes, [-1, 0, None]))
    def test_argtopk_sanity(self, dtype, idx_dtype, axis):
        x = tensor.vector(name='x', dtype=dtype)
        fn = theano.function([x], argtopk(x, 1, axis=axis, idx_dtype=idx_dtype))
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == np.asarray([0], dtype=idx_dtype)

    @utt.parameterized.expand(product(
        _dtypes, [-1, 0, None]))
    def test_topk_sanity(self, dtype, axis):
        x = tensor.vector(name='x', dtype=dtype)
        fn = theano.function([x], topk(x, 1, axis=axis))
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == xval

    @utt.parameterized.expand(product(
        _dtypes, _int_dtypes, [-1, 0, None]))
    def test_combined_sanity(self, dtype, idx_dtype, axis):
        x = tensor.vector(name='x', dtype=dtype)
        yv, yi = topk_and_argtopk(x, 1, axis=axis, idx_dtype=idx_dtype)
        fn = theano.function([x], [yv, yi])
        xval = np.asarray([1]).astype(dtype)
        yvval, yival = fn(xval)
        assert yival == np.asarray([0], dtype=idx_dtype)
        assert np.allclose(xval, yvval)

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, 10, -10, 'n//2', 'n-1', '-n', '1-n'),
            ('float64', 'float16', 'int16', 'int8')),
        ((2049, 1337, 'float64'),)))
    def test_topk_1d(self, size, k, dtype):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = topk(x, k)
        fn = theano.function([x], y)
        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.sort(xval)[idx]

        print(np.sort(yval))
        print(goal)
        assert yval.dtype == goal.dtype
        assert np.allclose(np.sort(yval), goal)

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, 10, -10, 'n//2', 'n-1', '-n', '1-n'),
            ('float32', 'int32'),
            ('int32', 'int64')),
        ((2049, 1337, 'float32', 'int32'),)))
    def test_argtopk_1d(self, size, k, dtype, idx_dtype):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = argtopk(x, k, idx_dtype=idx_dtype)
        fn = theano.function([x], y)
        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.argsort(xval)[idx].astype(idx_dtype)

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yval)] == xval[np.sort(goal)])

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, 10, -10, 'n//2', 'n-1', '-n', '1-n'),
            ('float32', 'int32'),
            ('int32', 'int64')),
        ((2049, 1337, 'float32', 'int32'),)))
    def test_combined_1d(self, size, k, dtype, idx_dtype):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        yv, yi = topk_and_argtopk(x, k, idx_dtype=idx_dtype)
        fn = theano.function([x], [yv, yi])
        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yvval, yival = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goali = np.argsort(xval)[idx].astype(idx_dtype)
        goalv = xval[goali]

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yival)] == xval[np.sort(goali)])
        assert np.allclose(np.sort(yvval), goalv)

    @utt.parameterized.expand(chain(
        product(
            (18, 62, 258),
            (1, -1, 'n//2'),
            ('int32', 'float32')),
        ((2048, 1337, 'float32'),)))
    def test_argtopk_1d_collision(self, size, k, dtype):
        # with non-unique kth max value
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = argtopk(x, k, idx_dtype='int32')
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
    def test_argtopk_nd(self, shp, k_, dtype, idx_dtype):
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
            y = argtopk(x, k, axis=axis, idx_dtype=idx_dtype)
            fn = theano.function([x], y)
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, dtype).reshape(shp)
            yval = fn(xval)
            idx = slice(-k, None) if k > 0 else slice(-k)
            l = axis % ndim
            r = ndim - l
            idx = (slice(None),) * l + (idx,) + (slice(None),) * (r - 1)
            goal = np.argsort(xval, axis=axis)[idx].astype(idx_dtype)

            print(dict(k=k, axis=axis, shp=shp))
            print('x:')
            print(xval)
            print('y:')
            print(np.sort(yval, axis=axis))
            print('goal:')
            print(np.sort(goal, axis=axis))
            # print(np.argsort(xval))
            assert np.all(np.sort(yval, axis=axis) == np.sort(goal, axis=axis))


class TopKInferShapeTester(utt.InferShapeTester):
    @utt.parameterized.expand(product(
        ((2, 3), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2, 4, 3, 1)),
        (1, -1, '(1+n)//2', 'n-1', '-n', '1-n')))
    def test_topk_infer_shape(self, shp, k_):
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
            y = topk(x, k, axis=axis)
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, theano.config.floatX).reshape(shp)
            self._compile_and_check(
                [x], [y], [xval], TopKOp)

    @utt.parameterized.expand(product(
        ((2, 3), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2, 4, 3, 1)),
        (1, -1, '(1+n)//2', 'n-1', '-n', '1-n')))
    def test_argtopk_infer_shape(self, shp, k_):
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
            y = argtopk(x, k, axis=axis, idx_dtype='int32')
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, theano.config.floatX).reshape(shp)
            self._compile_and_check(
                [x], [y], [xval], TopKOp)

    @utt.parameterized.expand(product(
        ((2, 3), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2, 4, 3, 1)),
        (1, -1, '(1+n)//2', 'n-1', '-n', '1-n')))
    def test_combined_infer_shape(self, shp, k_):
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
            yv, yi = topk_and_argtopk(x, k, axis=axis, idx_dtype='int32')
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, theano.config.floatX).reshape(shp)
            self._compile_and_check(
                [x], [yv, yi], [xval], TopKOp)
