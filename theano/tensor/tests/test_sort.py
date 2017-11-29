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

_all_dtypes = tensor.integer_dtypes + tensor.float_dtypes


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
        utt.assert_allclose(f(self.m_val), np.sort(self.m_val))

    def test2(self):
        a = tensor.dmatrix()
        axis = tensor.scalar()
        w = sort(a, axis)
        f = theano.function([a, axis], w)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            utt.assert_allclose(gv, gt)

    def test3(self):
        a = tensor.dvector()
        w2 = sort(a)
        f = theano.function([a], w2)
        gv = f(self.v_val)
        gt = np.sort(self.v_val)
        utt.assert_allclose(gv, gt)

    def test4(self):
        a = tensor.dmatrix()
        axis = tensor.scalar()
        l = sort(a, axis, "mergesort")
        f = theano.function([a, axis], l)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            utt.assert_allclose(gv, gt)

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
        utt.assert_allclose(gv, gt)

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
    utt.assert_allclose(gv, gt)

    # Example 2
    a = tensor.dmatrix()
    axis = tensor.lscalar()
    w = argsort(a, axis)
    f = theano.function([a, axis], w)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        utt.assert_allclose(gv, gt)

    # Example 3
    a = tensor.dvector()
    w2 = argsort(a)
    f = theano.function([a], w2)
    gv = f(v_val)
    gt = np.argsort(v_val)
    utt.assert_allclose(gv, gt)

    # Example 4
    a = tensor.dmatrix()
    axis = tensor.lscalar()
    l = argsort(a, axis, "mergesort")
    f = theano.function([a, axis], l)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        utt.assert_allclose(gv, gt)

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
    utt.assert_allclose(gv, gt)


def test_argsort_grad():
    # Testing grad of argsort
    data = np.random.rand(2, 3).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-1), [data])

    data = np.random.rand(2, 3, 4, 5).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-3), [data])

    data = np.random.rand(2, 3, 3).astype(theano.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=2), [data])


class Test_TopK(unittest.TestCase):
    mode = None
    op_class = TopKOp

    def setUp(self):
        pass

    @utt.parameterized.expand(product(
        _all_dtypes, tensor.integer_dtypes, [-1, 0, None], [False]))
    def test_argtopk_sanity(self, dtype, idx_dtype, axis, sorted):
        x = tensor.vector(name='x', dtype=dtype)
        fn = theano.function([x],
                             argtopk(x, 1, axis=axis, sorted=sorted, idx_dtype=idx_dtype),
                             mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == np.asarray([0], dtype=idx_dtype)
        assert yval.dtype == np.dtype(idx_dtype)

    @utt.parameterized.expand(product(
        _all_dtypes, [-1, 0, None], [False]))
    def test_topk_sanity(self, dtype, axis, sorted):
        x = tensor.vector(name='x', dtype=dtype)
        fn = theano.function([x], topk(x, 1, axis=axis, sorted=sorted), mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == xval
        assert yval.dtype == xval.dtype

    @utt.parameterized.expand(product(
        _all_dtypes, tensor.integer_dtypes, [-1, 0, None], [False]))
    def test_combined_sanity(self, dtype, idx_dtype, axis, sorted):
        x = tensor.vector(name='x', dtype=dtype)
        yv, yi = topk_and_argtopk(x, 1, axis=axis, sorted=sorted, idx_dtype=idx_dtype)
        fn = theano.function([x], [yv, yi], mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
        xval = np.asarray([1]).astype(dtype)
        yvval, yival = fn(xval)
        assert yival == np.asarray([0], dtype=idx_dtype)
        utt.assert_allclose(xval, yvval)
        assert yvval.dtype == xval.dtype
        assert yival.dtype == np.dtype(idx_dtype)

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, -10, 'n//2', 'n-1', '-n', '1-n'),
            ('float64', 'float16', 'int16', 'int8'),
            (False,)),
        ((2049, 1337, 'float64', False),)))
    def test_topk_1d(self, size, k, dtype, sorted):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = topk(x, k, sorted=sorted)
        fn = theano.function([x], y, mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
        # assert local_useless_topk opt is done properly
        assert 1 == len(fn.maker.fgraph.outputs[0].owner.outputs)

        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = (slice(-k, None) if k > 0 else slice(-k))
        goal = np.sort(xval)[idx]

        assert yval.dtype == goal.dtype
        utt.assert_allclose(goal, np.sort(yval))

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, -10, 'n//2', 'n-1', '-n'),
            ('float32', 'int32'),
            (False,),
            ('int32', 'int64')),
        ((2049, 1337, 'float32', False, 'int32'),)))
    def test_argtopk_1d(self, size, k, dtype, sorted, idx_dtype):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = argtopk(x, k, sorted=sorted, idx_dtype=idx_dtype)
        fn = theano.function([x], y, mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])

        # assert local_useless_topk opt is done properly
        assert 1 == len(fn.maker.fgraph.outputs[0].owner.outputs)

        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = (slice(-k, None) if k > 0 else slice(-k))
        goal = np.argsort(xval)[idx].astype(idx_dtype)

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yval)] == xval[np.sort(goal)])

    @utt.parameterized.expand(chain(
        product(
            (16, 61, 257),
            (1, -1, 10, 'n//2', 'n-1', '1-n'),
            ('float32', 'int32'),
            (False,),
            ('int32', 'int64')),
        ((2049, 1337, 'float32', False, 'int32'),)))
    def test_combined_1d(self, size, k, dtype, sorted, idx_dtype):
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        yv, yi = topk_and_argtopk(x, k, sorted=sorted, idx_dtype=idx_dtype)
        fn = theano.function([x], [yv, yi], mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yvval, yival = fn(xval)
        idx = (slice(-k, None) if k > 0 else slice(-k))
        goali = np.argsort(xval)[idx].astype(idx_dtype)
        goalv = xval[goali]

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yival)] == xval[np.sort(goali)])
        utt.assert_allclose(np.sort(yvval), goalv)

    @utt.parameterized.expand(chain(
        product(
            (18, 62, 258),
            (1, -1, 'n//2'),
            ('int32', 'float32'),
            (False,)),
        ((2048, 1337, 'float32', False),)))
    def test_argtopk_1d_collision(self, size, k, dtype, sorted):
        # with non-unique kth max value
        if isinstance(k, str):
            k = eval(k.replace('n', str(size)))

        x = theano.tensor.vector(name='x', dtype=dtype)
        y = argtopk(x, k, sorted=sorted, idx_dtype='int32')
        fn = theano.function([x], y, mode=self.mode)
        assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
        xval = np.repeat(np.random.uniform(-100., 100., size=size // 2).astype(dtype), 2)
        xval = xval[np.random.permutation(size)]
        yval = fn(xval)
        idx = (slice(-k, None) if k > 0 else slice(-k))
        goal = np.argsort(xval)[idx].astype('int32')
        utt.assert_allclose(np.sort(xval[yval]), np.sort(xval[goal]))

    @utt.parameterized.expand(product(
        ((17, 15), (2, 3, 5, 7, 11), (500, 5, 3)),  # NB: Test may fail with bigger sizes (e.g. (2017, 5, 3)) due to "too many resources requested" kernel error on some GPUs.
        (-1, '(1+n)//2', '-n', '1-n'),
        ('float32', 'int32'),
        (False,),
        ('int32', 'int64')))
    def test_argtopk_nd(self, shp, k_, dtype, sorted, idx_dtype):
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
            y = argtopk(x, k, axis=axis, sorted=sorted, idx_dtype=idx_dtype)
            fn = theano.function([x], y, mode=self.mode)
            assert any([isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes])
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, dtype).reshape(shp)
            yval = fn(xval)
            idx = slice(-k, None) if k > 0 else slice(-k)
            l = axis % ndim
            r = ndim - l
            idx = (slice(None),) * l + (idx,) + (slice(None),) * (r - 1)
            goal = np.argsort(xval, axis=axis)[idx].astype(idx_dtype)

            assert np.all(np.sort(yval, axis=axis) == np.sort(goal, axis=axis))

    @utt.parameterized.expand(product(
        ((257,), (17, 15), (5, 3, 5, 3), (2, 3, 5, 7, 11)),
        (1, -1, '(1+n)//2', 'n-1', '-n', '1-n'), (False,)))
    def test_grad(self, shp, k_, sorted):
        ndim = len(shp)
        for axis in range(-ndim, ndim):
            if isinstance(k_, str):
                k = eval(k_.replace('n', str(shp[axis])))
            else:
                k = k_

            if k == 0:
                continue

            # make input away from undefined gradient (where some inputs are equal)
            xval = gen_unique_vector(
                reduce(int.__mul__, shp),
                dtype=theano.config.floatX
            ).reshape(shp)
            utt.verify_grad(lambda x: topk(x, k, axis=axis, sorted=sorted), [xval], eps=1e-2)


class TopKInferShapeTester(utt.InferShapeTester):
    @utt.parameterized.expand(product(
        ((2, 3), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2, 4, 3, 1)),
        (1, '(1+n)//2', 'n-1', 'n')))
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
            yv, yi = topk_and_argtopk(x, k, axis=axis, sorted=False, idx_dtype='int32')
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, theano.config.floatX).reshape(shp)
            self._compile_and_check(
                [x], [yv, yi], [xval], TopKOp)
