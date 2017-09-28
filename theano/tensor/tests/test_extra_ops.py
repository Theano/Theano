from __future__ import absolute_import, print_function, division

import numpy as np

import theano
from theano.tests import unittest_tools as utt

from theano.tensor.extra_ops import (SearchsortedOp, searchsorted,
                                     CumOp, cumsum, cumprod,
                                     CpuContiguous, cpu_contiguous,
                                     bincount, DiffOp, diff, squeeze, compress,
                                     RepeatOp, repeat, Bartlett, bartlett,
                                     FillDiagonal, fill_diagonal,
                                     FillDiagonalOffset, fill_diagonal_offset,
                                     to_one_hot, Unique, unravel_index, UnravelIndex,
                                     ravel_multi_index, RavelMultiIndex)
from theano import tensor as T
from theano import config, tensor, function
from theano.tests.unittest_tools import attr


def test_cpu_contiguous():
    a = T.fmatrix('a')
    i = T.iscalar('i')
    a_val = np.asarray(np.random.rand(4, 5), dtype='float32')
    f = theano.function([a, i], cpu_contiguous(a.reshape((5, 4))[::i]))
    topo = f.maker.fgraph.toposort()
    assert any([isinstance(node.op, CpuContiguous) for node in topo])
    assert f(a_val, 1).flags['C_CONTIGUOUS']
    assert f(a_val, 2).flags['C_CONTIGUOUS']
    assert f(a_val, 3).flags['C_CONTIGUOUS']
    # Test the grad:

    theano.tests.unittest_tools.verify_grad(cpu_contiguous,
                                            [np.random.rand(5, 7, 2)])


class TestSearchsortedOp(utt.InferShapeTester):

    def setUp(self):
        super(TestSearchsortedOp, self).setUp()
        self.op_class = SearchsortedOp
        self.op = SearchsortedOp()

        self.x = T.vector('x')
        self.v = T.tensor3('v')

        self.a = 30 * np.random.random(50).astype(config.floatX)
        self.b = 30 * np.random.random((8, 10, 5)).astype(config.floatX)
        self.idx_sorted = np.argsort(self.a).astype('int32')

    def test_searchsortedOp_on_sorted_input(self):
        f = theano.function([self.x, self.v], searchsorted(self.x, self.v))
        assert np.allclose(np.searchsorted(self.a[self.idx_sorted], self.b),
                           f(self.a[self.idx_sorted], self.b))

        sorter = T.vector('sorter', dtype='int32')
        f = theano.function([self.x, self.v, sorter], self.x.searchsorted(self.v, sorter=sorter, side='right'))
        assert np.allclose(self.a.searchsorted(self.b, sorter=self.idx_sorted, side='right'),
                           f(self.a, self.b, self.idx_sorted))

        sa = self.a[self.idx_sorted]
        f = theano.function([self.x, self.v], self.x.searchsorted(self.v, side='right'))
        assert np.allclose(sa.searchsorted(self.b, side='right'), f(sa, self.b))

    def test_searchsortedOp_wrong_side_kwd(self):
        self.assertRaises(ValueError, searchsorted, self.x, self.v, side='asdfa')

    def test_searchsortedOp_on_no_1d_inp(self):
        no_1d = T.dmatrix('no_1d')
        self.assertRaises(ValueError, searchsorted, no_1d, self.v)
        self.assertRaises(ValueError, searchsorted, self.x, self.v, sorter=no_1d)

    def test_searchsortedOp_on_float_sorter(self):
        sorter = T.vector('sorter', dtype="float32")
        self.assertRaises(TypeError, searchsorted,
                          self.x, self.v, sorter=sorter)

    def test_searchsortedOp_on_int_sorter(self):
        compatible_types = ('int8', 'int16', 'int32')
        if theano.configdefaults.python_int_bitwidth() == 64:
            compatible_types += ('int64',)
        # 'uint8', 'uint16', 'uint32', 'uint64')
        for dtype in compatible_types:
            sorter = T.vector('sorter', dtype=dtype)
            f = theano.function([self.x, self.v, sorter],
                                searchsorted(self.x, self.v, sorter=sorter),
                                allow_input_downcast=True)
            assert np.allclose(np.searchsorted(self.a, self.b, sorter=self.idx_sorted),
                               f(self.a, self.b, self.idx_sorted))

    def test_searchsortedOp_on_right_side(self):
        f = theano.function([self.x, self.v],
                            searchsorted(self.x, self.v, side='right'))
        assert np.allclose(np.searchsorted(self.a, self.b, side='right'),
                           f(self.a, self.b))

    def test_infer_shape(self):
        # Test using default parameters' value
        self._compile_and_check([self.x, self.v],
                                [searchsorted(self.x, self.v)],
                                [self.a[self.idx_sorted], self.b],
                                self.op_class)

        # Test parameter ``sorter``
        sorter = T.vector('sorter', dtype="int32")
        self._compile_and_check([self.x, self.v, sorter],
                                [searchsorted(self.x, self.v, sorter=sorter)],
                                [self.a, self.b, self.idx_sorted],
                                self.op_class)

        # Test parameter ``side``
        la = np.ones(10).astype(config.floatX)
        lb = np.ones(shape=(1, 2, 3)).astype(config.floatX)
        self._compile_and_check([self.x, self.v],
                                [searchsorted(self.x, self.v, side='right')],
                                [la, lb],
                                self.op_class)

    def test_grad(self):
        utt.verify_grad(self.op, [self.a[self.idx_sorted], self.b])


class TestCumOp(utt.InferShapeTester):

    def setUp(self):
        super(TestCumOp, self).setUp()
        self.op_class = CumOp
        self.op = CumOp()

    def test_cum_op(self):
        x = T.tensor3('x')
        a = np.random.random((3, 5, 2)).astype(config.floatX)

        # Test axis out of bounds
        self.assertRaises(ValueError, cumsum, x, axis=3)
        self.assertRaises(ValueError, cumsum, x, axis=-4)
        self.assertRaises(ValueError, cumprod, x, axis=3)
        self.assertRaises(ValueError, cumprod, x, axis=-4)

        f = theano.function([x], [cumsum(x), cumprod(x)])
        s, p = f(a)
        assert np.allclose(np.cumsum(a), s)  # Test axis=None
        assert np.allclose(np.cumprod(a), p)  # Test axis=None

        for axis in range(-len(a.shape), len(a.shape)):
            f = theano.function([x], [cumsum(x, axis=axis), cumprod(x, axis=axis)])
            s, p = f(a)
            assert np.allclose(np.cumsum(a, axis=axis), s)
            assert np.allclose(np.cumprod(a, axis=axis), p)

    def test_infer_shape(self):
        x = T.tensor3('x')
        a = np.random.random((3, 5, 2)).astype(config.floatX)

        # Test axis=None
        self._compile_and_check([x],
                                [self.op(x)],
                                [a],
                                self.op_class)

        for axis in range(-len(a.shape), len(a.shape)):
            self._compile_and_check([x],
                                    [cumsum(x, axis=axis)],
                                    [a],
                                    self.op_class)

    def test_grad(self):
        a = np.random.random((3, 5, 2)).astype(config.floatX)

        utt.verify_grad(self.op_class(mode='add'), [a])  # Test axis=None
        utt.verify_grad(self.op_class(mode='mul'), [a])  # Test axis=None

        for axis in range(-len(a.shape), len(a.shape)):
            utt.verify_grad(self.op_class(axis=axis, mode='add'), [a], eps=4e-4)
            utt.verify_grad(self.op_class(axis=axis, mode='mul'), [a], eps=4e-4)


class TestBinCount(utt.InferShapeTester):
    def test_bincountFn(self):
        w = T.vector('w')

        def ref(data, w=None, minlength=None):
            size = int(data.max() + 1)
            if minlength:
                size = max(size, minlength)
            if w is not None:
                out = np.zeros(size, dtype=w.dtype)
                for i in range(data.shape[0]):
                    out[data[i]] += w[i]
            else:
                out = np.zeros(size, dtype=a.dtype)
                for i in range(data.shape[0]):
                    out[data[i]] += 1
            return out

        for dtype in ('int8', 'int16', 'int32', 'int64',
                      'uint8', 'uint16', 'uint32', 'uint64'):
            x = T.vector('x', dtype=dtype)

            a = np.random.randint(1, 51, size=(25)).astype(dtype)
            weights = np.random.random((25,)).astype(config.floatX)

            f1 = theano.function([x], bincount(x))
            f2 = theano.function([x, w], bincount(x, weights=w))

            assert (ref(a) == f1(a)).all()
            assert np.allclose(ref(a, weights), f2(a, weights))
            f3 = theano.function([x], bincount(x, minlength=55))
            f4 = theano.function([x], bincount(x, minlength=5))
            assert (ref(a, minlength=55) == f3(a)).all()
            assert (ref(a, minlength=5) == f4(a)).all()
            # skip the following test when using unsigned ints
            if not dtype.startswith('u'):
                a[0] = -1
                f5 = theano.function([x], bincount(x, assert_nonneg=True))
                self.assertRaises(AssertionError, f5, a)


class TestDiffOp(utt.InferShapeTester):
    nb = 10  # Number of time iterating for n

    def setUp(self):
        super(TestDiffOp, self).setUp()
        self.op_class = DiffOp
        self.op = DiffOp()

    def test_diffOp(self):
        x = T.matrix('x')
        a = np.random.random((30, 50)).astype(config.floatX)

        f = theano.function([x], diff(x))
        assert np.allclose(np.diff(a), f(a))

        for axis in range(len(a.shape)):
            for k in range(TestDiffOp.nb):
                g = theano.function([x], diff(x, n=k, axis=axis))
                assert np.allclose(np.diff(a, n=k, axis=axis), g(a))

    def test_infer_shape(self):
        x = T.matrix('x')
        a = np.random.random((30, 50)).astype(config.floatX)

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
        a = np.random.random(50).astype(config.floatX)

        theano.function([x], T.grad(T.sum(diff(x)), x))
        utt.verify_grad(self.op, [a])

        for k in range(TestDiffOp.nb):
            theano.function([x], T.grad(T.sum(diff(x, n=k)), x))
            utt.verify_grad(DiffOp(n=k), [a], eps=7e-3)


class SqueezeTester(utt.InferShapeTester):
    shape_list = [(1, 3),
                  (1, 2, 3),
                  (1, 5, 1, 1, 6)]
    broadcast_list = [[True, False],
                      [True, False, False],
                      [True, False, True, True, False]]

    def setUp(self):
        super(SqueezeTester, self).setUp()
        self.op = squeeze

    def test_op(self):
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)
            variable = tensor.TensorType(theano.config.floatX, broadcast)()

            f = theano.function([variable], self.op(variable))

            expected = np.squeeze(data)
            tested = f(data)

            assert tested.shape == expected.shape
            assert np.allclose(tested, expected)

    def test_infer_shape(self):
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)
            variable = tensor.TensorType(theano.config.floatX, broadcast)()

            self._compile_and_check([variable],
                                    [self.op(variable)],
                                    [data],
                                    tensor.DimShuffle,
                                    warn=False)

    def test_grad(self):
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)

            utt.verify_grad(self.op, [data])

    def test_var_interface(self):
        # same as test_op, but use a_theano_var.squeeze.
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)
            variable = tensor.TensorType(theano.config.floatX, broadcast)()

            f = theano.function([variable], variable.squeeze())

            expected = np.squeeze(data)
            tested = f(data)

            assert tested.shape == expected.shape
            assert np.allclose(tested, expected)


class CompressTester(utt.InferShapeTester):
    axis_list = [None,
                 -1,
                 0,
                 0,
                 0,
                 1]
    cond_list = [[1, 0, 1, 0, 0, 1],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [],
                 [0, 0, 0, 0],
                 [1, 1, 0, 1, 0]]
    shape_list = [(2, 3),
                  (4, 3),
                  (4, 3),
                  (4, 3),
                  (4, 3),
                  (3, 5)]

    def setUp(self):
        super(CompressTester, self).setUp()
        self.op = compress

    def test_op(self):
        for axis, cond, shape in zip(self.axis_list, self.cond_list,
                                     self.shape_list):
            cond_var = theano.tensor.ivector()
            data = np.random.random(size=shape).astype(theano.config.floatX)
            data_var = theano.tensor.matrix()

            f = theano.function([cond_var, data_var],
                                self.op(cond_var, data_var, axis=axis))

            expected = np.compress(cond, data, axis=axis)
            tested = f(cond, data)

            assert tested.shape == expected.shape
            assert np.allclose(tested, expected)


class TestRepeatOp(utt.InferShapeTester):
    def _possible_axis(self, ndim):
        return [None] + list(range(ndim)) + [-i for i in range(ndim)]

    def setUp(self):
        super(TestRepeatOp, self).setUp()
        self.op_class = RepeatOp
        self.op = RepeatOp()
        # uint64 always fails
        # int64 and uint32 also fail if python int are 32-bit
        ptr_bitwidth = theano.configdefaults.local_bitwidth()
        if ptr_bitwidth == 64:
            self.numpy_unsupported_dtypes = ('uint64',)
        if ptr_bitwidth == 32:
            self.numpy_unsupported_dtypes = ('uint32', 'int64', 'uint64')

    def test_repeatOp(self):
        for ndim in [1, 3]:
            x = T.TensorType(config.floatX, [False] * ndim)()
            a = np.random.random((10, ) * ndim).astype(config.floatX)

            for axis in self._possible_axis(ndim):
                for dtype in tensor.integer_dtypes:
                    r_var = T.scalar(dtype=dtype)
                    r = np.asarray(3, dtype=dtype)
                    if (dtype == 'uint64' or
                            (dtype in self.numpy_unsupported_dtypes and
                                r_var.ndim == 1)):
                        self.assertRaises(TypeError, repeat, x, r_var, axis=axis)
                    else:
                        f = theano.function([x, r_var],
                                            repeat(x, r_var, axis=axis))
                        assert np.allclose(np.repeat(a, r, axis=axis),
                                           f(a, r))

                        r_var = T.vector(dtype=dtype)
                        if axis is None:
                            r = np.random.randint(
                                1, 6, size=a.size).astype(dtype)
                        else:
                            r = np.random.randint(
                                1, 6, size=(10,)).astype(dtype)

                        if dtype in self.numpy_unsupported_dtypes and r_var.ndim == 1:
                            self.assertRaises(TypeError,
                                              repeat, x, r_var, axis=axis)
                        else:
                            f = theano.function([x, r_var],
                                                repeat(x, r_var, axis=axis))
                            assert np.allclose(np.repeat(a, r, axis=axis),
                                               f(a, r))

                        # check when r is a list of single integer, e.g. [3].
                        r = np.random.randint(
                            1, 11, size=()).astype(dtype) + 2
                        f = theano.function([x],
                                            repeat(x, [r], axis=axis))
                        assert np.allclose(np.repeat(a, r, axis=axis),
                                           f(a))
                        assert not np.any([isinstance(n.op, RepeatOp)
                                           for n in f.maker.fgraph.toposort()])

                        # check when r is  theano tensortype that broadcastable is (True,)
                        r_var = theano.tensor.TensorType(broadcastable=(True,),
                                                         dtype=dtype)()
                        r = np.random.randint(1, 6, size=(1,)).astype(dtype)
                        f = theano.function([x, r_var],
                                            repeat(x, r_var, axis=axis))
                        assert np.allclose(np.repeat(a, r[0], axis=axis),
                                           f(a, r))
                        assert not np.any([isinstance(n.op, RepeatOp)
                                           for n in f.maker.fgraph.toposort()])

    @attr('slow')
    def test_infer_shape(self):
        for ndim in [1, 3]:
            x = T.TensorType(config.floatX, [False] * ndim)()
            shp = (np.arange(ndim) + 1) * 3
            a = np.random.random(shp).astype(config.floatX)

            for axis in self._possible_axis(ndim):
                for dtype in ["int8", "uint8", "uint64"]:
                    r_var = T.scalar(dtype=dtype)
                    r = np.asarray(3, dtype=dtype)
                    if dtype in self.numpy_unsupported_dtypes:
                        r_var = T.vector(dtype=dtype)
                        self.assertRaises(TypeError, repeat, x, r_var)
                    else:
                        self._compile_and_check([x, r_var],
                                                [RepeatOp(axis=axis)(x, r_var)],
                                                [a, r],
                                                self.op_class)

                        r_var = T.vector(dtype=dtype)
                        if axis is None:
                            r = np.random.randint(
                                1, 6, size=a.size).astype(dtype)
                        elif a.size > 0:
                            r = np.random.randint(
                                1, 6, size=a.shape[axis]).astype(dtype)
                        else:
                            r = np.random.randint(
                                1, 6, size=(10,)).astype(dtype)

                        self._compile_and_check(
                            [x, r_var],
                            [RepeatOp(axis=axis)(x, r_var)],
                            [a, r],
                            self.op_class)

    def test_grad(self):
        for ndim in range(3):
            a = np.random.random((10, ) * ndim).astype(config.floatX)

            for axis in self._possible_axis(ndim):
                utt.verify_grad(lambda x: RepeatOp(axis=axis)(x, 3), [a])

    def test_broadcastable(self):
        x = T.TensorType(config.floatX, [False, True, False])()
        r = RepeatOp(axis=1)(x, 2)
        self.assertEqual(r.broadcastable, (False, False, False))
        r = RepeatOp(axis=1)(x, 1)
        self.assertEqual(r.broadcastable, (False, True, False))
        r = RepeatOp(axis=0)(x, 2)
        self.assertEqual(r.broadcastable, (False, True, False))


class TestBartlett(utt.InferShapeTester):

    def setUp(self):
        super(TestBartlett, self).setUp()
        self.op_class = Bartlett
        self.op = bartlett

    def test_perform(self):
        x = tensor.lscalar()
        f = function([x], self.op(x))
        M = np.random.randint(3, 51, size=())
        assert np.allclose(f(M), np.bartlett(M))
        assert np.allclose(f(0), np.bartlett(0))
        assert np.allclose(f(-1), np.bartlett(-1))
        b = np.array([17], dtype='uint8')
        assert np.allclose(f(b[0]), np.bartlett(b[0]))

    def test_infer_shape(self):
        x = tensor.lscalar()
        self._compile_and_check([x], [self.op(x)],
                                [np.random.randint(3, 51, size=())],
                                self.op_class)
        self._compile_and_check([x], [self.op(x)], [0], self.op_class)
        self._compile_and_check([x], [self.op(x)], [1], self.op_class)


class TestFillDiagonal(utt.InferShapeTester):

    rng = np.random.RandomState(43)

    def setUp(self):
        super(TestFillDiagonal, self).setUp()
        self.op_class = FillDiagonal
        self.op = fill_diagonal

    def test_perform(self):
        x = tensor.matrix()
        y = tensor.scalar()
        f = function([x, y], fill_diagonal(x, y))
        for shp in [(8, 8), (5, 8), (8, 5)]:
            a = np.random.rand(*shp).astype(config.floatX)
            val = np.cast[config.floatX](np.random.rand())
            out = f(a, val)
            # We can't use np.fill_diagonal as it is bugged.
            assert np.allclose(np.diag(out), val)
            assert (out == val).sum() == min(a.shape)

        # test for 3d tensor
        a = np.random.rand(3, 3, 3).astype(config.floatX)
        x = tensor.tensor3()
        y = tensor.scalar()
        f = function([x, y], fill_diagonal(x, y))
        val = np.cast[config.floatX](np.random.rand() + 10)
        out = f(a, val)
        # We can't use np.fill_diagonal as it is bugged.
        assert out[0, 0, 0] == val
        assert out[1, 1, 1] == val
        assert out[2, 2, 2] == val
        assert (out == val).sum() == min(a.shape)

    @attr('slow')
    def test_gradient(self):
        utt.verify_grad(fill_diagonal, [np.random.rand(5, 8),
                                        np.random.rand()],
                        n_tests=1, rng=TestFillDiagonal.rng)
        utt.verify_grad(fill_diagonal, [np.random.rand(8, 5),
                                        np.random.rand()],
                        n_tests=1, rng=TestFillDiagonal.rng)

    def test_infer_shape(self):
        z = tensor.dtensor3()
        x = tensor.dmatrix()
        y = tensor.dscalar()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [np.random.rand(8, 5),
                                 np.random.rand()],
                                self.op_class)
        self._compile_and_check([z, y], [self.op(z, y)],
                                # must be square when nd>2
                                [np.random.rand(8, 8, 8),
                                 np.random.rand()],
                                self.op_class,
                                warn=False)


class TestFillDiagonalOffset(utt.InferShapeTester):

    rng = np.random.RandomState(43)

    def setUp(self):
        super(TestFillDiagonalOffset, self).setUp()
        self.op_class = FillDiagonalOffset
        self.op = fill_diagonal_offset

    def test_perform(self):
        x = tensor.matrix()
        y = tensor.scalar()
        z = tensor.iscalar()

        f = function([x, y, z], fill_diagonal_offset(x, y, z))
        for test_offset in (-5, -4, -1, 0, 1, 4, 5):
            for shp in [(8, 8), (5, 8), (8, 5), (5, 5)]:
                a = np.random.rand(*shp).astype(config.floatX)
                val = np.cast[config.floatX](np.random.rand())
                out = f(a, val, test_offset)
                # We can't use np.fill_diagonal as it is bugged.
                assert np.allclose(np.diag(out, test_offset), val)
                if test_offset >= 0:
                    assert (out == val).sum() == min(min(a.shape),
                                                     a.shape[1] - test_offset)
                else:
                    assert (out == val).sum() == min(min(a.shape),
                                                     a.shape[0] + test_offset)

    def test_gradient(self):
        for test_offset in (-5, -4, -1, 0, 1, 4, 5):
            # input 'offset' will not be tested
            def fill_diagonal_with_fix_offset(a, val):
                return fill_diagonal_offset(a, val, test_offset)

            utt.verify_grad(fill_diagonal_with_fix_offset,
                            [np.random.rand(5, 8), np.random.rand()],
                            n_tests=1, rng=TestFillDiagonalOffset.rng)
            utt.verify_grad(fill_diagonal_with_fix_offset,
                            [np.random.rand(8, 5), np.random.rand()],
                            n_tests=1, rng=TestFillDiagonalOffset.rng)
            utt.verify_grad(fill_diagonal_with_fix_offset,
                            [np.random.rand(5, 5), np.random.rand()],
                            n_tests=1, rng=TestFillDiagonalOffset.rng)

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.dscalar()
        z = tensor.iscalar()
        for test_offset in (-5, -4, -1, 0, 1, 4, 5):
            self._compile_and_check([x, y, z], [self.op(x, y, z)],
                                    [np.random.rand(8, 5),
                                     np.random.rand(),
                                     test_offset],
                                    self.op_class)
            self._compile_and_check([x, y, z], [self.op(x, y, z)],
                                    [np.random.rand(5, 8),
                                     np.random.rand(),
                                     test_offset],
                                    self.op_class)


def test_to_one_hot():
    v = theano.tensor.ivector()
    o = to_one_hot(v, 10)
    f = theano.function([v], o)
    out = f([1, 2, 3, 5, 6])
    assert out.dtype == theano.config.floatX
    assert np.allclose(
        out,
        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])

    v = theano.tensor.ivector()
    o = to_one_hot(v, 10, dtype="int32")
    f = theano.function([v], o)
    out = f([1, 2, 3, 5, 6])
    assert out.dtype == "int32"
    assert np.allclose(
        out,
        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])


class test_Unique(utt.InferShapeTester):

    def setUp(self):
        super(test_Unique, self).setUp()
        self.op_class = Unique
        self.ops = [Unique(),
                    Unique(True),
                    Unique(False, True),
                    Unique(True, True),
                    Unique(False, False, True),
                    Unique(True, False, True),
                    Unique(False, True, True),
                    Unique(True, True, True)]

    def test_basic_vector(self):
        # Basic test for a vector.
        # Done by using the op and checking that it returns the right answer.

        x = theano.tensor.vector()
        inp = np.asarray([2, 1, 3, 2], dtype=config.floatX)
        list_outs_expected = [[np.unique(inp)],
                              np.unique(inp, True),
                              np.unique(inp, False, True),
                              np.unique(inp, True, True),
                              np.unique(inp, False, False, True),
                              np.unique(inp, True, False, True),
                              np.unique(inp, False, True, True),
                              np.unique(inp, True, True, True)]
        for op, outs_expected in zip(self.ops, list_outs_expected):
            f = theano.function(inputs=[x], outputs=op(x, return_list=True))
            outs = f(inp)
            # Compare the result computed to the expected value.
            for out, out_exp in zip(outs, outs_expected):
                utt.assert_allclose(out, out_exp)

    def test_basic_matrix(self):
        # Basic test for a matrix.
        # Done by using the op and checking that it returns the right answer.

        x = theano.tensor.matrix()
        inp = np.asarray([[2, 1], [3, 2], [2, 3]], dtype=config.floatX)
        list_outs_expected = [[np.unique(inp)],
                              np.unique(inp, True),
                              np.unique(inp, False, True),
                              np.unique(inp, True, True),
                              np.unique(inp, False, False, True),
                              np.unique(inp, True, False, True),
                              np.unique(inp, False, True, True),
                              np.unique(inp, True, True, True)]
        for op, outs_expected in zip(self.ops, list_outs_expected):
            f = theano.function(inputs=[x], outputs=op(x, return_list=True))
            outs = f(inp)
            # Compare the result computed to the expected value.
            for out, out_exp in zip(outs, outs_expected):
                utt.assert_allclose(out, out_exp)

    def test_infer_shape_vector(self):
        # Testing the infer_shape with a vector.

        x = theano.tensor.vector()

        for op in self.ops:
            if not op.return_inverse:
                continue
            if op.return_index:
                f = op(x)[2]
            else:
                f = op(x)[1]
            self._compile_and_check([x],
                                    [f],
                                    [np.asarray(np.array([2, 1, 3, 2]),
                                                dtype=config.floatX)],
                                    self.op_class)

    def test_infer_shape_matrix(self):
        # Testing the infer_shape with a matrix.

        x = theano.tensor.matrix()

        for op in self.ops:
            if not op.return_inverse:
                continue
            if op.return_index:
                f = op(x)[2]
            else:
                f = op(x)[1]
            self._compile_and_check([x],
                                    [f],
                                    [np.asarray(np.array([[2, 1], [3, 2], [2, 3]]),
                                                dtype=config.floatX)],
                                    self.op_class)


class test_unravel_index(utt.InferShapeTester):
    def test_unravel_index(self):
        def check(shape, index_ndim, order):
            indices = np.arange(np.product(shape))
            # test with scalars and higher-dimensional indices
            if index_ndim == 0:
                indices = indices[-1]
            elif index_ndim == 2:
                indices = indices[:, np.newaxis]
            indices_symb = theano.shared(indices)

            # reference result
            ref = np.unravel_index(indices, shape)

            def fn(i, d, nd=None):
                if nd is None:
                    return function([], unravel_index(i, d, order=order))
                else:
                    return function([], unravel_index(i, d, order=order, ndim=nd))

            # shape given as a tuple
            f_array_tuple = fn(indices, shape)
            f_symb_tuple = fn(indices_symb, shape)
            np.testing.assert_equal(ref, f_array_tuple())
            np.testing.assert_equal(ref, f_symb_tuple())

            # shape given as an array
            shape_array = np.array(shape)
            f_array_array = fn(indices, shape_array)
            np.testing.assert_equal(ref, f_array_array())

            # shape given as a theano variable
            shape_symb = theano.shared(shape_array)
            f_array_symb = fn(indices, shape_symb, len(shape))
            np.testing.assert_equal(ref, f_array_symb())

            # shape given as a Shape op (unravel_index will use get_vector_length
            # to infer the number of dimensions)
            indexed_array = theano.shared(np.random.uniform(size=shape_array))
            f_array_shape = fn(indices, indexed_array.shape)
            np.testing.assert_equal(ref, f_array_shape())

            # shape testing
            self._compile_and_check([],
                                    unravel_index(indices, shape_symb, order=order, ndim=len(shape)),
                                    [], UnravelIndex)

        for order in ('C', 'F'):
            for index_ndim in (0, 1, 2):
                check((3,), index_ndim, order)
                check((3, 4), index_ndim, order)
                check((3, 4, 5), index_ndim, order)

        # must specify ndim if length of dims is not fixed
        self.assertRaises(ValueError, unravel_index, theano.tensor.ivector(), theano.tensor.ivector())

        # must provide integers
        self.assertRaises(TypeError, unravel_index, theano.tensor.fvector(), (3, 4))
        self.assertRaises(TypeError, unravel_index, (3, 4), (3.4, 3.2))
        self.assertRaises(ValueError, unravel_index, (3, 4), (3, 3), ndim=5.4)

        # dims must be a 1D sequence
        self.assertRaises(TypeError, unravel_index, (3, 4), 3)
        self.assertRaises(TypeError, unravel_index, (3, 4), ((3, 4),))


class test_ravel_multi_index(utt.InferShapeTester):
    def test_ravel_multi_index(self):
        def check(shape, index_ndim, mode, order):
            multi_index = np.unravel_index(np.arange(np.product(shape)), shape, order=order)
            # create some invalid indices to test the mode
            if mode in ('wrap', 'clip'):
                multi_index = (multi_index[0] - 1,) + multi_index[1:]
            # test with scalars and higher-dimensional indices
            if index_ndim == 0:
                multi_index = tuple(i[-1] for i in multi_index)
            elif index_ndim == 2:
                multi_index = tuple(i[:, np.newaxis] for i in multi_index)
            multi_index_symb = [theano.shared(i) for i in multi_index]

            # reference result
            ref = np.ravel_multi_index(multi_index, shape, mode, order)

            def fn(mi, s):
                return function([], ravel_multi_index(mi, s, mode, order))

            # shape given as a tuple
            f_array_tuple = fn(multi_index, shape)
            f_symb_tuple = fn(multi_index_symb, shape)
            np.testing.assert_equal(ref, f_array_tuple())
            np.testing.assert_equal(ref, f_symb_tuple())

            # shape given as an array
            shape_array = np.array(shape)
            f_array_array = fn(multi_index, shape_array)
            np.testing.assert_equal(ref, f_array_array())

            # shape given as a theano variable
            shape_symb = theano.shared(shape_array)
            f_array_symb = fn(multi_index, shape_symb)
            np.testing.assert_equal(ref, f_array_symb())

            # shape testing
            self._compile_and_check([],
                                    [ravel_multi_index(multi_index, shape_symb, mode, order)],
                                    [], RavelMultiIndex)

        for mode in ('raise', 'wrap', 'clip'):
            for order in ('C', 'F'):
                for index_ndim in (0, 1, 2):
                    check((3,), index_ndim, mode, order)
                    check((3, 4), index_ndim, mode, order)
                    check((3, 4, 5), index_ndim, mode, order)

        # must provide integers
        self.assertRaises(TypeError, ravel_multi_index, (theano.tensor.fvector(), theano.tensor.ivector()), (3, 4))
        self.assertRaises(TypeError, ravel_multi_index, ((3, 4), theano.tensor.ivector()), (3.4, 3.2))

        # dims must be a 1D sequence
        self.assertRaises(TypeError, ravel_multi_index, ((3, 4),), ((3, 4),))
