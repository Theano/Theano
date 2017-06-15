from __future__ import absolute_import, print_function, division
from functools import partial
from itertools import product

import numpy as np
from six.moves import xrange

from theano import tensor as T
import theano
import theano.tensor.tests.test_extra_ops

from theano.tensor.extra_ops import CumOp
from theano.tests.unittest_tools import SkipTest
from theano.tests import unittest_tools as utt

from .config import mode_with_gpu, test_ctx_name
from ..extra_ops import GpuCumOp
from ..type import get_context

cum_modes = utt.parameterized.expand([('mul',), ('add',)])


class TestGpuCumOp(theano.tensor.tests.test_extra_ops.TestCumOp):
    mode = mode_with_gpu

    def setUp(self):
        super(TestGpuCumOp, self).setUp()
        test_ctx = get_context(test_ctx_name)
        if test_ctx.kind != b'cuda':
            raise SkipTest("Cuda specific tests")
        self.max_threads_dim0 = test_ctx.maxlsize0
        self.max_grid_size1 = test_ctx.maxgsize2
        self.op_class = CumOp

    @cum_modes
    def test_infer_shape(self, mode):
        # GpuCumOp is only defined for float32 for now, so we skip it
        # in the unsupported cases
        op_class = partial(self.op_class, mode=mode)
        gpucumop_supported_dtypes = ('float32',)
        if theano.config.floatX not in gpucumop_supported_dtypes:
            raise SkipTest('Gpucumop not implemented for dtype %s'
                           % theano.config.floatX)
        x = T.tensor3('x')
        a = np.random.random((3, 5, 2)).astype(theano.config.floatX)

        for axis in range(-len(a.shape), len(a.shape)):
            self._compile_and_check([x],
                                    [op_class(axis=axis)(x)],
                                    [a],
                                    GpuCumOp)

    @cum_modes
    def test_grad(self, mode):
        # no grad for GpuCumOp
        pass

    @cum_modes
    def test_Strides1D(self, mode):
        op_class = partial(self.op_class, mode=mode)
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        x = T.fvector('x')

        for axis in [0, None, -1]:
            a = np.random.random((42,)).astype("float32")
            cumop_function = theano.function(
                [x], op_class(axis=axis)(x), mode=self.mode)

            slicings = [slice(None, None, None),    # Normal strides
                        slice(None, None, 2),       # Stepped strides
                        slice(None, None, -1),      # Negative strides
                        ]

            # Cartesian product of all slicings to test.
            for slicing in product(slicings, repeat=x.ndim):
                f = theano.function([x], op_class(axis=axis)(x[slicing]),
                                    mode=self.mode)
                assert [n for n in f.maker.fgraph.toposort()
                        if isinstance(n.op, GpuCumOp)]
                utt.assert_allclose(np_func(a[slicing], axis=axis), f(a))
                utt.assert_allclose(np_func(a[slicing], axis=axis),
                                    cumop_function(a[slicing]))

    @cum_modes
    def test_Strides2D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        x = T.fmatrix('x')

        for axis in [0, 1, None, -1, -2]:
            a = np.random.random((42, 30)).astype("float32")
            cumop_function = theano.function(
                [x], op_class(axis=axis)(x), mode=self.mode)

            slicings = [slice(None, None, None),    # Normal strides
                        slice(None, None, 2),       # Stepped strides
                        slice(None, None, -1),      # Negative strides
                        ]

            # Cartesian product of all slicings to test.
            for slicing in product(slicings, repeat=x.ndim):
                f = theano.function([x], op_class(axis=axis)(x[slicing]),
                                    mode=self.mode)
                assert [n for n in f.maker.fgraph.toposort()
                        if isinstance(n.op, GpuCumOp)]
                utt.assert_allclose(np_func(a[slicing], axis=axis), f(a))
                utt.assert_allclose(np_func(a[slicing], axis=axis),
                                    cumop_function(a[slicing]))

    @cum_modes
    def test_Strides3D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        x = T.ftensor3('x')

        for axis in [0, 1, 2, None, -1, -2, -3]:
            a = np.random.random((42, 30, 25)).astype("float32")
            cumop_function = theano.function(
                [x], op_class(axis=axis)(x), mode=self.mode)

            slicings = [slice(None, None, None),    # Normal strides
                        slice(None, None, 2),       # Stepped strides
                        slice(None, None, -1),      # Negative strides
                        ]

            # Cartesian product of all slicings to test.
            for slicing in product(slicings, repeat=x.ndim):
                f = theano.function(
                    [x], op_class(axis=axis)(x[slicing]), mode=self.mode)
                assert [n for n in f.maker.fgraph.toposort()
                        if isinstance(n.op, GpuCumOp)]
                utt.assert_allclose(np_func(a[slicing], axis=axis), f(a))
                utt.assert_allclose(np_func(a[slicing], axis=axis),
                                    cumop_function(a[slicing]))

    @cum_modes
    def test_GpuCumOp1D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        block_max_size = self.max_threads_dim0 * 2

        x = T.fvector('x')
        f = theano.function([x], op_class(axis=0)(x), mode=self.mode)
        assert [n for n in f.maker.fgraph.toposort()
                if isinstance(n.op, GpuCumOp)]

        # Extensive testing for the first 1025 sizes
        a = np.random.random(1025).astype("float32")
        for i in xrange(a.shape[0]):
            utt.assert_allclose(np_func(a[:i]), f(a[:i]))

        # Use multiple GPU threadblocks
        a = np.random.random((block_max_size + 2, )).astype("float32")
        utt.assert_allclose(np_func(a), f(a))

        # Use recursive cumop
        a = np.ones((block_max_size * (block_max_size + 1) + 2,),
                    dtype="float32")
        utt.assert_allclose(np_func(a), f(a))

    @cum_modes
    def test_GpuCumOp2D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        block_max_size = self.max_threads_dim0 * 2

        x = T.fmatrix('x')
        for shape_axis, axis in zip([0, 1, 0, 1, 0], [0, 1, None, -1, -2]):
            f = theano.function([x], op_class(axis=axis)(x), mode=self.mode)
            assert [n for n in f.maker.fgraph.toposort()
                    if isinstance(n.op, GpuCumOp)]

            # Extensive testing for the first 1025 sizes
            a_shape = [5, 5]
            a_shape[shape_axis] = 1025
            a = np.random.random(a_shape).astype("float32")
            slices = [slice(None), slice(None)]
            for i in xrange(a.shape[shape_axis]):
                slices[shape_axis] = slice(i)
                fa = f(a[slices])
                npa = np_func(a[slices], axis=axis)
                utt.assert_allclose(npa, fa)

            # Use multiple GPU threadblocks
            a_shape = [5, 5]
            a_shape[shape_axis] = block_max_size + 2
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            # Use multiple GPU gridblocks
            a_shape = [4, 4]
            a_shape[1 - shape_axis] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a), rtol=5e-5)

            # Use recursive cumop
            a_shape = [3, 3]
            a_shape[shape_axis] = block_max_size * (block_max_size + 1) + 2
            a = np.random.random(a_shape).astype("float32")
            a = np.sign(a - 0.5).astype("float32")  # Avoid floating point error
            utt.assert_allclose(np_func(a, axis=axis), f(a))

    @cum_modes
    def test_GpuCumOp3D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        block_max_size = self.max_threads_dim0 * 2

        x = T.ftensor3('x')
        for shape_axis, axis in zip([0, 1, 2, 0, 2, 1, 0], [0, 1, 2, None, -1, -2, -3]):
            f = theano.function([x], op_class(axis=axis)(x), mode=self.mode)
            assert [n for n in f.maker.fgraph.toposort()
                    if isinstance(n.op, GpuCumOp)]

            # Extensive testing for the first 1025 sizes
            a_shape = [5, 5, 5]
            a_shape[shape_axis] = 1025
            a = np.random.rand(*a_shape).astype("float32")
            slices = [slice(None), slice(None), slice(None)]
            for i in xrange(a.shape[shape_axis]):
                slices[shape_axis] = slice(i)
                fa = f(a[slices])
                npa = np_func(a[slices], axis=axis)
                utt.assert_allclose(npa, fa)

            # Use multiple GPU threadblocks (along accumulation axis)
            a_shape = [2, 2, 2]
            a_shape[shape_axis] = block_max_size + 2
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            # Use multiple GPU gridblocks (not along accumulation axis)
            a_shape = [5, 5, 5]
            a_shape[(shape_axis + 1) % 3] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            if axis is None:
                # Avoid floating point error
                a = np.sign(a - 0.5).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            a_shape = [5, 5, 5]
            a_shape[(shape_axis + 2) % 3] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            if axis is None:
                # Avoid floating point error
                a = np.sign(a - 0.5).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            # Use recursive cumop (along accumulation axis)
            a_shape = [3, 3, 3]
            a_shape[shape_axis] = block_max_size * (block_max_size + 1) + 2
            a = np.random.random(a_shape).astype("float32")
            a = np.sign(a - 0.5).astype("float32")  # Avoid floating point error
            utt.assert_allclose(np_func(a, axis=axis), f(a))

    @cum_modes
    def test_GpuCumOp4D(self, mode):
        op_class = partial(self.op_class, mode=mode)
        # Should not use the GPU version.
        x = T.ftensor4('x')
        f = theano.function([x], op_class(axis=1)(x), mode=self.mode)
        assert [n for n in f.maker.fgraph.toposort()
                if isinstance(n.op, CumOp)]
