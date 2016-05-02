# Skip test if cuda_ndarray is not available.
from __future__ import absolute_import, print_function, division
import itertools

import numpy as np
from six.moves import xrange

from theano import tensor as T
import theano
import theano.tensor.tests.test_extra_ops

from theano.tensor.extra_ops import cumsum, CumsumOp
from theano.tests.unittest_tools import SkipTest
from theano.tests import unittest_tools as utt

from .config import mode_with_gpu, test_ctx_name
from ..extra_ops import GpuCumsum
from ..type import get_context


class TestGpuCumsum(theano.tensor.tests.test_extra_ops.TestCumsumOp):
    mode = mode_with_gpu

    def setUp(self):
        super(TestGpuCumsum, self).setUp()
        test_ctx = get_context(test_ctx_name)
        if test_ctx.kind != 'cuda':
            raise SkipTest("Cuda specific tests")
        self.max_threads_dim0 = test_ctx.maxlsize0
        self.max_grid_size1 = test_ctx.maxgsize2

    def test_Strides1D(self):
        x = T.fvector('x')

        for axis in [0, None, -1]:
            a = np.random.random((42,)).astype("float32")
            cumsum_function = theano.function([x], cumsum(x, axis=axis),
                                              mode=self.mode)

            slicings = [slice(None, None, None),    # Normal strides
                        slice(None, None, 2),       # Stepped strides
                        slice(None, None, -1),      # Negative strides
                        ]

            # Cartesian product of all slicings to test.
            for slicing in itertools.product(slicings, repeat=x.ndim):
                f = theano.function([x], cumsum(x[slicing], axis=axis),
                                    mode=self.mode)
                assert [n for n in f.maker.fgraph.toposort()
                        if isinstance(n.op, GpuCumsum)]
                utt.assert_allclose(np.cumsum(a[slicing], axis=axis), f(a))
                utt.assert_allclose(np.cumsum(a[slicing], axis=axis),
                                    cumsum_function(a[slicing]))

    def test_Strides2D(self):
        x = T.fmatrix('x')

        for axis in [0, 1, None, -1, -2]:
            a = np.random.random((42, 30)).astype("float32")
            cumsum_function = theano.function([x], cumsum(x, axis=axis),
                                              mode=self.mode)

            slicings = [slice(None, None, None),    # Normal strides
                        slice(None, None, 2),       # Stepped strides
                        slice(None, None, -1),      # Negative strides
                        ]

            # Cartesian product of all slicings to test.
            for slicing in itertools.product(slicings, repeat=x.ndim):
                f = theano.function([x], cumsum(x[slicing], axis=axis),
                                    mode=self.mode)
                assert [n for n in f.maker.fgraph.toposort()
                        if isinstance(n.op, GpuCumsum)]
                utt.assert_allclose(np.cumsum(a[slicing], axis=axis), f(a))
                utt.assert_allclose(np.cumsum(a[slicing], axis=axis),
                                    cumsum_function(a[slicing]))

    def test_Strides3D(self):
        x = T.ftensor3('x')

        for axis in [0, 1, 2, None, -1, -2, -3]:
            a = np.random.random((42, 30, 25)).astype("float32")
            cumsum_function = theano.function([x], cumsum(x, axis=axis),
                                              mode=self.mode)

            slicings = [slice(None, None, None),    # Normal strides
                        slice(None, None, 2),       # Stepped strides
                        slice(None, None, -1),      # Negative strides
                        ]

            # Cartesian product of all slicings to test.
            for slicing in itertools.product(slicings, repeat=x.ndim):
                f = theano.function([x], cumsum(x[slicing], axis=axis),
                                    mode=self.mode)
                assert [n for n in f.maker.fgraph.toposort()
                        if isinstance(n.op, GpuCumsum)]
                utt.assert_allclose(np.cumsum(a[slicing], axis=axis), f(a))
                utt.assert_allclose(np.cumsum(a[slicing], axis=axis),
                                    cumsum_function(a[slicing]))

    def test_GpuCumsum1D(self):
        block_max_size = self.max_threads_dim0 * 2

        x = T.fvector('x')
        f = theano.function([x], cumsum(x), mode=self.mode)
        assert [n for n in f.maker.fgraph.toposort()
                if isinstance(n.op, GpuCumsum)]

        # Extensive testing for the first 1025 sizes
        a = np.random.random(1025).astype("float32")
        for i in xrange(a.shape[0]):
            utt.assert_allclose(np.cumsum(a[:i]), f(a[:i]))

        # Use multiple GPU threadblocks
        a = np.random.random((block_max_size + 2, )).astype("float32")
        utt.assert_allclose(np.cumsum(a), f(a))

        # Use recursive cumsum
        a = np.ones((block_max_size * (block_max_size + 1) + 2,),
                    dtype="float32")
        utt.assert_allclose(np.cumsum(a), f(a))

    def test_GpuCumsum2D(self):
        block_max_size = self.max_threads_dim0 * 2

        x = T.fmatrix('x')
        for shape_axis, axis in zip([0, 1, 0, 1, 0], [0, 1, None, -1, -2]):
            f = theano.function([x], cumsum(x, axis=axis), mode=self.mode)
            assert [n for n in f.maker.fgraph.toposort()
                    if isinstance(n.op, GpuCumsum)]

            # Extensive testing for the first 1025 sizes
            a_shape = [5, 5]
            a_shape[shape_axis] = 1025
            a = np.random.random(a_shape).astype("float32")
            slices = [slice(None), slice(None)]
            for i in xrange(a.shape[shape_axis]):
                slices[shape_axis] = slice(i)
                fa = f(a[slices])
                npa = np.cumsum(a[slices], axis=axis)
                utt.assert_allclose(npa, fa)

            # Use multiple GPU threadblocks
            a_shape = [5, 5]
            a_shape[shape_axis] = block_max_size + 2
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a))

            # Use multiple GPU gridblocks
            a_shape = [4, 4]
            a_shape[1 - shape_axis] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a), rtol=5e-5)

            # Use recursive cumsum
            a_shape = [3, 3]
            a_shape[shape_axis] = block_max_size * (block_max_size + 1) + 2
            a = np.random.random(a_shape).astype("float32")
            a = np.sign(a - 0.5).astype("float32")  # Avoid floating point error
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a))

    def test_GpuCumsum3D(self):
        block_max_size = self.max_threads_dim0 * 2

        x = T.ftensor3('x')
        for shape_axis, axis in zip([0, 1, 2, 0, 2, 1, 0], [0, 1, 2, None, -1, -2, -3]):
            f = theano.function([x], cumsum(x, axis=axis), mode=self.mode)
            assert [n for n in f.maker.fgraph.toposort()
                    if isinstance(n.op, GpuCumsum)]

            # Extensive testing for the first 1025 sizes
            a_shape = [5, 5, 5]
            a_shape[shape_axis] = 1025
            a = np.random.rand(*a_shape).astype("float32")
            slices = [slice(None), slice(None), slice(None)]
            for i in xrange(a.shape[shape_axis]):
                slices[shape_axis] = slice(i)
                fa = f(a[slices])
                npa = np.cumsum(a[slices], axis=axis)
                utt.assert_allclose(npa, fa)

            # Use multiple GPU threadblocks (along accumulation axis)
            a_shape = [2, 2, 2]
            a_shape[shape_axis] = block_max_size + 2
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a))

            # Use multiple GPU gridblocks (not along accumulation axis)
            a_shape = [5, 5, 5]
            a_shape[(shape_axis + 1) % 3] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            if axis is None:
                # Avoid floating point error
                a = np.sign(a - 0.5).astype("float32")
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a))

            a_shape = [5, 5, 5]
            a_shape[(shape_axis + 2) % 3] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            if axis is None:
                # Avoid floating point error
                a = np.sign(a - 0.5).astype("float32")
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a))

            # Use recursive cumsum (along accumulation axis)
            a_shape = [3, 3, 3]
            a_shape[shape_axis] = block_max_size * (block_max_size + 1) + 2
            a = np.random.random(a_shape).astype("float32")
            a = np.sign(a - 0.5).astype("float32")  # Avoid floating point error
            utt.assert_allclose(np.cumsum(a, axis=axis), f(a))

    def test_GpuCumsum4D(self):
        # Should not use the GPU version.
        x = T.ftensor4('x')
        f = theano.function([x], cumsum(x, axis=1), mode=self.mode)
        assert [n for n in f.maker.fgraph.toposort()
                if isinstance(n.op, CumsumOp)]
