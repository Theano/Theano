# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available is False:
    raise SkipTest('Optional package cuda disabled')

import theano.tensor.tests.test_extra_ops
from theano.sandbox.cuda.extra_ops import GpuCumsum

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

from theano import tensor as T
import numpy as np
import theano
from theano import config
from theano.tensor.extra_ops import cumsum


class TestGpuCumsum(theano.tensor.tests.test_extra_ops.TestCumsumOp):
    mode = mode_with_gpu
    op = GpuCumsum

    def setUp(self):
        super(TestGpuCumsum, self).setUp()

        # Fetch some useful properties on the device
        cuda = theano.sandbox.cuda
        device_id = cuda.use.device_number
        if device_id is None:
            cuda.use("gpu",
                     force=False,
                     default_to_move_computation_to_gpu=False,
                     move_shared_float32_to_gpu=False,
                     enable_cuda=False,
                     test_driver=True)
            device_id = cuda.use.device_number
        cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
        prop = cuda_ndarray.device_properties(device_id)
        self.max_threads_dim0 = prop['maxThreadsDim0']
        self.max_grid_size1 = prop['maxGridSize1']

    def test_Strides1D(self):
        x = T.fvector('x')

        # Stepped strides
        f = theano.function([x], cumsum(x[::2]), mode=self.mode)
        a = np.random.randint(10, size=(42,)).astype("float32")
        assert np.allclose(np.cumsum(a[::2]), f(a))

        # Negative strides
        f = theano.function([x], cumsum(x[::-1]), mode=self.mode)
        a = np.random.randint(10, size=(42,)).astype("float32")
        assert np.allclose(np.cumsum(a[::-1]), f(a))

    def test_GpuCumsum1D(self):
        block_max_size = self.max_threads_dim0 * 2

        x = T.fvector('x')
        f = theano.function([x], cumsum(x), mode=self.mode)

        # Extensive testing for the first 1k sizes
        a = np.ones((int(1e3),), dtype="float32")
        for i in xrange(a.shape[0]):
            assert np.allclose(np.cumsum(a[:i]), f(a[:i]))

        # Use multiple GPU threadblocks
        a = np.random.random((block_max_size+2,)).astype("float32")
        assert np.allclose(np.cumsum(a), f(a))

        # Use recursive cumsum
        a = np.ones((block_max_size*(block_max_size+1)+2,),
                    dtype="float32")
        assert np.allclose(np.cumsum(a), f(a))

    def test_GpuCumsum2D(self):
        block_max_size = self.max_threads_dim0 * 2

        x = T.fmatrix('x')
        for axis in xrange(2):
            f = theano.function([x], cumsum(x, axis=axis), mode=self.mode)

            # Extensive testing for the first 1k sizes
            a_shape = [11, 11]
            a_shape[axis] = int(1e3)
            a = np.ones(a_shape, dtype="float32")
            slices = [slice(None), slice(None)]
            for i in xrange(a.shape[axis]):
                slices[axis] = slice(i)
                fa = f(a[slices])
                npa = np.cumsum(a[slices], axis=axis)
                assert np.allclose(npa, fa)

            # Use multiple GPU threadblocks
            a_shape = [11, 11]
            a_shape[axis] = block_max_size+2
            a = np.ones(a_shape, dtype="float32")
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Use multiple GPU gridblocks
            a_shape = [11, 11]
            a_shape[1-axis] = self.max_grid_size1+1
            a = np.ones(a_shape, dtype="float32")
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Use recursive cumsum
            a_shape = [11, 11]
            a_shape[axis] = block_max_size*(block_max_size+1)+2
            a = np.ones(a_shape, dtype="float32")
            assert np.allclose(np.cumsum(a, axis=axis), f(a))
