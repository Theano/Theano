import unittest
from theano.tensor.scipy_ndimage import Zoom
from theano.tests.unittest_tools import verify_grad
import theano as T
from scipy.ndimage.interpolation import zoom
from numpy.testing import assert_array_equal
import numpy as np

def getFn():
    x   = T.tensor.tensor4()
    ref = T.tensor.tensor4()
    op = Zoom()
    return T.function([x, ref], op(x, ref))

class T_SharedRandomStreams(unittest.TestCase):
    def test_Zoom_single_channel_factor_2(self):
        NCHANS = 1
        ZFACTOR = 2
        insize  = (1, NCHANS, 32, 32)
        outsize = (1, NCHANS, int(insize[2] * ZFACTOR), int(insize[3] * ZFACTOR))
        #
        x   = np.random.uniform(size=insize).astype(np.float32)
        ref = np.random.uniform(size=outsize).astype(np.float32)
        f = getFn()
        #
        out = f(x, ref)
        assert out.shape == outsize
        for i in range(NCHANS):
            expected = zoom(x, (1, 1, ZFACTOR, ZFACTOR), order=0)
            assert_array_equal(out[0,i], expected[0,i])

    def test_Zoom_single_channel_factor_1p875(self):
        NCHANS = 1
        ZFACTOR = 1.875
        insize  = (1, NCHANS, 32, 32)
        outsize = (1, NCHANS, int(insize[2] * ZFACTOR), int(insize[3] * ZFACTOR))
        #
        x   = np.random.uniform(size=insize).astype(np.float32)
        ref = np.random.uniform(size=outsize).astype(np.float32)
        f = getFn()
        #
        out = f(x, ref)
        assert out.shape == outsize
        for i in range(NCHANS):
            expected = zoom(x, (1, 1, ZFACTOR, ZFACTOR), order=0)
            assert_array_equal(out[0,i], expected[0,i])

    def test_Zoom_single_channel_factor_1(self):
        NCHANS = 1
        ZFACTOR = 1
        insize  = (1, NCHANS, 32, 32)
        outsize = (1, NCHANS, int(insize[2] * ZFACTOR), int(insize[3] * ZFACTOR))
        #
        x   = np.random.uniform(size=insize).astype(np.float32)
        ref = np.random.uniform(size=outsize).astype(np.float32)
        f = getFn()
        #
        out = f(x, ref)
        assert out.shape == outsize
        for i in range(NCHANS):
            expected = zoom(x, (1, 1, ZFACTOR, ZFACTOR), order=0)
            assert_array_equal(out[0,i], expected[0,i])

    def test_Zoom_three_channels_factor_2(self):
        NCHANS = 3
        ZFACTOR = 1
        insize  = (1, NCHANS, 32, 32)
        outsize = (1, NCHANS, int(insize[2] * ZFACTOR), int(insize[3] * ZFACTOR))
        #
        x   = np.random.uniform(size=insize).astype(np.float32)
        ref = np.random.uniform(size=outsize).astype(np.float32)
        f = getFn()
        #
        out = f(x, ref)
        assert out.shape == outsize
        for i in range(NCHANS):
            expected = zoom(x, (1, 1, ZFACTOR, ZFACTOR), order=0)
            assert_array_equal(out[0,i], expected[0,i])

    def test_Zoom_gradient_single_channel_factor_1(self):
        NCHANS = 1
        ZFACTOR = 1
        insize  = (1, NCHANS, 32, 32)
        outsize = (1, NCHANS, int(insize[2] * ZFACTOR), int(insize[3] * ZFACTOR))
        #
        x   = np.random.uniform(size=insize).astype(np.float32)
        ref = np.random.uniform(size=outsize).astype(np.float32)
        op = Zoom()
        #
        verify_grad(op,[x, ref])

    # TODO: Test failing
    def test_Zoom_gradient_single_channel_factor_2(self):
        NCHANS = 1
        ZFACTOR = 2
        insize  = (1, NCHANS, 32, 32)
        outsize = (1, NCHANS, int(insize[2] * ZFACTOR), int(insize[3] * ZFACTOR))
        # inputs
        x   = np.random.uniform(size=insize).astype(np.float32)
        ref = np.random.uniform(size=outsize).astype(np.float32)
        # op
        op = Zoom()
        #
        verify_grad(op,[x, ref])
