from __future__ import absolute_import, print_function, division
import unittest
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt

import theano.gpuarray.fft

from .config import mode_with_gpu

# Skip tests if pygpu is not available.
from nose.plugins.skip import SkipTest
from theano.gpuarray.fft import pygpu_available, skcuda_available, pycuda_available
if not pygpu_available:  # noqa
    raise SkipTest('Optional package pygpu not available')
if not skcuda_available:  # noqa
    raise SkipTest('Optional package scikit-cuda not available')
if not pycuda_available:  # noqa
    raise SkipTest('Optional package pycuda not available')

# Transform sizes
N = 32


class TestFFT(unittest.TestCase):

    def test_1Dfft(self):
        inputs_val = np.random.random((1, N)).astype('float32')

        x = T.matrix('x', dtype='float32')
        rfft = theano.gpuarray.fft.curfft(x)
        f_rfft = theano.function([x], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft(inputs_val)
        res_rfft_comp = (np.asarray(res_rfft[:, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, 1]))

        rfft_ref = np.fft.rfft(inputs_val, axis=1)

        utt.assert_allclose(rfft_ref, res_rfft_comp)

        m = rfft.type()
        irfft = theano.gpuarray.fft.cuirfft(m)
        f_irfft = theano.function([m], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft(res_rfft)

        utt.assert_allclose(inputs_val, np.asarray(res_irfft))

        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return theano.gpuarray.fft.curfft(inp)
        inputs_val = np.random.random((1, N)).astype('float32')
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return theano.gpuarray.fft.cuirfft(inp)
        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype('float32')
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

    def test_rfft(self):
        inputs_val = np.random.random((1, N, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        rfft = theano.gpuarray.fft.curfft(inputs)
        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        rfft_ref = np.fft.rfftn(inputs_val, axes=(1, 2))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

    def test_irfft(self):
        inputs_val = np.random.random((1, N, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        fft = theano.gpuarray.fft.curfft(inputs)
        f_fft = theano.function([], fft, mode=mode_with_gpu)
        res_fft = f_fft()

        m = fft.type()
        ifft = theano.gpuarray.fft.cuirfft(m)
        f_ifft = theano.function([m], ifft, mode=mode_with_gpu)
        res_ifft = f_ifft(res_fft)

        utt.assert_allclose(inputs_val, np.asarray(res_ifft))

        inputs_val = np.random.random((1, N, N, 2)).astype('float32')
        inputs = theano.shared(inputs_val)

        irfft = theano.gpuarray.fft.cuirfft(inputs)
        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()
        inputs_ref = inputs_val[..., 0] + inputs_val[..., 1] * 1j

        irfft_ref = np.fft.irfftn(inputs_ref, axes=(1, 2))

        utt.assert_allclose(irfft_ref, res_irfft, atol=1e-4, rtol=1e-4)

    def test_type(self):
        inputs_val = np.random.random((1, N)).astype('float64')
        inputs = theano.shared(inputs_val)

        with self.assertRaises(AssertionError):
            theano.gpuarray.fft.curfft(inputs)
        with self.assertRaises(AssertionError):
            theano.gpuarray.fft.cuirfft(inputs)

    def test_norm(self):
        inputs_val = np.random.random((1, N, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        # Unitary normalization
        rfft = theano.gpuarray.fft.curfft(inputs, norm='ortho')
        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        rfft_ref = np.fft.rfftn(inputs_val, axes=(1, 2))

        utt.assert_allclose(rfft_ref / N, res_rfft_comp, atol=1e-4, rtol=1e-4)

        # No normalization
        rfft = theano.gpuarray.fft.curfft(inputs, norm='no_norm')
        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

        # Inverse FFT inputs
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype('float32')
        inputs = theano.shared(inputs_val)
        inputs_ref = inputs_val[:, :, :, 0] + 1j * inputs_val[:, :, :, 1]

        # Unitary normalization inverse FFT
        irfft = theano.gpuarray.fft.cuirfft(inputs, norm='ortho')
        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()

        irfft_ref = np.fft.irfftn(inputs_ref, axes=(1, 2))

        utt.assert_allclose(irfft_ref * N, res_irfft, atol=1e-4, rtol=1e-4)

        # No normalization inverse FFT
        irfft = theano.gpuarray.fft.cuirfft(inputs, norm='no_norm')
        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()

        utt.assert_allclose(irfft_ref * N**2, res_irfft, atol=1e-4, rtol=1e-4)

    def test_grad(self):
        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return theano.gpuarray.fft.curfft(inp)
        inputs_val = np.random.random((1, N, N)).astype('float32')
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return theano.gpuarray.fft.cuirfft(inp)
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype('float32')
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_rfft(inp):
            return theano.gpuarray.fft.curfft(inp, norm='ortho')
        inputs_val = np.random.random((1, N, N)).astype('float32')
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return theano.gpuarray.fft.cuirfft(inp, norm='no_norm')
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype('float32')
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

    def test_odd(self):
        M = N - 1

        inputs_val = np.random.random((1, M, M)).astype('float32')
        inputs = theano.shared(inputs_val)

        rfft = theano.gpuarray.fft.curfft(inputs)
        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()

        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        rfft_ref = np.fft.rfftn(inputs_val, s=(M, M), axes=(1, 2))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

        m = rfft.type()
        ifft = theano.gpuarray.fft.cuirfft(m, is_odd=True)
        f_ifft = theano.function([m], ifft, mode=mode_with_gpu)
        res_ifft = f_ifft(res_rfft)

        utt.assert_allclose(inputs_val, np.asarray(res_ifft))

        inputs_val = np.random.random((1, M, M // 2 + 1, 2)).astype('float32')
        inputs = theano.shared(inputs_val)

        irfft = theano.gpuarray.fft.cuirfft(inputs, norm='ortho', is_odd=True)
        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()

        inputs_ref = inputs_val[:, :, :, 0] + 1j * inputs_val[:, :, :, 1]
        irfft_ref = np.fft.irfftn(inputs_ref, s=(M, M), axes=(1, 2)) * M

        utt.assert_allclose(irfft_ref, res_irfft, atol=1e-4, rtol=1e-4)

        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return theano.gpuarray.fft.curfft(inp)
        inputs_val = np.random.random((1, M, M)).astype('float32')
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return theano.gpuarray.fft.cuirfft(inp, is_odd=True)
        inputs_val = np.random.random((1, M, M // 2 + 1, 2)).astype('float32')
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_rfft(inp):
            return theano.gpuarray.fft.curfft(inp, norm='ortho')
        inputs_val = np.random.random((1, M, M)).astype('float32')
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return theano.gpuarray.fft.cuirfft(inp, norm='no_norm', is_odd=True)
        inputs_val = np.random.random((1, M, M // 2 + 1, 2)).astype('float32')
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

    def test_params(self):
        inputs_val = np.random.random((1, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        self.assertRaises(ValueError, theano.gpuarray.fft.curfft, inputs, norm=123)

        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype('float32')
        inputs = theano.shared(inputs_val)

        self.assertRaises(ValueError, theano.gpuarray.fft.cuirfft, inputs, norm=123)
        self.assertRaises(ValueError, theano.gpuarray.fft.cuirfft, inputs, is_odd=123)
