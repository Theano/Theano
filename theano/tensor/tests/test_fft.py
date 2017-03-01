from __future__ import absolute_import, print_function, division
import numpy as np
import unittest

import theano
from theano import tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor import fft

N = 16


class TestFFT(unittest.TestCase):

    def test_rfft_float(self):
        # Test that numpy's default float64 output is cast to theano input type
        eps = 1e-1

        def f_rfft(inp):
            return fft.rfft(inp)
        inputs_val = np.random.random((1, N)).astype(theano.config.floatX)
        utt.verify_grad(f_rfft, [inputs_val], eps=eps)

        def f_irfft(inp):
            return fft.irfft(inp)
        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype(theano.config.floatX)
        utt.verify_grad(f_irfft, [inputs_val], eps=eps)

    def test_1Drfft(self):
        inputs_val = np.random.random((1, N)).astype(theano.config.floatX)

        x = T.matrix('x')
        rfft = fft.rfft(x)
        f_rfft = theano.function([x], rfft)
        res_rfft = f_rfft(inputs_val)
        res_rfft_comp = (np.asarray(res_rfft[:, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, 1]))

        rfft_ref = np.fft.rfft(inputs_val, axis=1)

        utt.assert_allclose(rfft_ref, res_rfft_comp)

        m = rfft.type()
        print(m.ndim)
        irfft = fft.irfft(m)
        f_irfft = theano.function([m], irfft)
        res_irfft = f_irfft(res_rfft)

        utt.assert_allclose(inputs_val, np.asarray(res_irfft))

        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return fft.rfft(inp)
        inputs_val = np.random.random((1, N)).astype(theano.config.floatX)
        utt.verify_grad(f_rfft, [inputs_val], eps=eps)

        def f_irfft(inp):
            return fft.irfft(inp)
        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype(theano.config.floatX)
        utt.verify_grad(f_irfft, [inputs_val], eps=eps)

    def test_rfft(self):
        inputs_val = np.random.random((1, N, N)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)

        rfft = fft.rfft(inputs)
        f_rfft = theano.function([], rfft)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        rfft_ref = np.fft.rfftn(inputs_val, axes=(1, 2))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

    def test_irfft(self):
        inputs_val = np.random.random((1, N, N)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)

        rfft = fft.rfft(inputs)
        f_rfft = theano.function([], rfft)
        res_fft = f_rfft()

        m = rfft.type()
        irfft = fft.irfft(m)
        f_irfft = theano.function([m], irfft)
        res_irfft = f_irfft(res_fft)

        utt.assert_allclose(inputs_val, np.asarray(res_irfft))

        inputs_val = np.random.random((1, N, N, 2)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)

        irfft = fft.irfft(inputs)
        f_irfft = theano.function([], irfft)
        res_irfft = f_irfft()
        inputs_ref = inputs_val[..., 0] + inputs_val[..., 1] * 1j

        irfft_ref = np.fft.irfftn(inputs_ref, axes=(1, 2))

        utt.assert_allclose(irfft_ref, res_irfft, atol=1e-4, rtol=1e-4)

    def test_norm_rfft(self):
        inputs_val = np.random.random((1, N, N)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)

        # Unitary normalization
        rfft = fft.rfft(inputs, norm='ortho')
        f_rfft = theano.function([], rfft)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        rfft_ref = np.fft.rfftn(inputs_val, axes=(1, 2))

        utt.assert_allclose(rfft_ref / N, res_rfft_comp, atol=1e-4, rtol=1e-4)

        # No normalization
        rfft = fft.rfft(inputs, norm='no_norm')
        f_rfft = theano.function([], rfft)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, :, 1]))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

        # Inverse FFT inputs
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)
        inputs_ref = inputs_val[..., 0] + 1j * inputs_val[..., 1]

        # Unitary normalization inverse FFT
        irfft = fft.irfft(inputs, norm='ortho')
        f_irfft = theano.function([], irfft)
        res_irfft = f_irfft()

        irfft_ref = np.fft.irfftn(inputs_ref, axes=(1, 2))

        utt.assert_allclose(irfft_ref * N, res_irfft, atol=1e-4, rtol=1e-4)

        # No normalization inverse FFT
        irfft = fft.irfft(inputs, norm='no_norm')
        f_irfft = theano.function([], irfft)
        res_irfft = f_irfft()

        utt.assert_allclose(irfft_ref * N**2, res_irfft, atol=1e-4, rtol=1e-4)

    def test_params(self):
        inputs_val = np.random.random((1, N)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)

        self.assertRaises(ValueError, fft.rfft, inputs, norm=123)

        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype(theano.config.floatX)
        inputs = theano.shared(inputs_val)

        self.assertRaises(ValueError, fft.irfft, inputs, norm=123)
        self.assertRaises(ValueError, fft.irfft, inputs, is_odd=123)

    def test_grad_rfft(self):
        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return fft.rfft(inp)
        inputs_val = np.random.random((1, N, N)).astype(theano.config.floatX)
        utt.verify_grad(f_rfft, [inputs_val], eps=eps)

        def f_irfft(inp):
            return fft.irfft(inp)
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype(theano.config.floatX)
        utt.verify_grad(f_irfft, [inputs_val], eps=eps)

        def f_rfft(inp):
            return fft.rfft(inp, norm='ortho')
        inputs_val = np.random.random((1, N, N)).astype(theano.config.floatX)
        utt.verify_grad(f_rfft, [inputs_val], eps=eps)

        def f_irfft(inp):
            return fft.irfft(inp, norm='no_norm')
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype(theano.config.floatX)
        utt.verify_grad(f_irfft, [inputs_val], eps=eps)
