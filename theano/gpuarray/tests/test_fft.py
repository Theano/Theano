from __future__ import absolute_import, print_function, division
import unittest
import numpy as np

import theano
from theano.tests import unittest_tools as utt

# Skip tests if pygpu is not available.
from nose.plugins.skip import SkipTest
from theano.gpuarray.fft import pygpu_available, scikits_cuda_available
if not pygpu_available:  # noqa
    raise SkipTest('Optional package pygpu not available')
if not scikits_cuda_available:  # noqa
    raise SkipTest('Optional package scikits.cuda not available')

import theano.gpuarray.fft
import theano.tensor.fourier

from .config import mode_with_gpu


class TestFFT(unittest.TestCase):

    def test_fft(self):
        N = 64
        inputs_val = np.random.random((1, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        fft_ref = theano.tensor.fourier.fft(inputs, N, 1)
        fft = theano.gpuarray.fft.cufft(inputs)

        f_ref = theano.function([], fft_ref)
        f_fft = theano.function([], fft, mode=mode_with_gpu)

        res_ref = f_ref()
        res_fft = f_fft()

        res_fft_comp = (np.asarray(res_fft[:, :, 0]) +
                        1j * np.asarray(res_fft[:, :, 1]))

        utt.assert_allclose(res_ref[0][0:N / 2 + 1], res_fft_comp)

    def test_ifft(self):
        N = 64
        inputs_val = np.random.random((1, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        fft = theano.gpuarray.fft.cufft(inputs)
        f_fft = theano.function([], fft, mode=mode_with_gpu)
        res_fft = f_fft()

        m = fft.type()
        ifft = theano.gpuarray.fft.cuifft(m)
        f_ifft = theano.function([m], ifft, mode=mode_with_gpu)
        res_ifft = f_ifft(res_fft)

        utt.assert_allclose(inputs_val, np.asarray(res_ifft) / N)

    def test_type(self):
        N = 64
        inputs_val = np.random.random((1, N)).astype('float64')
        inputs = theano.shared(inputs_val)

        with self.assertRaises(AssertionError):
            theano.gpuarray.fft.cufft(inputs)
        with self.assertRaises(AssertionError):
            theano.gpuarray.fft.cuifft(inputs)
