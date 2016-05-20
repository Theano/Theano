from __future__ import absolute_import, print_function, division
import unittest
import numpy as np

import theano
import theano.tensor
from theano.tests import unittest_tools as utt

import theano.gpuarray.fft
import numpy.fft

from .config import mode_with_gpu

# Skip tests if pygpu is not available.
from nose.plugins.skip import SkipTest
from theano.gpuarray.fft import pygpu_available, scikits_cuda_available
from theano.gpuarray.fft import pycuda_available
if not pygpu_available:  # noqa
    raise SkipTest('Optional package pygpu not available')
if not scikits_cuda_available:  # noqa
    raise SkipTest('Optional package scikits.cuda not available')
if not pycuda_available:  # noqa
    raise SkipTest('Optional package pycuda not available')


class TestFFT(unittest.TestCase):

    def test_rfft(self):
        N = 64
        inputs_val = np.random.random((1, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        rfft = theano.gpuarray.fft.curfft(inputs)
        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = (np.asarray(res_rfft[:, :, 0]) +
                         1j * np.asarray(res_rfft[:, :, 1]))

        rfft_ref = numpy.fft.rfft(inputs_val, N, 1, norm='ortho')

        utt.assert_allclose(rfft_ref, res_rfft_comp)

    def test_irfft(self):
        N = 64
        inputs_val = np.random.random((1, N)).astype('float32')
        inputs = theano.shared(inputs_val)

        fft = theano.gpuarray.fft.curfft(inputs)
        f_fft = theano.function([], fft, mode=mode_with_gpu)
        res_fft = f_fft()

        m = fft.type()
        ifft = theano.gpuarray.fft.cuirfft(m)
        f_ifft = theano.function([m], ifft, mode=mode_with_gpu)
        res_ifft = f_ifft(res_fft)

        utt.assert_allclose(inputs_val, np.asarray(res_ifft))

    def test_type(self):
        N = 64
        inputs_val = np.random.random((1, N)).astype('float64')
        inputs = theano.shared(inputs_val)

        with self.assertRaises(AssertionError):
            theano.gpuarray.fft.curfft(inputs)
        with self.assertRaises(AssertionError):
            theano.gpuarray.fft.cuirfft(inputs)
