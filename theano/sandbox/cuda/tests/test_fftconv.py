import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')
import theano.sandbox.cuda.fftconv

from theano.sandbox.cuda import float32_shared_constructor as shared

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestConv2dFFT(unittest.TestCase):
    def run_conv(self, inputs_shape, filters_shape, border_mode,
                 autopad=False):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv_ref = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                                  border_mode=border_mode)
        conv_fft = theano.sandbox.cuda.fftconv.FFTConv2D(
            border_mode=border_mode, autopad=autopad)(inputs, filters)

        f_ref = theano.function([], conv_ref)
        f_fft = theano.function([], conv_fft, mode=mode_with_gpu)

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)

    def test_valid(self):
        self.run_conv(inputs_shape=(5, 3, 7, 6),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='valid')
        self.run_conv(inputs_shape=(5, 3, 7, 7),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='valid', autopad=True)

    def test_full(self):
        self.run_conv(inputs_shape=(5, 3, 7, 6),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='full')
        self.run_conv(inputs_shape=(5, 3, 7, 7),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='full', autopad=True)

    def test_opt_valid(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters)

        mode = mode_with_gpu.including('conv_fft_valid')

        f_ref = theano.function([], conv)
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.FFTConv2D)
                   for n in topo) == 1


        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)

    def test_opt_full(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                              border_mode='full')

        mode = mode_with_gpu.including('conv_fft_full')

        f_ref = theano.function([], conv)
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.FFTConv2D)
                   for n in topo) == 1

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)
