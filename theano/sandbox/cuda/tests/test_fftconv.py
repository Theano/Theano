import unittest
import numpy

import theano
from theano import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda.float32_shared_constructor as shared

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestConv2dFFT(unittest.TestCase):
    def setUp(self):
        self._prev = theano.confg.enable_conv2d_fft
        theano.confg.enable_conv2d_fft = True

    def tearDown(self):
        theano.confg.enable_conv2d_fft = self._prev

    def run_conv(self, inputs_shape, filters_shape, pad=False, **other_args):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv_ref = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                                  **other_args)
        conv_fft = theano.sandbox.cuda.fftconv.conv2d_fft(inputs, filters,
                                                          pad_last_dim=pad,
                                                          **other_args)

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
                      border_mode='valid', pad=True)

    def test_full(self):
        self.run_conv(inputs_shape=(5, 3, 7, 6),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='full')
        self.run_conv(inputs_shape=(5, 3, 7, 7),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='full', pad=True)

    def test_opt(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters)

        f_ref = theano.function([], conv)
        f_fft = theano.function([], conv, mode=mode_with_gpu)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert len(op for op in topo
                   if isinstance(op, theano.sandbox.cuda.fftconv.CuFFTOp)) == 1

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)
