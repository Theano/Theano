from __future__ import absolute_import, print_function, division
import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if not cuda_ndarray.cuda_available:  # noqa
    raise SkipTest('Optional package cuda not available')
from theano.misc.pycuda_init import pycuda_available
if not pycuda_available:  # noqa
    raise SkipTest('Optional package pycuda not available')
from theano.sandbox.cuda.fftconv import scikits_cuda_available
if not scikits_cuda_available:  # noqa
    raise SkipTest('Optional package scikits.cuda not available')

from theano.sandbox.cuda import float32_shared_constructor as shared
import theano.sandbox.cuda.fftconv

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestConv2dFFT(unittest.TestCase):
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
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2, topo

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
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2, topo

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)

    def test_opt_nofft_valid(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                              version='no_fft')

        mode = mode_with_gpu.including('conv_fft_valid')

        f_fft = theano.function([], conv, mode=mode)

        # make sure we that no CuFFTOp has been inserted
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 0

    def test_opt_nofft_full(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                              border_mode='full',
                                              version='no_fft')

        mode = mode_with_gpu.including('conv_fft_full')

        f_fft = theano.function([], conv, mode=mode)

        # make sure we that no CuFFTOp has been inserted
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 0


class TestConv3dFFT(unittest.TestCase):

    def run_conv_valid(self, inputs_shape, filters_shape, pad=False):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[0]).astype('float32'))

        # Flip filter as conv3D compute correlation
        filters_flip = filters[:, ::-1, ::-1, ::-1, :]
        # filters_flip = filters
        conv_ref = theano.tensor.nnet.conv3D(V=inputs, W=filters_flip,
                                             b=bias, d=(1, 1, 1))

        conv_fft = theano.sandbox.cuda.fftconv.conv3d_fft(
            inputs.dimshuffle(0, 4, 1, 2, 3),
            filters.dimshuffle(0, 4, 1, 2, 3),
            border_mode="valid",
            pad_last_dim=pad)
        conv_fft = conv_fft.dimshuffle(0, 2, 3, 4, 1)

        f_ref = theano.function([], conv_ref, mode="FAST_RUN")
        mode = mode_with_gpu
        mode.check_py_code = False
        f_fft = theano.function([], conv_fft, mode=mode)

        res_ref = f_ref()
        res_fft = f_fft()
        utt.assert_allclose(res_ref, res_fft, rtol=1e-05, atol=1e-05)

    def run_conv_full(self, inputs_shape, filters_shape, pad=False):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[4]).astype('float32'))

        conv_ref = theano.tensor.nnet.convTransp3D(
            W=filters, b=bias, d=(1, 1, 1),
            H=inputs)

        filters = filters.dimshuffle(4, 0, 1, 2, 3)
        inputs = inputs.dimshuffle(0, 4, 1, 2, 3)
        conv_fft = theano.sandbox.cuda.fftconv.conv3d_fft(inputs, filters,
                                                          border_mode="full",
                                                          pad_last_dim=pad)
        conv_fft = conv_fft.dimshuffle(0, 2, 3, 4, 1)

        f_ref = theano.function([], conv_ref)
        f_fft = theano.function([], conv_fft, mode=mode_with_gpu)

        res_ref = f_ref()
        res_fft = f_fft()
        utt.assert_allclose(res_ref, res_fft, rtol=1e-04, atol=1e-04)

    def test_valid(self):
        self.run_conv_valid(inputs_shape=(16, 20, 32, 16, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            pad=True)
        self.run_conv_valid(inputs_shape=(16, 20, 32, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            pad=True)

    def test_full(self):
        self.run_conv_full(inputs_shape=(16, 15, 21, 16, 10),
                           filters_shape=(10, 6, 12, 4, 1),
                           pad=True)
        self.run_conv_full(inputs_shape=(16, 15, 21, 12, 10),
                           filters_shape=(10, 6, 12, 4, 1),
                           pad=True)

    def test_opt_conv3d_fft(self):
        inputs_shape = (16, 20, 32, 16, 1)
        filters_shape = (10, 6, 12, 4, 1)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[0]).astype('float32'))

        conv = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                         b=bias, d=(1, 1, 1))
        mode = mode_with_gpu.including('conv3d_fft')
        mode.check_py_code = False

        f_ref = theano.function([], conv, mode="FAST_RUN")
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)

    def test_opt_convgrad3d_fft(self):
        inputs_shape = (2, 17, 15, 16, 1)
        filters_shape = (10, 6, 7, 4, 1)
        dCdH_shape = (inputs_shape[0],
                      inputs_shape[1] - filters_shape[1] + 1,
                      inputs_shape[2] - filters_shape[2] + 1,
                      inputs_shape[3] - filters_shape[3] + 1,
                      filters_shape[0])

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        dCdH_val = numpy.random.random(dCdH_shape).astype('float32')

        inputs = shared(inputs_val)
        dCdH = shared(dCdH_val)

        conv = theano.tensor.nnet.convGrad3D(V=inputs, dCdH=dCdH,
                                             WShape=filters_shape,
                                             d=(1, 1, 1))
        mode = mode_with_gpu.including('convgrad3d_fft')
        mode.check_py_code = False

        f_ref = theano.function([], conv, mode="FAST_RUN")
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft, rtol=1e-04, atol=1e-04)

    def test_opt_convtransp3d_fft(self):
        inputs_shape = (2, 9, 16, 12, 10)
        filters_shape = (10, 3, 8, 4, 1)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')
        bias = shared(numpy.zeros(filters_shape[4]).astype('float32'))

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.convTransp3D(W=filters, b=bias, d=(1, 1, 1),
                                               H=inputs)
        mode = mode_with_gpu.including('convtransp3d_fft')

        f_ref = theano.function([], conv)
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft, rtol=1e-04, atol=1e-04)
