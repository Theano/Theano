from __future__ import absolute_import, print_function, division
import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
from theano.tensor.nnet.corr3d import Corr3dMM, Corr3dMM_gradWeights, Corr3dMM_gradInputs
from theano.sandbox.cuda import float32_shared_constructor as shared
from theano.sandbox.cuda.blas import (
    GpuCorr3dMM, GpuCorr3dMM_gradWeights, GpuCorr3dMM_gradInputs)
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import theano.sandbox.cuda as cuda_ndarray
if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda not available')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


# python reference implementation of a 3D convolution
# see also: theano.tensor.nnet.tests.test_conv3d2d
# expects: (batch, 0, channels, 1, 2)
class TestCorr3DMM(unittest.TestCase):

    def run_conv_valid(self, inputs_shape, filters_shape,
                       border_mode='valid',
                       filter_dilation=(1, 1, 1),
                       subsample=(1, 1, 1),
                       verify_grad=False):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv_ref = Corr3dMM(border_mode=border_mode,
                            filter_dilation=filter_dilation,
                            subsample=subsample)(
                                inputs.dimshuffle(0, 4, 1, 2, 3),
                                filters.dimshuffle(0, 4, 1, 2, 3))
        conv_ref = conv_ref.dimshuffle(0, 2, 3, 4, 1)
        f_ref = theano.function([], conv_ref, mode='FAST_RUN')

        conv = GpuCorr3dMM(border_mode=border_mode,
                           filter_dilation=filter_dilation,
                           subsample=subsample)(
                               inputs.dimshuffle(0, 4, 1, 2, 3),
                               filters.dimshuffle(0, 4, 1, 2, 3))
        conv = conv.dimshuffle(0, 2, 3, 4, 1)
        f = theano.function([], conv, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

        if verify_grad:
            utt.verify_grad(GpuCorr3dMM(border_mode=border_mode,
                                        filter_dilation=filter_dilation,
                                        subsample=subsample),
                            [inputs_val.transpose(0, 4, 1, 2, 3),
                             filters_val.transpose(0, 4, 1, 2, 3)],
                            mode=mode_with_gpu)

    def test_valid(self):
        self.run_conv_valid(inputs_shape=(16, 20, 12, 16, 1),
                            filters_shape=(10, 6, 12, 4, 1))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            subsample=(2, 2, 2))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            subsample=(2, 2, 2))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            subsample=(3, 3, 3))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            subsample=(3, 3, 3))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            subsample=(3, 2, 1))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            subsample=(1, 2, 3))

    def test_border_mode(self):
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            border_mode='valid')
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            border_mode='half')
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            border_mode='full')
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            border_mode=(0, 0, 0))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            border_mode=(1, 2, 3))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            border_mode=(3, 2, 1))

    def test_filter_dilation(self):
        inputs_shape = [16, 20, 12, 15, 1]
        filters_shape = [10, 6, 5, 4, 1]

        for filter_dilation in [(2, 1, 1), (1, 2, 1), (1, 1, 2)]:
            for border_mode in ['valid', 'half', 'full']:
                self.run_conv_valid(inputs_shape=inputs_shape,
                                    filters_shape=filters_shape,
                                    filter_dilation=filter_dilation,
                                    border_mode=border_mode)

    def test_verify_gradients(self):
        # use a small example to check the gradients
        inputs_shape = [2, 7, 9, 6, 1]
        filters_shape = [1, 3, 3, 2, 1]

        for filter_dilation in [(2, 1, 1), (1, 2, 1), (1, 1, 2)]:
            for border_mode in ['valid', 'half', 'full', (2, 1, 3)]:
                self.run_conv_valid(inputs_shape=inputs_shape,
                                    filters_shape=filters_shape,
                                    filter_dilation=filter_dilation,
                                    border_mode=border_mode,
                                    verify_grad=True)

    def run_gradweight(self, inputs_shape, filters_shape, dCdH_shape,
                       subsample=(1, 1, 1)):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        dCdH_val = numpy.random.random(dCdH_shape).astype('float32')
        inputs = shared(inputs_val)
        dCdH = shared(dCdH_val)

        img = gpu_contiguous(inputs.dimshuffle(0, 4, 1, 2, 3))
        topgrad = gpu_contiguous(dCdH.dimshuffle(0, 4, 1, 2, 3))
        if (subsample == (1, 1, 1)):
            conv_ref = Corr3dMM_gradWeights(subsample=subsample)(
                img, topgrad)
            conv_gemm = GpuCorr3dMM_gradWeights(subsample=subsample)(
                img, topgrad)
        else:
            conv_ref = Corr3dMM_gradWeights(subsample=subsample)(
                img, topgrad, shape=filters_shape[1:4])
            conv_gemm = GpuCorr3dMM_gradWeights(subsample=subsample)(
                img, topgrad, shape=filters_shape[1:4])

        f_ref = theano.function([], conv_ref, mode='FAST_RUN')
        f = theano.function([], conv_gemm, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

    def test_gradweight(self):
        self.run_gradweight(inputs_shape=(16, 10, 12, 16, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            dCdH_shape=(16, 5, 1, 13, 10),
                            subsample=(1, 1, 1))
        self.run_gradweight(inputs_shape=(16, 20, 10, 16, 1),
                            filters_shape=(10, 6, 4, 4, 1),
                            dCdH_shape=(16, 8, 4, 7, 10),
                            subsample=(2, 2, 2))
        self.run_gradweight(inputs_shape=(16, 20, 10, 16, 1),
                            filters_shape=(10, 6, 3, 4, 1),
                            dCdH_shape=(16, 5, 3, 5, 10),
                            subsample=(3, 3, 3))
        self.run_gradweight(inputs_shape=(16, 20, 12, 16, 1),
                            filters_shape=(10, 6, 12, 4, 1),
                            dCdH_shape=(16, 8, 1, 5, 10),
                            subsample=(2, 1, 3))

    def run_gradinput(self, inputs_shape, filters_shape,
                      subsample=(1, 1, 1)):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        bottom_height = (inputs_shape[1] - 1) * subsample[0] + filters_shape[1]
        bottom_width = (inputs_shape[2] - 1) * subsample[1] + filters_shape[2]
        bottom_depth = (inputs_shape[3] - 1) * subsample[2] + filters_shape[3]
        bottom_shape = theano.shared(numpy.array([bottom_height, bottom_width, bottom_depth]))

        weight = gpu_contiguous(filters.dimshuffle(0, 4, 1, 2, 3))
        top = gpu_contiguous(inputs.dimshuffle(0, 4, 1, 2, 3))
        if (subsample == (1, 1, 1)):
            conv_ref = Corr3dMM_gradInputs(subsample=subsample)(
                kern=weight, topgrad=top)
            conv_gemm = GpuCorr3dMM_gradInputs(subsample=subsample)(
                kern=weight, topgrad=top)
        else:
            conv_ref = Corr3dMM_gradInputs(subsample=subsample)(
                kern=weight, topgrad=top,
                shape=bottom_shape)
            conv_gemm = GpuCorr3dMM_gradInputs(subsample=subsample)(
                kern=weight, topgrad=top,
                shape=bottom_shape)

        f_ref = theano.function([], conv_ref, mode='FAST_RUN')
        f = theano.function([], conv_gemm, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

    def test_gradinput(self):
        self.run_gradinput(inputs_shape=(16, 15, 12, 12, 10),
                           filters_shape=(10, 6, 12, 4, 1))
        self.run_gradinput(inputs_shape=(16, 15, 12, 12, 10),
                           filters_shape=(10, 6, 12, 4, 1),
                           subsample=(2, 2, 2))
        self.run_gradinput(inputs_shape=(16, 15, 12, 12, 10),
                           filters_shape=(10, 6, 12, 4, 1),
                           subsample=(3, 3, 3))
        self.run_gradinput(inputs_shape=(16, 15, 12, 12, 10),
                           filters_shape=(10, 6, 12, 4, 1),
                           subsample=(3, 1, 2))

    def test_opt_conv3d_gemm(self):
        inputs_shape = (16, 20, 32, 16, 1)
        filters_shape = (10, 6, 12, 4, 1)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[0]).astype('float32'))

        conv = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                         b=bias, d=(1, 1, 1))
        mode = mode_with_gpu.including('conv3d_gemm')
        mode.check_py_code = False

        f_ref = theano.function([], conv, mode="FAST_RUN")
        f_gemm = theano.function([], conv, mode=mode)

        # make sure we inserted the gemm trickery
        topo = f_gemm.maker.fgraph.toposort()
        assert sum(isinstance(n.op, GpuCorr3dMM) for n in topo) > 0

        res_ref = f_ref()
        res_gemm = f_gemm()
        utt.assert_allclose(res_ref, res_gemm)

    def test_opt_convgrad3d_gemm(self):
        inputs_shape = (16, 10, 12, 16, 1)
        filters_shape = (10, 6, 12, 4, 1)
        dCdH_shape = (16, 5, 1, 13, 10)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        dCdH_val = numpy.random.random(dCdH_shape).astype('float32')

        inputs = shared(inputs_val)
        dCdH = shared(dCdH_val)

        conv = theano.tensor.nnet.convGrad3D(V=inputs, dCdH=dCdH,
                                             WShape=filters_shape,
                                             d=(1, 1, 1))
        mode = mode_with_gpu.including('convgrad3d_gemm')

        f_ref = theano.function([], conv)
        f_gemm = theano.function([], conv, mode=mode)

        # make sure we inserted the gemm trickery
        topo = f_gemm.maker.fgraph.toposort()
        assert sum(isinstance(n.op, GpuCorr3dMM_gradWeights) for n in topo) > 0

        res_ref = f_ref()
        res_gemm = f_gemm()
        utt.assert_allclose(res_ref, res_gemm)

    def test_opt_convtransp3d_gemm(self):
        inputs_shape = (16, 15, 12, 12, 10)
        filters_shape = (10, 6, 12, 4, 1)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')
        bias = shared(numpy.zeros(filters_shape[4]).astype('float32'))

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.convTransp3D(W=filters, b=bias, d=(1, 1, 1),
                                               H=inputs)
        mode = mode_with_gpu.including('convtransp3d_gemm')

        f_ref = theano.function([], conv)
        f_gemm = theano.function([], conv, mode=mode)

        # make sure we inserted the gemm trickery
        topo = f_gemm.maker.fgraph.toposort()
        assert sum(isinstance(n.op, GpuCorr3dMM_gradInputs) for n in topo) > 0

        res_ref = f_ref()
        res_gemm = f_gemm()
        utt.assert_allclose(res_ref, res_gemm)
