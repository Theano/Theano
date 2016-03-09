from __future__ import absolute_import, print_function, division

import numpy
import theano
from theano.tensor.nnet.tests import test_abstract_conv
from theano.sandbox.cuda import float32_shared_constructor as gpu_shared

from theano.sandbox.cuda.dnn import (
    dnn_available,
    GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI)
from theano.sandbox.cuda.blas import (
    GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs)
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda
if not cuda.cuda_available:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestDnnConv2d(test_abstract_conv.BaseTestConv2d):
    def setUp(self):
        super(TestDnnConv2d, self).setUp()
        # provide_shape is not used by the CuDNN impementation
        self.provide_shape = [False]
        self.shared = gpu_shared

    def tcase(self, i, f, s, b, flip, provide_shape):
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        mode = mode_with_gpu
        o = self.get_output_shape(i, f, s, b)
        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, mode=mode,
                     provide_shape=provide_shape, border_mode=b,
                     filter_flip=flip, target_op=GpuDnnConv)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s,
                            verify_grad=True, mode=mode,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip, target_op=GpuDnnConvGradW)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s,
                           verify_grad=True, mode=mode,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip, target_op=GpuDnnConvGradI)


class TestCorrMMConv2d(test_abstract_conv.BaseTestConv2d):
    def setUp(self):
        super(TestCorrMMConv2d, self).setUp()
        self.shared = gpu_shared
        self.mode = mode_with_gpu.excluding('cudnn')

    def tcase(self, i, f, s, b, flip, provide_shape):
        mode = self.mode
        o = self.get_output_shape(i, f, s, b)
        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, mode=mode,
                     provide_shape=provide_shape, border_mode=b,
                     filter_flip=flip,
                     target_op=(GpuCorrMM,
                                GpuCorrMM_gradWeights,
                                GpuCorrMM_gradInputs))
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s,
                            verify_grad=True, mode=mode,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip,
                            target_op=GpuCorrMM_gradWeights)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s,
                           verify_grad=True, mode=mode,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip,
                           target_op=GpuCorrMM_gradInputs)


class TestDnnConvTypes(test_abstract_conv.TestConvTypes):
    def setUp(self):
        self.input = cuda.ftensor4()
        self.filters = cuda.ftensor4()
        self.topgrad = cuda.ftensor4()
        self.constant_tensor = cuda.CudaNdarray(
            numpy.zeros((3, 5, 7, 11), dtype='float32'))
