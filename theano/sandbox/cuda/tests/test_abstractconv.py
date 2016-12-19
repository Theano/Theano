from __future__ import absolute_import, print_function, division

import numpy
import theano
from theano.tensor.nnet.tests import test_abstract_conv
from theano.sandbox.cuda import float32_shared_constructor as gpu_shared

from theano.sandbox.cuda.dnn import (
    dnn_available,
    GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI,
    GpuDnnConv3d, GpuDnnConv3dGradW, GpuDnnConv3dGradI)
from theano.sandbox.cuda.blas import (
    GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs,
    GpuCorr3dMM, GpuCorr3dMM_gradWeights, GpuCorr3dMM_gradInputs)
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises

import theano.sandbox.cuda as cuda
if not cuda.cuda_available:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestDnnConv2d(test_abstract_conv.BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        test_abstract_conv.BaseTestConv2d.setup_class()
        # provide_shape is not used by the cuDNN impementation
        cls.provide_shape = [False]
        cls.shared = staticmethod(gpu_shared)

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        if fd != (1, 1):
            raise SkipTest("No dilation implementation for cuDNN ConvOp.")
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        mode = mode_with_gpu
        o = self.get_output_shape(i, f, s, b, fd)
        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, mode=mode,
                     provide_shape=provide_shape, border_mode=b,
                     filter_flip=flip, target_op=GpuDnnConv,
                     filter_dilation=fd)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s,
                            verify_grad=True, mode=mode,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip, target_op=GpuDnnConvGradW,
                            filter_dilation=fd)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s,
                           verify_grad=True, mode=mode,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip, target_op=GpuDnnConvGradI,
                           filter_dilation=fd)

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False):
        if fd != (1, 1):
            raise SkipTest("No dilation implementation for cuDNN ConvOp.")
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        mode = mode_with_gpu

        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=True, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip, target_op=GpuDnnConvGradI,
                               filter_dilation=fd)
        else:
            assert_raises((RuntimeError, ValueError),
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s,
                          verify_grad=False, mode=mode,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip, target_op=GpuDnnConvGradI,
                          ref=None,
                          filter_dilation=fd)


class TestDnnConv3d(test_abstract_conv.BaseTestConv3d):
    @classmethod
    def setup_class(cls):
        test_abstract_conv.BaseTestConv3d.setup_class()
        # provide_shape is not used by the cuDNN impementation
        cls.provide_shape = [False]
        cls.shared = staticmethod(gpu_shared)

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        if fd != (1, 1, 1):
            raise SkipTest("No dilation implementation for cuDNN ConvOp.")
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        mode = mode_with_gpu
        o = self.get_output_shape(i, f, s, b, fd)
        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, mode=mode,
                     provide_shape=provide_shape, border_mode=b,
                     filter_flip=flip, target_op=GpuDnnConv3d,
                     filter_dilation=fd)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s,
                            verify_grad=True, mode=mode,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip, target_op=GpuDnnConv3dGradW,
                            filter_dilation=fd)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s,
                           verify_grad=True, mode=mode,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip, target_op=GpuDnnConv3dGradI,
                           filter_dilation=fd)

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False):
        if fd != (1, 1, 1):
            raise SkipTest("No dilation implementation for cuDNN ConvOp.")
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        mode = mode_with_gpu

        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=True, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip, target_op=GpuDnnConvGradI,
                               filter_dilation=fd)
        else:
            assert_raises((RuntimeError, ValueError),
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s,
                          verify_grad=False, mode=mode,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip, target_op=GpuDnnConvGradI,
                          ref=None,
                          filter_dilation=fd)


class TestCorrMMConv2d(test_abstract_conv.BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        test_abstract_conv.BaseTestConv2d.setup_class()
        cls.shared = staticmethod(gpu_shared)
        cls.mode = mode_with_gpu.excluding('cudnn')

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        mode = self.mode
        o = self.get_output_shape(i, f, s, b, fd)
        self.run_fwd(inputs_shape=i, filters_shape=f,
                     subsample=s, verify_grad=True, mode=mode,
                     provide_shape=provide_shape, border_mode=b,
                     filter_flip=flip, target_op=(GpuCorrMM,
                                                  GpuCorrMM_gradWeights,
                                                  GpuCorrMM_gradInputs),
                     filter_dilation=fd)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s,
                            verify_grad=True, mode=mode,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip,
                            target_op=GpuCorrMM_gradWeights,
                            filter_dilation=fd)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s,
                           verify_grad=True, mode=mode,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip,
                           target_op=GpuCorrMM_gradInputs,
                           filter_dilation=fd)

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False):
        mode = self.mode
        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=True, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=GpuCorrMM_gradInputs,
                               filter_dilation=fd)
        else:
            assert_raises(ValueError,
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s,
                          verify_grad=False, mode=mode,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip,
                          target_op=GpuCorrMM_gradInputs,
                          ref=None,
                          filter_dilation=fd)


class TestCorrMMConv3d(test_abstract_conv.BaseTestConv3d):
    @classmethod
    def setup_class(cls):
        test_abstract_conv.BaseTestConv3d.setup_class()
        cls.shared = staticmethod(gpu_shared)
        cls.mode = mode_with_gpu.excluding('cudnn')

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        mode = self.mode
        o = self.get_output_shape(i, f, s, b, fd)
        self.run_fwd(inputs_shape=i, filters_shape=f,
                     subsample=s, verify_grad=True, mode=mode,
                     provide_shape=provide_shape, border_mode=b,
                     filter_flip=flip, target_op=(GpuCorr3dMM,
                                                  GpuCorr3dMM_gradWeights,
                                                  GpuCorr3dMM_gradInputs),
                     filter_dilation=fd)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s,
                            verify_grad=True, mode=mode,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip,
                            target_op=GpuCorr3dMM_gradWeights,
                            filter_dilation=fd)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s,
                           verify_grad=True, mode=mode,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip,
                           target_op=GpuCorr3dMM_gradInputs,
                           filter_dilation=fd)

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False):
        mode = self.mode
        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=True, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=GpuCorr3dMM_gradInputs,
                               filter_dilation=fd)
        else:
            assert_raises(ValueError,
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s,
                          verify_grad=False, mode=mode,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip,
                          target_op=GpuCorr3dMM_gradInputs,
                          ref=None,
                          filter_dilation=fd)


class TestDnnConvTypes(test_abstract_conv.TestConvTypes):
    def setUp(self):
        self.input = cuda.ftensor4()
        self.filters = cuda.ftensor4()
        self.topgrad = cuda.ftensor4()
        self.constant_tensor = cuda.CudaNdarray(
            numpy.zeros((3, 5, 7, 11), dtype='float32'))
