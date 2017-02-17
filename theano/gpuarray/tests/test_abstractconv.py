from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
from nose.tools import assert_raises

import numpy as np

from theano.tensor.nnet.tests import test_abstract_conv
from ..type import GpuArrayType, gpuarray_shared_constructor, get_context
from ..dnn import dnn_available, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI
from ..blas import (
    GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs,
    GpuCorr3dMM, GpuCorr3dMM_gradWeights, GpuCorr3dMM_gradInputs)

from .config import mode_with_gpu, test_ctx_name
from pygpu import gpuarray

gpu_ftensor4 = GpuArrayType(dtype='float32', broadcastable=(False,) * 4)


class TestDnnConv2d(test_abstract_conv.BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        test_abstract_conv.BaseTestConv2d.setup_class()
        cls.shared = staticmethod(gpuarray_shared_constructor)
        # provide_shape is not used by the cuDNN impementation
        cls.provide_shape = [False]

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        if not dnn_available(test_ctx_name):
            raise SkipTest(dnn_available.msg)
        mode = mode_with_gpu

        if fd != (1, 1):
            raise SkipTest("Doesn't have CUDNN implementation")
        o = self.get_output_shape(i, f, s, b, fd)

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

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False):
        if not dnn_available(test_ctx_name):
            raise SkipTest(dnn_available.msg)
        if fd != (1, 1):
            raise SkipTest("Doesn't have CUDNN implementation")
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
        cls.shared = staticmethod(gpuarray_shared_constructor)
        # provide_shape is not used by the cuDNN impementation
        cls.provide_shape = [False]

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        if not dnn_available(test_ctx_name):
            raise SkipTest(dnn_available.msg)
        mode = mode_with_gpu

        if fd != (1, 1, 1):
            raise SkipTest("Doesn't have CUDNN implementation")
        o = self.get_output_shape(i, f, s, b, fd)

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

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False):
        if not dnn_available(test_ctx_name):
            raise SkipTest(dnn_available.msg)
        if fd != (1, 1, 1):
            raise SkipTest("Doesn't have CUDNN implementation")
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
        cls.shared = staticmethod(gpuarray_shared_constructor)
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
        cls.shared = staticmethod(gpuarray_shared_constructor)
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
        self.input = gpu_ftensor4()
        self.filters = gpu_ftensor4()
        self.topgrad = gpu_ftensor4()
        self.constant_tensor = gpuarray.array(
            np.zeros((3, 5, 7, 11), dtype='float32'),
            context=get_context(test_ctx_name))


class TestConv2dTranspose(test_abstract_conv.TestConv2dTranspose):
    mode = mode_with_gpu
