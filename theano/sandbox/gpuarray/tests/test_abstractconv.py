from nose.plugins.skip import SkipTest

from theano.tensor.nnet.tests import test_abstract_conv
from ..type import GpuArrayType, gpuarray_shared_constructor
from ..dnn import dnn_available, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI

from .config import mode_with_gpu, test_ctx_name

gpu_ftensor4 = GpuArrayType(dtype='float32', broadcastable=(False,) * 4)


class TestDnnConv2d(test_abstract_conv.BaseTestConv2d):
    def setUp(self):
        super(TestDnnConv2d, self).setUp()
        self.shared = gpuarray_shared_constructor
        # provide_shape is not used by the CuDNN impementation
        self.provide_shape = [False]

    def tcase(self, i, f, s, b, flip, provide_shape):
        if not dnn_available(test_ctx_name):
            raise SkipTest(dnn_available.msg)
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


class TestDnnConvTypes(test_abstract_conv.TestConvTypes):
    def setUp(self):
        self.input = gpu_ftensor4()
        self.filters = gpu_ftensor4()
        self.topgrad = gpu_ftensor4()
