import itertools

from nose.plugins.skip import SkipTest

from theano.tensor.nnet.tests import test_abstract_conv
from ..type import GpuArrayType
from ..dnn import dnn_available, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI

from .config import mode_with_gpu, test_ctx_name

gpu_ftensor4 = GpuArrayType(dtype='float32', broadcastable=(False,) * 4)


class TestDnnConv2d(test_abstract_conv.TestConv2d):
    def test_dnn_conv(self):
        if not dnn_available(test_ctx_name):
            raise SkipTest(dnn_available.msg)
        mode = mode_with_gpu
        # provide_shape is not used by the CuDNN impementation
        provide_shape = False

        for (i, f), s, b, flip in itertools.product(
                zip(self.inputs_shapes, self.filters_shapes),
                self.subsamples,
                self.border_modes,
                self.filter_flip):
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
