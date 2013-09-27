from theano.tensor.tests.test_subtensor import T_subtensor

from theano.sandbox.gpuarray.basic_ops import (HostFromGpu, GpuFromHost)
from theano.sandbox.gpuarray.subtensor import GpuSubtensor

from theano.sandbox.gpuarray.type import gpuarray_shared_constructor

from theano.sandbox.gpuarray.tests.test_basic_ops import mode_with_gpu

from theano.compile import DeepCopyOp

from theano import tensor

class G_subtensor(T_subtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        T_subtensor.__init__(self, name,
                             shared=gpuarray_shared_constructor,
                             sub=GpuSubtensor,
                             mode=mode_with_gpu,
                             # avoid errors with limited devices
                             dtype='float32',
                             ignore_topo=(HostFromGpu,GpuFromHost,DeepCopyOp))
        assert self.sub == GpuSubtensor
