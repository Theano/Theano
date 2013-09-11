from theano.tensor.tests.test_subtensor import T_subtensor

from theano.sandbox.gpuarray.basic_ops import (HostFromGpu, GpuFromHost)
from theano.sandbox.gpuarray.subtensor import GpuSubtensor

from theano.sandbox.gpuarray.type import gpuarray_shared_constructor

from theano.sandbox.gpuarray.tests.test_basic_ops import mode_with_gpu

class G_subtensor(T_subtensor):
    def shortDescription(self):
        return None

    shared = staticmethod(gpuarray_shared_constructor)
    sub = GpuSubtensor
    mode = mode_with_gpu
    dtype = 'float32' # avoid errors on gpus which do not support float64
    ignore_topo = (HostFromGpu, GpuFromHost)
    fast_compile = False
    ops = (GpuSubtensor,)

    def __init__(self, name):
        T_subtensor.__init__(self, name)
