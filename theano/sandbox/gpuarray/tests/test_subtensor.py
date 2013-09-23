from theano.tensor.tests.test_subtensor import T_subtensor

from theano.sandbox.gpuarray.basic_ops import (HostFromGpu, GpuFromHost)
from theano.sandbox.gpuarray.subtensor import GpuSubtensor

from theano.sandbox.gpuarray.type import gpuarray_shared_constructor

from theano.sandbox.gpuarray.tests.test_basic_ops import mode_with_gpu

from theano import tensor

class G_subtensor(T_subtensor):
    def shortDescription(self):
        return None

    shared = staticmethod(gpuarray_shared_constructor)
    sub = GpuSubtensor
    mode = mode_with_gpu
    dtype = 'float32' # avoid errors on gpus which do not support float64
    ignore_topo = (HostFromGpu, GpuFromHost)
    fast_compile = False
    ops = (GpuSubtensor,
           tensor.IncSubtensor, tensor.AdvancedSubtensor1,
           tensor.AdvancedIncSubtensor1)

    #inc_sub = cuda.GpuIncSubtensor
    #adv_sub1 = cuda.GpuAdvancedSubtensor1
    #adv_incsub1 = cuda.GpuAdvancedIncSubtensor1
    inc_sub = tensor.IncSubtensor
    adv_sub1 = tensor.AdvancedSubtensor1
    adv_incsub1 = tensor.AdvancedIncSubtensor1

    def __init__(self, name):
        # We must call T_subtensor parten __init__, otherwise, sub is overrieded
        ret = super(T_subtensor, self).__init__(name)
        assert self.sub == GpuSubtensor
        return ret
