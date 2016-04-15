from __future__ import absolute_import, print_function, division
import theano
import numpy
from theano import Op, Apply, config
from theano.tensor.extra_ops import CumsumOp

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from .opt import register_opt as register_gpu_opt, op_lifter
from .type import GpuArrayType


class GpuCumsum(CumsumOp, Op):
    """
    Parameters
    ----------
    axis
        Can not be None. If you want the array flatten, do it before.
    """
    SUPPORTED_NDIMS = 3

    def __init__(self, axis):
        self.axis = axis
        # not sure if this should be here
        self.max_threads_dim0 = None
        self.max_grid_size1 = None
        self.max_gride_size2 = None


    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, self.axis)


    def c_code_cache_version(self):
        return (1,)


    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']


    def make_node(self, x): 
        if x.ndim > GpuCumsum.SUPPORTED_NDIMS:
            raise NotImplementedError('Only cumsum on 1D, 2D and\
                                       3D arrays are supported right now!')
        if self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError('axis(={1}) out of bounds'.format(self.axis))

        x_ = as_gpuarray_variable(x, infer_context_name(x_))
        return Apply(self, [x_], [x_.type()])


    # ask Arnaud about this
    def make_thunk(self, node, storage_map, comput_map, no_recycling):
        pass


    # copied from neighbour.py
    def perform(self, node, inp, out, ctx):
        # Disable the perform method from the CPU version
        Op.perform(self, node, inp, out, ctx)


    def gpu_kernels(self, node, nodename):
        kernels = []
        # cumadd
        k_name = "k_cumadd"   
        k_var = "k_cumadd_" + nodename
        params =  
        dtype_x = node.inputs[0].dtype
        flags = Kernel.get_flags(dtype_x)
        code = """
        KERNEL void %(kname)s(float* input, float* output, size_t inputStrides[3],
                              size_t[3] outputStrides, int offsetY, int offsetZ, 
                              int beforeLastElementIdx, int lastElementIdx){
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            int dataOffsetY_input = idY * inputStrides[1] + idZ * inputStrides.[3];
            int dataOffsetY_output = idY * outputStrides[1] + idZ * outputStrides[2];
            int idx_last_input = lastElementIdx*inputStrides[0] + dataOffsetY_input;
            int idx_last_output = lastElementIdx*outputStrides[0] + dataOffsetY_output;
            int idx_beforelast = beforeLastElementIdx*outputStrides[0] + dataOffsetY_output;
            output[idx_last_output] = input[idx_last_input] + output[idx_beforelast];
            }
        """ % locals()
        kernels.append(Kernel(code=code, name="k_cumadd", params=params,
                              flags=flags, objvar=k_var))
        # finalCumSum
        k_name = "k_finalCumSum" 
        k_var = "k_finalCumSum_" + nodename
        params = 
        code = """
        """       
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels

    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.context.kind != 'cuda':
            raise NotImplementedError("cuda only")
