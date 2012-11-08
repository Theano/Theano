import numpy

import theano
from theano import Variable, Constant, tensor
from theano.compile import SharedVariable

try:
    # Let this be importable for documentation purposes
    import pygpu.gpuarray
    from basic_ops import host_from_gpu, gpu_from_host
except ImportError:
    pass

from type import GpuArrayType

class _operators(tensor.basic._tensor_py_operators):

    def _as_TensorVariable(self):
        return host_from_gpu(self)
    # XXX: don't forget to add _as_CudaNdarrayVariable() when we
    #      figure out how to do it.
    def _as_GpuArrayVariable(self):
        return self

    dtype = property(lambda s: s.type.dtype)
    broadcastable = property(lambda s: s.type.broadcastable)
    ndim = property(lambda s: s.type.ndim)


class GpuArrayVariable(_operators, Variable):
    pass


GpuArrayType.Variable = GpuArrayVariable


class GpuArrayConstant(_operators, Constant):
    def signature(self):
        return GpuArraySignature((self.type, numpy.asarray(self.data)))

    def __str__(self):
        if self.name is not None:
            return self.name
        return "GpuArrayConstant{%s}" % numpy.asarray(self.data)


GpuArrayType.Constant = GpuArrayConstant


class GpuArraySharedVariable(_operators, SharedVariable):
    def get_value(self, borrow=False, return_internal_type=False):
        if return_internal_type:
            if borrow:
                return self.container.value
            else:
                return self.container.value.copy()
        else:
            return numpy.asarray(self.container.value)

    def set_value(self, value, borrow=False):
        self.container.value = pygpu.gpuarray.array(value, copy=(not borrow))

    def __getitem__(self, *args):
        return _operators.__getitem__(self, *args)


GpuArrayType.SharedVariable = GpuArraySharedVariable


def gpuarray_shared_constructor(value, name=None, strict=False,
                                allow_downcast=None, borrow=False,
                                broadcastable=None):
    """SharedVariable constructor for GpuArrayType"""
    if globals.kind is None:
        raise RuntimeError("pygpu is not initialized")

    if not isinstance(value, (numpy.ndarray, pygpu.gpuarray.GpuArray)):
        raise TypeError('ndarray or GpuArray required')
    
    if broadcastable is None:
        broadcastable = (False,) * value.ndim
    type = GpuArrayType(value.dtype, broadcastable)
    deviceval = pygpu.gpuarray.array(value, copy=(not borrow))
    return GpuArraySharedVariable(type=type, value=deviceval, name=name,
                                  strict=strict)
