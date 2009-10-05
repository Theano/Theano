import numpy

from theano import Op, Type, Apply, Variable, Constant
from theano import tensor
from theano.compile.sandbox.sharedvalue import shared, SharedVariable, shared_constructor

from .type import CudaNdarrayType
from .type_support import filter as type_support_filter

from .basic_ops import HostFromGpu, GpuFromHost

class _operators(tensor.basic._tensor_py_operators):
    """Define a few properties and conversion methods for CudaNdarray Variables.

    The default implementation of arithemetic operators is to build graphs of TensorType
    variables. 

    The optimization pass (specialization) will insert pure GPU implementations.
    This approach relieves the Cuda-Ops of having to deal with input argument checking and
    gradients.
    """
    def _as_TensorVariable(self):
        return HostFromGpu()(self)
    def _as_CudaNdarrayVariable(self):
        return self

    dtype = property(lambda s:'float32')
    broadcastable = property(lambda s:s.type.broadcastable)
    ndim = property(lambda s:s.type.ndim)


class CudaNdarrayVariable(Variable, _operators):
    pass
CudaNdarrayType.Variable = CudaNdarrayVariable

class CudaNdarrayConstantSignature(tensor.TensorConstantSignature):
    pass

class CudaNdarrayConstant(Constant, _operators):
    def signature(self):
        return CudaNdarrayConstantSignature((self.type, numpy.asarray(self.data)))
CudaNdarrayType.Constant = CudaNdarrayConstant

class CudaNdarraySharedVariable(SharedVariable, _operators):

    def __getvalue(self):
        return numpy.asarray(self.container.value)
    def __setvalue(self, value):
        self.container.value = value #container does the filtering 
    value = property(__getvalue, __setvalue)

    def filter_update(self, other):
        if hasattr(other, '_as_CudaNdarrayVariable'):
            return other._as_CudaNdarrayVariable()

        if not isinstance(other.type, tensor.TensorType):
            raise TypeError('Incompatible type', other.type)
        if (other.type.dtype != self.dtype):
            raise TypeError('Incompatible dtype', (self.dtype, other.type.dtype))
        if (other.type.broadcastable != self.broadcastable):
            raise TypeError('Incompatible broadcastable', (self, (self.broadcastable,
                other.type.broadcastable)))
        return GpuFromHost()(other)

CudaNdarrayType.SharedVariable = CudaNdarraySharedVariable

def shared_constructor(value, name, strict=False, broadcastable=None):
    """SharedVariable Constructor for TensorType"""

    #TODO: what should strict mean in this context, since we always have to make a copy?
    if strict:
        _value = value
    else:
        _value = numpy.asarray(value, dtype='float32')

    if not isinstance(_value, numpy.ndarray):
        raise TypeError('ndarray required')
    if _value.dtype.num != CudaNdarrayType.typenum:
        raise TypeError('float32 ndarray required')

    if broadcastable is None:
        broadcastable = [b==1 for b in value.shape]
    type = CudaNdarrayType(broadcastable=broadcastable)
    return CudaNdarraySharedVariable(type=type, value=_value, name=name, strict=strict)




def unset_shared_for_numpy():
    raise NotImplementedError()

def set_shared_for_numpy():
    """
    Set the gpu_tensor_constructor as the handler for ndarray
    """
    shared_constructor(shared_constructor)

