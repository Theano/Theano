import copy

import numpy

import theano
from theano import Variable, Constant
from theano import tensor
from theano.compile import SharedVariable

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import filter as type_support_filter

from theano.sandbox.cuda.basic_ops import HostFromGpu, GpuFromHost

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
    def __str__(self):
        if self.name is not None:
            return self.name
        return "CudaNdarray{"+str(numpy.asarray(self.data))+"}"
CudaNdarrayType.Constant = CudaNdarrayConstant

class CudaNdarraySharedVariable(SharedVariable, _operators):

    def get_value(self, borrow=False, return_internal_type=False):
        if return_internal_type: # return a cuda_ndarray
            if borrow:
                return self.container.value
            else:
                return copy.deepcopy(self.container.value)
        else: #return an ndarray
            return numpy.asarray(self.container.value)

    def set_value(self, value, borrow=False):
        if not borrow:
            #TODO: check for cuda_ndarray type
            if not isinstance(value, numpy.ndarray):
                # in case this is a cuda_ndarray, we copy it
                value = copy.deepcopy(value)
        self.container.value = value # this will copy a numpy ndarray

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

def cuda_shared_constructor(value, name=None, strict=False,
        allow_downcast=None, borrow=False, broadcastable=None):
    """SharedVariable Constructor for TensorType"""

    # THIS CONSTRUCTOR TRIES TO CAST VALUE TO A FLOAT32, WHICH THEN GOES ONTO THE CARD
    # SO INT shared vars, float64 shared vars, etc. all end up on the card.
    # THIS IS NOT THE DEFAULT BEHAVIOUR THAT WE WANT. 
    # SEE float32_shared_constructor

    #TODO: what should strict mean in this context, since we always have to make a copy?
    if strict:
        _value = value
    else:
        _value = theano._asarray(value, dtype='float32')

    if not isinstance(_value, numpy.ndarray):
        raise TypeError('ndarray required')
    if _value.dtype.num != CudaNdarrayType.typenum:
        raise TypeError('float32 ndarray required')

    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = CudaNdarrayType(broadcastable=broadcastable)
    print "trying to return?"
    try:
        rval = CudaNdarraySharedVariable(type=type, value=_value, name=name, strict=strict)
    except Exception, e:
        print "ERROR", e
        raise
    return rval

def float32_shared_constructor(value, name=None, strict=False,
        allow_downcast=None, borrow=False, broadcastable=None):
    """SharedVariable Constructor for TensorType"""

    # if value isn't a float32 ndarray, then raise
    if not isinstance(value, numpy.ndarray):
        raise TypeError('ndarray required')
    if value.dtype.num != CudaNdarrayType.typenum:
        raise TypeError('float32 ndarray required')

    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = CudaNdarrayType(broadcastable=broadcastable)
    deviceval = type_support_filter(value, broadcastable, False, None)
    try:
        rval = CudaNdarraySharedVariable(type=type, value=deviceval, name=name, strict=strict)
    except Exception, e:
        print "ERROR", e
        raise
    return rval

