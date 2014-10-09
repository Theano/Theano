import copy

import numpy

import theano
from theano import Variable, Constant
from theano import tensor
from theano.compile import SharedVariable

from theano.sandbox.cuda.type import CudaNdarrayType
try:
    # We must do those import to be able to create the full doc when nvcc
    # is not available
    from theano.sandbox.cuda import filter as type_support_filter
    from theano.sandbox.cuda.basic_ops import HostFromGpu, GpuFromHost
except ImportError:
    pass

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


class CudaNdarrayVariable(_operators, Variable):
    pass
CudaNdarrayType.Variable = CudaNdarrayVariable

class CudaNdarrayConstantSignature(tensor.TensorConstantSignature):
    pass

class CudaNdarrayConstant(_operators, Constant):
    def signature(self):
        return CudaNdarrayConstantSignature((self.type, numpy.asarray(self.data)))
    def __str__(self):
        if self.name is not None:
            return self.name
        try:
            data = str(numpy.asarray(self.data))
        except Exception, e:
            data = "error while transferring the value: " + str(e)
        return "CudaNdarrayConstant{"+data+"}"
CudaNdarrayType.Constant = CudaNdarrayConstant

CudaNdarrayType.SharedVariable = tensor.sharedvar.TensorSharedVariable

# This is a legacy constructor from before the time 
# CudaNdarraySharedVariable and TensorSharedVariable weren't merged.
# It is used by unit tests for CUDA. It hard codes the visible type
# of the TensorSharedVariable to be an instance of CudaNdarrayType.
def float32_shared_constructor(value, name=None, strict=False,
        allow_downcast=None, borrow=False, broadcastable=None):
    """SharedVariable Constructor for CudaNdarrayType"""
    if not isinstance(value, (numpy.ndarray, theano.sandbox.cuda.CudaNdarray)):
        raise TypeError('ndarray or CudaNdarray required')

    # if no broadcastable is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
        
    deviceval = None
    get_value_return_ndarray = True
    if isinstance(value, theano.sandbox.cuda.CudaNdarray):
        get_value_return_ndarray = False
        if borrow:
            deviceval = value
        else:
            deviceval = value.copy()
    else:
        deviceval = numpy.array(value, copy=(not borrow))
    
    # Force casting of numpy.array to cudandarray
    type = CudaNdarrayType(broadcastable=broadcastable)
    rval = None
    try:
        rval = tensor.sharedvar.TensorSharedVariable(
            type=type, value=deviceval, name=name, 
            strict=strict, allow_downcast=allow_downcast
        )
    except Exception, e:
        print "ERROR", e
        raise

    rval.get_value_return_ndarray = get_value_return_ndarray
    
    return rval
