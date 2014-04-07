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
        return "CudaNdarrayConstant{"+str(numpy.asarray(self.data))+"}"
CudaNdarrayType.Constant = CudaNdarrayConstant


