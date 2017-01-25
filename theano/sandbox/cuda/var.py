from __future__ import absolute_import, print_function, division
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
    from theano.sandbox.cuda.basic_ops import HostFromGpu
except ImportError:
    pass


class _operators(tensor.basic._tensor_py_operators):
    """
    Define a few properties and conversion methods for CudaNdarray Variables.

    The default implementation of arithemetic operators is to build graphs of
    TensorType variables.

    The optimization pass (specialization) will insert pure GPU implementations.
    This approach relieves the Cuda-Ops of having to deal with input argument
    checking and gradients.

    """

    def _as_TensorVariable(self):
        return HostFromGpu()(self)

    def _as_CudaNdarrayVariable(self):
        return self

    dtype = property(lambda s: 'float32')
    broadcastable = property(lambda s: s.type.broadcastable)
    ndim = property(lambda s: s.type.ndim)


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
        except Exception as e:
            data = "error while transferring the value: " + str(e)
        return "CudaNdarrayConstant{" + data + "}"
CudaNdarrayType.Constant = CudaNdarrayConstant


class CudaNdarraySharedVariable(_operators, SharedVariable):
    """
    Shared Variable interface to CUDA-allocated arrays.

    """

    get_value_return_ndarray = True

    def get_value(self, borrow=False, return_internal_type=False):
        """
        Return the value of this SharedVariable's internal array.

        """
        if return_internal_type or not self.get_value_return_ndarray:
            # return a cuda_ndarray
            if borrow:
                return self.container.value
            else:
                return copy.deepcopy(self.container.value)
        else:  # return an ndarray
            return numpy.asarray(self.container.value)

    def set_value(self, value, borrow=False):
        """
        Assign `value` to the GPU-allocated array.

        """
        if not borrow:
            # TODO: check for cuda_ndarray type
            if not isinstance(value, numpy.ndarray):
                # in case this is a cuda_ndarray, we copy it
                value = copy.deepcopy(value)
        self.container.value = value  # this will copy a numpy ndarray

    def __getitem__(self, *args):
        # Defined to explicitly use the implementation from `_operators`, since
        # the definition in `SharedVariable` is only meant to raise an error.
        return _operators.__getitem__(self, *args)


CudaNdarrayType.SharedVariable = CudaNdarraySharedVariable


def cuda_shared_constructor(value, name=None, strict=False,
                            allow_downcast=None, borrow=False,
                            broadcastable=None, target='gpu'):
    """
    SharedVariable Constructor for CudaNdarrayType.

    """
    if target != 'gpu':
        raise TypeError('not for gpu')

    # THIS CONSTRUCTOR TRIES TO CAST VALUE TO A FLOAT32, WHICH THEN GOES ONTO THE CARD
    # SO INT shared vars, float64 shared vars, etc. all end up on the card.
    # THIS IS NOT THE DEFAULT BEHAVIOUR THAT WE WANT.
    # SEE float32_shared_constructor

    # TODO: what should strict mean in this context, since we always have to make a copy?
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
    print("trying to return?")
    try:
        rval = CudaNdarraySharedVariable(type=type, value=_value, name=name, strict=strict)
    except Exception as e:
        print("ERROR", e)
        raise
    return rval


def float32_shared_constructor(value, name=None, strict=False,
                               allow_downcast=None, borrow=False,
                               broadcastable=None, target='gpu'):
    """
    SharedVariable Constructor for CudaNdarrayType from numpy.ndarray or
    CudaNdarray.

    """
    if target != 'gpu':
        raise TypeError('not for gpu')
    if theano.sandbox.cuda.use.device_number is None:
        theano.sandbox.cuda.use("gpu",
                                force=True,
                                default_to_move_computation_to_gpu=False,
                                move_shared_float32_to_gpu=False,
                                enable_cuda=False)

    # if value isn't a float32 ndarray, or a CudaNdarray then raise

    if not isinstance(value, (numpy.ndarray, theano.sandbox.cuda.CudaNdarray)):
        raise TypeError('ndarray or CudaNdarray required')
    if isinstance(value, numpy.ndarray) and value.dtype.num != CudaNdarrayType.typenum:
        raise TypeError('float32 ndarray required')

    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = CudaNdarrayType(broadcastable=broadcastable)
    get_value_return_ndarray = True
    if isinstance(value, theano.sandbox.cuda.CudaNdarray):
        get_value_return_ndarray = False
        if borrow:
            deviceval = value
        else:
            deviceval = value.copy()
    else:
        # type.broadcastable is guaranteed to be a tuple, which this next
        # function requires
        deviceval = type_support_filter(value, type.broadcastable, False, None)

    try:
        rval = CudaNdarraySharedVariable(type=type, value=deviceval, name=name, strict=strict)
    except Exception as e:
        print("ERROR", e)
        raise

    rval.get_value_return_ndarray = get_value_return_ndarray

    return rval
