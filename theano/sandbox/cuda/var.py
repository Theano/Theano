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
        return "CudaNdarrayConstant{"+str(numpy.asarray(self.data))+"}"
CudaNdarrayType.Constant = CudaNdarrayConstant

class CudaNdarraySharedVariable(SharedVariable, _operators):
    """
    Shared Variable interface to CUDA-allocated arrays
    """

    get_value_return_ndarray = True

    def get_value(self, borrow=False, return_internal_type=False):
        """
        Return the value of this SharedVariable's internal array.

        :param borrow:
                permit the return of internal storage, when used in conjunction with
                ``return_internal_type=True``
        :param return_internal_type:
                True to return the internal ``cuda_ndarray`` instance rather than a ``numpy.ndarray``
                (Default False)

        By default ``get_value()`` copies from the GPU to a ``numpy.ndarray`` and returns that
        host-allocated array.

        ``get_value(False,True)`` will return a GPU-allocated copy of the original GPU array.

        ``get_value(True,True)`` will return the original GPU-allocated array without any
        copying.

        """
        if return_internal_type or not self.get_value_return_ndarray:
            # return a cuda_ndarray
            if borrow:
                return self.container.value
            else:
                return copy.deepcopy(self.container.value)
        else: #return an ndarray
            return numpy.asarray(self.container.value)

    def set_value(self, value, borrow=False):
        """
        Assign `value` to the GPU-allocated array.

        :param borrow: ``True`` permits reusing `value` itself, ``False`` requires that this function
                       copies `value` into internal storage.

        :note:

            Prior to Theano 0.3.1, set_value did not work in-place on the GPU. This meant that sometimes,
            GPU memory for the new value would be allocated before the old memory was released. If you're
            running near the limits of GPU memory, this could cause you to run out of GPU memory.

            Beginning with Theano 0.3.1, set_value will work in-place on the GPU, if the following conditions
            are met:

            * The destination on the GPU must be c_contiguous.
            * The source is on the CPU.
            * The old value must have the same dtype as the new value (which is a given for now,
              since only float32 is supported).
            * The old and new value must have the same shape.
            * The old value is being completely replaced by the new value (not partially modified,
              e.g. by replacing some subtensor of it).
            * You change the value of the shared variable via set_value, not via the .value
              accessors. You should not use the .value accessors anyway, since they will soon be
              deprecated and removed.

            It is also worth mentioning that, for efficient transfer to the GPU, Theano will make the new data
            ``c_contiguous``. This can require an extra copy of the data on the host.

            The inplace on gpu memory work when borrow is either True or False.
        """
        if not borrow:
            #TODO: check for cuda_ndarray type
            if not isinstance(value, numpy.ndarray):
                # in case this is a cuda_ndarray, we copy it
                value = copy.deepcopy(value)
        self.container.value = value # this will copy a numpy ndarray

    def __getitem__(self, *args):
        # Defined to explicitly use the implementation from `_operators`, since
        # the definition in `SharedVariable` is only meant to raise an error.
        return _operators.__getitem__(self, *args)


CudaNdarrayType.SharedVariable = CudaNdarraySharedVariable

def cuda_shared_constructor(value, name=None, strict=False,
        allow_downcast=None, borrow=False, broadcastable=None):
    """SharedVariable Constructor for CudaNdarrayType"""

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
    """SharedVariable Constructor for CudaNdarrayType from numpy.ndarray or CudaNdarray"""

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
    except Exception, e:
        print "ERROR", e
        raise

    rval.get_value_return_ndarray = get_value_return_ndarray

    return rval
