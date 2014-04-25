import traceback
import numpy

"""
TODO:
toGPU, etc.
dtype, etc
get/set
constructors
unit tests
"""

import theano.tensor.basic
from theano.tensor.basic import TensorType, _tensor_py_operators
from theano.compile import shared_constructor, SharedVariable

cuda = None

def init_cuda():
    global cuda
    if cuda is None:
        try:
            import theano.sandbox.cuda
            #from theano.sandbox.cuda.type import CudaNdarrayType
            cuda = theano.sandbox.cuda
        except ImportError:
            return False
    return True


def load_shared_variable(val):
    """This function is only here to keep some pickles loading
    after a failed fix done in August 2011.
    It can be removed after sufficient time has passed."""
    return tensor_constructor(val)


# _tensor_py_operators is first to have its version of __{gt,ge,lt,le}__
class TensorSharedVariable(_tensor_py_operators, SharedVariable):
    dtype = property(lambda s:s.container.value.type)
    broadcastable = property(lambda s:s.type.broadcastable)
    ndim = property(lambda s:s.type.ndim)
    get_value_return_ndarray = True
    
    def __init__(self, name, type, value, strict,
                 allow_downcast=None, container=None):
        """
        :param name: The name for this variable (see `Variable`).

        :param type: The type for this variable (see `Variable`).

        :param value: A value to associate with this variable (a new
        container will be created).

        :param strict: True -> assignments to .value will not be cast
        or copied, so they must have the correct type.

        :param allow_downcast: Only applies if `strict` is False.
        True -> allow assigned value to lose precision when cast during assignment.
        False -> never allow precision loss.
        None -> only allow downcasting of a Python float to a scalar floatX.

        :param container: The container to use for this
        variable. Illegal to pass this as well as a value.

        :note: For more user-friendly constructor, see `shared`

        """
        # true if cuda import works
        self._gpu_capable = init_cuda()
        
        if not isinstance(type, self.validTypes()):
            raise TypeError(
                "TensorSharedVariable only accepts type of instance "
                "TensorType or CudaNdarrayType."
            )
        
        if container is None:
            container = Container(
                    type,
                    storage=[type.filter(
                                value, strict=strict,
                                allow_downcast=allow_downcast
                            )],
                    readonly=False,
                    strict=strict,
                    allow_downcast=allow_downcast
            )
        else:
            if not isinstance(container.type, self.validTypes()):
                raise TypeError(
                    "TensorSharedVariable only accepts instance Type "
                    "TensorType or CudaNdarrayType."
                )
                    
        assert container is not None 
        
        super(TensorSharedVariable, self).__init__(
            type=container.type, name=name, owner=None, index=None, 
            container=container
        )
            
        
        # if the user provides a different type than that in container
        if (type != self.container.type):
            # convert container to new type
            if isinstance(type, cuda.type.CudaNdarrayType):
                self.toGPU()
            else:
                self.toCPU()
        else:
            self.type = self.container.type
        
    def validTypes(self):
        valid_types = TensorType
        if self._gpu_capable:
            valid_types = (TensorType, cuda.type.CudaNdarrayType)
        return valid_types
    
    def toGPU(self):
        assert self._gpu_capable, "No CUDA-capable device detected"
        if isinstance(self.container.type, TensorType):
            if self.container.value.dtype.num != cuda.type.CudaNdarrayType.typenum:
                raise TypeError('float32 required for GPU usage')
            self._setContainer(
                self.container.castClone(
                    cuda.type.CudaNdarrayType(
                        broadcastable=broadcastable
                    )
                )
            )
            return True
        return False
    
    def toCPU(self):
        if self._isCudaType():
            self._setContainer(
                self.container.castClone(
                    TensorType(self.type.dtype, self.type.broadcastable)
                )
            )
            return True
        return False
        
    def _setContainer(self, container):
        del self.container
        self.container = container
        self.type = self.container.type
    
    def _isCudaType(self, type=None):
        type = type or self.container.type
        return self._gpu_capable and isinstance(type, cuda.type.CudaNdarrayType)
        
    
    """Define a few properties and conversion methods for CudaNdarray Variables.

    The default implementation of arithemetic operators is to build graphs of TensorType
    variables.

    The optimization pass (specialization) will insert pure GPU implementations.
    This approach relieves the Cuda-Ops of having to deal with input argument checking and
    gradients.
    """
    def _as_TensorVariable(self):
        if self._isCudaType():
            return cuda.basic_ops.HostFromGpu()(self)
        assert(isinstance(self.type, TensorType))
        return super(TensorSharedVariable, self)._as_TensorVariable(self)
    
    def _as_CudaNdarrayVariable(self):
        assert(self._isCudaType())
        return self

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
        if isinstance(self.container.type, TensorType):
            if borrow:
                return self.container.value
            else:
                return copy.deepcopy(self.container.value)
        else:
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
        if borrow:
            self.container.value = value
        else:
            if self._isCudaType() and isinstance(value, numpy.ndarray):
                # the container will catch this and copy the ndarray 
                # into gpu memory (refer to container.__set__)
                self.container.value = value
            else:
                self.container.value = copy.deepcopy(value)
        

    def __getitem__(self, *args):
        # Defined to explicitly use the implementation from `_operators`, since
        # the definition in `SharedVariable` is only meant to raise an error.
        return _operators.__getitem__(self, *args)
    
    # For pickling
    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # if shared variable was on cuda when pickled, then
        # will sent values back to cuda in any further unpickle if
        # gpu device is detected
        d._was_cuda = d._was_cuda or False
        if self._isCudaType():
            d.type = TensorType(
                d.container.type.dtype, d.container.type.broadcastable
            )
            d.container = d.container.castClone(d.type)
            d._was_cuda = True
            
        return d

    # For unpickling
    def __setstate__(self, d):
        self.__dict__.update(d)
        self._gpu_capable = init_cuda()
        if self._was_cuda:
            self.toGPU()


@shared_constructor
def tensor_constructor(value, name=None, strict=False, allow_downcast=None,
                       borrow=False, broadcastable=None):
    """SharedVariable Constructor for TensorType

    :note: Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is
    ``(False,)*len(value.shape)``.  The optional `broadcastable`
    argument will override this default.

    """
    if not isinstance(value, (numpy.ndarray, theano.sandbox.cuda.CudaNdarray)):
        raise TypeError('ndarray or CudaNdarray required')

    # if no broadcastable is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
        
    type = None
    deviceval = None
    get_value_return_ndarray = True
    if init_cuda() and isinstance(value, cuda.CudaNdarray):
        type = cuda.type.CudaNdarrayType(broadcastable=broadcastable)
        get_value_return_ndarray = False
        if borrow:
            deviceval = value
        else:
            deviceval = value.copy()
    else:
        type = TensorType(value.dtype, broadcastable=broadcastable)
        deviceval = numpy.array(value, copy=(not borrow))
    
    rval = None
    try:
        rval = TensorSharedVariable(
            type=type, value=deviceval, name=name, 
            strict=strict, allow_downcast=allow_downcast
        )
    except Exception, e:
        print "ERROR", e
        raise

    rval.get_value_return_ndarray = get_value_return_ndarray
    
    return rval


# TensorSharedVariable brings in the tensor operators, is not ideal, but works
# as long as we dont do purely scalar-scalar operations
# _tensor_py_operators is first to have its version of __{gt,ge,lt,le}__
#
# N.B. THERE IS ANOTHER CLASS CALLED ScalarSharedVariable in the
# theano.scalar.sharedvar file.  It is not registered as a shared_constructor,
# this one is.
class ScalarSharedVariable(_tensor_py_operators, SharedVariable):
    pass


@shared_constructor
def scalar_constructor(value, name=None, strict=False, allow_downcast=None):
    """SharedVariable constructor for scalar values. Default: int64 or float64.

    :note: We implement this using 0-d tensors for now.

    """
    if not isinstance(value, (numpy.number, float, int, complex)):
        raise TypeError()
    try:
        dtype = value.dtype
    except Exception:
        dtype = numpy.asarray(value).dtype

    dtype = str(dtype)
    value = theano._asarray(value, dtype=dtype)
    tensor_type = TensorType(dtype=str(value.dtype), broadcastable=[])

    try:
        # Do not pass the dtype to asarray because we want this to fail if
        # strict is True and the types do not match.
        rval = ScalarSharedVariable(type=tensor_type,
                value=numpy.array(value, copy=True),
                name=name, strict=strict, allow_downcast=allow_downcast)
        return rval
    except Exception:
        traceback.print_exc()
        raise
