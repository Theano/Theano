
import numpy
import theano.tensor.basic
from basic import TensorType, _tensor_py_operators
from theano.compile import shared_constructor, SharedVariable

class TensorSharedVariable(SharedVariable, _tensor_py_operators):
    pass

@shared_constructor
def tensor_constructor(value, name=None, strict=False, broadcastable=None):
    """SharedVariable Constructor for TensorType
    
    :note: Regarding the inference of the broadcastable pattern... 
    The default is to assume that the value might be resized in any dimension, so the default
    broadcastable is ``(False,)*len(value.shape)``.  The optional `broadcastable` argument will
    override this default.
    
    """
    if not isinstance(value, numpy.ndarray):
        raise TypeError()

    # if no broadcastable is given, then the default is to assume that the value might be
    # resized in any dimension in the future.
    # 
    if broadcastable is None:
        broadcastable = (False,)*len(value.shape)
    type = TensorType(value.dtype, broadcastable=broadcastable)
    return TensorSharedVariable(type=type, value=value, name=name, strict=strict)

# TensorSharedVariable brings in the tensor operators, is not ideal, but works as long as we
# dont do purely scalar-scalar operations 
class ScalarSharedVariable(SharedVariable, _tensor_py_operators):
    pass

@shared_constructor
def scalar_constructor(value, name=None, strict=False, dtype=None):
    """SharedVariable constructor for scalar values. Default: int64 or float64. 

    :note: We implement this using 0-d tensors for now.
    
    """  
    if not isinstance (value, (numpy.number, float, int)):
        raise TypeError()
    if dtype is None:
        if isinstance(value, float):
            dtype = 'float64'
        elif isinstance(value, int):
            dtype = 'int64'
        else:
            dtype = type(value).__name__

    tensor_type = TensorType(dtype=dtype, broadcastable=[])

    try:
        # Do not pass the dtype to asarray because we want this to fail if
        # strict is True and the types do not match.
        rval = ScalarSharedVariable(type=tensor_type,
                value=numpy.asarray(value),
                name=name, strict=strict)
        return rval
    except:
        traceback.print_exc()
        raise

