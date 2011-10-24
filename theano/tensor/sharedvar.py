import traceback
import numpy
import theano.tensor.basic
from basic import TensorType, _tensor_py_operators, autocast_int, autocast_float
from theano.compile import shared_constructor, SharedVariable
from theano import config

def load_shared_variable(val):
    """This function is only here to keep some pickles loading
    after a failed fix done in August 2011.
    It can be removed after sufficient time has passed."""
    return tensor_constructor(val)

# _tensor_py_operators is first to have its version of __{gt,ge,lt,le}__
class TensorSharedVariable(_tensor_py_operators, SharedVariable):
    pass

@shared_constructor
def tensor_constructor(value, name=None, strict=False, allow_downcast=None, borrow=False, broadcastable=None):
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
    return TensorSharedVariable(type=type,
            value=numpy.array(value,copy=(not borrow)),
            name=name,
            strict=strict,
            allow_downcast=allow_downcast)

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
    if not isinstance (value, (numpy.number, float, int, complex)):
        raise TypeError()
    try:
        dtype=value.dtype
    except:
        dtype=numpy.asarray(value).dtype

    dtype=str(dtype)
    value = theano._asarray(value, dtype=dtype)
    tensor_type = TensorType(dtype=str(value.dtype), broadcastable=[])

    try:
        # Do not pass the dtype to asarray because we want this to fail if
        # strict is True and the types do not match.
        rval = ScalarSharedVariable(type=tensor_type,
                value=numpy.array(value, copy=True),
                name=name, strict=strict, allow_downcast=allow_downcast)
        return rval
    except:
        traceback.print_exc()
        raise
