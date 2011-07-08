"""A shared variable container for true scalars - for internal use.
"""
__authors__   = "James Bergstra"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"

import numpy
from theano.compile import shared_constructor, SharedVariable
from basic import Scalar, _scalar_py_operators

class ScalarSharedVariable(_scalar_py_operators, SharedVariable):
    pass

# this is not installed in the default shared variable registry so that
# scalars are typically 0-d tensors.
# still, in case you need a shared variable scalar, you can get one
# by calling this function directly.
def shared(value, name=None, strict=False, allow_downcast=None):
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
    value = getattr(numpy, dtype)(value)
    scalar_type = Scalar(dtype=dtype)
    rval = ScalarSharedVariable(
            type=scalar_type,
            value=value,
            name=name, strict=strict, allow_downcast=allow_downcast)
    return rval
