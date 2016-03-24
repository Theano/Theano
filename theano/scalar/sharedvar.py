"""
A shared variable container for true scalars - for internal use.

Why does this file exist?
-------------------------

Scalars are used to index subtensors.  Subtensor indexing is the heart of what
looks like the new scan mechanism.  This little file made it possible to catch
up to the Python interpreter in benchmarking tests.

We don't want to encourage people to use scalars (rather than 0-d tensors), but
the reason is just to keep the docs simple, not because scalars are bad.  If we
just don't register this shared variable constructor to handle any values by
default when calling theano.shared(value) then users must really go out of their
way (as scan does) to create a shared variable of this kind.

"""
from __future__ import absolute_import, print_function, division
import numpy
from six import integer_types

from theano.compile import SharedVariable
from .basic import Scalar, _scalar_py_operators

__authors__ = "James Bergstra"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"


class ScalarSharedVariable(_scalar_py_operators, SharedVariable):
    pass

# this is not installed in the default shared variable registry so that
# scalars are typically 0-d tensors.
# still, in case you need a shared variable scalar, you can get one
# by calling this function directly.


def shared(value, name=None, strict=False, allow_downcast=None):
    """
    SharedVariable constructor for scalar values. Default: int64 or float64.

    Notes
    -----
    We implement this using 0-d tensors for now.

    """
    if not isinstance(value, (numpy.number, float, integer_types, complex)):
        raise TypeError()
    try:
        dtype = value.dtype
    except AttributeError:
        dtype = numpy.asarray(value).dtype

    dtype = str(dtype)
    value = getattr(numpy, dtype)(value)
    scalar_type = Scalar(dtype=dtype)
    rval = ScalarSharedVariable(
        type=scalar_type,
        value=value,
        name=name,
        strict=strict,
        allow_downcast=allow_downcast)
    return rval
