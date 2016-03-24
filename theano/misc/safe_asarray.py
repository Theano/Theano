"""
Helper function to safely convert an array to a new data type.
"""
from __future__ import absolute_import, print_function, division

import numpy

import theano

__docformat__ = "restructuredtext en"


def _asarray(a, dtype, order=None):
    """Convert the input to a Numpy array.

    This function is almost identical to ``numpy.asarray``, but it should be
    used instead of its numpy counterpart when a data type is provided in
    order to perform type conversion if required.
    The reason is that ``numpy.asarray`` may not actually update the array's
    data type to the user-provided type. For more information see ticket
    http://projects.scipy.org/numpy/ticket/870.

    In that case, we check that both dtype have the same string
    description (byte order, basic type, and number of bytes), and
    return a view with the desired dtype.

    This function's name starts with a '_' to indicate that it is meant to be
    used internally. It is imported so as to be available directly through
        theano._asarray
    """
    if str(dtype) == 'floatX':
        dtype = theano.config.floatX
    dtype = numpy.dtype(dtype)  # Convert into dtype object.
    rval = numpy.asarray(a, dtype=dtype, order=order)
    # Note that dtype comparison must be done by comparing their `num`
    # attribute. One cannot assume that two identical data types are pointers
    # towards the same object (e.g. under Windows this appears not to be the
    # case).
    if rval.dtype.num != dtype.num:
        # Type mismatch between the data type we asked for, and the one
        # returned by numpy.asarray.
        # If both types have the same string description (byte order, basic
        # type, and number of bytes), then it is safe to return a view.
        if (dtype.str == rval.dtype.str):
            # Silent fix.
            return rval.view(dtype=dtype)
        else:
            # Unexpected mismatch: better know what is going on!
            raise TypeError(
                'numpy.array did not return the data type we '
                'asked for (%s %s #%s), instead it returned type '
                '%s %s #%s: function '
                'theano._asarray may need to be modified to handle this '
                'data type.' %
                (dtype, dtype.str, dtype.num, rval.dtype, rval.str,
                 rval.dtype.num))
    else:
        return rval
