"""
Helper function to safely convert an array to a new data type.
"""

__docformat__ = "restructuredtext en"

import numpy

def _asarray(a, dtype=None, order=None):
    """Convert the input to a Numpy array.

    This function is almost identical to ``numpy.asarray``, but it should be
    used instead of its numpy counterpart when a data type is provided in
    order to perform type conversion if required.
    The reason is that ``numpy.asarray`` may not actually update the array's
    data type to the user-provided type. For more information see ticket
    http://projects.scipy.org/numpy/ticket/870.

    Currently, this issue has only been causing trouble when the target
    data type is 'int32', on some computers. As a result, this is the only
    situation where we do more than a simple call to ``numpy.asarray``. If it
    turns out that a similar problem can occur for more data type, this
    function should be updated accordingly.

    This function's name starts with a '_' to indicate that it is meant to be
    used internally. It is imported so as to be available directly through
        theano._asarray
    """
    rval = numpy.asarray(a, dtype=dtype, order=order)
    if dtype is numpy.int32 or dtype == 'int32':
        # Make sure the type is properly set to the correct type.
        return rval.view(dtype=numpy.int32)
    else:
        # Using ``numpy.asarray`` should work just fine.
        return rval
