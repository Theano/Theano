"""Define the tensor toplevel"""
__docformat__ = "restructuredtext en"


import warnings

from basic import *

import opt
import opt_uncanonicalize
import blas
import blas_scipy
import xlogx

import raw_random, randomstreams
import shared_randomstreams
from randomstreams import \
    RandomStreams

random = RandomStreams(seed=0xBAD5EED)
"""Imitate the numpy.random symbol with a tensor.random one"""

from elemwise import \
    DimShuffle, Elemwise, CAReduce

import sharedvar # adds shared-variable constructors

# We import as `_shared` instead of `shared` to avoid confusion between
# `theano.shared` and `tensor._shared`.
from sharedvar import tensor_constructor as _shared


def shared(*args, **kw):
    """
    Backward-compatibility wrapper around `tensor._shared`.

    Once the deprecation warning has been around for long enough, this function
    can be deleted.
    """
    # Note that we do not use the DeprecationWarning class because it is
    # ignored by default since python 2.7.
    warnings.warn('`tensor.shared` is deprecated. You should probably be using'
                  ' `theano.shared` instead (if you *really* intend to call '
                  '`tensor.shared`, you can get rid of this warning by using '
                  '`tensor._shared`).',
                  stacklevel=2)
    return _shared(*args, **kw)


import nnet # used for softmax, sigmoid, etc.

from tensor_grad import Rop, Lop, grad, numeric_grad, verify_grad


