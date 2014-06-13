"""Define the tensor toplevel"""
__docformat__ = "restructuredtext en"

import warnings

from theano.tensor.basic import *
from theano.tensor.subtensor import *
from theano.tensor.type_other import *
from theano.tensor.var import (
    AsTensorError, _tensor_py_operators, TensorVariable,
    TensorConstantSignature, TensorConstant)

from theano.tensor import opt
from theano.tensor import opt_uncanonicalize
from theano.tensor import blas
from theano.tensor import blas_scipy
from theano.tensor import blas_c
from theano.tensor import xlogx

# These imports cannot be performed here because the modules depend on tensor.  This is done at the
# end of theano.__init__.py instead.
#from theano.tensor import raw_random
#from theano.tensor import randomstreams
#from theano.tensor import shared_randomstreams
#from theano.tensor.randomstreams import \
#    RandomStreams

#random = RandomStreams(seed=0xBAD5EED, no_warn = True)
#"""Imitate the numpy.random symbol with a tensor.random one"""

from theano.tensor.elemwise import DimShuffle, Elemwise, CAReduce

from theano.tensor import sharedvar  # adds shared-variable constructors

# We import as `_shared` instead of `shared` to avoid confusion between
# `theano.shared` and `tensor._shared`.
from theano.tensor.sharedvar import tensor_constructor as _shared

from theano.tensor.io import *

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


from theano.tensor import nnet  # used for softmax, sigmoid, etc.

from theano.gradient import Rop, Lop, grad, numeric_grad, verify_grad, \
    jacobian, hessian, consider_constant

from theano.tensor.sort import sort, argsort
from theano.tensor.extra_ops import (DiffOp, bincount, squeeze,
                       repeat, bartlett, fill_diagonal, fill_diagonal_offset,
                       cumsum, cumprod)
