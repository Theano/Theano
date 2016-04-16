"""Define the tensor toplevel"""
from __future__ import absolute_import, print_function, division

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
from theano.tensor import nlinalg

# These imports cannot be performed here because the modules depend on tensor.  This is done at the
# end of theano.__init__.py instead.
#from theano.tensor import raw_random
#from theano.tensor import shared_randomstreams

from theano.tensor.elemwise import DimShuffle, Elemwise, CAReduce

from theano.tensor import sharedvar  # adds shared-variable constructors

# We import as `_shared` instead of `shared` to avoid confusion between
# `theano.shared` and `tensor._shared`.
from theano.tensor.sharedvar import tensor_constructor as _shared

from theano.tensor.io import *

from theano.tensor import nnet  # used for softmax, sigmoid, etc.

from theano.gradient import Rop, Lop, grad, numeric_grad, verify_grad, \
    jacobian, hessian, consider_constant

from theano.tensor.sort import sort, argsort
from theano.tensor.extra_ops import (DiffOp, bincount, squeeze,
                       repeat, bartlett, fill_diagonal, fill_diagonal_offset,
                       cumsum, cumprod)

# SpecifyShape is defined in theano.compile, but should be available in tensor
from theano.compile import SpecifyShape, specify_shape
