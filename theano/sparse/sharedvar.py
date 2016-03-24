from __future__ import absolute_import, print_function, division
import copy
import scipy.sparse
from theano.compile import shared_constructor, SharedVariable
from theano.sparse.basic import SparseType, _sparse_py_operators


class SparseTensorSharedVariable(_sparse_py_operators, SharedVariable):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)


@shared_constructor
def sparse_constructor(value, name=None, strict=False, allow_downcast=None,
                       borrow=False, format=None):
    """
    SharedVariable Constructor for SparseType.

    writeme

    """
    if not isinstance(value, scipy.sparse.spmatrix):
        raise TypeError("Expected a sparse matrix in the sparse shared variable constructor. Received: ",
                        value.__class__)

    if format is None:
        format = value.format
    type = SparseType(format=format, dtype=value.dtype)
    if not borrow:
        value = copy.deepcopy(value)
    return SparseTensorSharedVariable(type=type, value=value, name=name,
                                      strict=strict, allow_downcast=allow_downcast)
