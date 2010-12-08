"""
Helper function to detect memory sharing for ndarray AND sparse type.
numpy version support only ndarray.
"""

__docformat__ = "restructuredtext en"

import numpy
from theano.tensor.basic import TensorType

try:
    import scipy.sparse
except ImportError:
    #scipy not imported, their can be only ndarray
    def may_share_memory(a, b, raise_other_type=True):
        if not isinstance(a, numpy.ndarray) or not isinstance(b, numpy.ndarray):
            if raise_other_type:
                raise TypeError("may_share_memory support only ndarray when scipy is not available")
            return False
        return numpy.may_share_memory(a,b)
else:
    #scipy imported, their can be ndarray and sparse type
    from theano.sparse.basic import _is_sparse, SparseType
    def may_share_memory(a, b, raise_other_type=True):
        a_ndarray = isinstance(a, numpy.ndarray)
        b_ndarray = isinstance(b, numpy.ndarray)
        try:
            a_sparse = _is_sparse(a)
            b_sparse = _is_sparse(b)
        except NotImplementedError:
            if raise_other_type:
                raise TypeError("may_share_memory support only ndarray and scipy.sparse type")
            return False

        if not(a_ndarray or a_sparse) or not(b_ndarray or b_sparse):
            if raise_other_type:
                raise TypeError("may_share_memory support only ndarray and scipy.sparse type")
            return False

        if a_ndarray and b_ndarray:
            return TensorType.may_share_memory(a,b)
        return SparseType.may_share_memory(a,b)
