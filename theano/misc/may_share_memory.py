"""
Function to detect memory sharing for ndarray AND sparse type AND CudaNdarray.
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
        except NotImplementedError:
            a_sparse = False
        try:
            b_sparse = _is_sparse(b)
        except NotImplementedError:
            b_sparse = False

        a_cuda = False
        b_cuda = False
        if a.__class__.__name__ == "CudaNdarray":
            a_cuda = True
        if b.__class__.__name__ == "CudaNdarray":
            b_cuda = True

        if not(a_ndarray or a_sparse or a_cuda) or not(b_ndarray or b_sparse or b_cuda):
            if raise_other_type:
                raise TypeError("may_share_memory support only ndarray and scipy.sparse and CudaNdarray type")
            return False

        if a_ndarray and b_ndarray:
            return TensorType.may_share_memory(a,b)
        if a_cuda and b_cuda:
            from theano.sandbox.cuda.type import CudaNdarrayType
            return CudaNdarrayType.may_share_memory(a,b)
        if a_cuda or b_cuda:
            return False
        return SparseType.may_share_memory(a,b)
