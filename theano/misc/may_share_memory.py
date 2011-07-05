"""
Function to detect memory sharing for ndarray AND sparse type AND CudaNdarray.
numpy version support only ndarray.
"""

__docformat__ = "restructuredtext en"

import numpy
from theano.tensor.basic import TensorType

try:
    import scipy.sparse
    from theano.sparse.basic import SparseType
    def _is_sparse(a):
        return scipy.sparse.issparse(a)
except ImportError:
    #scipy not imported, their can be only ndarray and cudandarray
    def _is_sparse(a):
        return False

from theano.sandbox import cuda
if cuda.cuda_available:
    def _is_cuda(a):
        return isinstance(a, cuda.CudaNdarray)
else:
    def _is_cuda(a):
        return False

def may_share_memory(a, b, raise_other_type=True):
    a_ndarray = isinstance(a, numpy.ndarray)
    b_ndarray = isinstance(b, numpy.ndarray)
    a_sparse = _is_sparse(a)
    b_sparse = _is_sparse(b)
    a_cuda = _is_cuda(a)
    b_cuda = _is_cuda(b)

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
