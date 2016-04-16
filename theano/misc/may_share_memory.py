"""
Function to detect memory sharing for ndarray AND sparse type AND CudaNdarray.
numpy version support only ndarray.
"""
from __future__ import absolute_import, print_function, division

import numpy
from theano.tensor.basic import TensorType

try:
    import scipy.sparse
    from theano.sparse.basic import SparseType

    def _is_sparse(a):
        return scipy.sparse.issparse(a)
except ImportError:
    # scipy not imported, their can be only ndarray and cudandarray
    def _is_sparse(a):
        return False

from theano.sandbox import cuda
from theano.sandbox import gpuarray

if cuda.cuda_available:
    from theano.sandbox.cuda.type import CudaNdarrayType

    def _is_cuda(a):
        return isinstance(a, cuda.CudaNdarray)
else:
    def _is_cuda(a):
        return False

__docformat__ = "restructuredtext en"


if gpuarray.pygpu:
    def _is_gpua(a):
        return isinstance(a, gpuarray.pygpu.gpuarray.GpuArray)
else:
    def _is_gpua(a):
        return False


def may_share_memory(a, b, raise_other_type=True):
    a_ndarray = isinstance(a, numpy.ndarray)
    b_ndarray = isinstance(b, numpy.ndarray)
    if a_ndarray and b_ndarray:
        return TensorType.may_share_memory(a, b)
    a_cuda = _is_cuda(a)
    b_cuda = _is_cuda(b)
    if a_cuda and b_cuda:
        return CudaNdarrayType.may_share_memory(a, b)
    a_gpua = _is_gpua(a)
    b_gpua = _is_gpua(b)
    if a_gpua and b_gpua:
        return gpuarray.pygpu.gpuarray.may_share_memory(a, b)

    a_sparse = _is_sparse(a)
    b_sparse = _is_sparse(b)
    if (not(a_ndarray or a_sparse or a_cuda or a_gpua) or
            not(b_ndarray or b_sparse or b_cuda or b_gpua)):
        if raise_other_type:
            raise TypeError("may_share_memory support only ndarray"
                            " and scipy.sparse, CudaNdarray or GpuArray type")
        return False

    if a_cuda or b_cuda or a_gpua or b_gpua:
        return False
    return SparseType.may_share_memory(a, b)
