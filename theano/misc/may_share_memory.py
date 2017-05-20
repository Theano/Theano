"""
Function to detect memory sharing for ndarray AND sparse type AND GpuArray.
numpy version support only ndarray.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from theano.tensor.basic import TensorType

try:
    import scipy.sparse
    from theano.sparse.basic import SparseType

    def _is_sparse(a):
        return scipy.sparse.issparse(a)
except ImportError:
    # scipy not imported, their can be only ndarray and gpuarray
    def _is_sparse(a):
        return False

from theano import gpuarray

if gpuarray.pygpu:
    def _is_gpua(a):
        return isinstance(a, gpuarray.pygpu.gpuarray.GpuArray)
else:
    def _is_gpua(a):
        return False

__docformat__ = "restructuredtext en"


def may_share_memory(a, b, raise_other_type=True):
    a_ndarray = isinstance(a, np.ndarray)
    b_ndarray = isinstance(b, np.ndarray)
    if a_ndarray and b_ndarray:
        return TensorType.may_share_memory(a, b)
    a_gpua = _is_gpua(a)
    b_gpua = _is_gpua(b)
    if a_gpua and b_gpua:
        return gpuarray.pygpu.gpuarray.may_share_memory(a, b)

    a_sparse = _is_sparse(a)
    b_sparse = _is_sparse(b)
    if (not(a_ndarray or a_sparse or a_gpua) or
            not(b_ndarray or b_sparse or b_gpua)):
        if raise_other_type:
            raise TypeError("may_share_memory support only ndarray"
                            " and scipy.sparse or GpuArray type")
        return False

    if a_gpua or b_gpua:
        return False
    return SparseType.may_share_memory(a, b)
