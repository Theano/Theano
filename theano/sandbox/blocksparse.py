from __future__ import absolute_import, print_function, division
import warnings
from theano.tensor.nnet.blocksparse import (
    SparseBlockGemv, SparseBlockOuter, sparse_block_dot, sparse_block_gemv,
    sparse_block_gemv_inplace, sparse_block_outer, sparse_block_outer_inplace)

__all__ = [SparseBlockGemv, SparseBlockOuter, sparse_block_dot,
           sparse_block_gemv, sparse_block_gemv_inplace, sparse_block_outer,
           sparse_block_outer_inplace]

warnings.warn("DEPRECATION: theano.sandbox.blocksparse does not exist anymore,"
              "it has been moved to theano.tensor.nnet.blocksparse.",
              category=DeprecationWarning)
