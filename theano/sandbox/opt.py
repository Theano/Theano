"""
Optimizations addressing the ops in sandbox root directory
"""

from theano import compile  # to register the optimizer built by this file
from theano import gof
from theano.sandbox.blocksparse import (
    SparseBlockGemv,
    SparseBlockOuter,
    sparse_block_gemv_inplace,
    sparse_block_outer_inplace)


@gof.local_optimizer([SparseBlockGemv], inplace=True)
def local_inplace_sparse_block_gemv(node):
    """
        SparseBlockGemv(inplace=False) -> SparseBlockGemv(inplace=True)
    """
    if isinstance(node.op, SparseBlockGemv) and not node.op.inplace:
        new_node = sparse_block_gemv_inplace(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_sparse_block_gemv',
                       gof.TopoOptimizer(
                           local_inplace_sparse_block_gemv,
                           failure_callback=gof.TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([SparseBlockOuter], inplace=True)
def local_inplace_sparse_block_outer(node):
    """
        SparseBlockOuter(inplace=False) -> SparseBlockOuter(inplace=True)
    """
    if isinstance(node.op, SparseBlockOuter) and not node.op.inplace:
        new_node = sparse_block_outer_inplace(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_sparse_block_outer',
                       gof.TopoOptimizer(
                           local_inplace_sparse_block_outer,
                           failure_callback=gof.TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG
