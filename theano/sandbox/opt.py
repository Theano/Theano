"""
Optimizations addressing the ops in sandbox root directory
"""

import bisect
import logging

from theano.compile import optdb
from theano.gof import local_optimizer, EquilibriumOptimizer
from theano.sandbox.blocksparse import (
    SparseBlockGemv, 
    SparseBlockOuter,
    cpu_sparse_block_gemv,
    cpu_sparse_block_outer)


_logger = logging.getLogger('theano.sandbox.opt')


def register_meta_opt(op_class, order, *tags):

    def call(fct):
        idx = bisect.bisect_left((order, fct), 
                                 op_class.registered_opts)
        op_class.registered_opts.insert(idx, (order, fct))
        optdb.register("meta_%s.%s" % (str(op_class.__name__), str(fct.__name__)), 
                       EquilibriumOptimizer([fct],max_use_ratio=1), order, *tags)

    return call


@register_meta_opt(SparseBlockGemv, 48.56, "fast_run")
@local_optimizer([SparseBlockGemv])
def cpu_sparse_block_gemv_opt(node):
    """
        TODO: WRITEME
    """
    if node.op.inplace:
        _logger.warning("CPU version of sparse_block_gemv does not support"
                        "inplace")
    return [cpu_sparse_block_gemv(*node.inputs)]


@register_meta_opt(SparseBlockOuter, 48.56, "fast_run")
@local_optimizer([SparseBlockOuter])
def cpu_sparse_block_outer_opt(node):
    """
        TODO: WRITEME
    """
    if node.op.inplace:
        _logger.warning("CPU version of sparse_block_outer does not support"
                        "inplace")
    return [cpu_sparse_block_outer(*node.inputs)]
