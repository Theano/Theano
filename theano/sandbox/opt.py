"""
Optimizations addressing the ops in sandbox root directory
"""

import logging
from theano.sandbox.blocksparse import (
    SparseBlockGemv, 
    SparseBlockOuter,
    cpu_sparse_block_gemv)


_logger = logging.getLogger('theano.sandbox.opt')


def register_meta_opt(op_class, order, *tags):

    def call(fct):
        idx = bisect.bisect_left((order, fct), 
                op_class.registered_opt_priorities)
        op_class.registered_opts.insert(idx, (order, fct))
        optdb.register("meta_%s_%s" % (str(op_class), str(fct.__name__)), fct,
                            order=order, *tags)

    return call


@register_meta_opt(SparseBlockGemv, 48.56)
@local_optimizer([SparseBlockGemv])
def cpu_sparse_block_gemv(node):
    """
        TODO: WRITEME
    """
    if node.op.inplace:
        _logger.warning("CPU version of sparse_block_gemv does not support"
                        "inplace")
    return [cpu_sparse_block_gemv(*node.inputs)]


@register_meta_opt(SparseBlockOuter, 48.56)
@local_optimizer([SparseBlockOuter])
def cpu_sparse_block_outer(node):
    """
        TODO: WRITEME
    """
    if node.op.inplace:
        _logger.warning("CPU version of sparse_block_outer does not support"
                        "inplace")
    return [cpu_sparse_block_outer(*node.inputs)]
