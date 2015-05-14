"""
Optimizations addressing the ops in sandbox root directory
"""

import logging
_logger = logging.getLogger('theano.sandbox.opt')

def register_meta_opt(op_class, priority, local_ops):

    def call(fct):
        for local_op in local_ops:
            op_class.register_opt[local_op].append(fct)

    return call

def meta_opts(op_class, op):
    return op_class.registered_opts[op]


@register_meta_opt(SparseBlockGemv, 100, [SparseBlockGemv]):
def cpu_sparse_block_gemv():
    # TODO: WRITEME
    pass

@register_meta_opt(SparseBlockGemv, 100, [SparseBlockGemv, HostFromGpu]): # cuda/opt.py
def gpu_sparse_block_gemv():
    # TODO: Move to cuda/opt.py
    # TODO: WRITEME
    pass

@register_specialize
def meta_sparse_block_gemv(node):
    for registered_opt in meta_opts(SparseBlockGemv, node.op): # list = [gpu_sparse_block_gemv]
        new_node = registered_opt(node)
        if new_node is not None:
            return [new_node]






