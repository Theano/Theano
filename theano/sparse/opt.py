from itertools import izip

import theano
from theano import gof
from theano.sparse import (CSC, CSR, csm_properties, Remove0,
                           register_specialize)


@gof.local_optimizer([None])
def local_inplace_remove0(node):
    """
    Optimization to insert inplace versions of Remove0.
    """
    if isinstance(node.op, Remove0) and not node.op.inplace:
        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
theano.compile.optdb.register('local_inplace_remove0',
                              gof.TopoOptimizer(local_inplace_remove0,
    failure_callback=gof.TopoOptimizer.warn_inplace),
                              60, 'fast_run', 'inplace')


@gof.local_optimizer([csm_properties])
def skip_pack_csc01(node):
    """if we find csm_properties(CSM(*args)), then we can replace that with the
    *args directly"""
    if node.op == csm_properties:
        csm, = node.inputs
        if csm.owner and (csm.owner.op == CSC or csm.owner.op == CSR):
            # csm.owner.inputs could be broadcastable. In that case, we have
            # to adjust the broadcasting flag here.
            ret_var = [theano.tensor.patternbroadcast(i, o.broadcastable)
                    for i, o in izip(csm.owner.inputs, node.outputs)]
            return ret_var

    return False
register_specialize(skip_pack_csc01)
