import theano
from theano import gof
from theano.sparse import Remove0


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
