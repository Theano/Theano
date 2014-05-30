from theano import gof
from theano import compile
from theano.gof import TopoOptimizer
from theano.typed_list.basic import (Reverse,
                    Append, Extend, Insert)


@gof.local_optimizer([Reverse], inplace=True)
def local_inplace_reverse(node):
    if isinstance(node.op, Reverse) and not node.op.inplace:
        new_op = node.op.__class__(
            inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_reverse',
                       TopoOptimizer(local_inplace_reverse,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([Append], inplace=True)
def local_inplace_append(node):
    if isinstance(node.op, Append) and not node.op.inplace:
        new_op = node.op.__class__(
            inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_append',
                       TopoOptimizer(local_inplace_append,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([Extend], inplace=True)
def local_inplace_extend(node):
    if isinstance(node.op, Extend) and not node.op.inplace:
        new_op = node.op.__class__(
            inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_extend',
                       TopoOptimizer(local_inplace_extend,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([Insert], inplace=True)
def local_inplace_insert(node):
    if isinstance(node.op, Insert) and not node.op.inplace:
        new_op = node.op.__class__(
            inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_insert',
                       TopoOptimizer(local_inplace_insert,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')  # DEBUG
