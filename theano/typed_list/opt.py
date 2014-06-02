from theano import gof
from theano import compile
from theano.gof import TopoOptimizer
from theano.typed_list.basic import (Reverse,
                    Append, Extend, Insert)


def generic_opt_creator(op):

    @gof.local_optimizer([op], inplace=True)
    def generic_inplace_opt(node):
            if isinstance(node.op, op) and not node.op.inplace:
                new_op = node.op.__class__(
                    inplace=True)
                new_node = new_op(*node.inputs)
                return [new_node]
            return False

    return generic_inplace_opt


local_inplace_reverse = generic_opt_creator(Reverse)
compile.optdb.register('local_inplace_reverse',
                       TopoOptimizer(local_inplace_reverse,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')


local_inplace_append = generic_opt_creator(Append)
compile.optdb.register('local_inplace_append',
                       TopoOptimizer(local_inplace_append,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')


local_inplace_extend = generic_opt_creator(Extend)
compile.optdb.register('local_inplace_extend',
                       TopoOptimizer(local_inplace_extend,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')


local_inplace_insert = generic_opt_creator(Insert)
compile.optdb.register('local_inplace_insert',
                       TopoOptimizer(local_inplace_insert,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')  # DEBUG
