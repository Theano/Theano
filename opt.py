
from gof import opt
from elemwise import Broadcast, DimShuffle
from gof.python25 import any, all


class InplaceOptimizer(opt.OpSpecificOptimizer):

    opclass = Broadcast
    
    def apply_on_op(self, env, op):
        baseline = op.inplace_pattern
        candidate_outputs = [i for i in xrange(len(op.outputs)) if i not in baseline]
        candidate_inputs = [i for i in xrange(len(op.inputs)) if i not in baseline.values()]
        for candidate_output in candidate_outputs:
            for candidate_input in candidate_inputs:
                inplace_pattern = dict(baseline, **{candidate_output: candidate_input})
                try:
                    new_op = Broadcast(op.scalar_opclass, op.inputs, inplace_pattern)
                    env.replace_all(dict(zip(op.outputs, new_op.outputs)))
                except:
                    continue
                candidate_inputs.remove(candidate_input)
                op = new_op
                break

inplace_optimizer = InplaceOptimizer()



class DimShuffleLifter(opt.Optimizer):
    """
    "Lifts" DimShuffle through Broadcast operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Broadcast(x, y)) => Broadcast(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)

    After this transform, clusters of Broadcast operations are
    void of DimShuffle operations.
    """

    def apply(self, env):

        seen = set()

        def merge(ord1, ord2):
            return [x == 'x' and 'x' or ord1[x] for x in ord2]
        
        def lift(r):
            if r in seen:
                return
            seen.add(r)
            op = r.owner
            if op is None \
                    or op in env.inputs \
                    or op in env.orphans():
                return
            if isinstance(op, DimShuffle):
                in_op = op.inputs[0].owner
                if isinstance(in_op, DimShuffle):
                    new_order = [x == 'x' and 'x' or in_op.new_order[x] for x in op.new_order]
                    if new_order == range(len(new_order)):
                        repl = in_op.inputs[0]
                    else:
                        repl = DimShuffle(in_op.inputs[0], new_order).out
                    env.replace(r, repl)
                    lift(repl)
                    return
                elif isinstance(in_op, Broadcast):
                    repl = Broadcast(in_op.scalar_opclass,
                                     [DimShuffle(input, op.new_order).out for input in in_op.inputs],
                                     in_op.inplace_pattern).out
                    env.replace(r, repl)
                    r = repl
                    op = r.owner
            for next_r in op.inputs:
                lift(next_r)

        for output in env.outputs:
            lift(output)

lift_dimshuffle = DimShuffleLifter()


def find_cliques(env, through_broadcast = False):

    def seek_from(r):
        op = r.owner
        if r in env.inputs \
                or r in env.orphans() \
                or op is None \
                or not isinstance(op, Broadcast):
            return None

        ret = set()

        if not through_broadcast:
            if any(any(bc) and not all(bc)
                   for bc in zip(*[input.broadcastable for input in op.inputs])):
                ret.update(op.inputs)
                return ret
        
        for input in op.inputs:
            res = seek_from(input)
            if res is None:
                ret.add(input)
            else:
                ret.update(res)
        
        return ret
    
    cliques = []

    def find_cliques_helper(r):
        if r in env.inputs or r in env.orphans():
            return
        clique_inputs = seek_from(r)
        if clique_inputs is None:
            op = r.owner
            if op is not None:
                for input in op.inputs:
                    find_cliques_helper(input)
        else:
            cliques.append((clique_inputs, r))
            for input in clique_inputs:
                find_cliques_helper(input)

    for output in env.outputs:
        find_cliques_helper(output)

    # todo: merge the cliques if possible

    return cliques




# class ElemwisePatternOptimizer(opt.Optimizer):

#     def __init__(self, scalar_opt):
#         self.







# def synchronize(env1, env2, equiv, transform):

#     class Synchronize(Listener, Constraint):
        
#         def on_import(self, op1):
#             if op1 not in equiv:
#                 equiv[op1] = transform(op1)

#         def on_prune(self, op1):
#             if op1 in equiv:
#                 del equiv[op1]










"""
This variable is used in compile.prog as the optimizer for all programs built
using either compile.single, compile.to_func, and compile.prog.

Old code::
	if 0:
	    def optimizer(lst):
	        begin = gof.SeqOptimizer([])
	        end   = gof.SeqOptimizer([gof.DummyRemover])
	        seq_opt = gof.SeqOptimizer(begin + lst + end)
	        return gof.PythonOpt(gof.MergeOptMerge(seq_opt))
	
	if 0:
	    optimizer_begin = gof.SeqOptimizer([opt for name, opt in [
	             ['double_transpose_eliminator', pattern_opt((transpose, (transpose, 'x')), 'x')],
	
	             ['addxx_to_twice',              pattern_opt((add_elemwise, 'x', 'x'), (twice, 'x'))],
	
	             ['twice_to_itwice',             op_sub(twice, itwice)],
	
	             ['mulxx_to_sqr',                pattern_opt((mul_elemwise, 'x', 'x'), (sqr, 'x'))],
	
	             ['sqr_to_isqr',                 op_sub(sqr, isqr)],
	
	             ['add_to_iadd',                 op_sub(add_elemwise, iadd_elemwise)],
	
	             ['add_to_iadd_reverse',         pattern_opt((add_elemwise, 'x', 'y'),
	                 (iadd_elemwise, 'y', 'x'))]]])
	#         ['remove_copies',               gof.OpRemover(array_copy)],
	#         [None,                          gof.DummyRemover] # has to be at the end
"""
