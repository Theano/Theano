
from gof import opt
from elemwise2 import Broadcast

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





















"""
This variable is used in compile.prog as the optimizer for all programs built
using either compile.single, compile.to_func, and compile.prog.


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
