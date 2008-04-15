
from gof import opt, Env
import gof
from elemwise import Broadcast, DimShuffle
from gof.python25 import any, all
import scalar


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
                or not isinstance(op, Broadcast) \
                or len(op.outputs) > 1:
            # todo: handle multiple-output broadcast ops
            #       (needs to update the clique's outputs)
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
            cliques.append((clique_inputs, [r]))
            for input in clique_inputs:
                find_cliques_helper(input)

    for output in env.outputs:
        find_cliques_helper(output)

    # todo: merge the cliques if possible

    return cliques


class CliqueOptimizer(opt.Optimizer):

    def __init__(self, through_broadcast = False, scalar_optimizer = None, make_composite = False):
        self.through_broadcast = through_broadcast
        self.scalar_optimizer = scalar_optimizer
        self.make_composite = make_composite

    def apply(self, env):
        if self.scalar_optimizer is None and not self.make_composite:
            # there's nothing to do with the cliques...
            return
        cliques = find_cliques(env, self.through_broadcast)
        opt = self.scalar_optimizer

        def build_scalar_clique(r, env, equiv):
            if r in equiv:
                return equiv[r]
            op = r.owner
            if r in env.inputs or r in env.orphans():
                s = scalar.Scalar(dtype = r.dtype)
                _r = r
                if isinstance(r.owner, DimShuffle) and all(x == 'x' for x in r.owner.new_order):
                    _r = r.owner.inputs[0]
                if (getattr(r, 'constant', False) or getattr(_r, 'constant', False)) \
                       and _r.broadcastable == ():
                    s.data = _r.data
                    s.constant = True
                equiv[r] = s
                return s
            s_op = op.scalar_opclass(*[build_scalar_clique(input, env, equiv) for input in op.inputs])
            equiv[op] = s_op
            for output, s_output in zip(op.outputs, s_op.outputs):
                equiv[output] = s_output
            return equiv[r]

        for c_in, c_out in cliques:
            equiv = dict()
            g = Env(c_in, c_out)
            for output in c_out:
                build_scalar_clique(output, g, equiv)
            s_g = Env([equiv[r] for r in g.inputs],
                      [equiv[r] for r in g.outputs])
            if opt is not None:
                equiv2 = dict()
                for k, v in equiv.items():
                    equiv2[v] = k
                def transform(op, equiv):
                    return Broadcast(op.__class__, [equiv[input] for input in op.inputs])
                s_g.add_feature(sync_to(env, equiv2, transform))
                opt.optimize(s_g)
            if self.make_composite:
                def follow_inplace(r):
                    op = r.owner
                    if op is None or r in g.inputs or r in g.orphans():
                        return None
                    assert isinstance(op, Broadcast)
                    destroyed = op.destroy_map().get(r, None)
                    if destroyed is None:
                        return None
                    else:
                        r2 = destroyed[0]
                        ret = follow_inplace(r2)
                        if ret is None:
                            return r2
                        else:
                            return ret
                inplace_pattern = {}
                for i, output in enumerate(g.outputs):
                    destroyed = follow_inplace(output)
                    if destroyed is not None and destroyed in g.inputs:
                        inplace_pattern[i] = g.inputs.index(destroyed)
                C = scalar.composite(s_g.inputs, s_g.outputs)
                ec = Broadcast(C, g.inputs, inplace_pattern = inplace_pattern)
                env.replace_all(dict((o, eco) for o, eco in zip(c_out, ec.outputs)))


def sync_to(target, equiv, transform):

    class Synchronize(gof.Listener, gof.Constraint):

        def __init__(self, source):
            self.source = source
            self.target = target
            self.equiv = equiv
            self.transform = transform
            self.inconsistencies = []

        def on_import(self, op1):
            if op1 not in self.equiv:
                op2 = self.transform(op1, self.equiv)
                self.equiv[op1] = op2
                for o1, o2 in zip(op1.outputs, op2.outputs):
                    self.equiv[o1] = o2

        def on_prune(self, op1):
            if op1 in self.equiv:
                op2 = self.equiv[op1]
                del self.equiv[op1]
                for o1, o2 in zip(op1.outputs, op2.outputs):
                    del self.equiv[o1]

        def on_rewire(self, clients1, r1, new_r1):
            if (new_r1, r1) in self.inconsistencies:
                self.inconsistencies.remove((new_r1, r1))
                return
            if not self.source.clients(r1):
                try:
                    target.replace(self.equiv[r1], self.equiv[new_r1])
                except:
                    self.inconsistencies.append((r1, new_r1))

        def validate(self):
            if self.inconsistencies:
                raise InconsistencyError("Could not synchronize when replacing the following pairs: %s" % self.inconsistencies)
            return True

    return Synchronize











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
