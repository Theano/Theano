
import gof
from elemwise import Elemwise, DimShuffle
import scalar
import tensor as T

# gemm: (d,a,b,c,s) -> d = d*s + a*dot(b,c)
# Transforms d -= a * dot(b, c) into gemm(d, -a, b, c, 1.0)
gemm_pattern_1 = gof.PatternSub((T.sub_inplace,
                                 'd',
                                 (T.mul,
                                  dict(pattern = (T.DimShuffle((), ['x', 'x'], inplace = True), 'a'),
                                       allow_multiple_clients = True),
                                  (T.dot, 'b', 'c'))),
                                (T.gemm, 'd', (T.neg, 'a'), 'b', 'c', T.constant(1.0)),
                                allow_multiple_clients = False)

# gemm: (d,a,b,c,s) -> d = d*s + a*dot(b,c)
# Transforms dot(a, b) into gemm(zeros(2)(hstack(shape(a)[:1], shape(b)[1:])), 1.0, a, b, 1.0)
dot_to_gemm = gof.PatternSub((T.dot, 'a', 'b'),
                             (T.gemm, (T.Zeros(2),
                                       (T.vertical_stack,
                                        (T.Subtensor([slice(0, 1)]), (T.shape, 'a')),
                                        (T.Subtensor([slice(1, 2)]), (T.shape, 'b')))),
                              T.constant(1.0), 'a', 'b', T.constant(1.0)),
                             allow_multiple_clients = False)


class InplaceOptimizer(gof.Optimizer):
    """
    Usage: inplace_optimizer.optimize(env)
    
    Attempts to replace all Broadcast ops by versions of them
    that operate inplace. It operates greedily: for each Broadcast
    Op that is encountered, for each output, tries each input to
    see if it can operate inplace on that input. If so, makes the
    change and go to the next output or Broadcast Op.

    Examples:
      x + y + z -> x += y += z
      (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)
    """
    
    def apply(self, env):
        for node in list(env.nodes):
            op = node.op
            if not isinstance(op, Elemwise):
                continue
            baseline = op.inplace_pattern
            candidate_outputs = [i for i in xrange(len(node.outputs)) if i not in baseline]
            candidate_inputs = [i for i in xrange(len(node.inputs)) if i not in baseline.values()]
            for candidate_output in candidate_outputs:
                for candidate_input in candidate_inputs:
                    inplace_pattern = dict(baseline, **{candidate_output: candidate_input})
                    try:
                        new = Elemwise(op.scalar_op, inplace_pattern).make_node(op.inputs)
                        env.replace_all_validate(dict(zip(node.outputs, new.outputs)))
                    except:
                        continue
                    candidate_inputs.remove(candidate_input)
                    node = new
                    baseline = inplace_pattern
                    break

    def add_requirements(self, env):
        env.extend(gof.toolbox.ReplaceValidate)

inplace_optimizer = InplaceOptimizer()



class DimShuffleLifter(gof.LocalOptimizer):
    """
    Usage: lift_dimshuffle.optimize(env)
    
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.
    """

    def transform(self, node):
        op = node.op
        if not isinstance(op, DimShuffle):
            return False

        input = node.inputs[0]
        inode = input.owner
        if inode and isinstance(inode.op, Elemwise):
            return inode.op.make_node(*[DimShuffle(input.type.broadcastable,
                                                   op.new_order,
                                                   op.inplace)(input) for input in inode.inputs]).outputs
        if inode and isinstance(inode.op, DimShuffle):
            new_order = [x == 'x' and 'x' or inode.op.new_order[x] for x in op.new_order]
            inplace = op.inplace and inode.op.inplace
            iinput = inode.inputs[0]
            if new_order == range(len(new_order)):
                return [iinput]
            else:
                return DimShuffle(iinput.type.broadcastable, new_order, inplace).make_node(iinput).outputs

lift_dimshuffle = gof.TopoOptimizer(DimShuffleLifter(), order = 'out_to_in')



class Canonizer(gof.Optimizer):
    """
    Simplification tool.

    Usage: Canonizer(main, inverse, reciprocal, mainfn, invfn, recfn, transform)
    
    * main: a suitable Op class that is commutative, associative and takes
            one to an arbitrary number of inputs, e.g. Add or Mul
    * inverse: an Op class such that inverse(main(x, y), y) == x
               e.g. Sub or Div
    * reciprocal: a function such that main(x, reciprocal(y)) == inverse(x, y)
                  e.g. Neg or Inv

    * mainfn, invfn, recfn: functions that behave just like the previous three
                            Ops, but on true scalars (e.g. their impl)

    * transform: a function that maps (numerator, denominatur) where numerator
                 and denominator are lists of Result instances, to new lists
                 where further simplifications may have been applied.

    Examples:
      add_canonizer = Canonizer(Add, Sub, Neg, lambda *inputs: sum(inputs), ...)
      mul_canonizer = Canonizer(Mul, Div, Inv, lambda *inputs: product(inputs), ...)
    
    Examples of optimizations mul_canonizer can perform:
      x / x -> 1
      (x * y) / x -> y
      x / y / x -> 1 / y
      x / y / z -> x / (y * z)
      x / (y / z) -> (x * z) / y
      (a / b) * (b / c) * (c / d) -> a / d
      (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
      2 * x / 2 -> x
    """

    def __init__(self, main, inverse, reciprocal, mainfn, invfn, recfn, transform = None):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.mainfn = mainfn
        self.invfn = invfn
        self.recfn = recfn
        self.neutral = mainfn()
        self.transform = transform

    def apply(self, env):

        def edge(r):
            return r.owner is None
        def follow(r):
            return None if r.owner is None else r.owner.inputs

        def canonize(r):
            
            next = follow(r)
            if next is None:
                return
            
            def flatten(r, nclients_check = True):
                # Collapses a tree of main/inverse/reciprocal Ops (aka Mul/Div/Inv or Add/Sub/Neg)
                # into a list of numerators and a list of denominators
                # e.g. (x*(1/y))*(x/(z/a)) aka Mul(Mul(x, (Inv, y)), Div(x, Div(z, a))) -> [x, x, a], [z, y]

                if edge(r):
                    return [r], []
                node = r.owner
                op = node.op
                
                results = [r2.type == r.type and flatten(r2) or ([r2], []) for r2 in node.inputs]
                if op == self.main and (not nclients_check or env.nclients(r) == 1):
                    nums = [x[0] for x in results]
                    denums = [x[1] for x in results]
                elif op == self.inverse and (not nclients_check or env.nclients(r) == 1):
                    # num, denum of the second argument are added to the denum, num respectively
                    nums = [results[0][0], results[1][1]]
                    denums = [results[0][1], results[1][0]]
                elif op == self.reciprocal and (not nclients_check or env.nclients(r) == 1):
                    # num, denum of the sole argument are added to the denum, num respectively
                    nums = [results[0][1]]
                    denums = [results[0][0]]
                else:
                    return [r], []

                return reduce(list.__add__, nums), reduce(list.__add__, denums)

            num, denum = flatten(r, False)

            if (num, denum) == ([r], []):
                for input in (follow(r) or []):
                    canonize(input)
                return

            # Terms that are both in the num and denum lists cancel each other
            for d in list(denum):
                if d in list(num):
                    # list.remove only removes the element once
                    num.remove(d)
                    denum.remove(d)

            # We identify the constants in num and denum
            numct, num = gof.utils.partition(lambda factor: isinstance(factor, gof.Constant) and factor.data is not None, num)
            denumct, denum = gof.utils.partition(lambda factor: isinstance(factor, gof.Constant) and factor.data is not None, denum)

            #print numct, num
            #print denumct, denum
            print num, denum

            # All constants in num and denum are combined into a single constant which we add to num (unless it's a neutral constant)
            v = self.invfn(self.mainfn(*[x.data for x in numct]), self.mainfn(*[x.data for x in denumct]))
            if v != self.neutral:
                num.insert(0, C(v))

            # We optimize the num and denum lists further if requested
            if self.transform is not None:
                num, denum = self.transform(env, num, denum)

            def make(factors):
                # Combines the factors using self.main (aka Mul) depending
                # on the number of elements.
                n = len(factors)
                if n == 0:
                    return None
                elif n == 1:
                    return factors[0]
                else:
                    return self.main(*factors)

            numr, denumr = make(num), make(denum)
            
            if numr is None:
                if denumr is None:
                    # Everything cancelled each other so we're left with
                    # the neutral element.
                    new_r = gof.Constant(r.type, self.neutral)
                else:
                    # There's no numerator so we use reciprocal
                    new_r = self.reciprocal(denumr)
            else:
                if denumr is None:
                    new_r = numr
                else:
                    new_r = self.inverse(numr, denumr)

            # Hopefully this won't complain!
            env.replace(r, new_r)

            for factor in num + denum:
                canonize(factor)

        for output in env.outputs:
            canonize(output)


_mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
_divfn = lambda x, y: x / y
_invfn = lambda x: 1 / x
mul_canonizer = Canonizer(T.mul, T.div, T.inv, _mulfn, _divfn, _invfn)




















# class DimShuffleLifter(opt.Optimizer):
#     """
#     Usage: lift_dimshuffle.optimize(env)
    
#     "Lifts" DimShuffle through Broadcast operations and merges
#     consecutive DimShuffles. Basically, applies the following
#     transformations on the whole graph:

#     DimShuffle(Broadcast(x, y)) => Broadcast(DimShuffle(x), DimShuffle(y))
#     DimShuffle(DimShuffle(x)) => DimShuffle(x)

#     After this transform, clusters of Broadcast operations are
#     void of DimShuffle operations.
#     """

#     def apply(self, env):

#         seen = set()
        
#         def lift(r):
#             if r in seen:
#                 return
#             seen.add(r)
#             if env.edge(r):
#                 return
#             op = r.owner
#             if isinstance(op, DimShuffle):
#                 in_op = op.inputs[0].owner
#                 if isinstance(in_op, DimShuffle):
#                     # DimShuffle(DimShuffle(x)) => DimShuffle(x)
#                     new_order = [x == 'x' and 'x' or in_op.new_order[x] for x in op.new_order]
#                     if new_order == range(len(new_order)):
#                         repl = in_op.inputs[0]
#                     else:
#                         repl = DimShuffle(in_op.inputs[0], new_order).out
#                     env.replace(r, repl)
#                     lift(repl)
#                     return
#                 elif isinstance(in_op, Broadcast):
#                     # DimShuffle(Broadcast(x, y)) => Broadcast(DimShuffle(x), DimShuffle(y))
#                     repl = Broadcast(in_op.scalar_opclass,
#                                      [DimShuffle(input, op.new_order).out for input in in_op.inputs],
#                                      in_op.inplace_pattern).out
#                     env.replace(r, repl)
#                     r = repl
#                     op = r.owner
#             for next_r in op.inputs:
#                 lift(next_r)

#         for output in env.outputs:
#             lift(output)

# lift_dimshuffle = DimShuffleLifter()





# def find_cliques(env, through_broadcast = False):
#     """
#     Usage: find_cliques(env, through_broadcast = False)

#     Returns a list of pairs where each pair contains a list
#     of inputs and a list of outputs such that Env(inputs, outputs)
#     contains nothing but Broadcast Ops.

#     If through_broadcast is False, the cliques will only be
#     allowed to broadcast over the inputs, which means, for
#     example, that vector operations will not be mixed with
#     matrix operations.
#     """

#     def seek_from(r):
#         # walks through the graph until it encounters a
#         # non-Broadcast operation or (if through_broadcast
#         # is False) a Result which needs to be broadcasted.
        
#         op = r.owner
#         if env.edge(r) \
#                 or not isinstance(op, Broadcast) \
#                 or len(op.outputs) > 1:
#             # todo: handle multiple-output broadcast ops
#             #       (needs to update the clique's outputs)
#             return None

#         ret = set()

#         if not through_broadcast:
#             # check each dimension over all the inputs - if the broadcastable
#             # fields are not all 0 or all 1 for a particular dimension, then
#             # broadcasting will be performed along it on the inputs where the
#             # value is 1 and we will stop.
#             if any(any(bc) and not all(bc)
#                    for bc in zip(*[input.broadcastable for input in op.inputs])):
#                 ret.update(op.inputs)
#                 return ret
        
#         for input in op.inputs:
#             res = seek_from(input)
#             if res is None:
#                 # input is a leaf of our search
#                 ret.add(input)
#             else:
#                 ret.update(res)
        
#         return ret
    
#     cliques = []

#     def find_cliques_helper(r):
#         if env.edge(r):
#             return
#         clique_inputs = seek_from(r)
#         if clique_inputs is None:
#             # Not in a clique, keep going
#             op = r.owner
#             if op is not None:
#                 for input in op.inputs:
#                     find_cliques_helper(input)
#         else:
#             # We found a clique, add it to the list and
#             # jump to the leaves.
#             cliques.append((clique_inputs, [r]))
#             for input in clique_inputs:
#                 find_cliques_helper(input)

#     for output in env.outputs:
#         find_cliques_helper(output)

#     # todo: merge the cliques if possible

#     return cliques


# class CliqueOptimizer(opt.Optimizer):
#     """
#     Usage: CliqueOptimizer(through_broadcast = False,
#                            scalar_optimizer = None,
#                            make_composite = False).optimize(env)

#     Finds cliques of Broadcast operations in the env and does either
#     or both of two things:
    
#     * Apply scalar_optimizer on the clique as if the clique was a
#       group of scalar operations. scalar_optimizer can be any optimization
#       which applies on scalars. If it is None, no optimization is done.
#     * Replace the clique with a single Op, optimized to perform the
#       computations properly. If make_composite is False, no such replacement
#       is done.

#     Note: it is recommended to run the lift_dimshuffle optimization before
#     this one.
#     """

#     def __init__(self, through_broadcast = False, scalar_optimizer = None, make_composite = False):
#         self.through_broadcast = through_broadcast
#         self.scalar_optimizer = scalar_optimizer
#         self.make_composite = make_composite

#     def apply(self, env):
#         if self.scalar_optimizer is None and not self.make_composite:
#             # there's nothing to do with the cliques...
#             return
        
#         cliques = find_cliques(env, self.through_broadcast)
#         opt = self.scalar_optimizer

#         def build_scalar_clique(r, env, equiv):
#             # Maps a clique of Broadcast Ops to a clique of Scalar Ops with the same
#             # structure and equivalent operations. equiv contains the mapping.
#             if r in equiv:
#                 return equiv[r]
#             op = r.owner
#             if env.edge(r):
#                 # For each leave we make a Scalar of the corresponding dtype
#                 s = scalar.Scalar(dtype = r.dtype)
#                 _r = r
#                 if isinstance(r.owner, DimShuffle) and all(x == 'x' for x in r.owner.new_order):
#                     _r = r.owner.inputs[0]
#                 if (getattr(r, 'constant', False) or getattr(_r, 'constant', False)) \
#                        and _r.broadcastable == ():
#                     # If we have a constant tensor we map it to a constant scalar.
#                     s.data = _r.data
#                     s.constant = True
#                 equiv[r] = s
#                 return s
#             s_op = op.scalar_opclass(*[build_scalar_clique(input, env, equiv) for input in op.inputs])
#             equiv[op] = s_op
#             for output, s_output in zip(op.outputs, s_op.outputs):
#                 equiv[output] = s_output
#             return equiv[r]

#         for c_in, c_out in cliques:
#             equiv = dict()
#             g = Env(c_in, c_out)
#             for output in c_out:
#                 build_scalar_clique(output, g, equiv)
#             s_g = Env([equiv[r] for r in g.inputs],
#                       [equiv[r] for r in g.outputs])
#             if opt is not None:
#                 equiv2 = dict() # reverse mapping, from Scalar Op to Tensor Op
#                 for k, v in equiv.items():
#                     equiv2[v] = k
#                 def transform(op, equiv):
#                     # We get a scalar op and we return an equivalent op on tensors.
#                     return Broadcast(op.__class__, [equiv[input] for input in op.inputs])
#                 s_g.add_feature(sync_to(env, equiv2, transform)) # Any change to s_g will now be transferred to g
#                 opt.optimize(s_g)
#             if self.make_composite:
#                 def follow_inplace(r):
#                     # Tries to find the earliest r2 in g such that r destroys r2
#                     # If no such r2 is found, returns None
#                     op = r.owner
#                     if op is None or r in g.inputs or r in g.orphans():
#                         return None
#                     assert isinstance(op, Broadcast)
#                     destroyed = op.destroy_map().get(r, None)
#                     if destroyed is None:
#                         return None
#                     else:
#                         r2 = destroyed[0]
#                         ret = follow_inplace(r2)
#                         if ret is None:
#                             return r2
#                         else:
#                             return ret
#                 inplace_pattern = {}
#                 for i, output in enumerate(g.outputs):
#                     destroyed = follow_inplace(output)
#                     if destroyed is not None and destroyed in g.inputs:
#                         # we transfer the inplace operation only if it is
#                         # an input that is destroyed
#                         inplace_pattern[i] = g.inputs.index(destroyed)
#                 C = scalar.composite(s_g.inputs, s_g.outputs)
#                 ec = Broadcast(C, g.inputs, inplace_pattern = inplace_pattern)
#                 env.replace_all(dict((o, eco) for o, eco in zip(c_out, ec.outputs)))


# def sync_to(target, equiv, transform):
#     """
#     Usage: sync_to(target, equiv, transform)
#     * target: an Env
#     * equiv: a dictionary that maps results and ops to results and ops
#              in target
#     * transform: a function that takes (op, equiv) as inputs and
#                  returns a new op.
    
#     Returns a Feature that can be added to an Env and mirrors all
#     modifications to that env with modifications to the target env.
#     """

#     class Synchronize(gof.Listener, gof.Constraint):

#         def __init__(self, source):
#             self.source = source
#             self.target = target
#             self.equiv = equiv
#             self.transform = transform
#             self.inconsistencies = []

#         def on_import(self, op1):
#             if op1 not in self.equiv:
#                 op2 = self.transform(op1, self.equiv)
#                 self.equiv[op1] = op2
#                 for o1, o2 in zip(op1.outputs, op2.outputs):
#                     self.equiv[o1] = o2

#         def on_prune(self, op1):
#             if op1 in self.equiv:
#                 op2 = self.equiv[op1]
#                 del self.equiv[op1]
#                 for o1, o2 in zip(op1.outputs, op2.outputs):
#                     del self.equiv[o1]

#         def on_rewire(self, clients1, r1, new_r1):
#             if (new_r1, r1) in self.inconsistencies:
#                 self.inconsistencies.remove((new_r1, r1))
#                 return
#             if not self.source.clients(r1):
#                 try:
#                     target.replace(self.equiv[r1], self.equiv[new_r1])
#                 except:
#                     self.inconsistencies.append((r1, new_r1))

#         def validate(self):
#             if self.inconsistencies:
#                 raise InconsistencyError("Could not synchronize when replacing the following pairs: %s" % self.inconsistencies)
#             return True

#     return Synchronize

