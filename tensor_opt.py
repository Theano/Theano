
import gof
from gof import opt
from elemwise import Elemwise, DimShuffle
import scalar
import tensor as T
import numpy as N
import operator

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


@gof.optimizer
def inplace_optimizer(self, env):
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


######################
# DimShuffle lifters #
######################

@gof.local_optimizer
def local_dimshuffle_lift(node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.
    """
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

dimshuffle_lift = gof.TopoOptimizer(local_dimshuffle_lift, order = 'out_to_in')


#################
# Shape lifters #
#################

@gof.local_optimizer
def local_shape_lift_elemwise(node):
    """
    shape(elemwise_op(..., x, ...)) -> shape(x)

    Where x contains the maximal shape information.
    """
    if not opt.check_chain(node, T.shape, T.Elemwise):
        return False
    
    output = node.inputs[0]
    parent = output.owner

    for input in parent.inputs:
        if input.type.broadcastable == output.type.broadcastable:
            return T.shape.make_node(input).outputs

    return False

@gof.local_optimizer
def local_shape_lift_sum(node):
    """
    shape(sum{n}(x)) -> [shape(x)[0], ..., shape(x)[n-1], shape(x)[n+1], ...]
    """
    if not opt.check_chain(node, T.shape, T.Sum):
        return False

    input = node.inputs[0].owner.inputs[0]
    axis = node.inputs[0].owner.op.axis
    if axis is None:# or len(axis) != 1:
        axis = range(input.type.ndim)


    ish = T.shape(input)
    return T.make_lvector.make_node(*(ish[i] for i in xrange(input.type.ndim) if i not in axis)).outputs
#    return T.vertical_stack.make_node(ish[:axis], ish[axis+1:]).outputs

@gof.local_optimizer
def local_shape_lift_dot(node):
    """
    shape(dot(a, b)) -> [shape(a)[0], shape(b)[1]]
    """
    if not opt.check_chain(node, T.shape, T.dot):
        return False
    a, b = node.inputs[0].owner.inputs
    return T.make_lvector.make_node(T.shape(a)[0], T.shape(b)[1]).outputs

local_shape_lift = opt.LocalOptGroup(local_shape_lift_elemwise,
                                     local_shape_lift_sum,
                                     local_shape_lift_dot)


################
# Fill lifters #
################

def encompasses_broadcastable(b1, b2):
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2):]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))

def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]

@gof.local_optimizer
def local_fill_lift(node):
    """
    fill(f(a), b) -> fill(a, b)
    If a.type == f(a).type.

    fill(a, b) -> b
    If a.type == b.type.
    """
    if not opt.check_chain(node, T.fill):
        return False

    model, filling = node.inputs

    mb, fb = model.type.broadcastable, filling.type.broadcastable
    if model.type.dtype == filling.type.dtype and encompasses_broadcastable(fb, mb):
        return [filling]

    parent = model.owner
    if parent is None:
        return False
    for input in parent.inputs:
        if input.type == model.type:
            return [T.fill(input, filling)]

    return False


##################
# Subtensor opts #
##################


@gof.local_optimizer
def local_subtensor_make_vector(node):
    """
    [a,b,c][0] -> a
    [a,b,c][0:2] -> [a,b]

    If the index or slice is constant.
    """
    if not opt.check_chain(node, T.Subtensor, T.MakeVector):
        return False

    idxlist = node.op.idx_list
    if len(idxlist) != 1:
        return False
    idx = idxlist[0]
    if isinstance(idx, int):
        return [node.inputs[0].owner.inputs[idx]]
    try:
        return T.make_vector(*(node.owner.inputs[0].owner.inputs.__getslice__(idx)))
    except TypeError:
        return False
        

##################
# Middleman cuts #
##################

@gof.local_optimizer
def local_fill_cut(node):
    """
    f(fill(a,b), c) -> f(b, c)
    If c.type == a.type.
    """

    if not opt.check_chain(node, T.Elemwise):
        return False
    
    output = node.outputs[0]
    try:
        reference = [input
                     for input in node.inputs
                     if input.type == output.type and (not input.owner or input.owner.op != T.fill)][0]
    except IndexError:
        return False

    new_inputs = []
    for input in node.inputs:
        if opt.check_chain(input, T.fill):
            model, filling = input.owner.inputs
            if encompasses_broadcastable(reference.type.broadcastable,
                                         filling.type.broadcastable):
                new_inputs.append(filling)
                continue
        new_inputs.append(input)

    if new_inputs == node.inputs:
        return False
    return node.op.make_node(*new_inputs).outputs

@gof.local_optimizer
def local_fill_sink(node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(a, fill(c, f(b, d, e)))
    """
    if not (node.op and isinstance(node.op, T.Elemwise) and node.op != T.fill):
        return False
    models = []
    inputs = []
    for input in node.inputs:
        if input.owner and input.owner.op == T.fill:
            models.append(input.owner.inputs[0])
            inputs.append(input.owner.inputs[1])
        else:
            inputs.append(input)
    if inputs == node.inputs:
        return False
    c = node.op(*inputs)
    for model in models:
        c = T.fill(model, c)
    return [c]


################
# Canonization #
################

class Canonizer(gof.LocalOptimizer):

    def __init__(self, main, inverse, reciprocal, calculate):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate

    def get_num_denum(self, input):
        if input.owner is None or input.owner.op not in [self.main, self.inverse, self.reciprocal]:
            return [input], []
        num = []
        denum = []
        parent = input.owner
        pairs = [self.get_num_denum(input) for input in parent.inputs]
        if parent.op == self.main:
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return T.as_tensor(self.calculate([], []))
        if not ln:
            return self.reciprocal(self.merge_num_denum(denum, []))
        if not ld:
            if ln == 1:
                return num[0]
            else:
                return self.main(*num)
        return self.inverse(self.merge_num_denum(num, []),
                            self.merge_num_denum(denum, []))

    def get_constant(self, v):
        if isinstance(v, gof.Constant):
            return v.data
        if v.owner and isinstance(v.owner.op, DimShuffle):
            return self.get_constant(v.owner.inputs[0])
        return None

    def simplify(self, num, denum):
        return self.simplify_constants(*self.simplify_factors(num, denum))

    def simplify_factors(self, num, denum):
        for v in list(num):
            if v in denum:
                num.remove(v)
                denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum):
        num, denum = list(orig_num), list(orig_denum)
        numct, denumct = [], []
        ncc, dcc = 0, 0
        for v in orig_num:
            ct = self.get_constant(v)
            if ct is not None:
                ncc += 1
                num.remove(v)
                numct.append(ct)
        for v in orig_denum:
            ct = self.get_constant(v)
            if ct is not None:
                dcc += 1
                denum.remove(v)
                denumct.append(ct)
        ct = self.calculate(numct, denumct, aslist = True)
        if len(ct) and ncc == 1 and dcc == 0:
            return orig_num, orig_denum
        return ct + num, denum

    def transform(self, node):
        op = node.op
        inputs = node.inputs
        out = node.outputs[0]
        if op not in [self.main, self.inverse, self.reciprocal]:
            return False

        iops = set(input.owner.op for input in inputs if input.owner)
        reorg = False
        if op == self.main:
            reorg = len(iops.intersection([self.main, self.inverse, self.reciprocal])) != 0
        elif op == self.inverse:
            reorg = len(iops.intersection([self.inverse, self.reciprocal])) != 0
        elif op == self.reciprocal:
            reorg = len(iops.intersection([self.inverse, self.reciprocal])) != 0

        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = list(orig_num), list(orig_denum)
        num, denum = self.simplify(num, denum)

        if not reorg and orig_num == num and orig_denum == denum:
            return False

        new = self.merge_num_denum(num, denum)
        if new.type != out.type:
            new = T.fill(out, new)
        return [new]

def mul_calculate(num, denum, aslist = False):
    v = reduce(N.multiply, num, 1.0) / reduce(N.multiply, denum, 1.0)
    if aslist:
        if N.all(v == 1):
            return []
        else:
            return [v]
    return v

local_mul_canonizer = Canonizer(T.mul, T.div, T.inv, mul_calculate)
mul_canonizer = gof.TopoOptimizer(gof.LocalOptGroup(local_mul_canonizer, local_fill_sink), order = 'in_to_out')

def add_calculate(num, denum, aslist = False):
    v = reduce(N.add, num, 0.0) - reduce(N.add, denum, 0.0)
    if aslist:
        if N.all(v == 0):
            return []
        else:
            return [v]
    return v

local_add_canonizer = Canonizer(T.add, T.sub, T.neg, add_calculate)
add_canonizer = gof.TopoOptimizer(gof.LocalOptGroup(local_add_canonizer, local_fill_sink), order = 'in_to_out')


##################
# Distributivity #
##################


def distribute_greedy(pos_pairs, neg_pairs, num, denum, minscore = 0):
    score = len(num) + len(denum) # score is number of operations saved, higher is better
    new_pos_pairs = itertools.starmap(local_mul_canonizer.simplify,
                                      [(n+num, d+denum) for (n, d) in plus_pairs])
    new_neg_pairs = itertools.starmap(local_mul_canonizer.simplify,
                                      [(n+num, d+denum) for (n, d) in plus_pairs])
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs + new_neg_pairs):
        # We calculate how many operations we are saving with the new num and denum
        score += len(n) + len(d) - len(nn) - len(dd)
    if score < minscore:
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs


@gof.local_optimizer
def local_greedy_distributor(node):
    """
    The following expressions are simplified:
    ((a/x + b/y) * x * y) --> a*y + b*x
    ((a/x + b) * x) --> a + b*x

    The following expressions are not:
    ((a + b) * x) -X-> a*x + b*x
    """
    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False
    new_num = []
    for entry in num:
        pos, neg = local_add_canonizer.get_num_denum(entry)
        if len(pos) == 1 and not neg:
            new_num.append(entry)
            continue
    pos_pairs = map(local_mul_canonizer.get_num_denum, pos)
    neg_pairs = map(local_mul_canonizer.get_num_denum, neg)
            























# class Canonizer(gof.LocalOptimizer):

#     def __init__(self, main, inverse, reciprocal, simplify_constants, constant_op):
#         self.main = main
#         self.inverse = inverse
#         self.reciprocal = reciprocal
#         self.simplify_constants = simplify_constants
#         self.constant_op = constant_op

#     def get_num_denum(self, input, depth):
#         if depth == 0 or input.owner is None or input.owner.op not in [self.main, self.inverse, self.reciprocal]:
#             return [input], []
#         num = []
#         denum = []
#         parent = input.owner
#         pairs = [self.get_num_denum(input, depth - 1) for input in parent.inputs]
#         if parent.op == self.main:
#             num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
#             denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
#         elif parent.op == self.inverse:
#             num = pairs[0][0] + pairs[1][1]
#             denum = pairs[0][1] + pairs[1][0]
#         elif parent.op == self.reciprocal:
#             num = pairs[0][1]
#             denum = pairs[0][0]
#         return num, denum

#     def deep_num_denum(self, node):
#         op = node.op
#         if op == self.main:
#             num, denum = self.get_num_denum(inputs)
#         elif op == self.inverse:
#             assert len(inputs) == 2
#             n1, d1 = self.get_num_denum(inputs[:1])
#             n2, d2 = self.get_num_denum(inputs[1:])
#             num, denum = n1+d2, d1+n2
#         elif op == self.reciprocal:
#             denum, num = self.get_num_denum(inputs)
#         else:
#             num, denum = [node.outputs[0]], []
#         return num, denum

#     def get_num_denum(self, inputs):
#         num = []
#         denum = []
#         for input in inputs:
#             if input.owner is None:
#                 num.append(input)
#                 continue
#             parent = input.owner
#             if parent.op == self.main:
#                 num += parent.inputs
#             elif parent.op == self.inverse:
#                 num += parent.inputs[:1]
#                 denum += parent.inputs[1:]
#             elif parent.op == self.reciprocal:
#                 denum += parent.inputs
#             else:
#                 num.append(input)
#         return num, denum

#     def merge_num_denum(self, num, denum, outtype):
#         ln, ld = len(num), len(denum)
#         if not ln and not ld:
#             return outtype.filter(self.simplify_constants([], []))
#         if not ln:
#             return self.reciprocal(self.merge_num_denum(denum, [], outtype))
#         if not ld:
#             if ln == 1:
#                 return num[0]
#             else:
#                 return self.main(*num)
#         return self.inverse(self.merge_num_denum(num, [], outtype),
#                             self.merge_num_denum(denum, [], outtype))

#     def get_constant(self, v):
#         if isinstance(v, gof.Constant):
#             return v.data
#         if v.owner and isinstance(v.owner.op, DimShuffle):
#             return self.get_constant(v.owner.inputs[0])
#         return None

#     def simplify(self, num, denum):
#         numct, denumct = [], []
#         ncc, dcc = 0, 0
#         for v in list(num):
#             if v in denum:
#                 num.remove(v)
#                 denum.remove(v)
#                 continue
#             ct = self.get_constant(v)
#             if ct is not None:
#                 ncc += 1
#                 num.remove(v)
#                 numct.append(ct)
#         for v in list(denum):
#             ct = self.get_constant(v)
#             if ct is not None:
#                 dcc += 1
#                 denum.remove(v)
#                 denumct.append(ct)
#         ct = self.simplify_constants(numct, denumct)
#         if ct is None:
#             return ncc+dcc>0, None, num, denum
#         ctop = self.constant_op.get(ct)
#         if ctop is not None:
#             return True, ctop, num, denum
#         return not (ncc==1 and dcc==0), None, [ct]+num, denum

#     def transform(self, node):
#         op = node.op
#         inputs = node.inputs
#         if op == self.main:
#             num, denum = self.get_num_denum(inputs)
#         elif op == self.inverse:
#             assert len(inputs) == 2
#             n1, d1 = self.get_num_denum(inputs[:1])
#             n2, d2 = self.get_num_denum(inputs[1:])
#             num, denum = n1+d2, d1+n2
#         elif op == self.reciprocal:
#             denum, num = self.get_num_denum(inputs)
#         else:
#             return False
#         change, ctop, num2, denum2 = self.simplify(num, denum)
#         if change:
#             num, denum = num2, denum2
# #         print node, ct, num, denum
# #         ctop = ct != [] and self.constant_op.get(ct[0], None)
# #         if not ctop:
# #             num = ct + num
#         new = self.merge_num_denum(num, denum, node.outputs[0].type)
#         if ctop:
#             new = ctop(new)
#         print new.owner.op, op, new.owner.inputs, inputs
#         if new.owner and new.owner.op == op and all((new_input.owner   new.owner.inputs == inputs:
#             return False
#         return [new]































# @gof.local_optimizer
# def local_cut_middlemen(node):
#     op = node.op
#     if isinstance(op, Elemwise):
#         aaaaaaa




        



# # @gof.local_optimizer
# # def local_merge_mul(node):
# #     op = node.op
# #     if op != mul:
# #         return False
# #     num, denum = _get_num_denum(node.inputs)
# #     if num == node.inputs and denum == []:
# #         return False
# #     return _

































# class Lift(gof.LocalOptimizer):

#     def __init__(self, op, lifters, chooser):
#         self.op = op
#         self.lifters = lifters
#         self.chooser = chooser

#     def op_key(self):
#         return self.op

#     def transform(self, node):
#         if not node.op == self.op:
#             return False
#         candidates = [node.inputs[0]]
#         seen = set(candidates)
        
#         while True:
#             candidate = candidates.pop()
#             for lifter in self.lifters:
#                 new_candidates = lifter(candidate)
#                 if not new_candidates:
#                     break
#             else:
#                 candidates.append(candidate)
                

#         new_op = self.op(self.chooser(candidates))
#         return new_op




# class Canonizer(gof.Optimizer):
#     """
#     Simplification tool.

#     Usage: Canonizer(main, inverse, reciprocal, mainfn, invfn, recfn, transform)
    
#     * main: a suitable Op class that is commutative, associative and takes
#             one to an arbitrary number of inputs, e.g. Add or Mul
#     * inverse: an Op class such that inverse(main(x, y), y) == x
#                e.g. Sub or Div
#     * reciprocal: a function such that main(x, reciprocal(y)) == inverse(x, y)
#                   e.g. Neg or Inv

#     * mainfn, invfn, recfn: functions that behave just like the previous three
#                             Ops, but on true scalars (e.g. their impl)

#     * transform: a function that maps (numerator, denominatur) where numerator
#                  and denominator are lists of Result instances, to new lists
#                  where further simplifications may have been applied.

#     Examples:
#       add_canonizer = Canonizer(Add, Sub, Neg, lambda *inputs: sum(inputs), ...)
#       mul_canonizer = Canonizer(Mul, Div, Inv, lambda *inputs: product(inputs), ...)
    
#     Examples of optimizations mul_canonizer can perform:
#       x / x -> 1
#       (x * y) / x -> y
#       x / y / x -> 1 / y
#       x / y / z -> x / (y * z)
#       x / (y / z) -> (x * z) / y
#       (a / b) * (b / c) * (c / d) -> a / d
#       (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
#       2 * x / 2 -> x
#     """

#     def __init__(self, main, inverse, reciprocal, mainfn, invfn, recfn, transform = None):
#         self.main = main
#         self.inverse = inverse
#         self.reciprocal = reciprocal
#         self.mainfn = mainfn
#         self.invfn = invfn
#         self.recfn = recfn
#         self.neutral = mainfn()
#         self.transform = transform

#     def apply(self, env):

#         def edge(r):
#             return r.owner is None
#         def follow(r):
#             return None if r.owner is None else r.owner.inputs

#         def canonize(r):
            
#             next = follow(r)
#             if next is None:
#                 return
            
#             def flatten(r, nclients_check = True):
#                 # Collapses a tree of main/inverse/reciprocal Ops (aka Mul/Div/Inv or Add/Sub/Neg)
#                 # into a list of numerators and a list of denominators
#                 # e.g. (x*(1/y))*(x/(z/a)) aka Mul(Mul(x, (Inv, y)), Div(x, Div(z, a))) -> [x, x, a], [z, y]

#                 if edge(r):
#                     return [r], []
#                 node = r.owner
#                 op = node.op
                
#                 results = [r2.type == r.type and flatten(r2) or ([r2], []) for r2 in node.inputs]
#                 if op == self.main and (not nclients_check or env.nclients(r) == 1):
#                     nums = [x[0] for x in results]
#                     denums = [x[1] for x in results]
#                 elif op == self.inverse and (not nclients_check or env.nclients(r) == 1):
#                     # num, denum of the second argument are added to the denum, num respectively
#                     nums = [results[0][0], results[1][1]]
#                     denums = [results[0][1], results[1][0]]
#                 elif op == self.reciprocal and (not nclients_check or env.nclients(r) == 1):
#                     # num, denum of the sole argument are added to the denum, num respectively
#                     nums = [results[0][1]]
#                     denums = [results[0][0]]
#                 else:
#                     return [r], []

#                 return reduce(list.__add__, nums), reduce(list.__add__, denums)

#             num, denum = flatten(r, False)

#             if (num, denum) == ([r], []):
#                 for input in (follow(r) or []):
#                     canonize(input)
#                 return

#             # Terms that are both in the num and denum lists cancel each other
#             for d in list(denum):
#                 if d in list(num):
#                     # list.remove only removes the element once
#                     num.remove(d)
#                     denum.remove(d)

#             # We identify the constants in num and denum
#             numct, num = gof.utils.partition(lambda factor: isinstance(factor, gof.Constant) and factor.data is not None, num)
#             denumct, denum = gof.utils.partition(lambda factor: isinstance(factor, gof.Constant) and factor.data is not None, denum)

#             #print numct, num
#             #print denumct, denum
#             print num, denum

#             # All constants in num and denum are combined into a single constant which we add to num (unless it's a neutral constant)
#             v = self.invfn(self.mainfn(*[x.data for x in numct]), self.mainfn(*[x.data for x in denumct]))
#             if v != self.neutral:
#                 num.insert(0, C(v))

#             # We optimize the num and denum lists further if requested
#             if self.transform is not None:
#                 num, denum = self.transform(env, num, denum)

#             def make(factors):
#                 # Combines the factors using self.main (aka Mul) depending
#                 # on the number of elements.
#                 n = len(factors)
#                 if n == 0:
#                     return None
#                 elif n == 1:
#                     return factors[0]
#                 else:
#                     return self.main(*factors)

#             numr, denumr = make(num), make(denum)
            
#             if numr is None:
#                 if denumr is None:
#                     # Everything cancelled each other so we're left with
#                     # the neutral element.
#                     new_r = gof.Constant(r.type, self.neutral)
#                 else:
#                     # There's no numerator so we use reciprocal
#                     new_r = self.reciprocal(denumr)
#             else:
#                 if denumr is None:
#                     new_r = numr
#                 else:
#                     new_r = self.inverse(numr, denumr)

#             # Hopefully this won't complain!
#             env.replace(r, new_r)

#             for factor in num + denum:
#                 canonize(factor)

#         for output in env.outputs:
#             canonize(output)


# _mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
# _divfn = lambda x, y: x / y
# _invfn = lambda x: 1 / x
# mul_canonizer = Canonizer(T.mul, T.div, T.inv, _mulfn, _divfn, _invfn)




















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

