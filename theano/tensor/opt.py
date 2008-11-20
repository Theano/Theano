
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0


from .. import gof
from ..gof import opt
from elemwise import Elemwise, DimShuffle
from .. import scalar
import basic as T
import inplace as I
import numpy as N
import operator
import itertools
import sys
from .. import compile  #to register the optimizer built by this file


# Utilities

def out2in(*local_opts):
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order = 'out_to_in',
                             failure_callback = lambda exc,opt,pairs: None)

def in2out(*local_opts, **kwargs):
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order = 'in_to_out',
                             failure_callback = lambda exc,opt,pairs: None,
                             **kwargs)


# gemm: (d,a,b,c,s) -> d = d*s + a*dot(b,c)
# Transforms d -= a * dot(b, c) into gemm(d, -a, b, c, 1.0)
gemm_pattern_1 = gof.PatternSub((T.sub,
                                 'd',
                                 (T.mul,
                                  dict(pattern = (T.DimShuffle((), ['x', 'x'], inplace = True), 'a'),
                                       allow_multiple_clients = True),
                                  (T.dot, 'b', 'c'))),
                                (T.gemm, 'd', (T.neg, 'a'), 'b', 'c', T.constant(1.0)),
                                allow_multiple_clients = False)

# gemm: (d,a,b,c,s) -> d = d*s + a*dot(b,c)
# Transforms dot(a, b) into gemm(zeros(2)(hstack(shape(a)[:1], shape(b)[1:])), 1.0, a, b, 1.0)
# The construction of the 'gemm' node may fail if, for example, a and b are not both matrices.
dot_to_gemm = gof.PatternSub((T.dot, 'a', 'b'),
                             (T.gemm, (T.Zeros(2),
                                       (T.stack,
                                        (T.Subtensor([slice(0, 1)]), (T.shape, 'a')),
                                        (T.Subtensor([slice(1, 2)]), (T.shape, 'b')))),
                              T.constant(1.0), 'a', 'b', T.constant(1.0)),
                              allow_multiple_clients = False)



def _insert_inplace_optimizer(env):
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
                    new = Elemwise(
                        op.scalar_op.__class__(
                            scalar.transfer_type(*[inplace_pattern.get(i, None) for i in xrange(len(node.outputs))])),
                        inplace_pattern).make_node(*node.inputs)
                    env.replace_all_validate(zip(node.outputs, new.outputs))
                except Exception, e:
                    continue
                candidate_inputs.remove(candidate_input)
                node = new
                baseline = inplace_pattern
                break
insert_inplace_optimizer = gof.optimizer(_insert_inplace_optimizer)

inplace_optimizer = gof.InplaceOptimizer(
    gof.SeqOptimizer(out2in(gemm_pattern_1),
                     insert_inplace_optimizer,
                     failure_callback = gof.warn))
compile.optdb.register('inplace_opt', inplace_optimizer, 99, 'fast_run', 'inplace')


def register_canonicalize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['canonicalize'].register(name, lopt, 'fast_run', *tags)

def register_specialize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['specialize'].register(name, lopt, 'fast_run', *tags)

######################
# DimShuffle lifters #
######################

@gof.local_optimizer([None, None])
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

register_canonicalize(local_dimshuffle_lift)



#################
# Shape lifters #
#################

@gof.local_optimizer([T.shape, None])
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

register_canonicalize(local_shape_lift_elemwise)


@gof.local_optimizer([T.shape, None])
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

register_canonicalize(local_shape_lift_sum)


@gof.local_optimizer([T.shape, T.dot])
def local_shape_lift_dot(node):
    """
    shape(dot(a, b)) -> [shape(a)[0], shape(b)[1]]
    """
    if not opt.check_chain(node, T.shape, T.dot):
        return False
    a, b = node.inputs[0].owner.inputs
    return T.make_lvector.make_node(T.shape(a)[0], T.shape(b)[1]).outputs

register_canonicalize(local_shape_lift_dot)


# local_shape_lift = opt.LocalOptGroup(local_shape_lift_elemwise,
#                                      local_shape_lift_sum,
#                                      local_shape_lift_dot)


################
# Fill lifters #
################

def encompasses_broadcastable(b1, b2):
    """
    Returns True if the broadcastable patterns b1 and b2 are such that b2 is
    broadcasted to b1's shape and not the opposite.

    :param b1: the broadcastable attribute of a tensor type
    :param b2: the broadcastable attribute of a tensor type
    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2):]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))

def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]

@gof.local_optimizer([T.fill, None])
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
        return False# [filling]

    parent = model.owner
    if parent is None or not isinstance(parent, T.Elemwise):
        return False
    for input in parent.inputs:
        if input.type == model.type:
            return [T.fill(input, filling)]

    return False

register_canonicalize(local_fill_lift)


##################
# Subtensor opts #
##################


@gof.local_optimizer([None, None])
def local_subtensor_make_vector(node):
    """
    [a,b,c][0] -> a
    [a,b,c][0:2] -> [a,b]

    If the index or slice is constant.
    """
    if not opt.check_chain(node, T.Subtensor, T.MakeVector):
        return False
    
    joined_r = node.inputs[0]

    try: 
        #check that join is being used to join scalars
        veclen = T.join.vec_length(joined_r)
    except:
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

register_canonicalize(local_subtensor_make_vector)


##################
# Middleman cuts #
##################

@gof.local_optimizer([None, T.fill])
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

register_canonicalize(local_fill_cut)

register_canonicalize(gof.OpRemove(T.tensor_copy), name='remove_tensor_copy' )

@gof.local_optimizer([None, T.fill])
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

register_canonicalize(local_fill_sink)


################
# Canonization #
################

class Canonizer(gof.LocalOptimizer):
    """
    Simplification tool.

    Usage: Canonizer(main, inverse, reciprocal, calculate)
    
    * main: a suitable Op class that is commutative, associative and takes
            one to an arbitrary number of inputs, e.g. Add or Mul
    * inverse: an Op class such that inverse(main(x, y), y) == x
               e.g. Sub or Div
    * reciprocal: a function such that main(x, reciprocal(y)) == inverse(x, y)
                  e.g. Neg or Inv

    * calculate: function that takes a list of numpy.ndarray instances for
                 the numerator, another list for the denumerator, and calculates
                 inverse(main(*num), main(*denum)). It takes a keyword argument,
                 aslist. If True, the value should be returned as a list of one
                 element, unless the value is such that value = main(). In that
                 case, the return value should be an empty list.

    The result is a local_optimizer. It is best used with a TopoOptimizer in
    in_to_out order.

    Examples:
      T = theano.tensor
      add_canonizer = Canonizer(T.add, T.sub, T.neg, lambda n, d: sum(n) - sum(d))
      mul_canonizer = Canonizer(T.mul, T.div, T.inv, lambda n, d: prod(n) / prod(d))
    
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

    def __init__(self, main, inverse, reciprocal, calculate, use_reciprocal = True):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

    def tracks(self):
        #return [[None], [None, None], [None]*3, [None]*4, [None]*5]
        return [[self.main, None], [self.inverse, None], [self.reciprocal, None]]

    def get_num_denum(self, input):
        if input.owner is None or input.owner.op not in [self.main, self.inverse, self.reciprocal]:
            if input.owner and isinstance(input.owner.op, T.DimShuffle):
                return self.get_num_denum(input.owner.inputs[0])
            else:
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
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist = False)]
        if not ld:
            if ln == 1:
                if isinstance(num[0], gof.Result):
                    return num[0]
                else:
                    return T.as_tensor(num[0])
            else:
                return self.main(*num)
        return self.inverse(self.merge_num_denum(num, []),
                            self.merge_num_denum(denum, []))

    @classmethod
    def get_constant(cls, v):
        if isinstance(v, N.generic):
            return v
        if isinstance(v, gof.Constant):
            return v.data
        if v.owner and isinstance(v.owner.op, DimShuffle):
            return cls.get_constant(v.owner.inputs[0])
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
        if self.use_reciprocal or num:
            ct = self.calculate(numct, denumct, aslist = True)
        else:
            ct = [self.calculate(numct, denumct, aslist = False)]
#         if len(ct) and ncc == 1 and dcc == 0:
#             return orig_num, orig_denum
        if orig_num and len(numct) == 1 and ct and N.all(ct == self.get_constant(orig_num[0])):
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

        def same(x, y):
            return len(x) == len(y) and all(N.all(xe == ye) for xe, ye in zip(x, y))

        if not reorg and same(orig_num, num) and same(orig_denum, denum):
            return False

        new = self.merge_num_denum(num, denum)
        if new.type != out.type:
            #new = T.fill(out, new)
            new = T.fill(out, T.Elemwise(scalar.Identity(scalar.specific_out(getattr(scalar, out.type.dtype))))(new))
        return [new]

    def __str__(self):
        return getattr(self, 'name', 'Canonizer(%s, %s, %s)' % (self.main, self.inverse, self.reciprocal))


def mul_calculate(num, denum, aslist = False):
    v = reduce(N.multiply, num, 1.0) / reduce(N.multiply, denum, 1.0)
    if aslist:
        if N.all(v == 1):
            return []
        else:
            return [v]
    return v

local_mul_canonizer = Canonizer(T.mul, T.div, T.inv, mul_calculate, False)

@gof.local_optimizer([T.neg])
def local_neg_to_mul(node):
    if node.op == T.neg:
        return [-1 * node.inputs[0]]
    else:
        return False
register_canonicalize(local_neg_to_mul)

@gof.local_optimizer([T.mul])
def local_mul_to_neg(node):
    if node.op == T.mul and N.all(local_mul_canonizer.get_constant(node.inputs[0]) == -1.0):
        return [-local_mul_canonizer.merge_num_denum(node.inputs[1:], [])]
    else:
        return False
register_specialize(local_mul_to_neg)

@gof.local_optimizer([T.div])
def local_div_to_inv(node):
    if node.op == T.div and N.all(local_mul_canonizer.get_constant(node.inputs[0]) == 1.0):
        return [T.inv(local_mul_canonizer.merge_num_denum(node.inputs[1:], []))]
    else:
        return False
register_specialize(local_div_to_inv)

@gof.local_optimizer([T.inv])
def local_inv_canon(node):
    if node.op == T.inv:
        return [T.pow(node.inputs[0], -1.0)]
    else:
        return False
register_canonicalize(local_inv_canon)

@gof.local_optimizer([T.pow])
def local_pow_canonicalize(node):
    if node.op == T.pow:
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 1.0):
            return [T.fill(node.inputs[1], node.inputs[0])]
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 0.0):
            #extra fills here are to make sure the size of the output stays constant.
            return [T.fill(node.inputs[0], T.fill(node.inputs[1], 1.0))]
    else:
        return False
register_canonicalize(local_pow_canonicalize)

@gof.local_optimizer([T.pow])
def local_pow_specialize(node):
    #here, we are past the point of canonicalization, so we don't want to put in un-necessary fills.
    if node.op == T.pow:
        #the idea here is that we have pow(x, y)
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable, ysym.type.broadcastable):
            if N.all(y == 2.0):
                return [T.sqr(xsym)]
            if N.all(y == 1.0):
                return [xsym]
            if N.all(y == 0.0):
                return [T.fill(xsym, 1.0)]
            if N.all(y == 0.5):
                return [T.sqrt(xsym)]
            if N.all(y == -0.5):
                return [T.inv(T.sqrt(xsym))]
            if N.all(y == -1.0):
                return [T.inv(xsym)]
            if N.all(y == -2.0):
                return [T.inv(T.sqr(xsym))]
    else:
        return False
register_specialize(local_pow_specialize)

if 0: #TODO: replace this with a c version of any InplaceDimShuffle
    class _TransposeInplace(T.Op):
        view_map = {0: [0]}
        
        def make_node(self, input):
            return T.Apply(self, [input], 
                    [T.tensor(dtype = input.type.dtype,
                        broadcastable = reversed(input.type.broadcastable))])
        
        def perform(self, node, (x, ), (z, )):
            z[0] = x.T
        
        def c_code(self, node, name, (x, ), (z, ), sub):
            return """
            PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
            if (%(z)s) {
                Py_XDECREF(%(z)s);
            }
            %(z)s = transposed;
            """ % locals()

        def __str__(self):
            return "_TransposeInplace"
    _transpose_inplace = _TransposeInplace()

    @gof.local_optimizer([T.DimShuffle([False,False],[1,0],inplace=True)])
    def local_dimshuffle_transposeinplace(node):
        if node.op == T.DimShuffle([False,False],[1,0],inplace=True):
            return [_transpose_inplace(node.inputs[0])]
        return False
    register_specialize(local_dimshuffle_transposeinplace)

register_canonicalize(local_mul_canonizer, name = 'local_mul_canonizer')


# neg_to_mul = out2in(gof.LocalOptGroup(local_neg_to_mul))
# mul_to_neg = out2in(gof.LocalOptGroup(local_mul_to_neg))

mul_canonizer = in2out(gof.LocalOptGroup(local_mul_canonizer, local_fill_cut, local_fill_sink))



def add_calculate(num, denum, aslist = False):
    v = reduce(N.add, num, 0.0) - reduce(N.add, denum, 0.0)
    if aslist:
        if N.all(v == 0):
            return []
        else:
            return [v]
    return v

local_add_canonizer = Canonizer(T.add, T.sub, T.neg, add_calculate)
add_canonizer = in2out(gof.LocalOptGroup(local_add_canonizer, local_fill_cut, local_fill_sink))

register_canonicalize(local_add_canonizer, name = 'local_add_canonizer')


##################
# Distributivity #
##################


def distribute_greedy(pos_pairs, neg_pairs, num, denum, minscore = 0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    score = len(num) + div_cost * len(denum) # score is number of operations saved, higher is better
    new_pos_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n+num, d+denum) for (n, d) in pos_pairs]))
    new_neg_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n+num, d+denum) for (n, d) in neg_pairs]))
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs + new_neg_pairs):
        # We calculate how many operations we are saving with the new num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs

def attempt_distribution(factor, num, denum):
    # we try to insert each num and each denum in the factor
    # returns: changes?, new_factor, new_num, new_denum
    # if there are changes, new_num and new_denum contain all the numerators
    # and denumerators that could not be distributed in the factor
    pos, neg = local_add_canonizer.get_num_denum(factor)
    if len(pos) == 1 and not neg:
        return False, factor, num, denum
    pos_pairs = map(local_mul_canonizer.get_num_denum, pos)
    neg_pairs = map(local_mul_canonizer.get_num_denum, neg)
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs, neg_pairs, [n], [])
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs, neg_pairs, [], [d])
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return change, local_add_canonizer.merge_num_denum(
            list(itertools.starmap(local_mul_canonizer.merge_num_denum, pos_pairs)),
            list(itertools.starmap(local_mul_canonizer.merge_num_denum, neg_pairs))), num, denum

@gof.local_optimizer([T.mul, T.add, T.mul], [T.mul, T.sub, T.mul],
                     [T.mul, T.add, T.div], [T.mul, T.sub, T.div])
def local_greedy_distributor(node):
    """
    This optimization tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ((a/x + b/y) * x * y) --> a*y + b*x
    2. ((a/x + b) * x) --> a + b*x

    The following expressions are not simplified:
    3. ((a + b) * x) -/-> a*x + b*x

    This optimization aims to reduce computational cost. It may also
    increase numerical stability, e.g. when x and/or y tend to 0 in
    example 1.
    """
    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(candidate, num, denum)
        change |= _change
        if change:
            new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(candidate, denum, num)
        change |= _change
        if change:
            new_denum.append(candidate)

    if not change:
        return False

    new_num += num
    new_denum += denum

    return [local_mul_canonizer.merge_num_denum(new_num, new_denum)]

register_canonicalize(local_greedy_distributor)



@gof.local_optimizer([None])
def constant_folding(node):
    for input in node.inputs:
        if not isinstance(input, gof.Constant):
            return False
    storage = [[None] for output in node.outputs]
    node.op.perform(node, [x.data for x in node.inputs], storage)
    return [gof.Constant(output.type, s[0]) for s, output in zip(storage, node.outputs)]

register_canonicalize(constant_folding)






# def _math_optimizer():
#     pass_1 = in2out(local_fill_sink)
#     pass_2 = out2in(local_dimshuffle_lift, local_shape_lift, local_fill_lift)#, local_fill_cut)
#     pass_3 = out2in(local_subtensor_make_vector, local_fill_cut)
    
#     canonizer = in2out(local_add_canonizer,
#                        local_mul_canonizer,
#                        local_fill_sink)

#     pass_4 = out2in(local_greedy_distributor)

#     return gof.SeqOptimizer(pass_1,
#                             pass_2,
#                             pass_3,
#                             neg_to_mul,
#                             canonizer,
#                             pass_4,
#                             mul_to_neg)

# math_optimizer = _math_optimizer()






# compile.register_optimizer('math', 
#         gof.MergeOptMerge(
#             gof.PureThenInplaceOptimizer(
#                 math_optimizer,
#                 inplace_optimizer)))


# compile.register_mode('SANITY_CHECK', compile.Mode('c&py', 'math'))
# compile.register_mode('FAST_RUN', compile.Mode('c|py', 'math'))
# compile.register_mode('EXPENSIVE_OPTIMIZATIONS', compile.Mode('c|py', 'math'))


# @gof.local_optimizer
# def local_clique_fusion(node):
#     aaaaaaaaaaaaaaaaaaaaaaa










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

