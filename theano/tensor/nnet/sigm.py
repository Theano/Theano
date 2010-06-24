"""Ops and optimizations: sigmoid, softplus

These functions implement special cases of exp and log to improve numerical stability.
"""
import numpy

from theano import gof
from theano import scalar
from theano import printing
from theano.tensor import basic as tensor
from theano.printing import pprint, debugprint
from theano.tensor import elemwise
from theano.tensor import opt
from theano.compile import optdb


############
#
# SCALAR OPS
#

class ScalarSigmoid(scalar.UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return 1.0 
        return 1.0 / (1.0 + numpy.exp(-x))
    def impl(self, x):
        return ScalarSigmoid.st_impl(x)
    def grad(self, (x,), (gz,)):
        y = scalar_sigmoid(x)
        return [gz * y * (1.0 - y)]
    def c_code(self, node, name, (x,), (z,), sub):
        if node.inputs[0].type == scalar.float32:
            # These constants were obtained by looking at the output of python commands like:
            #  for i in xrange(750):
            #      print i, repr( theano._asarray(1.0, dtype=dt) / (theano._asarray(1.0, dtype=dt) + numpy.exp(-theano._asarray([i,-i], dtype=dt))))
            # the boundary checks prevent us from generating inf
            return """%(z)s = %(x)s < -88.0f ? 0.0 : %(x)s > 15.0f ? 1.0f : 1.0f /(1.0f + exp(-%(x)s));""" % locals()
        elif node.inputs[0].type == scalar.float64:
            return """%(z)s = %(x)s < -709.0 ? 0.0 : %(x)s > 19.0 ? 1.0 : 1.0 /(1.0+exp(-%(x)s));""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')
    def c_code_cache_version(self):
        v = super(ScalarSigmoid, self).c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v
scalar_sigmoid = ScalarSigmoid(scalar.upgrade_to_float, name='scalar_sigmoid')
sigmoid = elemwise.Elemwise(scalar_sigmoid, name='sigmoid')

sigmoid_inplace = elemwise.Elemwise(
        ScalarSigmoid(scalar.transfer_type(0)),
        inplace_pattern={0:0},
        name='sigmoid_inplace',
        )

pprint.assign(sigmoid, printing.FunctionPrinter('sigmoid'))


class ScalarSoftplus(scalar.UnaryScalarOp):
    @staticmethod
    def static_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return x
        return numpy.log1p(numpy.exp(x))
    def impl(self, x):
        return ScalarSoftplus.static_impl(x)
    def grad(self, (x,), (gz,)):
        return [gz * scalar_sigmoid(x)]
    def c_code(self, node, name, (x,), (z,), sub):
        if node.inputs[0].type == scalar.float32:
            # These constants were obtained by looking at the output of python commands like:
            #  for i in xrange(750):
            #      print i, repr( numpy.log1p(numpy.exp(theano._asarray([i,-i], dtype=dt))))
            # the boundary checks prevent us from generating inf
            return """%(z)s = %(x)s < -103.0f ? 0.0 : %(x)s > 14.0f ? %(x)s : log1p(exp(%(x)s));""" % locals()
        elif node.inputs[0].type == scalar.float64:
            return """%(z)s = %(x)s < -745.0 ? 0.0 : %(x)s > 16.0 ? %(x)s : log1p(exp(%(x)s));""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')
    def c_code_cache_version(self):
        v = super(ScalarSoftplus, self).c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v
scalar_softplus = ScalarSoftplus(scalar.upgrade_to_float, name='scalar_softplus')
softplus = elemwise.Elemwise(scalar_softplus, name='softplus')

pprint.assign(softplus, printing.FunctionPrinter('softplus'))

def _skip_mul_1(r):
    if r.owner and r.owner.op == tensor.mul:
        not_is_1 = [i for i in r.owner.inputs if not _is_1(i) ]
        if len(not_is_1)==1:
            return not_is_1[0]

logsigm_to_softplus = gof.PatternSub(
    (tensor.log, (sigmoid, 'x')),
    (tensor.neg, (softplus, (tensor.neg, 'x'))),
    allow_multiple_clients = True,
    skip_identities_fn=_skip_mul_1)

def _is_1(expr):
    """rtype bool. True iff expr is a constant close to 1
    """
    try:
        v = opt.get_constant_value(expr)
        return numpy.allclose(v, 1)
    except TypeError:
        return False

log1msigm_to_softplus = gof.PatternSub(
    (tensor.log, 
        (tensor.sub,
            dict(pattern='y', constraint = _is_1),
            (sigmoid, 'x'))),
    (tensor.neg, (softplus, 'x')),
    allow_multiple_clients = True,
    skip_identities_fn=_skip_mul_1)

log1pexp_to_softplus = gof.PatternSub(
    (tensor.log1p, 
     (tensor.exp, 'x')),
    (softplus, 'x'),
    allow_multiple_clients = True)

opt.register_stabilize(logsigm_to_softplus, name = 'logsigm_to_softplus')
opt.register_stabilize(log1msigm_to_softplus, name = 'log1msigm_to_softplus')
opt.register_stabilize(log1pexp_to_softplus, name = 'log1pexp_to_softplus')

def is_1pexp(t):
    # if t is of form (1+exp(x)), return x
    # else return None
    if t.owner and t.owner.op == tensor.add:
        scalars, scalar_inputs, nonconsts = \
                opt.scalarconsts_rest(t.owner.inputs)
        # scalar_inputs are potentially dimshuffled and fill'd scalars
        if len(nonconsts) == 1:
            maybe_exp = nonconsts[0]
            if maybe_exp.owner and maybe_exp.owner.op == tensor.exp:
                return False, maybe_exp.owner.inputs[0]
    return None

def is_exp(t):
    # if t is of form (exp(x)) then return x
    # else return None
    neg = False
    if t.owner and t.owner.op == tensor.neg:
        t = t.owner.inputs[0]
        neg = True
    if t.owner and t.owner.op == tensor.exp:
        return neg, t.owner.inputs[0]

def partition_num_or_denom(r, f):
    if r.owner and r.owner.op == tensor.mul:
        a = r.owner.inputs
    else:
        a = [r]

    # ugly 2.4-compatible thing
    f_terms = []
    neg = False
    rest = []
    for t in a:
        f_t = f(t)
        if f_t is None:
            rest.append(t)
        else:
            neg_t, f_t = f_t
            f_terms.append(f_t)
            neg ^= neg_t #bit flip if neg_t is true
    return f_terms, rest, neg


@opt.register_stabilize
@gof.local_optimizer([tensor.true_div])
def local_exp_over_1_plus_exp(node):
    """exp(x)/(1+exp(x)) -> sigm(x)
    c/(1+exp(x)) -> c*sigm(-x)
    """
    # this optimization should be done for numerical stability
    # so we don't care to check client counts
    if node.op == tensor.true_div:

        #find all the exp() terms in the numerator
        num, denom = node.inputs
        num_exp_x, num_rest, num_neg = partition_num_or_denom(num, is_exp)
        denom_1pexp, denom_rest, denom_neg = partition_num_or_denom(denom, is_1pexp)

        sigmoids = []
        for t in denom_1pexp:
            if t in num_exp_x:
                # case: exp(x) /(1+exp(x))
                sigmoids.append(sigmoid(t))
                del num_exp_x[num_exp_x.index(t)]
            else:
                # case: 1/(1+exp(x))
                sigmoids.append(sigmoid(-t))

        if not sigmoids: # we didn't find any.  abort
            return
        # put the new numerator together
        new_num = sigmoids + [tensor.exp(t) for t in num_exp_x] + num_rest
        if len(new_num) == 1:
            new_num = new_num[0]
        else:
            new_num = tensor.mul(*new_num)

        if num_neg ^ denom_neg:
            new_num = -new_num

        if len(denom_rest) == 0:
            return [new_num]
        elif len(denom_rest) == 1:
            return [new_num / denom_rest[0]]
        else:
            return [new_num / tensor.mul(*denom_rest)]

@opt.register_stabilize
@gof.local_optimizer([tensor.mul])
def local_sigm_times_exp(node):
    """
    exp(x)*sigm(-x) -> -sigm(x)
    """
    # this is a numerical stability thing, so we dont check clients
    if node.op == tensor.mul:
        exp_x = []
        exp_minus_x = []
        sigm_x = []
        sigm_minus_x = []
        other = []
        neg = False
        for i in node.inputs:
            while i.owner and i.owner.op == tensor.neg:
                neg ^= True
                i = i.owner.inputs[0]
            if i.owner and i.owner.op == tensor.exp:
                exp_arg = i.owner.inputs[0]
                if exp_arg.owner and exp_arg.owner.op == tensor.neg:
                    exp_minus_x.append(exp_arg.owner.inputs[0])
                else:
                    exp_x.append(exp_arg)
            elif i.owner and i.owner.op == sigmoid:
                sigm_arg = i.owner.inputs[0]
                if sigm_arg.owner and sigm_arg.owner.op == tensor.neg:
                    sigm_minus_x.append(sigm_arg.owner.inputs[0])
                else:
                    sigm_x.append(sigm_arg)
            else:
                other.append(i)

        # remove matched pairs in exp_x and sigm_minus_x
        did_something = False
        for i in exp_x:
            if i in sigm_minus_x:
                del sigm_minus_x[sigm_minus_x.index(i)]
                other.append(sigmoid(i))
                did_something = True
            else:
                other.append(i)

        # remove matched pairs in exp_minus_x and sigm_x
        for i in exp_minus_x:
            if i in sigm_x:
                del sigm_x[sigm_x.index(i)]
                other.append(sigm(-i))
                did_something = True
            else:
                other.append(i)
        if did_something:
            terms = other + [sigmoid(x) for x in sigm_x] \
                    + [sigmoid(-x) for x in sigm_minus_x]
            if len(terms)>1:
                rval = tensor.mul(*terms)
            else:
                rval = terms[0]
            
            if neg:
                return [-rval]
            else:
                return [rval]


@opt.register_stabilize
@gof.local_optimizer([tensor.inv])
def local_inv_1_plus_exp(node):
    """
    1/(1+exp(x)) -> sigm(-x)
    """
    # this optimization should be done for numerical stability
    # so we don't care to check client counts
    if node.op == tensor.inv:
        inv_arg = node.inputs[0]
        if inv_arg.owner and inv_arg.owner.op == tensor.add:
            scalars, scalar_inputs, nonconsts = \
                    opt.scalarconsts_rest(inv_arg.owner.inputs)
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if len(nonconsts) == 1:
                if nonconsts[0].owner and nonconsts[0].owner.op == tensor.exp:
                    if scalars and numpy.allclose(numpy.sum(scalars), 1):
                        return opt._fill_chain(
                                sigmoid(tensor.neg(nonconsts[0].owner.inputs[0])),
                                scalar_inputs)

# Registration is below, and conditional.
@gof.local_optimizer([tensor.sub])
def local_1msigmoid(node):
    """
    1-sigm(x) -> sigm(-x)
    """
    if node.op == tensor.sub:
        sub_l, sub_r = node.inputs
        if len(sub_r.clients) > 1:
            return # graph is using both sigm and 1-sigm
        if sub_r.owner and sub_r.owner.op == sigmoid:
            try:
                val_l = opt.get_constant_value(sub_l)
            except Exception, e:
                return
            if numpy.allclose(numpy.sum(val_l), 1):
                return [sigmoid(-sub_r.owner.inputs[0])]

register_local_1msigmoid = False
# This is False because the Stabilize pattern above 
# is looking for 1-sigm.  Also Canonizer turns neg into *(-1) and so 
# this optimization might set off an unwanted chain of things.
# OTH - this transformation can be seen as pushing normal arithmetic either  below or above the
# sigmoidal nonlinearity... so if the canonicalized form had anything to say about that then it
# would be a consideration... anyway leaving False for now.
              
if register_local_1msigmoid:
    opt.register_canonicalize(local_1msigmoid)

if 0:
    # This code is if'd out because it is not complete,
    # and it isn't obviously a good idea anyway.
    # The motivation here was to identify the last exp() node
    # in the SciPy2010 article, which was not optimized away at the time of publication,
    # so the example is actually not numerically stable, even though it should be.
    @opt.register_stabilize
    @gof.local_optimizer([tensor.mul])
    def local_sigm_gest(node):
        print "CANONICALIZE"
        print sigm_canonicalize(node)

    def sigm_canonicalize(node):
        add = tensor.add
        mul = tensor.mul
        div = tensor.true_div

        if node.op == tensor.add:
            rval = []
            for i in node.inputs:
                rval += sigm_canonicalize(i)
            return rval
        if node.op == tensor.mul:
            rval = sigm_canonicalize(node.inputs[0])
            for i in node.inputs[1:]:
                old_rval = rval
                rval = []
                for t1 in sigm_canonicalize(i):
                    for t0 in old_rval:
                        assert t1.owner.op == div
                        assert t0.owner.op == div
                        t0top, t0bot = t0.owner.inputs
                        t1top, t1bot = t1.owner.inputs
                        rval.append(div(mul(*(t0top+t1top)), mul(*(t0bot+t1bot))))

                        if len(rval) > 100:
                            # This loop can be exponentially long.
                            # aborting
                            return []
        elif len(node.outputs)>1:
            return []
        else:
            return [node.outputs[0]]

