"""Ops and optimizations: sigmoid, softplus

These functions implement special cases of exp and log to improve numerical stability.
"""

import warnings
from itertools import imap

import numpy

import theano
from theano import config, gof, printing, scalar
from theano.compile import optdb
from theano.configparser import AddConfigVar, BoolParam
from theano.printing import pprint, debugprint
from theano.tensor import basic as tensor
from theano.tensor import elemwise, opt, NotScalarConstantError


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

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        y = scalar_sigmoid(x)
        rval = gz * y * (1.0 - y)

        assert rval.type.dtype.find('float') != -1

        return [rval]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
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
        inplace_pattern={0: 0},
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

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * scalar_sigmoid(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
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
scalar_softplus = ScalarSoftplus(scalar.upgrade_to_float, name=                                                                                                                                                                                                        'scalar_softplus')
softplus = elemwise.Elemwise(scalar_softplus, name='softplus')

pprint.assign(softplus, printing.FunctionPrinter('softplus'))


def _skip_mul_1(r):
    if r.owner and r.owner.op == tensor.mul:
        not_is_1 = [i for i in r.owner.inputs if not _is_1(i)]
        if len(not_is_1) == 1:
            return not_is_1[0]

logsigm_to_softplus = gof.PatternSub(
    (tensor.log, (sigmoid, 'x')),
    (tensor.neg, (softplus, (tensor.neg, 'x'))),
    allow_multiple_clients=True,
    skip_identities_fn=_skip_mul_1)


def _is_1(expr):
    """rtype bool. True iff expr is a constant close to 1
    """
    try:
        v = opt.get_scalar_constant_value(expr)
        return numpy.allclose(v, 1)
    except tensor.NotScalarConstantError:
        return False

log1msigm_to_softplus = gof.PatternSub(
    (tensor.log,
        (tensor.sub,
            dict(pattern='y', constraint=_is_1),
            (sigmoid, 'x'))),
    (tensor.neg, (softplus, 'x')),
    allow_multiple_clients=True,
    skip_identities_fn=_skip_mul_1)

log1pexp_to_softplus = gof.PatternSub(
    (tensor.log1p,
     (tensor.exp, 'x')),
    (softplus, 'x'),
    allow_multiple_clients=True)

opt.register_stabilize(logsigm_to_softplus, name='logsigm_to_softplus')
opt.register_stabilize(log1msigm_to_softplus, name='log1msigm_to_softplus')
opt.register_stabilize(log1pexp_to_softplus, name='log1pexp_to_softplus')


def is_1pexp(t):
    """
    If 't' is of the form (1+exp(x)), return (False, x).
    Else return None.
    """
    if t.owner and t.owner.op == tensor.add:
        scalars, scalar_inputs, nonconsts = \
                opt.scalarconsts_rest(t.owner.inputs)
        # scalar_inputs are potentially dimshuffled and fill'd scalars
        if len(nonconsts) == 1:
            maybe_exp = nonconsts[0]
            if maybe_exp.owner and maybe_exp.owner.op == tensor.exp:
                # Verify that the constant terms sum to 1.
                if scalars:
                    scal_sum = scalars[0]
                    for s in scalars[1:]:
                        scal_sum = scal_sum + s
                    if numpy.allclose(scal_sum, 1):
                        return False, maybe_exp.owner.inputs[0]
                # Before 7987b51 there used to be a bug where *any* constant
                # was considered as if it was equal to 1, and thus this
                # function would incorrectly identify it as (1 + exp(x)).
                if config.warn.identify_1pexp_bug:
                    warnings.warn(
                        'Although your current code is fine, please note that '
                        'Theano versions prior to 0.5 (more specifically, '
                        'prior to commit 7987b51 on 2011-12-18) may have '
                        'yielded an incorrect result. To remove this warning, '
                        'either set the `warn.identify_1pexp_bug` config '
                        'option to False, or `warn.ignore_bug_before` to at '
                        'least \'0.4.1\'.')
    return None


AddConfigVar('warn.identify_1pexp_bug',
        'Warn if Theano versions prior to 7987b51 (2011-12-18) could have '
        'yielded a wrong result due to a bug in the is_1pexp function',
        BoolParam(theano.configdefaults.warn_default('0.4.1')),
        in_c_key=False)


def is_exp(var):
    """
    Match a variable with either of the `exp(x)` or `-exp(x)` patterns.

    :param var: The Variable to analyze.

    :return: A pair (b, x) with `b` a boolean set to True if `var` is of the
    form `-exp(x)` and False if `var` is of the form `exp(x)`. If `var` cannot
    be cast into either form, then return `None`.
    """
    neg = False
    neg_info = is_neg(var)
    if neg_info is not None:
        neg = True
        var = neg_info
    if var.owner and var.owner.op == tensor.exp:
        return neg, var.owner.inputs[0]


def is_mul(var):
    """
    Match a variable with `x * y * z * ...`.

    :param var: The Variable to analyze.

    :return: A list [x, y, z, ...] if `var` is of the form `x * y * z * ...`,
    or None if `var` cannot be cast into this form.
    """
    if var.owner and var.owner.op == tensor.mul:
        return var.owner.inputs
    else:
        return None


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
            neg ^= neg_t  # bit flip if neg_t is true
    return f_terms, rest, neg


def is_neg(var):
    """
    Match a variable with the `-x` pattern.

    :param var: The Variable to analyze.

    :return: `x` if `var` is of the form `-x`, or None otherwise.
    """
    apply = var.owner
    if not apply:
        return None
    # First match against `tensor.neg`.
    if apply.op == tensor.neg:
        return apply.inputs[0]
    # Then match against a multiplication by -1.
    if apply.op == tensor.mul and len(apply.inputs) >= 2:
        for idx, mul_input in enumerate(apply.inputs):
            try:
                constant = opt.get_scalar_constant_value(mul_input)
                is_minus_1 = numpy.allclose(constant, -1)
            except NotScalarConstantError:
                is_minus_1 = False
            if is_minus_1:
                # Found a multiplication by -1.
                if len(apply.inputs) == 2:
                    # Only return the other input.
                    return apply.inputs[1 - idx]
                else:
                    # Return the multiplication of all other inputs.
                    return tensor.mul(*(apply.inputs[0:idx] +
                                        apply.inputs[idx + 1:]))
    # No match.
    return None


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
        denom_1pexp, denom_rest, \
            denom_neg = partition_num_or_denom(denom, is_1pexp)

        sigmoids = []
        for t in denom_1pexp:
            if t in num_exp_x:
                # case: exp(x) /(1+exp(x))
                sigmoids.append(sigmoid(t))
                del num_exp_x[num_exp_x.index(t)]
            else:
                # case: 1/(1+exp(x))
                sigmoids.append(sigmoid(-t))

        if not sigmoids:  # we didn't find any.  abort
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


def parse_mul_tree(root):
    """
    Parse a tree of multiplications starting at the given root.

    :param root: The variable at the root of the tree.

    :return: A tree where each non-leaf node corresponds to a multiplication
    in the computation of `root`, represented by the list of its inputs. Each
    input is a pair [n, x] with `n` a boolean value indicating whether
    sub-tree `x` should be negated.

    Examples:
        x * y               -> [False, [[False, x], [False, y]]]
        -(x * y)            -> [True, [[False, x], [False, y]]]
        -x * y              -> [False, [[True, x], [False, y]]]
        -x                  -> [True, x]
        (x * y) * -z        -> [False, [[False, [[False, x], [False, y]]],
                                        [True, z]]]
    """
    # Is it a multiplication?
    mul_info = is_mul(root)
    if mul_info is None:
        # Is it a negation?
        neg_info = is_neg(root)
        if neg_info is None:
            # Keep the root "as is".
            return [False, root]
        else:
            # Recurse, inverting the negation.
            neg, sub_tree = parse_mul_tree(neg_info)
            return [not neg, sub_tree]
    else:
        # Recurse into inputs.
        return [False, map(parse_mul_tree, mul_info)]


def replace_leaf(arg, leaves, new_leaves, op, neg):
    """
    Attempts to replace a leaf of a multiplication tree.

    We search for a leaf in `leaves` whose argument is `arg`, and if we find
    one, we remove it from `leaves` and add to `new_leaves` a leaf with
    argument `arg` and variable `op(arg)`.

    :param arg: The argument of the leaf we are looking for.

    :param leaves: List of leaves to look into. Each leaf should be a pair
    (x, l) with `x` the argument of the Op found in the leaf, and `l` the
    actual leaf as found in a multiplication tree output by `parse_mul_tree`
    (i.e. a pair [boolean, variable]).

    :param new_leaves: If a replacement occurred, then the leaf is removed from
    `leaves` and added to the list `new_leaves` (after being modified by `op`).

    :param op: A function that, when applied to `arg`, returns the Variable
    we want to replace the original leaf variable with.

    :param neg: If True, then the boolean value associated to the leaf should
    be swapped. If False, then this value should remain unchanged.

    :return: True if a replacement occurred, or False otherwise.
    """
    for idx, x in enumerate(leaves):
        if x[0] == arg:
            x[1][0] ^= neg
            x[1][1] = op(arg)
            leaves.pop(idx)
            new_leaves.append(x)
            return True
    return False


def simplify_mul(tree):
    """
    Simplify a multiplication tree.

    :param tree: A multiplication tree (as output by `parse_mul_tree`).

    :return: A multiplication tree computing the same output as `tree` but
    without useless multiplications by 1 nor -1 (identified by leaves of the
    form [False, None] or [True, None] respectively). Useless multiplications
    (with less than two inputs) are also removed from the tree.
    """
    neg, inputs = tree
    if isinstance(inputs, list):
        # Recurse through inputs.
        s_inputs = []
        for s_i in imap(simplify_mul, inputs):
            if s_i[1] is None:
                # Multiplication by +/-1.
                neg ^= s_i[0]
            else:
                s_inputs.append(s_i)
        if not s_inputs:
            # The multiplication is empty.
            rval = [neg, None]
        elif len(s_inputs) == 1:
            # The multiplication has a single input.
            s_inputs[0][0] ^= neg
            rval = s_inputs[0]
        else:
            rval = [neg, s_inputs]
    else:
        rval = tree
    #print 'simplify_mul: %s -> %s' % (tree, rval)
    return rval


def compute_mul(tree):
    """
    Compute the Variable that is the output of a multiplication tree.

    This is the inverse of the operation performed by `parse_mul_tree`, i.e.
        compute_mul(parse_mul_tree(tree)) == tree

    :param tree: A multiplication tree (as output by `parse_mul_tree`).

    :return: A Variable that computes the multiplication represented by the
    tree.
    """
    neg, inputs = tree
    if inputs is None:
        raise AssertionError(
            'Function `compute_mul` found a missing leaf, did you forget to '
            'call `simplify_mul` on the tree first?')
    elif isinstance(inputs, list):
        # Recurse through inputs.
        rval = tensor.mul(*map(compute_mul, inputs))
    else:
        rval = inputs
    if neg:
        rval = -rval
    return rval


def perform_sigm_times_exp(tree, exp_x=None, exp_minus_x=None, sigm_x=None,
                           sigm_minus_x=None, parent=None, child_idx=None,
                           full_tree=None):
    """
    Core processing of the `local_sigm_times_exp` optimization.

    This recursive function operates on a multiplication tree as output by
    `parse_mul_tree`. It walks through the tree and modifies it in-place
    by replacing matching pairs (exp, sigmoid) with the desired optimized
    version.

    :param tree: The sub-tree to operate on.

    :exp_x: List of arguments x so that `exp(x)` exists somewhere in the whole
    multiplication tree. Each argument is a pair (x, leaf) with `x` the
    argument of the exponential, and `leaf` the corresponding leaf in the
    multiplication tree (of the form [n, exp(x)] -- see `parse_mul_tree`).
    If None, this argument is initialized to an empty list.

    :param exp_minus_x: Similar to `exp_x`, but for `exp(-x)`.

    :param sigm_x: Similar to `exp_x`, but for `sigmoid(x)`.

    :param sigm_minus_x: Similar to `exp_x`, but for `sigmoid(-x)`.

    :param parent: Parent of `tree` (None if `tree` is the global root).

    :param child_idx: Index of `tree` in its parent's inputs (None if `tree` is
    the global root).

    :param full_tree: The global multiplication tree (should not be set except
    by recursive calls to this function). Used for debugging only.

    :return: True if a modification was performed somewhere in the whole
    multiplication tree, or False otherwise.
    """

    if exp_x is None:
        exp_x = []
    if exp_minus_x is None:
        exp_minus_x = []
    if sigm_x is None:
        sigm_x = []
    if sigm_minus_x is None:
        sigm_minus_x = []
    if full_tree is None:
        full_tree = tree
    if False:  # Debug code.
        print '<perform_sigm_times_exp>'
        print '  full_tree   = %s' % full_tree
        print '  tree        = %s' % tree
        print '  exp_x       = %s' % exp_x
        print '  exp_minus_x = %s' % exp_minus_x
        print '  sigm_x      = %s' % sigm_x
        print '  sigm_minus_x= %s' % sigm_minus_x
    neg, inputs = tree
    if isinstance(inputs, list):
        # Recurse through inputs of the multiplication.
        rval = False
        for sub_idx, sub_tree in enumerate(inputs):
            rval |= perform_sigm_times_exp(
                    tree=sub_tree, parent=tree, child_idx=sub_idx,
                    exp_x=exp_x, exp_minus_x=exp_minus_x, sigm_x=sigm_x,
                    sigm_minus_x=sigm_minus_x, full_tree=full_tree)
        return rval
    else:
        # Reached a leaf: if it is an exponential or a sigmoid, then we
        # first attempt to find a match in leaves already visited.
        # If there is such a match, we modify the already-visited leaf
        # accordingly: for instance if we visited a leaf sigmoid(x), then
        # find later a -exp(-x), we replace the previous leaf by
        # -sigmoid(-x) and remove the -exp(-x) from the tree.
        # If no match is found, then we register this leaf so that it can
        # be found later while walking the tree.
        var = inputs
        keep_it = False
        exp_info = is_exp(var)
        if exp_info is not None:
            exp_neg, exp_arg = exp_info
            neg ^= exp_neg
            neg_arg = is_neg(exp_arg)
            if neg_arg is None:
                if not replace_leaf(exp_arg, sigm_minus_x, sigm_x,
                                    sigmoid, neg):
                    exp_x.append((exp_arg, tree))
                    keep_it = True
            else:
                if not replace_leaf(neg_arg, sigm_x, sigm_minus_x,
                                    lambda x: sigmoid(-x), neg):
                    exp_minus_x.append((neg_arg, tree))
                    keep_it = True
        elif var.owner and var.owner.op == sigmoid:
            sigm_arg = var.owner.inputs[0]
            neg_arg = is_neg(sigm_arg)
            if neg_arg is None:
                if not replace_leaf(sigm_arg, exp_minus_x, sigm_minus_x,
                                    lambda x: sigmoid(-x), neg):
                    sigm_x.append((sigm_arg, tree))
                    keep_it = True
            else:
                if not replace_leaf(neg_arg, exp_x, sigm_x, sigmoid, neg):
                    sigm_minus_x.append((neg_arg, tree))
                    keep_it = True
        else:
            # It is not an exponential nor a sigmoid.
            keep_it = True
        if not keep_it:
            # Delete this leaf, i.e. replace it by [False, None] (corresponding
            # to a multiplication by 1).
            assert parent is not None
            parent[1][child_idx] = [False, None]
        return not keep_it


@opt.register_stabilize
@gof.local_optimizer([tensor.mul])
def local_sigm_times_exp(node):
    """
    exp(x) * sigm(-x) -> sigm(x)
    exp(-x) * sigm(x) -> sigm(-x)
    """
    # Bail early if it is not a multiplication.
    if node.op != tensor.mul:
        return None
    # Obtain tree of multiplications starting at this node.
    mul_tree = parse_mul_tree(node.outputs[0])
    # Perform core optimization.
    did_something = perform_sigm_times_exp(mul_tree)
    if not did_something:
        # No change.
        return None
    # The optimization may have introduced multiplications by 1 in the tree:
    # get rid of them.
    mul_tree = simplify_mul(mul_tree)
    # Recompute final output based on the updated tree.
    return [compute_mul(mul_tree)]


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
                                sigmoid(
                                    tensor.neg(nonconsts[0].owner.inputs[0])),
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
            return  # graph is using both sigm and 1-sigm
        if sub_r.owner and sub_r.owner.op == sigmoid:
            try:
                val_l = opt.get_scalar_constant_value(sub_l)
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
                        rval.append(div(mul(*(
                            t0top + t1top)), mul(*(t0bot + t1bot))))

                        if len(rval) > 100:
                            # This loop can be exponentially long.
                            # aborting
                            return []
        elif len(node.outputs) > 1:
            return []
        else:
            return [node.outputs[0]]
