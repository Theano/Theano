"""
Ops and optimizations: sigmoid, softplus.

These functions implement special cases of exp and log to improve numerical
stability.

"""
from __future__ import absolute_import, print_function, division

import warnings

import numpy as np

import theano
from theano import config, gof, printing, scalar
from theano.compat import imap
from theano.printing import pprint
from theano.tensor import basic as tensor
from theano.tensor import elemwise, opt, NotScalarConstantError
from theano.tensor.type import values_eq_approx_remove_inf
from theano.gof.opt import copy_stack_trace

############
#
# SCALAR OPS
#


class ScalarSigmoid(scalar.UnaryScalarOp):
    """
    This is just speed opt. Not for stability.

    """
    @staticmethod
    def st_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return 1.0
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return 1.0 / (1.0 + np.exp(-x, sig='f'))
        return 1.0 / (1.0 + np.exp(-x))

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
        # We add boundary checks prevent exp from generating inf or
        # 0. The reset of the logic always generate 0 or 1 in those
        # cases. This is a speed optimization.
        # The constants were obtained by looking at the output of
        # python commands like:
        #
        # import numpy, theano
        # dt='float32'  # or float64
        # for i in xrange(750):
        #     print i, repr(theano._asarray(1.0, dtype=dt) /
        #                   (theano._asarray(1.0, dtype=dt) +
        #                    numpy.exp(-theano._asarray([i,-i], dtype=dt))))

        # float16 limits: -11.0, 7.0f
        # We use the float32 limits for float16 for now as the
        # computation will happen in float32 anyway.
        if (node.inputs[0].type == scalar.float32 or
                node.inputs[0].type == scalar.float16):
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

    # This fct is disabled as it is slower then the normal code!
    def c_code_contiguous_disabled(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if (not theano.config.lib.amdlibm or
                node.inputs[0].dtype != node.outputs[0].dtype):
            raise theano.gof.utils.MethodNotDefined()
        dtype = node.inputs[0].dtype
        if dtype == 'float32' and self.amd_float32 is not None:
            dtype = 'float'
            fct = "amd_vrsa_expf"
        elif dtype == 'float64' and self.amd_float64 is not None:
            dtype = 'double'
            fct = "amd_vrda_exp"
        else:
            raise theano.gof.utils.MethodNotDefined()
        return """
        npy_intp n = PyArray_SIZE(%(z)s);
        %(dtype)s * x = (%(dtype)s*) PyArray_DATA(%(x)s);
        %(dtype)s * z = (%(dtype)s*) PyArray_DATA(%(z)s);
        // We block to keep the data in l1
        // normal l1 size = 32k: 32k/2(input + output)/8(nb bytes of double)=2k
        // We stay bellow the 2k limit to let space for
        // This is faster than the not blocking version
        for(int i=0;i<n;i+=2048){
            npy_intp nb = (n-i<2048)?n-i:2048;
            for(int j=0;j<nb;j++){
                z[i+j] = -x[i+j];
            }
            %(fct)s(nb, z+i, z+i);
            for(int j=0;j<nb;j++){
                z[i+j] = 1.0 /(1.0+z[i+j]);
            }
        }
        """ % locals()
        raise theano.gof.utils.MethodNotDefined()

    @staticmethod
    def gen_graph():
        """
        This method was used to generate the graph: sigmoid_prec.png in the doc.

        """
        data = np.arange(-15, 15, .1)
        val = 1 / (1 + np.exp(-data))

        def hard_sigmoid(x):
            return theano.tensor.nnet.hard_sigmoid(x)

        def ultra_fast_sigmoid(x):
            return theano.tensor.nnet.ultra_fast_sigmoid(x)

        val_hard = hard_sigmoid(data).eval()
        val_ultra = ultra_fast_sigmoid(data).eval()

        import matplotlib.pyplot as plt
        import os
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data, val)  # , 'o-')
        ax.plot(data, val_ultra)  # , '-')
        ax.plot(data, val_hard)  # , '-')
        ax.grid(True)
        ax.legend(("sigmoid", "ultra_fast", "hard"), "upper left")
        fname = os.path.join(os.path.dirname(theano.__file__), '..',
                             'doc', 'library', 'tensor', 'nnet',
                             'sigmoid_prec.png')
        plt.savefig(fname)
        print("New picture saved at", fname)
        print(val_ultra.max())
        print(val_ultra.min())


scalar_sigmoid = ScalarSigmoid(scalar.upgrade_to_float, name='scalar_sigmoid')
sigmoid = elemwise.Elemwise(scalar_sigmoid, name='sigmoid')

sigmoid_inplace = elemwise.Elemwise(
    ScalarSigmoid(scalar.transfer_type(0)),
    inplace_pattern={0: 0},
    name='sigmoid_inplace',
)

pprint.assign(sigmoid, printing.FunctionPrinter('sigmoid'))


class UltraFastScalarSigmoid(scalar.UnaryScalarOp):
    """
    This is just speed opt. Not for stability.

    """
    @staticmethod
    def st_impl(x):
        x = 0.5 * x
        # The if is a tanh approximate.
        if x >= 0:
            if x < 1.7:
                z = (1.5 * x / (1 + x))
            elif x < 3:
                z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7))
            else:
                z = 0.99505475368673
        else:
            xx = -x
            if xx < 1.7:
                z = (1.5 * xx / (1 + xx))
            elif xx < 3:
                z = (0.935409070603099 + 0.0458812946797165 * (xx - 1.7))
            else:
                z = 0.99505475368673
            z = -z

        return 0.5 * (z + 1.)

    def impl(self, x):
        return UltraFastScalarSigmoid.st_impl(x)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        dtype = node.outputs[0].type.dtype_specs()[1]

        return """
        %(dtype)s x = 0.5 * %(x)s;
   // The if is a tanh approximate.
   if(x>=0) {
        %(z)s = (x<1.7 ? (1.5*x/(1+x)) :
                         (x<3 ? (0.935409070603099 + 0.0458812946797165*(x-1.7)):
                         0.99505475368673));
    } else {
        %(dtype)s xx = -x;
        %(z)s = -(xx<1.7 ? (1.5*xx/(1+xx)) :
                           (xx<3 ? (0.935409070603099 + 0.0458812946797165*(xx-1.7)):
                                   0.99505475368673));
    }

        //%(z)s = 0.5*(ultrafasttanh(0.5*x)+1.);
        %(z)s = 0.5*(%(z)s+1.);
        """ % locals()

ultra_fast_scalar_sigmoid = UltraFastScalarSigmoid(
    scalar.upgrade_to_float, name='ultra_fast_scalar_sigmoid')
ultra_fast_sigmoid = elemwise.Elemwise(ultra_fast_scalar_sigmoid,
                                       name='ultra_fast_sigmoid')

ultra_fast_sigmoid_inplace = elemwise.Elemwise(
    UltraFastScalarSigmoid(scalar.transfer_type(0)),
    inplace_pattern={0: 0},
    name='ultra_fast_sigmoid_inplace',
)

pprint.assign(ultra_fast_sigmoid,
              printing.FunctionPrinter('ultra_fast_sigmoid'))


# @opt.register_uncanonicalize
@gof.local_optimizer([sigmoid])
def local_ultra_fast_sigmoid(node):
    """
    When enabled, change all sigmoid to ultra_fast_sigmoid.

    For example do mode.including('local_ultra_fast_sigmoid')
    or use the Theano flag optimizer_including=local_ultra_fast_sigmoid.

    This speeds up the sigmoid op by using an approximation.

    This is done after the stabilization and specialize phases
    to avoid interacting with them.

    """
    if (isinstance(node.op, tensor.Elemwise) and
            node.op.scalar_op == scalar_sigmoid):
        out = ultra_fast_sigmoid(node.inputs[0])
        copy_stack_trace(node.outputs[0], out)

        def values_eq_approx_remove_low_prec(a, b):
            # atol is found by trial/error.
            # Other test could fail without good reason.
            return tensor.TensorType.values_eq_approx(a, b, atol=0.02)
        # Let DebugMode know that there this opt approx the values.
        out.tag.values_eq_approx = values_eq_approx_remove_low_prec
        return [out]
theano.compile.optdb['uncanonicalize'].register("local_ultra_fast_sigmoid",
                                                local_ultra_fast_sigmoid)


def hard_sigmoid(x):
    """
    An approximation of sigmoid.

    More approximate and faster than ultra_fast_sigmoid.

    Approx in 3 parts: 0, scaled linear, 1.

    Removing the slope and shift does not make it faster.

    """
    # Use the same dtype as determined by "upgrade_to_float",
    # and perform computation in that dtype.
    out_dtype = scalar.upgrade_to_float(scalar.Scalar(dtype=x.dtype))[0].dtype
    slope = tensor.constant(0.2, dtype=out_dtype)
    shift = tensor.constant(0.5, dtype=out_dtype)
    x = (x * slope) + shift
    x = tensor.clip(x, 0, 1)
    return x


# @opt.register_uncanonicalize
@gof.local_optimizer([sigmoid])
def local_hard_sigmoid(node):
    if (isinstance(node.op, tensor.Elemwise) and
            node.op.scalar_op == scalar_sigmoid):
        out = hard_sigmoid(node.inputs[0])
        copy_stack_trace(node.outputs[0], out)

        def values_eq_approx_remove_low_prec(a, b):
            # atol is found by trial/error.
            # Other test could fail without good reason.
            return tensor.TensorType.values_eq_approx(a, b, atol=0.1)
        # Let DebugMode know that there this opt approx the values.
        out.tag.values_eq_approx = values_eq_approx_remove_low_prec
        return [out]
theano.compile.optdb['uncanonicalize'].register("local_hard_sigmoid",
                                                local_hard_sigmoid)


class ScalarSoftplus(scalar.UnaryScalarOp):
    """
    This helps numerical stability.
    """
    @staticmethod
    def static_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return x
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return np.log1p(np.exp(x, sig='f'))
        return np.log1p(np.exp(x))

    def impl(self, x):
        return ScalarSoftplus.static_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * scalar_sigmoid(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        # These constants were obtained by looking at the output of
        # python commands like:
        #  for i in xrange(750):
        #      print i, repr(numpy.log1p(numpy.exp(theano._asarray([i,-i], dtype=dt))))
        # the boundary checks prevent us from generating inf

        # float16 limits: -17.0, 6.0
        # We use the float32 limits for float16 for now as the
        # computation will happen in float32 anyway.
        if (node.inputs[0].type == scalar.float32 or
                node.inputs[0].type == scalar.float16):
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
scalar_softplus = ScalarSoftplus(scalar.upgrade_to_float,
                                 name='scalar_softplus')
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
    values_eq_approx=values_eq_approx_remove_inf,
    skip_identities_fn=_skip_mul_1)


def _is_1(expr):
    """

    Returns
    -------
    bool
        True iff expr is a constant close to 1.

    """
    try:
        v = opt.get_scalar_constant_value(expr)
        return np.allclose(v, 1)
    except tensor.NotScalarConstantError:
        return False

log1msigm_to_softplus = gof.PatternSub(
    (tensor.log,
        (tensor.sub,
            dict(pattern='y', constraint=_is_1),
            (sigmoid, 'x'))),
    (tensor.neg, (softplus, 'x')),
    allow_multiple_clients=True,
    values_eq_approx=values_eq_approx_remove_inf,
    skip_identities_fn=_skip_mul_1)


log1pexp_to_softplus = gof.PatternSub(
    (tensor.log1p,
     (tensor.exp, 'x')),
    (softplus, 'x'),
    values_eq_approx=values_eq_approx_remove_inf,
    allow_multiple_clients=True)

log1p_neg_sigmoid = gof.PatternSub(
    (tensor.log1p,
     (tensor.neg, (sigmoid, 'x'))),
    (tensor.neg, (softplus, 'x')),
    values_eq_approx=values_eq_approx_remove_inf,
    allow_multiple_clients=True)

opt.register_stabilize(logsigm_to_softplus, name='logsigm_to_softplus')
opt.register_stabilize(log1msigm_to_softplus, name='log1msigm_to_softplus')
opt.register_stabilize(log1pexp_to_softplus, name='log1pexp_to_softplus')
opt.register_stabilize(log1p_neg_sigmoid, name='log1p_neg_sigmoid,')


def is_1pexp(t, only_process_constants=True):
    """

    Returns
    -------
    object
        If 't' is of the form (1+exp(x)), return (False, x).
        Else return None.

    """
    if t.owner and t.owner.op == tensor.add:
        scalars, scalar_inputs, nonconsts = \
            opt.scalarconsts_rest(t.owner.inputs,
                                  only_process_constants=only_process_constants)
        # scalar_inputs are potentially dimshuffled and filled with scalars
        if len(nonconsts) == 1:
            maybe_exp = nonconsts[0]
            if maybe_exp.owner and maybe_exp.owner.op == tensor.exp:
                # Verify that the constant terms sum to 1.
                if scalars:
                    scal_sum = scalars[0]
                    for s in scalars[1:]:
                        scal_sum = scal_sum + s
                    if np.allclose(scal_sum, 1):
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


def is_exp(var):
    """
    Match a variable with either of the `exp(x)` or `-exp(x)` patterns.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    tuple
        A pair (b, x) with `b` a boolean set to True if `var` is of the
        form `-exp(x)` and False if `var` is of the form `exp(x)`. If `var`
        cannot be cast into either form, then return `None`.

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

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    object
        A list [x, y, z, ...] if `var` is of the form `x * y * z * ...`,
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

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    object
        `x` if `var` is of the form `-x`, or None otherwise.

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
                is_minus_1 = np.allclose(constant, -1)
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
    """
    exp(x)/(1+exp(x)) -> sigm(x)
    c/(1+exp(x)) -> c*sigm(-x)

    """
    # this optimization should be done for numerical stability
    # so we don't care to check client counts
    if node.op == tensor.true_div:

        # find all the exp() terms in the numerator
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
            copy_stack_trace(node.outputs[0], sigmoids[-1])

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

        copy_stack_trace(num, new_num)

        if len(denom_rest) == 0:
            return [new_num]
        elif len(denom_rest) == 1:
            out = new_num / denom_rest[0]
        else:
            out = new_num / tensor.mul(*denom_rest)

        copy_stack_trace(node.outputs[0], out)
        return [out]


def parse_mul_tree(root):
    """
    Parse a tree of multiplications starting at the given root.

    Parameters
    ----------
    root
        The variable at the root of the tree.

    Returns
    -------
    object
        A tree where each non-leaf node corresponds to a multiplication
        in the computation of `root`, represented by the list of its inputs.
        Each input is a pair [n, x] with `n` a boolean value indicating whether
        sub-tree `x` should be negated.

    Examples
    --------
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
        return [False, list(map(parse_mul_tree, mul_info))]


def replace_leaf(arg, leaves, new_leaves, op, neg):
    """
    Attempt to replace a leaf of a multiplication tree.

    We search for a leaf in `leaves` whose argument is `arg`, and if we find
    one, we remove it from `leaves` and add to `new_leaves` a leaf with
    argument `arg` and variable `op(arg)`.

    Parameters
    ----------
    arg
        The argument of the leaf we are looking for.
    leaves
        List of leaves to look into. Each leaf should be a pair
        (x, l) with `x` the argument of the Op found in the leaf, and `l` the
        actual leaf as found in a multiplication tree output by `parse_mul_tree`
        (i.e. a pair [boolean, variable]).
    new_leaves
        If a replacement occurred, then the leaf is removed from `leaves`
        and added to the list `new_leaves` (after being modified by `op`).
    op
        A function that, when applied to `arg`, returns the Variable
        we want to replace the original leaf variable with.
    neg : bool
        If True, then the boolean value associated to the leaf should
        be swapped. If False, then this value should remain unchanged.

    Returns
    -------
    bool
        True if a replacement occurred, or False otherwise.

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

    Parameters
    ----------
    tree
        A multiplication tree (as output by `parse_mul_tree`).

    Returns
    -------
    object
        A multiplication tree computing the same output as `tree` but without
        useless multiplications by 1 nor -1 (identified by leaves of the form
        [False, None] or [True, None] respectively). Useless multiplications
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
    # print 'simplify_mul: %s -> %s' % (tree, rval)
    return rval


def compute_mul(tree):
    """
    Compute the Variable that is the output of a multiplication tree.

    This is the inverse of the operation performed by `parse_mul_tree`, i.e.
    compute_mul(parse_mul_tree(tree)) == tree.

    Parameters
    ----------
    tree
        A multiplication tree (as output by `parse_mul_tree`).

    Returns
    -------
    object
        A Variable that computes the multiplication represented by the tree.

    """
    neg, inputs = tree
    if inputs is None:
        raise AssertionError(
            'Function `compute_mul` found a missing leaf, did you forget to '
            'call `simplify_mul` on the tree first?')
    elif isinstance(inputs, list):
        # Recurse through inputs.
        rval = tensor.mul(*list(map(compute_mul, inputs)))
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

    Parameters
    ----------
    tree
        The sub-tree to operate on.
    exp_x
        List of arguments x so that `exp(x)` exists somewhere in the whole
        multiplication tree. Each argument is a pair (x, leaf) with `x` the
        argument of the exponential, and `leaf` the corresponding leaf in the
        multiplication tree (of the form [n, exp(x)] -- see `parse_mul_tree`).
        If None, this argument is initialized to an empty list.
    exp_minus_x
        Similar to `exp_x`, but for `exp(-x)`.
    sigm_x
        Similar to `exp_x`, but for `sigmoid(x)`.
    sigm_minus_x
        Similar to `exp_x`, but for `sigmoid(-x)`.
    parent
        Parent of `tree` (None if `tree` is the global root).
    child_idx
        Index of `tree` in its parent's inputs (None if `tree` is the global
        root).
    full_tree
        The global multiplication tree (should not be set except by recursive
        calls to this function). Used for debugging only.

    Returns
    -------
    bool
        True if a modification was performed somewhere in the whole multiplication
        tree, or False otherwise.

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
        print('<perform_sigm_times_exp>')
        print('  full_tree   = %s' % full_tree)
        print('  tree        = %s' % tree)
        print('  exp_x       = %s' % exp_x)
        print('  exp_minus_x = %s' % exp_minus_x)
        print('  sigm_x      = %s' % sigm_x)
        print('  sigm_minus_x= %s' % sigm_minus_x)
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

    todo: add stack traces to the intermediate variables
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
    out = compute_mul(mul_tree)
    # keep the stack trace
    copy_stack_trace(node.outputs[0], out)
    return [out]


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
                opt.scalarconsts_rest(inv_arg.owner.inputs, only_process_constants=True)
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if len(nonconsts) == 1:
                if nonconsts[0].owner and nonconsts[0].owner.op == tensor.exp:
                    if scalars and np.allclose(np.sum(scalars), 1):
                        out = opt._fill_chain(
                            sigmoid(
                                tensor.neg(nonconsts[0].owner.inputs[0])),
                            scalar_inputs)
                        # keep combined stack traces of
                        #     exp(x):           nonconsts[0],
                        #     1 + exp(x):       inv_arg,
                        #     1 / (1 + exp(x)): node.outputs[0]
                        copy_stack_trace(
                            [nonconsts[0], inv_arg, node.outputs[0]], out)
                        return out

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
            except tensor.NotScalarConstantError:
                return
            if np.allclose(np.sum(val_l), 1):
                out = sigmoid(-sub_r.owner.inputs[0])
                copy_stack_trace([sub_r, node.outputs[0]], out)
                return [out]

register_local_1msigmoid = False
# This is False because the Stabilize pattern above
# is looking for 1-sigm.  Also Canonizer turns neg into *(-1) and so
# this optimization might set off an unwanted chain of things.
# OTH - this transformation can be seen as pushing normal arithmetic either  below or above the
# sigmoidal nonlinearity... so if the canonicalized form had anything to say about that then it
# would be a consideration... anyway leaving False for now.

if register_local_1msigmoid:
    opt.register_canonicalize(local_1msigmoid)
