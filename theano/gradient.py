"""Driver for gradient calculations."""

__authors__ = "James Bergstra, Razvan Pascanu, Arnaud Bergeron"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"

import __builtin__
import logging
import warnings
_logger = logging.getLogger('theano.gradient')
import sys

import numpy  # for numeric_grad

import theano
from theano.raise_op import Raise

from theano import gof
from theano.gof import Variable
from theano.gof.python25 import all
import theano.gof.utils

_msg_retType = 'op.grad(...) returned a non-list'
_msg_badlen = 'op.grad(...) returned wrong number of gradients'


def format_as(use_list, use_tuple, outputs):
    """
    Formats the outputs according to the flags `use_list` and `use_tuple`.
    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    assert not (use_list and use_tuple), \
        "Both flags cannot be simultaneously True"
    if (use_list or use_tuple) and not isinstance(outputs, (list, tuple)):
        if use_list:
            return [outputs]
        else:
            return (outputs,)
    elif not (use_list or use_tuple) and isinstance(outputs, (list, tuple)):
        assert len(outputs) == 1, \
            "Wrong arguments. Expected a one element list"
        return outputs[0]
    elif use_list or use_tuple:
        if use_list:
            return list(outputs)
        else:
            return tuple(outputs)
    else:
        return outputs


def grad_sources_inputs(sources, graph_inputs, warn_type=True):
    """
    A gradient source is a pair (``v``, ``g_v``), in which ``v`` is
    a `Variable`, and ``g_v`` is a `Variable` that is a gradient wrt
    ``v``. More specifically, ``g_v`` is the gradient of an external
    scalar cost, ``cost`` (that is not explicitly used), wrt ``v``.

    This function traverses the graph backward from the ``r`` sources,
    calling ``op.grad(...)`` for all ops with some non-None gradient
    on an output, to compute gradients of ``cost`` wrt intermediate
    variables and ``graph_inputs``.

    The ``op.grad(...)`` functions are called like this:

    .. code-block:: python

        op.grad(op.inputs[:], [total_gradient(v) for v in op.outputs])

    This call to ``op.grad`` should return a list or tuple: one symbolic
    gradient per input. These gradients represent the gradients of
    the same implicit ``cost`` mentionned above, wrt ``op.inputs``.  Note
    that this is **not** the same as the gradient of ``op.outputs`` wrt
    ``op.inputs``.

    If ``op`` has a single input, then ``op.grad`` should return a list
    or tuple of length 1.
    For each input wrt to which ``op`` is not differentiable, it should
    return ``None`` instead of a `Variable` instance.

    If a source ``r`` receives a gradient from another source ``r2``,
    then the effective gradient on ``r`` is the sum of both gradients.



    :type sources: list of pairs of Variable: (v, gradient-on-v) to
                   initialize the total_gradient dictionary
    :param sources: gradients to back-propagate using chain rule
    :type graph_inputs: list of Variable
    :param graph_inputs: variables considered to be constant
        (do not backpropagate through them)
    :type warn_type: bool
    :param warn_type: True will trigger warnings via the logging module when
       the gradient on an expression has a different type than the original
       expression

    :rtype: dictionary whose keys and values are of type Variable
    :return: mapping from each Variable encountered in the backward
        traversal to the gradient with respect to that Variable.

    It is assumed that there is some objective J shared between all members of
    sources, so that for each v, gradient-on-v is the gradient of J with
    respect to v

    """
    gmap = {}
    for (r, g_r) in sources:
        if not hasattr(r, 'type'):
            raise TypeError('sources must be Variables', r)
        if g_r is not None:
            if r in gmap:
                gmap[r] = gmap[r] + g_r
            else:
                gmap[r] = g_r

    graph_outputs = gof.utils.uniq([r for r, g in sources])

    if graph_inputs is None:
        graph_inputs = gof.graph.inputs(graph_outputs)

    for node in gof.graph.io_toposort(graph_inputs,
                                      graph_outputs).__reversed__():
        g_outputs = [gmap.get(o, None) for o in node.outputs]

        #if all output gradients are None, continue
        if all(map(lambda x: x is None, g_outputs)): continue

        output_arg = g_outputs
        input_arg = node.inputs

        # Each Op's grad function requires inputs and output_grads
        # If the Op destroys any input, but the grad expression uses it,
        # then chances are the resulting graph will have a dependency
        # cycle.  We avoid this cycle by passing (symbolic) copies of
        # each destroyed input.
        try:
            dinputs = [node.inputs[x[0]] for x in node.op.destroy_map.values()]
        except AttributeError:
            dinputs = []

        new_input_arg = []
        for input in input_arg:
            if input in dinputs and hasattr(input, 'copy'):
                new_input_arg.append(input.copy())
            else:
                new_input_arg.append(input)
        input_arg = new_input_arg

        #note that this function is not in a try-except block
        # the rationale:
        #  If the op implements grad, then any exception should be passed to
        #  the caller
        #  If the op doesn't implement grad, this entire function should fail.
        #  Other possibilities:
        #    * return a partial back-prop
        #
        op_grad = node.op.grad(input_arg, output_arg)
        if not isinstance(op_grad, (list, tuple)):
            raise ValueError(_msg_retType, node.op)
        g_inputs = op_grad
        assert isinstance(g_inputs, (list, tuple))
        if len(g_inputs) != len(node.inputs):
            raise ValueError(_msg_badlen,
                    node.op,
                    len(g_inputs),
                    len(node.inputs))
        for ii, (r, g_r) in enumerate(zip(node.inputs, g_inputs)):
            if warn_type:
                if g_r and (getattr(r, 'type', 0) != getattr(g_r, 'type', 1)):
                    r_type = getattr(r, 'type', None)
                    g_r_type = getattr(g_r, 'type', None)
                    _logger.warning('%s.grad returned a different type (%s) '
                            'for input %i of type (%s)',
                            node.op, g_r_type, ii, r_type)
            if g_r and len(sources) == 1 and sources[0][0].name and r.name:
                g_r.name = "(d%s/d%s)" % (sources[0][0].name, r.name)
            if g_r is not None:
                assert r is not None
                if r in gmap:
                    gmap[r] = gmap[r] + g_r
                else:
                    gmap[r] = g_r
    return gmap


def unimplemented_grad(op, x_pos, x):
    """
    DO NOT USE. Remove this function after all usage of it has been
    removed from theano.

    Return an un-computable symbolic variable of type `x.type`.

    If any function tries to compute this un-computable variable, an exception
    (NotImplementedError) will be raised indicating that the gradient on the
    `x_pos`'th input of `op` has not been implemented.
    """
    msg = '%s.grad not implemented for input %i' % (op, x_pos)
    return Raise(msg=msg)(x)

########################
# R Operator
########################


def Rop(f, wrt, eval_points):
    """
    Computes the R operation on `f` wrt to `wrt` evaluated at points given
    in `eval_points`. Mathematically this stands for the jacobian of `f` wrt
    to `wrt` right muliplied by the eval points.

    :type f: Variable or list of Variables
             `f` stands for the output of the computational graph to which you
             want to apply the R operator
    :type wrt: Variable or list of `Variables`s
               variables for which you compute the R operator of the expression
               described by `f`
    :type eval_points: Variable or list of Variables
                       evalutation points for each of the variables in `wrt`
    :rtype: Variable or list/tuple of Variables depending on type of f
    :return: symbolic expression such that
        R_op[i] = sum_j ( d f[i] / d wrt[j]) eval_point[j]
        where the indices in that expression are magic multidimensional
        indices that specify both the position within a list and all
        coordinates of the tensor element in the last.
        If `wrt` is a list/tuple, then return a list/tuple with the results.
    """
    from theano.tensor import as_tensor_variable
    using_list = isinstance(f, list)
    using_tuple = isinstance(f, tuple)
    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]

    if not isinstance(eval_points, (list, tuple)):
        eval_points = [eval_points]

    if not isinstance(f, (list, tuple)):
        f = [f]

    assert len(wrt) == len(eval_points)

    # Check that each element of wrt corresponds to an element
    # of eval_points with the same dimensionality.
    for pack in enumerate(zip(wrt, eval_points)):
        i = pack[0]
        wrt_elem, eval_point = pack[1]

        wrt_elem = as_tensor_variable(wrt_elem)
        eval_point = as_tensor_variable(eval_point)

        wrt_dim = len(wrt_elem.type.broadcastable)
        eval_dim = len(eval_point.type.broadcastable)

        if wrt_dim != eval_dim:
            raise ValueError('Element ' +
                             str(i) +
                             ' of wrt/eval_point have mismatched ' +
                             'dimensionality: ' +
                             str(wrt_dim) +
                             ' versus ' +
                             str(eval_dim))

    seen_nodes = {}

    def _traverse(node):
        """ TODO: writeme """
        if node is None:
            return None
        else:
            op = node.op
            inputs = node.inputs

            # Compute the evaluation points corresponding to each of the
            # inputs of the node
            local_eval_points = []
            for inp in inputs:
                if inp in wrt:
                    local_eval_points.append(eval_points[wrt.index(inp)])
                elif inp.owner is None:
                    local_eval_points.append(inp.zeros_like())
                elif inp.owner in seen_nodes:

                    local_eval_points.append(
                        seen_nodes[inp.owner][inp.owner.outputs.index(inp)])

                else:
                    # We actually need to compute the R_op for this node

                    _traverse(inp.owner)
                    local_eval_points.append(
                        seen_nodes[inp.owner][inp.owner.outputs.index(inp)])
            for x, y in zip(inputs, local_eval_points):
                if y is not None:
                    assert (as_tensor_variable(x).type ==
                            as_tensor_variable(y).type)

            seen_nodes[node] = op.R_op(node.inputs, local_eval_points)
            return None

    # Populate the dictionary
    for out in f:
        _traverse(out.owner)

    rval = []
    for out in f:
        if out in wrt:
            rval.append(eval_points[wrt.index(out)])
        elif seen_nodes[out.owner][out.owner.outputs.index(out)] is None:
            raise ValueError(('The function is not differentiable with '
                              'respect to the provided inputs !'))
        else:
            rval.append(seen_nodes[out.owner][out.owner.outputs.index(out)])

    return format_as(using_list, using_tuple, rval)


def Lop(f, wrt, eval_points, consider_constant=None, warn_type=False,
         disconnected_inputs='raise'):
    """
    Computes the L operation on `f` wrt to `wrt` evaluated at points given
    in `eval_points`. Mathematically this stands for the jacobian of `f` wrt
    to `wrt` left muliplied by the eval points.

    :type f: Variable or list of Variables
        `f` stands for the output of the computational graph to which you
        want to apply the L operator
    :type wrt: Variable or list of `Variables`s
        variables for which you compute the L operator of the expression
        described by `f`
    :type eval_points: Variable or list of Variables
                        evalutation points for each of the variables in `f`

    :rtype: Variable or list/tuple of Variables depending on type of f
    :return: symbolic expression such that
        L_op[i] = sum_i ( d f[i] / d wrt[j]) eval_point[i]
        where the indices in that expression are magic multidimensional
        indices that specify both the position within a list and all
        coordinates of the tensor element in the last
        If `f` is a list/tuple, then return a list/tuple with the results.
    """
    if consider_constant is None:
        consider_constant = []

    if type(eval_points) not in (list, tuple):
        eval_points = [eval_points]

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if not isinstance(f, (list, tuple)):
        f = [f]

    inputs = gof.graph.inputs(f)
    gmap = grad_sources_inputs(
        zip(f, eval_points),
        list(inputs) + list(consider_constant),
        warn_type=warn_type)

    # Note : If p is not in gmap there can be several reasons, among which
    # is the fact that p might not be part of the computational graph. A
    # simple example is that for a+b for e.g. a[0] is not part of the graph,
    # so Theano does not know how to compute TT.grad(TT.sum(a+b), a[0])
    # such subtle cases can be fixed by a more careful implementation of the
    # gradient, but for now Theano needs to throw an exception, and make the
    # user aware that it does not know how to compute that gradient
    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]
    ret = []
    for p in wrt:
        if p in gmap:
            ret.append(gmap[p])
        else:
            message = ("Lop method was asked to compute the gradient "
                    "with respect to a variable that is not part of "
                    "the computational graph of the cost, or is used "
                    "only by a non-differentiable operator: %s" % p)
            if disconnected_inputs == 'ignore':
                pass
            elif disconnected_inputs == 'warn':
                warnings.warn(message, stacklevel=1)
            elif disconnected_inputs == 'raise':
                raise ValueError(message)
            else:
                raise ValueError("Invalid value for keyword "
                        "'disconnected_inputs', valid values are "
                        "'ignore', 'warn' and 'raise'.")
            ret.append(p.zeros_like())

    return format_as(using_list, using_tuple, ret)


#########################
# Gradient
#########################

def grad(cost, wrt, g_cost=None, consider_constant=None, warn_type=False,
         disconnected_inputs='raise'):
    """
    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Variable or list of Variables.
    :type g_cost: Scalar Variable, or None.
    :param g_cost: an expression for the gradient through cost.  The default is
        ``ones_like(cost)``.
    :param consider_constant: a list of expressions not to backpropagate
        through

    :param warn_type: a value of True will cause warnings to be logged for any
        Op that emits a gradient that does not match its input type.

    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    :rtype: Variable or list/tuple of Variables (depending upon `wrt`)

    :return: symbolic expression of gradient of `cost` with respect to `wrt`.
             If an element of `wrt` is not differentiable with respect
             to the output, then a zero variable is returned.
             It returns an object of same type as `wrt`: a list/tuple
             or Variable in all cases.

    This function is a wrapper around the more general function
    `theano.gradient.grad_sources_inputs``.

    """
    if consider_constant is None:
        consider_constant = []
    else:
        #error checking on consider_constant: verify that it is a collection
        # of theano variables
        # this is important, if someone accidentally passes a nested data
        # structure with theano variables at the leaves, only the root will
        # be properly considered constant
        if not hasattr(consider_constant, '__iter__'):
            raise TypeError('consider_constant must be an iterable collection,'
                    ' got ' + str(type(consider_constant)))
        for elem in consider_constant:
            if not isinstance(elem, gof.Variable):
                raise TypeError('Elements of consider_constant must be '
                                'variables, but got ' + str(type(elem)))

    if not isinstance(cost, Variable):
        raise TypeError(('In grad(), cost argument should be '
                         'a Variable.'), cost)

    if cost.type.ndim:
        raise TypeError(
                'In theano.gradient.grad, "cost" argument should be a scalar,'
                ' but ndim is %i (should be 0). If you want to compute the'
                ' gradient of the sum of cost, you should use cost.sum().'
                % cost.type.ndim)

    if g_cost is None:
        from theano import tensor
        g_cost = tensor.ones_like(cost)
    inputs = gof.graph.inputs([cost])
    gmap = grad_sources_inputs(
        [(cost, g_cost)],
        list(inputs) + list(consider_constant),
        warn_type=warn_type)

    # Note : If p is not in gmap there can be several reasons, among which
    # is the fact that p might not be part of the computational graph. A
    # simple example is that for a+b for e.g. a[0] is not part of the graph,
    # so Theano does not know how to compute TT.grad(TT.sum(a+b), a[0])
    # such subtle cases can be fixed by a more careful implementation of the
    # gradient, but for now Theano needs to throw an exception, and make the
    # user aware that it does not know how to compute that gradient
    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]
    ret = []
    for p in wrt:
        if p in gmap:
            ret.append(gmap[p])
        else:
            message = ("grad method was asked to compute the gradient "
                    "with respect to a variable that is not part of "
                    "the computational graph of the cost, or is used "
                    "only by a non-differentiable operator: %s" % p)
            if disconnected_inputs == 'ignore':
                pass
            elif disconnected_inputs == 'warn':
                warnings.warn(message, stacklevel=1)
            elif disconnected_inputs == 'raise':
                raise ValueError(message)
            else:
                raise ValueError("Invalid value for keyword "
                        "'disconnected_inputs', valid values are "
                        "'ignore', 'warn' and 'raise'.")
            ret.append(p.zeros_like())

    return format_as(using_list, using_tuple, ret)


class numeric_grad(object):
    """
    Compute the numeric derivative of a scalar-valued function at a particular
    point.
    """

    # Note on step sizes and tolerances:
    #
    # There is a relationship between the step size and the function value and
    # the measurement error that is incurred due to rounding.  The finite
    # difference we measure is
    # delta = f(x0) - f(x0+eps)
    #
    # For maximum precision, f should be close to zero.
    # For every power of 2 that f departs from zero, we lose a bit of precision
    # in delta.
    #
    # Even in this case of maximum accuracy, there is a tradeoff between
    # stepsize and measurement error.
    # Taking small steps allows us to measure large derivatives accuractly,
    # but longer steps are required to measure small derivatives accurately.
    # However longer steps introduce bias into our measurement in general
    # for non-linear functions.
    #
    # It would be interesting to have a version of numeric grad that used an
    # adaptive stepsize.
    #
    # For now, we use a heuristic that catches very bad gradients, but is not
    # perfectly accurate.
    type_eps = {'float64': 1e-7,
            'float32': 3e-4,
            numpy.dtype('float64'): 1e-7,
            numpy.dtype('float32'): 3e-4}

    def __init__(self, f, pt, eps=None):
        """Return the gradient of f at pt.

        :param f: a differentiable function such that f(*pt) is a scalar
        :param pt: an ndarray, a list of ndarrays or tuple of ndarrays

        This function computes the gradient by a one-sided finite
        differences of a fixed step size (eps).

        It is assumed that f(...) will return a scalar.
        It is assumed that all f's inputs are numpy.ndarray objects.

        :param eps: the stepsize for the finite differencing.  None means
          input dtype-dependent. See `type_eps`.
        """

        def prod(inputs):
            rval = 1
            for i in inputs:
                rval *= i
            return rval

        packed_pt = False
        if not isinstance(pt, (list, tuple)):
            pt = [pt]
            packed_pt = True

        apt = [numpy.array(p) for p in pt]

        shapes = [p.shape for p in apt]
        dtypes = [str(p.dtype) for p in apt]

        # TODO: remove this eventually (why was this here in the first place ?)
        # In the case of CSM, the arguments are a mixture of floats and
        # integers...
        # if not dtypes == [dtypes[0]] * len(apt):
        #      raise TypeError('All function arguments must have same dtype')

        total_size = __builtin__.sum(prod(sh) for sh in shapes)

        working_dtype = __builtin__.min((self.type_eps[dt], dt)
                                        for dt in dtypes)[1]

        #create un-initialized memory
        x = numpy.ndarray((total_size,), dtype=working_dtype)
        gx = numpy.ndarray((total_size,), dtype=working_dtype)

        if eps is None:
            eps = __builtin__.max(self.type_eps[dt] for dt in dtypes)

        #set up aliases so that apt[i] is backed by memory in x
        # and self.gf is backed by memory in gx
        cur_pos = 0
        self.gf = []
        for i, p in enumerate(apt):
            p_size = prod(p.shape)
            # set up alias
            apt[i] = x[cur_pos: cur_pos + p_size].reshape(p.shape)
            self.gf.append(gx[cur_pos: cur_pos + p_size].reshape(p.shape))
            # initialize with p's value
            apt[i][...] = p
            cur_pos += p_size

        f_x = f(*[p.copy() for p in apt])

        # now iterate over the elements of x, and call f on apt.
        x_copy = x.copy()
        for i in xrange(total_size):
            x[:] = x_copy

            x[i] += eps
            f_eps = f(*apt)

            gx[i] = numpy.asarray((f_eps - f_x) / eps)

        if packed_pt:
            self.gf = self.gf[0]

    @staticmethod
    def abs_rel_err(a, b):
        """Return absolute and relative error between a and b.

        The relative error is a small number when a and b are close, relative
        to how big they are.

        Formulas used:
            abs_err = abs(a - b)
            rel_err = abs_err / max(abs(a) + abs(b), 1e-8)

        The denominator is clipped at 1e-8 to avoid dividing by 0 when a and b
        are both close to 0.

        The tuple (abs_err, rel_err) is returned
        """
        abs_err = abs(a - b)
        rel_err = abs_err / numpy.maximum(abs(a) + abs(b), 1e-8)
        return (abs_err, rel_err)

    def abs_rel_errors(self, g_pt):
        """Return the abs and rel error of gradient estimate `g_pt`

        `g_pt` must be a list of ndarrays of the same length as self.gf,
        otherwise a ValueError is raised.

        Corresponding ndarrays in `g_pt` and `self.gf` must have the same
        shape or ValueError is raised.

        """
        if len(g_pt) != len(self.gf):
            raise ValueError(
                    'argument has wrong number of elements',
                    len(g_pt))
        errs = []
        for i, (a, b) in enumerate(zip(g_pt, self.gf)):
            if a.shape != b.shape:
                raise ValueError(
                        'argument element %i has wrong shape %s' % (
                            i, str((a.shape, b.shape))))
            errs.append(numeric_grad.abs_rel_err(a, b))
        return errs

    def max_err(self, g_pt, abs_tol, rel_tol):
        """Find the biggest error between g_pt and self.gf.

        What is measured is the violation of relative and absolute errors,
        wrt the provided tolerances (abs_tol, rel_tol).
        A value > 1 means both tolerances are exceeded.

        Return the argmax of min(abs_err / abs_tol, rel_err / rel_tol) over
        g_pt, as well as abs_err and rel_err at this point.
        """
        pos = []
        errs = []
        abs_errs = []
        rel_errs = []

        abs_rel_errs = self.abs_rel_errors(g_pt)
        for abs_err, rel_err in abs_rel_errs:
            if not numpy.all(numpy.isfinite(abs_err)):
                raise ValueError('abs_err not finite', repr(abs_err))
            if not numpy.all(numpy.isfinite(rel_err)):
                raise ValueError('rel_err not finite', repr(rel_err))
            scaled_err = numpy.minimum(abs_err / abs_tol, rel_err / rel_tol)
            max_i = scaled_err.argmax()

            pos.append(max_i)
            errs.append(scaled_err.flatten()[max_i])
            abs_errs.append(abs_err.flatten()[max_i])
            rel_errs.append(rel_err.flatten()[max_i])

        # max over the arrays in g_pt
        max_arg = numpy.argmax(errs)
        max_pos = pos[max_arg]
        return (max_arg, pos[max_arg], abs_errs[max_arg], rel_errs[max_arg])


def verify_grad(fun, pt, n_tests=2, rng=None, eps=None, abs_tol=None,
                rel_tol=None, mode=None, cast_to_output_type=False):
    """ Test a gradient by Finite Difference Method. Raise error on failure.

    Example:
        >>> verify_grad(theano.tensor.tanh,
                        (numpy.asarray([[2,3,4], [-1, 3.3, 9.9]]),),
                        rng=numpy.random)

    Raises an Exception if the difference between the analytic gradient and
    numerical gradient (computed through the Finite Difference Method) of a
    random projection of the fun's output to a scalar exceeds the given
    tolerance.

    :param fun: a Python function that takes Theano variables as inputs,
        and returns a Theano variable. For instance, an Op instance with
        a single output.
    :param pt: the list of numpy.ndarrays to use as input values.
        These arrays must be either float32 or float64 arrays.
    :param n_tests: number of times to run the test
    :param rng: random number generator used to sample u, we test gradient
        of sum(u * fun) at pt
    :param eps: stepsize used in the Finite Difference Method (Default
        None is type-dependent)
    :param abs_tol: absolute tolerance used as threshold for gradient
        comparison
    :param rel_tol: relative tolerance used as threshold for gradient
        comparison

    :note: WARNING to unit-test writers: if `op` is a function that builds
        a graph, try to make it a SMALL graph.  Often verify grad is run
        in debug mode, which can be very slow if it has to verify a lot of
        intermediate computations.

    :note: This op does not support multiple outputs. In tests/test_scan.py
        there is an experimental verify_grad that covers that case as well
        by using random projections.
    """
    from theano import compile, shared
    import theano.tensor
    from theano.tensor import as_tensor_variable, cast, TensorType
    assert isinstance(pt, (list, tuple))
    pt = [numpy.array(p) for p in pt]

    for i, p in enumerate(pt):
        if p.dtype not in ('float32', 'float64'):
            raise TypeError(('verify_grad can work only with floating point '
                'inputs, but input %i has dtype "%s".') % (i, p.dtype))

    _type_tol = dict(  # relativ error tolerances for different types
            float32=1e-2,
            float64=1e-4)

    if abs_tol is None:
        abs_tol = __builtin__.max(_type_tol[str(p.dtype)] for p in pt)
    if rel_tol is None:
        rel_tol = __builtin__.max(_type_tol[str(p.dtype)] for p in pt)

    if rng is None:
        raise TypeError(('rng should be a valid instance of '
                        'numpy.random.RandomState. You may '
                         'want to use theano.tests.unittest'
                         '_tools.verify_grad instead of '
                         'theano.gradient.verify_grad.'))

    # We allow input downcast in function, because numeric_grad works in the
    # most precise dtype used among the inputs, so we may need to cast some.
    def function(inputs, output):
        if mode is None:
            f = compile.function(inputs, output, accept_inplace=True,
                    allow_input_downcast=True)
        else:
            f = compile.function(inputs, output, accept_inplace=True,
                    allow_input_downcast=True, mode=mode)
        return f

    tensor_pt = [TensorType(
            as_tensor_variable(p).dtype,
            as_tensor_variable(p).broadcastable)(name='input %i' % i)
        for i, p in enumerate(pt)]

    #fun can be either a function or an actual Op instance
    o_output = fun(*tensor_pt)

    if isinstance(o_output, list):
        raise NotImplementedError(('cant (yet) autotest gradient of fun '
                                   'with multiple outputs'))
        # we could make loop over outputs making random projections R for each,
        # but this doesn't handle the case where not all the outputs are
        # differentiable... so I leave this as TODO for now -JB.

    o_fn = function(tensor_pt, o_output)
    o_fn_out = o_fn(*[p.copy() for p in pt])

    if isinstance(o_fn_out, tuple) or isinstance(o_fn_out, list):
        raise TypeError('It seems like you are trying to use verify_grad '
                'on an op or a function which outputs a list: there should'
                ' be a single (array-like) output instead')

    # random_projection should not have elements too small,
    # otherwise too much precision is lost in numerical gradient
    def random_projection():
        plain = rng.rand(*o_fn_out.shape) + 0.5
        if cast_to_output_type:
            return numpy.array(plain, o_output.dtype)
        return plain

    t_r = shared(random_projection())

    # random projection of o onto t_r
    # This sum() is defined above, it's not the builtin sum.
    cost = theano.tensor.sum(t_r * o_output)

    cost_fn = function(tensor_pt, cost)

    #todo-- determine if this is actually needed
    g_cost = as_tensor_variable(1.0, name='g_cost')
    if cast_to_output_type:
        g_cost = cast(g_cost, o_output.dtype)

    symbolic_grad = grad(cost, tensor_pt, g_cost,
                         disconnected_inputs='ignore')

    grad_fn = function(tensor_pt, symbolic_grad)

    for test_num in xrange(n_tests):
        num_grad = numeric_grad(cost_fn, [p.copy() for p in pt], eps)

        analytic_grad = grad_fn(*[p.copy() for p in pt])

        # Since `tensor_pt` is a list, `analytic_grad` should be one too.
        assert isinstance(analytic_grad, list)

        max_arg, max_err_pos, max_abs_err, max_rel_err =\
                num_grad.max_err(analytic_grad, abs_tol, rel_tol)

        if max_abs_err > abs_tol and max_rel_err > rel_tol:
            raise verify_grad.E_grad(max_arg, max_err_pos,
                    max_abs_err, max_rel_err, abs_tol, rel_tol)

        #get new random projection for next test
        if test_num < n_tests - 1:
            t_r.set_value(random_projection(), borrow=True)


class GradientError(Exception):
    """This error is raised when a gradient is calculated, but incorrect."""
    def __init__(self, arg, err_pos, abs_err, rel_err, abs_tol, rel_tol):
        self.arg = arg
        self.err_pos = err_pos
        self.abs_err = abs_err
        self.rel_err = rel_err
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def __str__(self):
        # args may have been inserted by e.g. makeTester
        args_msg = ", ".join(str(a) for a in self.args)
        return """\
GradientError: numeric gradient and analytic gradient exceed tolerance:
        At position %i of argument %i,
            abs. error = %f,  abs. tolerance = %f
            rel. error = %f,  rel. tolerance = %f
Exception args: %s""" % (self.err_pos, self.arg,
                         self.abs_err, self.abs_tol,
                         self.rel_err, self.rel_tol,
                         args_msg)

verify_grad.E_grad = GradientError


def jacobian(expression, wrt, consider_constant=None, warn_type=False,
             disconnected_inputs='raise'):
    """
    :type expression: Vector (1-dimensional) Variable
    :type wrt: Variable or list of Variables

    :param consider_constant: a list of expressions not to backpropagate
        through

    :param warn_type: a value of True will cause warnings to be logged for any
        Op that emits a gradient that does not match its input type.

    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    :return: either a instance of Variable or list/tuple of Variables
            (depending upon `wrt`) repesenting the jacobian of `expression`
            with respect to (elements of) `wrt`. If an element of `wrt` is not
            differentiable with respect to the output, then a zero
            variable is returned. The return value is of same type
            as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    from theano.tensor import arange
    # Check inputs have the right format
    assert isinstance(expression, Variable), \
            "tensor.jacobian expects a Variable as `expression`"
    assert expression.ndim < 2, \
            ("tensor.jacobian expects a 1 dimensional variable as "
             "`expression`. If not use flatten to make it a vector")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    if expression.ndim == 0:
        # expression is just a scalar, use grad
        return format_as(using_list, using_tuple, grad(expression, wrt))

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            rval = grad(expr[idx],
                     inp,
                     consider_constant=consider_constant,
                     warn_type=warn_type,
                     disconnected_inputs=disconnected_inputs)
            rvals.append(rval)
        return rvals
    # Computing the gradients does not affect the random seeds on any random
    # generator used n expression (because during computing gradients we are
    # just backtracking over old values. (rp Jan 2012 - if anyone has a
    # counter example please show me)
    jacobs, updates = theano.scan(inner_function,
                            sequences=arange(expression.shape[0]),
                            non_sequences=[expression] + wrt)
    assert not updates, \
            ("Scan has returned a list of updates. This should not "
             "happen! Report this to theano-users (also include the "
             "script that generated the error)")
    return format_as(using_list, using_tuple, jacobs)


def hessian(cost, wrt, consider_constant=None, warn_type=False,
             disconnected_inputs='raise'):
    """
    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Vector (1-dimensional tensor) 'Variable' or list of
               vectors (1-dimensional tensors) Variables

    :param consider_constant: a list of expressions not to backpropagate
        through

    :param warn_type: a value of True will cause warnings to be logged for any
        Op that emits a gradient that does not match its input type.

    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    :return: either a instance of Variable or list/tuple of Variables
            (depending upon `wrt`) repressenting the Hessian of the `cost`
            with respect to (elements of) `wrt`. If an element of `wrt` is not
            differentiable with respect to the output, then a zero
            variable is returned. The return value is of same type
            as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    from theano.tensor import arange
    # Check inputs have the right format
    assert isinstance(cost, Variable), \
            "tensor.hessian expects a Variable as `cost`"
    assert cost.ndim == 0, \
            "tensor.hessian expects a 0 dimensional variable as `cost`"

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    hessians = []
    for input in wrt:
        assert isinstance(input, Variable), \
                "tensor.hessian expects a (list of) Variable as `wrt`"
        assert input.ndim == 1, \
                "tensor.hessian expects a (list of) 1 dimensional variable "\
                "as `wrt`"
        expr = grad(cost, input)
        hess, updates = theano.scan(lambda i, y, x: grad(
                            y[i],
                            x,
                            consider_constant=consider_constant,
                            warn_type=warn_type,
                            disconnected_inputs=disconnected_inputs),
                       sequences=arange(expr.shape[0]),
                       non_sequences=[expr, input])
        assert not updates, \
                ("Scan has returned a list of updates. This should not "
                 "happen! Report this to theano-users (also include the "
                 "script that generated the error)")
        hessians.append(hess)
    return format_as(using_list, using_tuple, hessians)
