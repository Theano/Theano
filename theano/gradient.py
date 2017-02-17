"""Driver for gradient calculations."""
from __future__ import absolute_import, print_function, division
from collections import OrderedDict
import six.moves.builtins as builtins
import logging
import time
import warnings

import numpy  # for numeric_grad
from six import itervalues

import theano

from theano import gof
from theano.gof import utils, Variable
from theano.compat import izip
from six.moves import xrange, reduce
from theano.gof.null_type import NullType, null_type
from theano.gof.op import get_debug_values
from theano.compile import ViewOp, FAST_RUN, DebugMode

np = numpy
__authors__ = "James Bergstra, Razvan Pascanu, Arnaud Bergeron, Ian Goodfellow"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"
_logger = logging.getLogger('theano.gradient')

# we can't do "import theano.tensor"
# tensor depends on theano.compile
# theano.compile depends on theano.gradient (this file)
# the reason theano.compile depends on theano.gradient
# is that theano.compile.builders contains the op from graph
# functionality and it uses theano.gradient to implement
# the new op's grad method
tensor = None

_msg_retType = 'op.grad(...) returned a non-list'

grad_time = 0


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


def grad_not_implemented(op, x_pos, x, comment=""):
    """
    Return an un-computable symbolic variable of type `x.type`.

    If any call to tensor.grad results in an expression containing this
    un-computable variable, an exception (NotImplementedError) will be
    raised indicating that the gradient on the
    `x_pos`'th input of `op` has not been implemented. Likewise if
    any call to theano.function involves this variable.

    Optionally adds a comment to the exception explaining why this
    gradient is not implemented.
    """

    return (NullType((
        "This variable is Null because the grad method for "
        "input %s (%s) of the %s op is not implemented. %s"
    ) % (x_pos, x, op, comment)))()


def grad_undefined(op, x_pos, x, comment=""):
    """
    Return an un-computable symbolic variable of type `x.type`.

    If any call to tensor.grad results in an expression containing this
    un-computable variable, an exception (GradUndefinedError) will be
    raised indicating that the gradient on the
    `x_pos`'th input of `op` is mathematically undefined. Likewise if
    any call to theano.function involves this variable.

    Optionally adds a comment to the exception explaining why this
    gradient is not defined.
    """

    return (NullType(
        (
            "This variable is Null because the grad method for "
            "input %s (%s) of the %s op is mathematically undefined. %s"
        ) % (x_pos, x, op, comment)))()


class DisconnectedType(theano.gof.type.Type):

    """ A type indicating that a variable is a result
        of taking the gradient of c with respect to x
        when c is not a function of x.
        A symbolic placeholder for 0, but to convey
        the extra information that this gradient is 0
        because it is disconnected.
    """

    def filter(self, data, strict=False, allow_downcast=None):
        raise AssertionError(
            (
                "If you're assigning to a DisconnectedType you're"
                " doing something wrong. It should only be used as"
                " a symbolic placeholder."
            ))

    def fiter_variable(self, other):
        raise AssertionError(
            (
                "If you're assigning to a DisconnectedType you're"
                " doing something wrong. It should only be used as"
                " a symbolic placeholder."
            ))

    def may_share_memory(a, b):
        return False

    def value_eq(a, b, force_same_dtype=True):
        raise AssertionError(
            (
                "If you're assigning to a DisconnectedType you're"
                " doing something wrong. It should only be used as"
                " a symbolic placeholder."
            ))

    def __str__(self):
        return 'DisconnectedType'


disconnected_type = DisconnectedType()


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
    :rtype: :class:`~theano.gof.Variable` or list/tuple of Variables depending on type of f
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
        if not isinstance(wrt_elem, gof.Variable):
            wrt_elem = as_tensor_variable(wrt_elem)
        if not isinstance(eval_point, gof.Variable):
            eval_point = as_tensor_variable(eval_point)

        try:

            if wrt_elem.type.ndim != eval_point.type.ndim:
                raise ValueError('Element ' +
                                 str(i) +
                                 ' of wrt/eval_point have mismatched ' +
                                 'dimensionality: ' +
                                 str(wrt_elem.type.ndim) +
                                 ' versus ' +
                                 str(eval_point.type.ndim))
        except AttributeError:
            # wrt_elem and eval_point don't always have ndim like random type
            # Tensor, Sparse and CudaNdArray have the ndim attribute
            pass

    seen_nodes = OrderedDict()

    def _traverse(node):
        """ TODO: writeme """

        if node is None:
            return

        op = node.op
        inputs = node.inputs

        # Compute the evaluation points corresponding to each of the
        # inputs of the node
        local_eval_points = []
        for inp in inputs:
            if inp in wrt:
                local_eval_points.append(eval_points[wrt.index(inp)])
            elif inp.owner is None:
                try:
                    local_eval_points.append(inp.zeros_like())
                except Exception:
                    # None should be used for non-differentiable
                    # arguments, like for example random states
                    local_eval_points.append(None)
            elif inp.owner in seen_nodes:

                local_eval_points.append(
                    seen_nodes[inp.owner][inp.owner.outputs.index(inp)])

            else:
                # We actually need to compute the R_op for this node

                _traverse(inp.owner)
                local_eval_points.append(
                    seen_nodes[inp.owner][inp.owner.outputs.index(inp)])
        same_type_eval_points = []
        for x, y in zip(inputs, local_eval_points):
            if y is not None:
                if not isinstance(x, gof.Variable):
                    x = as_tensor_variable(x)
                if not isinstance(y, gof.Variable):
                    y = as_tensor_variable(y)
                try:
                    y = x.type.filter_variable(y)
                except TypeError:
                    # This is a hack
                    # Originally both grad and Rop were written
                    # with the assumption that a variable and the
                    # gradient wrt that variable would have the same
                    # dtype. This was a bad assumption because the
                    # gradient wrt an integer can take on non-integer
                    # values.
                    # grad is now fixed, but Rop is not, so when grad
                    # does the right thing and violates this assumption
                    # we have to make it be wrong for Rop to keep working
                    # Rop should eventually be upgraded to handle integers
                    # correctly, the same as grad
                    y = theano.tensor.cast(y, x.type.dtype)
                    y = x.type.filter_variable(y)
                assert x.type == y.type
                same_type_eval_points.append(y)
            else:
                same_type_eval_points.append(y)

        seen_nodes[node] = op.R_op(node.inputs, same_type_eval_points)
    # end _traverse

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


def Lop(f, wrt, eval_points, consider_constant=None,
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

    :rtype: :class:`~theano.gof.Variable` or list/tuple of Variables depending on type of f
    :return: symbolic expression such that
        L_op[i] = sum_i ( d f[i] / d wrt[j]) eval_point[i]
        where the indices in that expression are magic multidimensional
        indices that specify both the position within a list and all
        coordinates of the tensor element in the last
        If `f` is a list/tuple, then return a list/tuple with the results.
    """
    if type(eval_points) not in (list, tuple):
        eval_points = [eval_points]

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if not isinstance(f, (list, tuple)):
        f = [f]

    # make copies of f and grads so we don't modify the client's copy
    f = list(f)
    grads = list(eval_points)

    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]

    assert len(f) == len(grads)
    known = OrderedDict(izip(f, grads))

    ret = grad(cost=None, known_grads=known,
               consider_constant=consider_constant, wrt=wrt,
               disconnected_inputs=disconnected_inputs)

    return format_as(using_list, using_tuple, ret)


#########################
# Gradient
#########################

def grad(cost, wrt, consider_constant=None,
         disconnected_inputs='raise', add_names=True,
         known_grads=None, return_disconnected='zero',
         null_gradients='raise'):
    """
    Return symbolic gradients for one or more variables with respect to some
    cost.

    For more information about how automatic differentiation works in Theano,
    see :mod:`gradient`. For information on how to implement the gradient of
    a certain Op, see :func:`grad`.

    Parameters
    ----------
    cost : :class:`~theano.gof.Variable` scalar (0-dimensional) tensor variable or None
        Value with respect to which we are differentiating.  May be
        `None` if known_grads is provided.
    wrt : :class:`~theano.gof.Variable` or list of Variables
        term[s] for which we want gradients
    consider_constant : list of variables
        expressions not to backpropagate through
    disconnected_inputs : {'ignore', 'warn', 'raise'}
        Defines the behaviour if some of the variables in `wrt` are
        not part of the computational graph computing `cost` (or if
        all links are non-differentiable). The possible values are:

        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise DisconnectedInputError.
    add_names : bool
        If True, variables generated by grad will be named
        (d<cost.name>/d<wrt.name>) provided that both cost and wrt
        have names
    known_grads : OrderedDict, optional
        A ordered dictionary mapping variables to their gradients. This is
        useful in the case where you know the gradient on some
        variables but do not know the original cost.
    return_disconnected : {'zero', 'None', 'Disconnected'}
        - 'zero' : If wrt[i] is disconnected, return value i will be
                   wrt[i].zeros_like()
        - 'None' : If wrt[i] is disconnected, return value i will be
                   None
        - 'Disconnected' : returns variables of type DisconnectedType
    null_gradients : {'raise', 'return'}
        Defines the behaviour if some of the variables in `wrt` have a
        null gradient. The possibles values are:

        - 'raise' : raise a NullTypeGradError exception
        - 'return' : return the null gradients

    Returns
    -------
    variable or list/tuple of variables (matches `wrt`)
        symbolic expression of gradient of `cost` with respect to each
        of the `wrt` terms.  If an element of `wrt` is not
        differentiable with respect to the output, then a zero
        variable is returned.

    """
    t0 = time.time()
    global tensor
    if tensor is None:
        from theano import tensor

    if cost is None:
        if known_grads is None:
            raise AssertionError("cost and known_grads can't both be None.")

    if cost is not None and isinstance(cost.type, NullType):
        raise ValueError("Can't differentiate a NaN cost."
                         "cost is NaN because " +
                         cost.type.why_null)

    if cost is not None and cost.ndim != 0:
        raise TypeError("cost must be a scalar.")

    if isinstance(wrt, set):
        raise TypeError("wrt must not be a set. sets have no defined "
                        "iteration order, so we can't return gradients in a"
                        "  matching order.")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)
    if not using_list and not using_tuple:
        wrt = [wrt]

    for elem in wrt:
        if not isinstance(elem, Variable):
            raise TypeError("Expected Variable, got " + str(elem) +
                            " of type " + str(type(elem)))

    outputs = []
    if cost is not None:
        outputs.append(cost)
    if known_grads is not None:
        outputs.extend(list(known_grads.keys()))

    var_to_app_to_idx = _populate_var_to_app_to_idx(
        outputs, wrt, consider_constant)

    # build a dict mapping var to the gradient of cost with respect to var
    grad_dict = OrderedDict()

    if known_grads is None:
        known_grads = OrderedDict()
    else:
        m = "known_grads must be an OrderedDict. "
        assert isinstance(known_grads, OrderedDict) or len(known_grads) <= 1, m

    # The gradient of the cost is 1 unless specified otherwise by known_grads.
    if cost is not None:
        if cost in known_grads:
            g_cost = known_grads[cost]
        else:
            g_cost = _float_ones_like(cost)
        # g_cost may be Disconnected or NullType. A creative use of the
        # function, sure, but nonetheless one we can and should support.
        # So before we try to cast it make sure it even has a dtype
        if (hasattr(g_cost.type, 'dtype') and
                cost.type.dtype in tensor.continuous_dtypes):
                # Here we enforce the constraint that floating point variables
                # have the same dtype as their gradient.
                g_cost = g_cost.astype(cost.type.dtype)
        # DO NOT enforce g_cost to be 0 if cost is an integer.
        # This is to be enforced by the Op.grad method for the
        # Op that outputs cost.
        if hasattr(g_cost.type, 'dtype'):
            assert g_cost.type.dtype in tensor.continuous_dtypes

        grad_dict[cost] = g_cost

    for var in known_grads:
        g_var = known_grads[var]

        if not hasattr(g_var, 'type'):
            raise TypeError('output grads must be theano variables.'
                            'Ambiguous whether %s should be made into tensor'
                            ' or sparse theano variable' % str(type(g_var)))

        if (not isinstance(g_var.type, (NullType, DisconnectedType)) and
                'float' not in str(g_var.type.dtype)):
            raise TypeError("Gradients must always be NullType, "
                            "DisconnectedType, or continuous, but grad was "
                            "given a known_grad of type " + str(g_var.type))

        # DO NOT check that these gradients are equal to 0 if var is int
        # The gradient is allowed to be non-zero on var in that case
        # Ops outputing var should not backpropagate its gradient further
        # but that is enforced elsewhere (grep for only_connected_to_int)

        grad_dict[var] = g_var

    def handle_disconnected(var):
            message = ("grad method was asked to compute the gradient "
                       "with respect to a variable that is not part of "
                       "the computational graph of the cost, or is used "
                       "only by a non-differentiable operator: %s" % var)
            if disconnected_inputs == 'ignore':
                pass
            elif disconnected_inputs == 'warn':
                warnings.warn(message, stacklevel=2)
            elif disconnected_inputs == 'raise':
                message = utils.get_variable_trace_string(var)
                raise DisconnectedInputError(message)
            else:
                raise ValueError("Invalid value for keyword "
                                 "'disconnected_inputs', valid values are "
                                 "'ignore', 'warn' and 'raise'.")

    # variables that do not influence the cost have zero gradient.
    # if wrt is such a variable, populate the grad_dict with this info
    # so that wrt not being in var_to_app_to_idx won't cause an error below
    # according to the flag, possibly raise an error if wrt is disconnected
    for elem in wrt:
        if elem not in var_to_app_to_idx and elem is not cost \
                and elem not in grad_dict:
            handle_disconnected(elem)
            grad_dict[elem] = disconnected_type()

    cost_name = None
    if add_names and cost is not None:
        cost_name = cost.name

    # Make sure we didn't initialize the grad_dict with any ints
    # The gradient may NEVER be an int, even if the variable is an int.
    # Read the Op contract and talk to Ian Goodfellow before changing this!
    for var in grad_dict:
        g = grad_dict[var]
        if hasattr(g.type, 'dtype'):
            assert g.type.dtype in tensor.float_dtypes

    rval = _populate_grad_dict(var_to_app_to_idx,
                               grad_dict, wrt, cost_name)

    for i in xrange(len(rval)):
        if isinstance(rval[i].type, NullType):
            if null_gradients == 'raise':
                raise NullTypeGradError("tensor.grad encountered a NaN. " +
                                        rval[i].type.why_null)
            else:
                assert null_gradients == 'return'
        if isinstance(rval[i].type, DisconnectedType):
            handle_disconnected(rval[i])
            if return_disconnected == 'zero':
                rval[i] = _float_zeros_like(wrt[i])
            elif return_disconnected == 'None':
                rval[i] = None
            else:
                assert return_disconnected == 'Disconnected'

    if using_tuple:
        rval = tuple(rval)
    elif not using_list:
        rval, = rval
    t1 = time.time()
    global grad_time
    grad_time += t1 - t0
    return rval


def subgraph_grad(wrt, end, start=None, cost=None, details=False):
    '''
    With respect to `wrt`, computes gradients of cost and/or from
    existing `start` gradients, up to the `end` variables of a
    symbolic digraph.  In other words, computes gradients for a
    subgraph of the symbolic theano function. Ignores all disconnected
    inputs.

    This can be useful when one needs to perform the gradient descent
    iteratively (e.g. one layer at a time in an MLP), or when a
    particular operation is not differentiable in theano
    (e.g. stochastic sampling from a multinomial). In the latter case,
    the gradient of the non-differentiable process could be
    approximated by user-defined formula, which could be calculated
    using the gradients of a cost with respect to samples (0s and
    1s). These gradients are obtained by performing a subgraph_grad
    from the `cost` or previously known gradients (`start`) up to the
    outputs of the stochastic process (`end`).  A dictionary mapping
    gradients obtained from the user-defined differentiation of the
    process, to variables, could then be fed into another
    subgraph_grad as `start` with any other `cost` (e.g. weight
    decay).

    In an MLP, we could use subgraph_grad to iteratively backpropagate:

    .. code-block:: python

        x, t = theano.tensor.fvector('x'), theano.tensor.fvector('t')
        w1 = theano.shared(np.random.randn(3,4))
        w2 = theano.shared(np.random.randn(4,2))
        a1 = theano.tensor.tanh(theano.tensor.dot(x,w1))
        a2 = theano.tensor.tanh(theano.tensor.dot(a1,w2))
        cost2 = theano.tensor.sqr(a2 - t).sum()
        cost2 += theano.tensor.sqr(w2.sum())
        cost1 = theano.tensor.sqr(w1.sum())

        params = [[w2],[w1]]
        costs = [cost2,cost1]
        grad_ends = [[a1], [x]]

        next_grad = None
        param_grads = []
        for i in xrange(2):
            param_grad, next_grad = theano.subgraph_grad(
                wrt=params[i], end=grad_ends[i],
                start=next_grad, cost=costs[i]
            )
            next_grad = dict(zip(grad_ends[i], next_grad))
            param_grads.extend(param_grad)

    :type wrt: list of variables
    :param wrt:
      Gradients are computed with respect to `wrt`.

    :type end: list of variables
    :param end:
      Theano variables at which to end gradient descent (they are
      considered constant in theano.grad).  For convenience, the
      gradients with respect to these variables are also returned.

    :type start: dictionary of variables
    :param start:
      If not None, a dictionary mapping variables to their
      gradients. This is useful when the gradient on some variables
      are known. These are used to compute the gradients backwards up
      to the variables in `end` (they are used as known_grad in
      theano.grad).

    :type cost: :class:`~theano.gof.Variable` scalar (0-dimensional) variable
    :param cost:
      Additional costs for which to compute the gradients.  For
      example, these could be weight decay, an l1 constraint, MSE,
      NLL, etc. May optionally be None if start is provided.  Warning
      : If the gradients of `cost` with respect to any of the `start`
      variables is already part of the `start` dictionary, then it may
      be counted twice with respect to `wrt` and `end`.

      .. warning::

        If the gradients of `cost` with respect to any of the `start`
        variables is already part of the `start` dictionary, then it
        may be counted twice with respect to `wrt` and `end`.


    :type details: bool
    :param details:
      When True, additionally returns the list of gradients from
      `start` and of `cost`, respectively, with respect to `wrt` (not
      `end`).

    :rtype: Tuple of 2 or 4 Lists of Variables

    :return: Returns lists of gradients with respect to `wrt` and `end`,
            respectively.

    .. versionadded:: 0.7
    '''
    assert ((cost is not None) or (start is not None))
    assert isinstance(end, list)
    assert isinstance(wrt, list)
    if start is not None:
        assert isinstance(start, dict)

    params = list(set(wrt + end))

    start_grads = None
    cost_grads = None
    if start is not None:
        start_grads = list(
            theano.grad(
                cost=None, wrt=params, known_grads=start,
                consider_constant=end,
                disconnected_inputs='ignore'
            )
        )

    if cost is not None:
        cost_grads = list(
            theano.grad(
                cost=cost, wrt=params,
                consider_constant=end,
                disconnected_inputs='ignore'
            )
        )

    grads = None
    if start is None:
        grads = cost_grads
    else:
        grads = start_grads
        if cost_grads is not None:
            for i in range(len(grads)):
                grads[i] += cost_grads[i]

    pgrads = OrderedDict(izip(params, grads))
    # separate wrt from end grads:
    wrt_grads = list(pgrads[k] for k in wrt)
    end_grads = list(pgrads[k] for k in end)

    if details:
        return wrt_grads, end_grads, start_grads, cost_grads

    return wrt_grads, end_grads


def _node_to_pattern(node):
    """ given an apply node, obtain its connection pattern
     this is just a wrapper around Op.connection_pattern
     that does type checking and supplies the default value
     if the method is not implemented
    """

    if hasattr(node.op, 'connection_pattern'):
        connection_pattern = node.op.connection_pattern(node)

        if not isinstance(connection_pattern, list):
            raise TypeError(
                "Op.connection_pattern should return " +
                ("list of list of bool, but for Op=%s" % node.op) +
                "got %s with type %s." % (connection_pattern,
                                          type(connection_pattern)))
        if len(connection_pattern) != len(node.inputs):
            raise ValueError(
                '%s.connection_pattern should have %d' %
                (node.op, len(node.inputs)) + ' rows but has %d.' %
                len(connection_pattern))
        for ii, output_pattern in enumerate(connection_pattern):
            if not isinstance(output_pattern, list):
                raise TypeError(
                    '%s.connection_pattern should return' %
                    node.op + ' a list of lists, but element %d' % ii +
                    'is %s of type %s.' % (output_pattern,
                                           type(output_pattern)))
    else:
        connection_pattern = [[True for output in node.outputs]
                              for ipt in node.inputs]
    assert isinstance(connection_pattern, list)
    assert len(connection_pattern) == len(node.inputs)
    for ii in xrange(len(node.inputs)):
        assert isinstance(connection_pattern[ii], list)
        assert len(connection_pattern[ii]) == len(node.outputs)
    return connection_pattern


def _populate_var_to_app_to_idx(outputs, wrt, consider_constant):
    """
    Helper function for grad function.

    outputs: a list of variables we want to take gradients of

    wrt: a list of variables we want to take the gradient with
        respect to.

    consider_constant: a list of variables not to backpropagate
        through.

    returns:

     var_to_app_to_idx:

      A dictionary mapping a variable to a second dictionary.
      The second dictionary maps apply nodes acting on this
      variable to the variable's index in the apply node's
      input list.

      This dictionary will only contain variables that
      meet two criteria:

       1) The elements of at least one output are a
          function of the elements of the variable

       2) The elements of the variable are a function of the
          elements of at least one member of wrt.

      This set is exactly the set of variables that connect
      the variables in wrt to the cost being differentiated.

      (A variable in consider_constant is not a function of
      anything)

    """

    # Validate and format consider_constant
    if consider_constant is None:
        consider_constant = []
    else:
        # error checking on consider_constant: verify that it is a collection
        # of theano variables
        # this is important, if someone accidentally passes a nested data
        # structure with theano variables at the leaves, only the root will
        # be properly considered constant
        try:
            iter(consider_constant)
        except TypeError:
            raise TypeError('consider_constant must be an iterable collection,'
                            ' got ' + str(type(consider_constant)))
        for elem in consider_constant:
            if not isinstance(elem, gof.Variable):
                raise TypeError('Elements of consider_constant must be '
                                'variables, but got ' + str(type(elem)))

    # var_to_app_to_idx[var][node] = [i,j] means node has
    # var as input at positions i and j
    var_to_app_to_idx = OrderedDict()

    # Set of variables that have been added to their true parents
    # ('true' here means that the elements of the variable are a function
    #  of the elements of the parent, according to the op's
    #  connection_pattern)
    # Note: we need to revisit the apply nodes repeatedly, because
    #       different outputs of the apply node are connected to
    #       different subsets of the inputs.
    accounted_for = set([])

    def account_for(var):
        # Don't visit the same variable twice
        if var in accounted_for:
            return
        accounted_for.add(var)

        # Constants are not a function of anything
        if var in consider_constant:
            return

        # Recursively add the variables that this variable is
        # a function of.
        if var.owner is not None:
            app = var.owner

            connection_pattern = _node_to_pattern(app)

            var_idx = app.outputs.index(var)

            for i, ipt in enumerate(app.inputs):

                # don't process ipt if it is not a true
                # parent of var
                if not connection_pattern[i][var_idx]:
                    continue

                if ipt not in var_to_app_to_idx:
                    # This object here *must* be an OrderedDict, because
                    # we iterate over its keys when adding up the terms of the
                    # gradient on ipt. If it is a regular dict, the grad method
                    # will return something that is analytically correct, but
                    # whose order of doing additions depends on the memory
                    # location of the apply nodes.
                    var_to_app_to_idx[ipt] = OrderedDict()
                app_to_idx = var_to_app_to_idx[ipt]
                if app not in app_to_idx:
                    app_to_idx[app] = []
                idx = app_to_idx[app]
                if i not in idx:
                    idx.append(i)
                account_for(ipt)

    # add all variables that are true ancestors of the cost
    for output in outputs:
        account_for(output)

    # determine which variables have elements of wrt as a true
    # ancestor. Do this with an upward pass starting from wrt,
    # following only true connections
    visited = set([])

    def visit(var):
        if var in visited:
            return
        if var not in var_to_app_to_idx:
            return
        visited.add(var)
        nodes = var_to_app_to_idx[var]
        for node in nodes:
            connection_pattern = _node_to_pattern(node)
            for idx in nodes[node]:
                for ii, output in enumerate(node.outputs):
                    if connection_pattern[idx][ii]:
                        visit(output)

    for elem in wrt:
        visit(elem)

    # Remove variables that don't have wrt as a true ancestor
    orig_vars = list(var_to_app_to_idx.keys())
    for var in orig_vars:
        if var not in visited:
            del var_to_app_to_idx[var]

    return var_to_app_to_idx


class NullTypeGradError(TypeError):
    """
    Raised when grad encounters a NullType.
    """


class DisconnectedInputError(ValueError):
    """
    Raised when grad is asked to compute the gradient
    with respect to a disconnected input and
    disconnected_inputs='raise'.
    """


def _populate_grad_dict(var_to_app_to_idx,
                        grad_dict, wrt, cost_name=None):
    """
        Helper function for grad function.

        var_to_app_to_idx: a dictionary mapping a variable to
                a second dictionary.
                the second dictionary maps apply nodes acting on
                this variable to the variable's index in the apply
                node's input list

        grad_dict: A dictionary mapping variables to their gradients.
                   Should be populated by grad function, which should:
                       -Set the gradient with respect to the cost to 1
                       -Load all gradients from known_grads, possibly
                        overriding the cost
                       -Set the gradient for disconnected
                        inputs to a variable with type DisconnectedType()

        wrt: the minimal set of variables that must be included in grad_dict

        cost_name: The name of the cost being differentiated, optional.
                    used to name the grad with respect to x as
                    (d<cost_name>/dx)

        returns: a list of gradients corresponding to wrt

    """
    # build a dict mapping node to the terms node contributes to each of
    # its inputs' gradients
    term_dict = OrderedDict()

    def access_term_cache(node):
        """ Populates term_dict[node] and returns it """

        if node not in term_dict:

            inputs = node.inputs

            output_grads = [access_grad_cache(var) for var in node.outputs]

            # list of bools indicating if each output is connected to the cost
            outputs_connected = [not isinstance(g.type, DisconnectedType)
                                 for g in output_grads]

            connection_pattern = _node_to_pattern(node)

            # list of bools indicating if each input is connected to the cost
            inputs_connected = [
                (True in [input_to_output and output_to_cost for
                          input_to_output, output_to_cost in
                          zip(input_to_outputs, outputs_connected)]) for
                input_to_outputs in connection_pattern
            ]

            # List of bools indicating if each output is an integer dtype
            output_is_int = [hasattr(output.type, 'dtype') and
                             output.type.dtype in theano.tensor.discrete_dtypes
                             for output in node.outputs]

            # List of bools indicating if each output is NullType
            ograd_is_nan = [isinstance(output.type, NullType)
                            for output in output_grads]

            # List of bools indicating if each input only has NullType outputs
            only_connected_to_nan = [
                (True not in
                 [in_to_out and out_to_cost and not out_nan
                  for in_to_out, out_to_cost, out_nan in
                  zip(in_to_outs, outputs_connected, ograd_is_nan)])
                for in_to_outs in connection_pattern]

            if True not in inputs_connected:
                # All outputs of this op are disconnected so we can skip
                # Calling the op's grad method and report that the inputs
                # are disconnected
                # (The op's grad method could do this too, but this saves the
                # implementer the trouble of worrying about this case)
                input_grads = [disconnected_type() for ipt in inputs]
            elif False not in only_connected_to_nan:
                # All inputs are only connected to nan gradients, so we don't
                # need to bother calling the grad method. We know the gradient
                # with respect to all connected inputs is nan.
                input_grads = []
                for connected in inputs_connected:
                    if connected:
                        input_grads.append(null_type())
                    else:
                        input_grads.append(disconnected_type())
            else:
                # At least one input of this op is connected to the cost so and
                # not all output gradients are undefined so we must
                # call the op's grad method

                # Each Op's grad function requires inputs and output_grads
                # If the Op destroys any input, but the grad expression uses
                # it, then chances are the resulting graph will have a
                # dependency cycle. We avoid this cycle by passing (symbolic)
                # copies of each destroyed input.
                try:
                    dinputs = [node.inputs[x[0]] for x in
                               itervalues(node.op.destroy_map)]
                except AttributeError:
                    dinputs = []

                def try_to_copy_if_needed(var):
                    if var in dinputs and hasattr(var, 'copy'):
                        return var.copy()
                    return var

                inputs = [try_to_copy_if_needed(ipt) for ipt in inputs]

                # Build a list of output gradients with the same dtype as
                # the corresponding output variable.
                # If an output is of a float dtype, we want to cast the
                # output gradient into the same dtype, to avoid having a
                # gradient graph with double precision (taking more memory,
                # and more computation).
                # If an output is of an integer dtype, then we just leave it
                # alone.
                # DO NOT force integer variables to have zero grad. This causes
                # bugs where we fail to detect disconnected or undefined
                # gradients.
                # DO NOT force integer variables to have integer dtype.
                # This is a violation of the op contract.
                new_output_grads = []
                for o, og in zip(node.outputs, output_grads):
                    o_dt = getattr(o.type, 'dtype', None)
                    og_dt = getattr(og.type, 'dtype', None)
                    if (o_dt not in theano.tensor.discrete_dtypes and
                            og_dt and o_dt != og_dt):
                        new_output_grads.append(og.astype(o_dt))
                    else:
                        new_output_grads.append(og)

                # Make sure that, if new_output_grads[i] has a floating point
                # dtype, it is the same dtype as outputs[i]
                for o, ng in zip(node.outputs, new_output_grads):
                    o_dt = getattr(o.type, 'dtype', None)
                    ng_dt = getattr(ng.type, 'dtype', None)
                    if (ng_dt is not None and
                            o_dt not in theano.tensor.discrete_dtypes):
                        assert ng_dt == o_dt

                # Someone who had obviously not read the Op contract tried
                # to modify this part of the function.
                # If you ever think it is a good idea to make an integer
                # valued gradient, please
                # 1) Read the Op contract again
                # 2) Talk to Ian Goodfellow
                # (Both of these sources will tell you not to do it)
                for ng in new_output_grads:
                    assert (getattr(ng.type, 'dtype', None)
                            not in theano.tensor.discrete_dtypes)

                # If config.compute_test_value is turned on, check that the
                # gradients on the outputs of this node have the right shape.
                # We also check the gradient on the inputs later--both checks
                # are needed, because some gradients are only ever specified
                # by the user, not computed by Op.grad, and some gradients are
                # only computed and returned, but never passed as another
                # node's output grads.
                for idx, packed in enumerate(izip(node.outputs,
                                             new_output_grads)):
                    orig_output, new_output_grad = packed
                    if not hasattr(orig_output, 'shape'):
                        continue
                    if isinstance(new_output_grad.type, DisconnectedType):
                        continue
                    for orig_output_v, new_output_grad_v in get_debug_values(
                            *packed):
                        o_shape = orig_output_v.shape
                        g_shape = new_output_grad_v.shape
                        if o_shape != g_shape:
                            raise ValueError(
                                "Got a gradient of shape " +
                                str(o_shape) + " on an output of shape " +
                                str(g_shape))

                input_grads = node.op.L_op(inputs, node.outputs,
                                           new_output_grads)

                if input_grads is None:
                    raise TypeError("%s.grad returned NoneType, "
                                    "expected iterable." % str(node.op))

                if len(input_grads) != len(inputs):
                    raise ValueError(("%s returned the wrong number of" +
                                      " gradient terms.") % str(node.op))
# We can not enforce this, as AdvancedSubtensor1 has an option to
# return the sparse grad for optimization reason.

                    #            for ig, i in zip(input_grads, inputs):
#                if (not isinstance(ig.type, (DisconnectedType, NullType)) and
#                    type(ig.type) != type(i.type)):
#                    raise ValueError(
#                        "%s returned the wrong type for gradient terms."
#                        " Sparse inputs must have sparse grads and dense"
#                        " inputs must have dense grad. Got %s, expected %s" %(
#                            str(node.op), ig.type, i.type))

            # must convert to list in case the op returns a tuple
            # we won't be able to post-process out the Nones if it does that
            input_grads = list(input_grads)

            # Need to propagate the NullType gradients; if an input grad is
            # not disconnected and the corresponding input is connected
            # to at least one output whose gradient is NullType then the input
            # grad should be NullType.
            for inp_idx in range(len(input_grads)):
                for out_idx in range(len(ograd_is_nan)):
                    if (ograd_is_nan[out_idx] and
                            connection_pattern[inp_idx][out_idx] and
                            not isinstance(input_grads[inp_idx].type,
                                           DisconnectedType)):
                        input_grads[inp_idx] = output_grads[out_idx]

            # Do type checking on the result

            # List of bools indicating if each input only has integer outputs
            only_connected_to_int = [
                (True not in
                 [in_to_out and out_to_cost and not out_int
                  for in_to_out, out_to_cost, out_int in
                  zip(in_to_outs, outputs_connected, output_is_int)])
                for in_to_outs in connection_pattern]

            for i, term in enumerate(input_grads):

                # Disallow Nones
                if term is None:
                    # We don't know what None means. in the past it has been
                    # used to mean undefined, zero, or disconnected.
                    # We therefore don't allow it because its usage has become
                    # so muddied.
                    raise TypeError(
                        ('%s.grad returned None for' +
                         ' a gradient term, '
                         'this is prohibited. Instead of None,'
                         'return zeros_like(input), disconnected_type(),'
                         ' or a NullType variable such as those made with '
                         'the grad_undefined or grad_unimplemented helper '
                         'functions.') % node.op)

                # Check that the gradient term for this input
                # has the right shape
                if hasattr(term, 'shape'):
                    orig_ipt = inputs[i]
                    for orig_ipt_v, term_v in get_debug_values(orig_ipt, term):
                        i_shape = orig_ipt_v.shape
                        t_shape = term_v.shape
                        if i_shape != t_shape:
                            raise ValueError(
                                "%s.grad returned object of "
                                "shape %s as gradient term on input %d "
                                "of shape %s" % (node.op, t_shape, i, i_shape))

                if not isinstance(term.type,
                                  (NullType, DisconnectedType)):
                    if term.type.dtype not in theano.tensor.float_dtypes:
                        raise TypeError(str(node.op) + '.grad illegally '
                                        ' returned an integer-valued variable.'
                                        ' (Input index %d, dtype %s)' % (
                                            i, term.type.dtype))

                    if only_connected_to_nan[i]:
                        assert isinstance(term.type, NullType)

                    if only_connected_to_int[i]:
                        # This term has only integer outputs and we know
                        # it's not undefined or disconnected
                        # The only other valid thing it can be is 0

                        is_zero = _is_zero(term)
                        assert is_zero in ['yes', 'no', 'maybe']
                        if is_zero == 'maybe':
                            msg = ("%s.grad returned %s of type %s for input"
                                   " %d. This input's only connections to "
                                   "the cost through this op are via "
                                   "integer-valued outputs so it should be "
                                   "NullType, DisconnectedType, or some form "
                                   "of zeros. It is not NullType or "
                                   "DisconnectedType and theano can't "
                                   "simplify it to a constant, so it's not "
                                   "verifiably zeros.")

                            msg %= (node.op, term, type(term), i)

                        elif is_zero == 'no':
                            msg = ("%s.grad returned %s of type %s for input"
                                   " %d. Since this input is only connected "
                                   "to integer-valued outputs, it should "
                                   "evaluate to zeros, but it evaluates to"
                                   "%s.")

                            msg %= (node.op, term, type(term), i,
                                    theano.get_scalar_constant_value(term))

                            raise ValueError(msg)

            # Check that op.connection_pattern matches the connectivity
            # logic driving the op.grad method
            for i, (ipt, ig, connected) in enumerate(
                zip(inputs, input_grads, inputs_connected)
            ):
                actually_connected = \
                    not isinstance(ig.type, DisconnectedType)

                if actually_connected and not connected:
                    msg = ("%s.grad returned %s of type %s for input %d."
                           " Expected DisconnectedType instance based on "
                           " the output of the op's connection_pattern "
                           "method.")
                    msg %= (str(node.op), str(ig), str(ig.type), i)
                    raise TypeError(msg)

                elif connected and not actually_connected:
                    msg = "%s.grad returned DisconnectedType for input %d."
                    msg %= (str(node.op), i)
                    if hasattr(node.op, 'connection_pattern'):
                        msg += (' Its connection_pattern method does not'
                                ' allow this.')
                        raise TypeError(msg)
                    else:
                        msg += (' You may want to implement a '
                                'connection_pattern method for it.')
                        warnings.warn(msg)

            # cache the result
            term_dict[node] = input_grads

        return term_dict[node]

    # populate grad_dict[var] and return it
    def access_grad_cache(var):
        if var not in grad_dict:
            # If var is not in grad_dict already, we must compute it
            if var in var_to_app_to_idx:
                null_terms = []
                terms = []
                node_to_idx = var_to_app_to_idx[var]
                for node in node_to_idx:
                    for idx in node_to_idx[node]:

                        term = access_term_cache(node)[idx]

                        if not isinstance(term, gof.Variable):
                            raise TypeError(
                                "%s.grad returned %s, expected"
                                " Variable instance." % (str(node.op),
                                                         type(term)))

                        if isinstance(term.type, NullType):
                            null_terms.append(term)
                            continue

                        # Don't try to sum up DisconnectedType placeholders
                        if isinstance(term.type, DisconnectedType):
                            continue

                        if hasattr(var, 'ndim') and term.ndim != var.ndim:
                            raise ValueError(
                                ("%s.grad returned a term with"
                                 " %d dimensions, but %d are required.") % (
                                     str(node.op), term.ndim, var.ndim))

                        terms.append(term)

                # Add up the terms to get the total gradient on this variable
                if len(null_terms) > 0:
                    # At least one term is a NullType : the total gradient
                    # will also be a NullType
                    grad_dict[var] = null_terms[0]
                elif len(terms) > 0:
                    # the next line is like sum(terms) but doesn't add an
                    # extraneous TensorConstant(0)
                    grad_dict[var] = reduce(lambda x, y: x + y, terms)
                else:
                    grad_dict[var] = disconnected_type()

                if cost_name is not None and var.name is not None:
                    grad_dict[var].name = '(d%s/d%s)' % (cost_name, var.name)
            else:
                # this variable isn't connected to the cost in the
                # computational graph
                grad_dict[var] = disconnected_type()
        # end if cache miss
        return grad_dict[var]

    rval = [access_grad_cache(elem) for elem in wrt]

    return rval


def _float_zeros_like(x):
    """ Like zeros_like, but forces the object to have a
    a floating point dtype """

    rval = x.zeros_like()

    if rval.type.dtype.find('float') != -1:
        return rval

    return rval.astype(theano.config.floatX)


def _float_ones_like(x):
    """ Like ones_like, but forces the object to have a
    floating point dtype """

    dtype = x.type.dtype
    if dtype not in tensor.float_dtypes:
        dtype = theano.config.floatX

    return x.ones_like(dtype=dtype)


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
                'float16': 1e-1,
                numpy.dtype('float64'): 1e-7,
                numpy.dtype('float32'): 3e-4,
                numpy.dtype('float16'): 1e-1}

    def __init__(self, f, pt, eps=None, out_type=None):
        """Return the gradient of f at pt.

        :param f: a differentiable function such that f(*pt) is a scalar
        :param pt: an ndarray, a list of ndarrays or tuple of ndarrays
        :param out_type: dtype of output, if complex (i.e. 'complex32' or
        'complex64')
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

        total_size = builtins.sum(prod(sh) for sh in shapes)

        working_dtype = builtins.min(
            (self.type_eps[dt], dt) for dt in dtypes)[1]

        # create un-initialized memory
        x = numpy.ndarray((total_size,), dtype=working_dtype)
        # (not out_type is None) --> (out_type is not None) ???
        if (out_type is not None) and (out_type.startswith('complex')):
            gx = numpy.ndarray((total_size,), dtype=out_type)
        else:
            gx = numpy.ndarray((total_size,), dtype=working_dtype)

        if eps is None:
            eps = builtins.max(self.type_eps[dt] for dt in dtypes)

        # set up aliases so that apt[i] is backed by memory in x
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

            # TODO: remove this when it is clear that the next
            # replacemement does not pose problems of its own.  It was replaced
            # for its inability to handle complex variables.
            # gx[i] = numpy.asarray((f_eps - f_x) / eps)

            gx[i] = ((f_eps - f_x) / eps)

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
        # The numpy.asarray are needed as if a or b is a sparse matrix
        # this would result in a numpy.matrix and not a numpy.ndarray
        # and the behave differently causing problem later.
        # In particular a_npy_matrix.flatten().shape == (1, n_element)
        abs_err = numpy.asarray(abs_err)
        rel_err = numpy.asarray(rel_err)
        return (abs_err, rel_err)

    def abs_rel_errors(self, g_pt):
        """Return the abs and rel error of gradient estimate `g_pt`

        `g_pt` must be a list of ndarrays of the same length as self.gf,
        otherwise a ValueError is raised.

        Corresponding ndarrays in `g_pt` and `self.gf` must have the same
        shape or ValueError is raised.

        """
        if len(g_pt) != len(self.gf):
            raise ValueError('argument has wrong number of elements',
                             len(g_pt))
        errs = []
        for i, (a, b) in enumerate(zip(g_pt, self.gf)):
            if a.shape != b.shape:
                raise ValueError('argument element %i has wrong shape %s' % (
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
        return (max_arg, max_pos, abs_errs[max_arg], rel_errs[max_arg])


def mode_not_debug(mode):
    if isinstance(mode, DebugMode):
        opt = mode.optimizer
        return FAST_RUN.clone(optimizer=opt)
    else:
        return mode


def verify_grad(fun, pt, n_tests=2, rng=None, eps=None,
                out_type=None, abs_tol=None,
                rel_tol=None, mode=None, cast_to_output_type=False,
                no_debug_ref=True):
    """Test a gradient by Finite Difference Method. Raise error on failure.

    Example:
        >>> verify_grad(theano.tensor.tanh,
        ...             (numpy.asarray([[2,3,4], [-1, 3.3, 9.9]]),),
        ...             rng=numpy.random)

    Raises an Exception if the difference between the analytic gradient and
    numerical gradient (computed through the Finite Difference Method) of a
    random projection of the fun's output to a scalar exceeds the given
    tolerance.

    :param fun: a Python function that takes Theano variables as inputs,
        and returns a Theano variable. For instance, an Op instance with
        a single output.
    :param pt: the list of numpy.ndarrays to use as input values.
        These arrays must be either float16, float32, or float64 arrays.
    :param n_tests: number of times to run the test
    :param rng: random number generator used to sample u, we test gradient
        of sum(u * fun) at pt
    :param eps: stepsize used in the Finite Difference Method (Default
        None is type-dependent)
        Raising the value of eps can raise or lower the absolute and
        relative errors of the verification depending on the
        Op. Raising eps does not lower the verification quality
        for linear operations. It
        is better to raise eps than raising abs_tol or rel_tol.
    :param out_type: dtype of output, if complex (i.e. 'complex32' or
        'complex64')
    :param abs_tol: absolute tolerance used as threshold for gradient
        comparison
    :param rel_tol: relative tolerance used as threshold for gradient
        comparison
    :param cast_to_output_type: if the output is float32 and
        cast_to_output_type is True, cast the random projection to
        float32. Otherwise it is float64. float16 is not handled here.
    :param no_debug_ref: Don't use DebugMode for the numerical
        gradient function.

    :note: This function does not support multiple outputs. In
        tests/test_scan.py there is an experimental verify_grad that
        covers that case as well by using random projections.

    """
    # The import is here to prevent circular import.
    from theano import compile, shared
    import theano.tensor
    from theano.tensor import as_tensor_variable, TensorType
    assert isinstance(pt, (list, tuple))
    pt = [numpy.array(p) for p in pt]

    for i, p in enumerate(pt):
        if p.dtype not in ('float16', 'float32', 'float64'):
            raise TypeError(
                ('verify_grad can work only with floating point '
                 'inputs, but input %i has dtype "%s".') % (i, p.dtype))

    _type_tol = dict(  # relative error tolerances for different types
        float16=5e-2,
        float32=1e-2,
        float64=1e-4)

    if abs_tol is None:
        abs_tol = builtins.max(_type_tol[str(p.dtype)] for p in pt)
    if rel_tol is None:
        rel_tol = builtins.max(_type_tol[str(p.dtype)] for p in pt)

    if rng is None:
        raise TypeError(('rng should be a valid instance of '
                        'numpy.random.RandomState. You may '
                         'want to use theano.tests.unittest'
                         '_tools.verify_grad instead of '
                         'theano.gradient.verify_grad.'))

    # We allow input downcast in function, because numeric_grad works in the
    # most precise dtype used among the inputs, so we may need to cast some.
    def function(inputs, output, name, mode=mode):
        f = compile.function(inputs, output, accept_inplace=True,
                             allow_input_downcast=True, mode=mode,
                             on_unused_input='ignore', name=name)
        return f

    tensor_pt = [
        TensorType(
            as_tensor_variable(p).dtype,
            as_tensor_variable(p).broadcastable)(name='input %i' % i)
        for i, p in enumerate(pt)]

    # fun can be either a function or an actual Op instance
    o_output = fun(*tensor_pt)

    if isinstance(o_output, list):
        raise NotImplementedError(('cant (yet) autotest gradient of fun '
                                   'with multiple outputs'))
        # we could make loop over outputs making random projections R for each,
        # but this doesn't handle the case where not all the outputs are
        # differentiable... so I leave this as TODO for now -JB.

    o_fn = function(tensor_pt, o_output, name='gradient.py fwd')
    o_fn_out = o_fn(*[p.copy() for p in pt])

    if isinstance(o_fn_out, tuple) or isinstance(o_fn_out, list):
        raise TypeError(
            'It seems like you are trying to use verify_grad '
            'on an op or a function which outputs a list: there should'
            ' be a single (array-like) output instead')

    # random_projection should not have elements too small,
    # otherwise too much precision is lost in numerical gradient
    def random_projection():
        plain = rng.rand(*o_fn_out.shape) + 0.5
        if cast_to_output_type and o_output.dtype == "float32":
            return numpy.array(plain, o_output.dtype)
        return plain

    t_r = shared(random_projection())
    t_r.name = 'random_projection'

    # random projection of o onto t_r
    # This sum() is defined above, it's not the builtin sum.
    cost = theano.tensor.sum(t_r * o_output)

    if no_debug_ref:
        mode_for_cost = mode_not_debug(mode)
    else:
        mode_for_cost = mode

    cost_fn = function(tensor_pt, cost, name='gradient.py cost',
                       mode=mode_for_cost)

    symbolic_grad = grad(cost, tensor_pt,
                         disconnected_inputs='ignore')

    grad_fn = function(tensor_pt, symbolic_grad,
                       name='gradient.py symbolic grad')

    for test_num in xrange(n_tests):
        try:
            num_grad = numeric_grad(cost_fn, [p.copy() for p in pt],
                                    eps, out_type)

            analytic_grad = grad_fn(*[p.copy() for p in pt])

            # Since `tensor_pt` is a list, `analytic_grad` should be one too.
            assert isinstance(analytic_grad, list)

            max_arg, max_err_pos, max_abs_err, max_rel_err = num_grad.max_err(
                analytic_grad, abs_tol, rel_tol)

            if max_abs_err > abs_tol and max_rel_err > rel_tol:

                raise verify_grad.E_grad(max_arg, max_err_pos,
                                         max_abs_err, max_rel_err,
                                         abs_tol, rel_tol)

            # get new random projection for next test
            if test_num < n_tests - 1:
                t_r.set_value(random_projection(), borrow=True)
        except Exception as e:
            e.args += ("\nThe error happened with the following inputs:", pt,
                       "\nThe value of eps is:", eps,
                       "\nThe out_type is:", out_type)
            raise


class GradientError(Exception):
    """This error is raised when a gradient is calculated, but incorrect."""
    def __init__(self, arg, err_pos, abs_err, rel_err, abs_tol, rel_tol):
        Exception.__init__(self)  # to be compatible with python2.4
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


def jacobian(expression, wrt, consider_constant=None,
             disconnected_inputs='raise'):
    """
    :type expression: Vector (1-dimensional) Variable
    :type wrt: Variable or list of Variables

    :param consider_constant: a list of expressions not to backpropagate
        through

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
        return format_as(using_list, using_tuple,
                         grad(expression,
                              wrt,
                              consider_constant=consider_constant,
                              disconnected_inputs=disconnected_inputs))

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            rval = grad(expr[idx],
                        inp,
                        consider_constant=consider_constant,
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


def hessian(cost, wrt, consider_constant=None,
            disconnected_inputs='raise'):
    """
    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Vector (1-dimensional tensor) 'Variable' or list of
               vectors (1-dimensional tensors) Variables

    :param consider_constant: a list of expressions not to backpropagate
        through

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
        expr = grad(cost, input, consider_constant=consider_constant,
                    disconnected_inputs=disconnected_inputs)

        # It is possible that the inputs are disconnected from expr,
        # even if they are connected to cost.
        # This should not be an error.
        hess, updates = theano.scan(lambda i, y, x: grad(
            y[i],
            x,
            consider_constant=consider_constant,
            disconnected_inputs='ignore'),
            sequences=arange(expr.shape[0]),
            non_sequences=[expr, input])
        assert not updates, \
            ("Scan has returned a list of updates. This should not "
             "happen! Report this to theano-users (also include the "
             "script that generated the error)")
        hessians.append(hess)
    return format_as(using_list, using_tuple, hessians)


def _is_zero(x):
    """
    Returns 'yes', 'no', or 'maybe' indicating whether x
    is always 0.
    'maybe' means that x is an expression that is complicated enough
    that we can't tell that it simplifies to 0.
    """
    if not hasattr(x, 'type'):
        return np.all(x == 0.)
    if isinstance(x.type, NullType):
        return 'no'
    if isinstance(x.type, DisconnectedType):
        return 'yes'

    no_constant_value = True
    try:
        constant_value = theano.get_scalar_constant_value(x)
        no_constant_value = False
    except theano.tensor.basic.NotScalarConstantError:
        pass

    if no_constant_value:
        return 'maybe'

    if constant_value != 0.:
        return 'no'

    return 'yes'


class ConsiderConstant(ViewOp):
    def grad(self, args, g_outs):
        return [g_out.zeros_like(g_out) for g_out in g_outs]


consider_constant_ = ConsiderConstant()


# I create a function only to have the doc show well.
def consider_constant(x):
    """
    DEPRECATED: use zero_grad() or disconnected_grad() instead.

    Consider an expression constant when computing gradients.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, it will not be backpropagated
    through. In other words, the gradient of the expression is
    truncated to 0.

    :param x: A Theano expression whose gradient should be truncated.

    :return: The expression is returned unmodified, but its gradient
        is now truncated to 0.

    .. versionadded:: 0.7
    """
    warnings.warn((
        "consider_constant() is deprecated, use zero_grad() or "
        "disconnected_grad() instead."), stacklevel=3)

    return consider_constant_(x)


class ZeroGrad(ViewOp):
    def grad(self, args, g_outs):
        return [g_out.zeros_like(g_out) for g_out in g_outs]


zero_grad_ = ZeroGrad()


def zero_grad(x):
    """
    Consider an expression constant when computing gradients.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, it will be backpropagated
    through with a value of zero. In other words, the gradient of
    the expression is truncated to 0.

    :param x: A Theano expression whose gradient should be truncated.

    :return: The expression is returned unmodified, but its gradient
        is now truncated to 0.
    """
    return zero_grad_(x)


class DisconnectedGrad(ViewOp):
    def grad(self, args, g_outs):
        return [disconnected_type() for g_out in g_outs]

    def R_op(self, inputs, eval_points):
        return [None]

    def connection_pattern(self, node):
        return [[False]]


disconnected_grad_ = DisconnectedGrad()


def disconnected_grad(x):
    """
    Consider an expression constant when computing gradients,
    while effectively not backpropagating through it.

    The expression itself is unaffected, but when its gradient is
    computed, or the gradient of another expression that this
    expression is a subexpression of, it will not be backpropagated
    through. This is effectively equivalent to truncating the gradient
    expression to 0, but is executed faster than zero_grad(), which stilll
    has to go through the underlying computational graph related to the
    expression.

    :param x: A Theano expression whose gradient should not be
              backpropagated through.

    :return: The expression is returned unmodified, but its gradient
        is now effectively truncated to 0.
    """
    return disconnected_grad_(x)


class GradClip(ViewOp):
    # See doc in user fct grad_clip
    __props__ = ()

    def __init__(self, clip_lower_bound, clip_upper_bound):
        # We do not put those member in __eq__ or __hash__
        # as they do not influence the perform of this op.
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [theano.tensor.clip(g_out, self.clip_lower_bound,
                                   self.clip_upper_bound)
                for g_out in g_outs]


def grad_clip(x, lower_bound, upper_bound):
    """
    This op do a view in the forward, but clip the gradient.

    This is an elemwise operation.

    :param x: the variable we want its gradient inputs clipped
    :param lower_bound: The lower bound of the gradient value
    :param upper_bound: The upper bound of the gradient value.

    :examples:

        x = theano.tensor.scalar()

        z = theano.tensor.grad(grad_clip(x, -1, 1)**2, x)
        z2 = theano.tensor.grad(x**2, x)

        f = theano.function([x], outputs = [z, z2])

        print(f(2.0))  # output (1.0, 4.0)

    :note: We register an opt in tensor/opt.py that remove the GradClip.
       So it have 0 cost in the forward and only do work in the grad.

    """
    return GradClip(lower_bound, upper_bound)(x)


class GradScale(ViewOp):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def grad(self, args, g_outs):
        return [self.multiplier * g_out for g_out in g_outs]


def grad_scale(x, multiplier):
    """
    This op scale or inverse the gradient in the backpropagation.

    :param x: the variable we want its gradient inputs scale
    :param multiplier: scale of the gradient

    :examples:

        x = theano.tensor.fscalar()
        fx = theano.tensor.sin(x)

        fp = theano.tensor.grad(fx, wrt=x)
        fprime = theano.function([x], fp)
        print(fprime(2))#-0.416

        f_inverse=grad_scale(fx,-1.)
        fpp = theano.tensor.grad(f_inverse, wrt=x)
        fpprime = theano.function([x], fpp)
        print(fpprime(2))#0.416
    """
    return GradScale(multiplier)(x)
