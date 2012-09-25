"""Driver for gradient calculations."""

__authors__ = "James Bergstra, Razvan Pascanu, Arnaud Bergeron, Ian Goodfellow"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"

import __builtin__
import logging
import warnings
_logger = logging.getLogger('theano.gradient')

import numpy  # for numeric_grad

import theano

from theano import gof
from theano.gof import Variable
from theano.gof.python25 import all
import theano.gof.utils
from theano.gof.null_type import NullType
from theano.printing import min_informative_str
# we can't do "import theano.tensor"
# tensor depends on theano.compile
# theano.compile depends on theano.gradient (this file)
# the reason theano.compile depends on theano.gradient
# is that theano.compile.builders contains the op from graph
# functionality and it uses theano.gradient to implement
# the new op's grad method
tensor = None

_msg_retType = 'op.grad(...) returned a non-list'


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

    return (NullType(
        (
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

    seen_nodes = {}

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
                except:
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
    #end _traverse

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

    # make copies of f and grads so we don't modify the client's copy
    f = list(f)
    grads = list(eval_points)

    for elem in consider_constant:
        assert elem not in f
        f.append(elem)
        grads.append(elem.zeros_like())

    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]

    arg1 = zip(f, eval_points)
    arg2 = list(wrt)

    gmap = grad_sources_inputs(
        arg1,
        arg2)

    # Note : If p is not in gmap there can be several reasons, among which
    # is the fact that p might not be part of the computational graph. A
    # simple example is that for a+b for e.g. a[0] is not part of the graph,
    # so Theano does not know how to compute TT.grad(TT.sum(a+b), a[0])
    # such subtle cases can be fixed by a more careful implementation of the
    # gradient, but for now Theano needs to throw an exception, and make the
    # user aware that it does not know how to compute that gradient
    ret = []
    for p in wrt:
        if p in gmap:
            ret.append(gmap[p])
        else:
            message = (
                "Lop method was asked to compute the gradient "
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
                raise ValueError(
                    "Invalid value for keyword "
                    "'disconnected_inputs', valid values are "
                    "'ignore', 'warn' and 'raise'.")
            ret.append(p.zeros_like())

    return format_as(using_list, using_tuple, ret)


#########################
# Gradient
#########################

def grad(cost, wrt, g_cost=None, consider_constant=None,
        disconnected_inputs='raise', add_names=True):
    """
    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Variable or list of Variables.
    :type g_cost: Scalar Variable, or None.
    :param g_cost: an expression for the gradient through cost.  The default is
        ``ones_like(cost)``.
    :param consider_constant: a list of expressions not to backpropagate
        through

    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    :type add_names: bool
    :param add_names: If True, variables generated by grad will be named
        (d<cost.name>/d<wrt.name>) provided that both cost and wrt have
        names

    :rtype: Variable or list/tuple of Variables (depending upon `wrt`)

    :return: symbolic expression of gradient of `cost` with respect to `wrt`.
             If an element of `wrt` is not differentiable with respect
             to the output, then a zero variable is returned.
             It returns an object of same type as `wrt`: a list/tuple
             or Variable in all cases.

    """
    global tensor
    if tensor is None:
        from theano import tensor

    if isinstance(cost.type, NullType):
        raise ValueError("Can't differentiate a NaN cost."
            "cost is NaN because " + \
                cost.type.why_null)

    if cost.ndim != 0:
        raise TypeError("cost must be a scalar.")

    if consider_constant is None:
        consider_constant = []
    else:
        # error checking on consider_constant: verify that it is a collection
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

    if isinstance(wrt, set):
        raise TypeError("wrt must not be a set. sets have no defined "
                "iteration order, so we can't return gradients in a matching"
                " order.")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)
    if not using_list and not using_tuple:
        wrt = [wrt]

    for elem in wrt:
        if not isinstance(elem, Variable):
            raise TypeError("Expected Variable, got " + str(elem) +
                    " of type "+str(type(elem)))

    var_to_node_to_idx = _populate_var_to_node_to_idx([cost], wrt)

    # build a dict mapping var to the gradient of cost with respect to var
    grad_dict = {}
    # by default, the gradient of the cost is 1
    if g_cost is None:
        g_cost = _float_ones_like(cost)
    grad_dict[cost] = g_cost

    # the gradient of the constants is 0
    for const in consider_constant:
        grad_dict[const] = DisconnectedType()()

    # variables that do not influence the cost have zero gradient.
    # if wrt is such a variable, populate the grad_dict with this info
    # so that wrt not being in var_to_node_to_idx won't cause an error below
    # according to the flag, possibly raise an error if wrt is disconnected
    for elem in wrt:
        if elem not in var_to_node_to_idx and elem is not cost:
            message = ("grad method was asked to compute the gradient "
                    "with respect to a variable that is not part of "
                    "the computational graph of the cost, or is used "
                    "only by a non-differentiable operator: %s" % elem)
            if disconnected_inputs == 'ignore':
                pass
            elif disconnected_inputs == 'warn':
                warnings.warn(message, stacklevel=2)
            elif disconnected_inputs == 'raise':
                raise ValueError(message)
            else:
                raise ValueError("Invalid value for keyword "
                        "'disconnected_inputs', valid values are "
                        "'ignore', 'warn' and 'raise'.")
            grad_dict[elem] = DisconnectedType()()

    cost_name = None
    if add_names:
        cost_name = cost.name

    # Make sure we didn't initialize the grad_dict with any ints
    for var in grad_dict:
        g = grad_dict[var]
        if hasattr(g.type, 'dtype'):
            assert g.type.dtype.find('float') != -1

    rval = _populate_grad_dict(var_to_node_to_idx,
            grad_dict, wrt, cost_name)

    for i in xrange(len(rval)):
        if isinstance(rval[i].type, DisconnectedType):
            rval[i] = _float_zeros_like(wrt[i])

    if using_tuple:
        rval = tuple(rval)
    elif not using_list:
        rval, = rval
    return rval


def _node_to_pattern(node):
    """ given an apply node, obtain its connection pattern
     this is just a wrapper around Op.connection_pattern
     that does type checking and supplies the default value
     if the method is not implemented
    """

    if hasattr(node.op, 'connection_pattern'):
        connection_pattern = node.op.connection_pattern(node)

        if not isinstance(connection_pattern, list):
            raise TypeError("Op.connection_pattern should return " + \
                    ("list of list of bool, but for Op=%s" % node.op) +\
                    "got %s with type %s." % (connection_pattern,
                        type(connection_pattern)))
        if len(connection_pattern) != len(node.inputs):
            raise ValueError('%s.connection_pattern should have %d' %
                    (node.op, len(node.inputs)) + ' rows but has %d.' %
                    len(connection_pattern))
        for ii, output_pattern in enumerate(connection_pattern):
            if not isinstance(output_pattern, list):
                raise TypeError('%s.connection_pattern should return' %
                        node.op + ' a list of lists, but element %d' % ii\
                        + 'is %s of type %s.' % (output_pattern,
                            type(output_pattern)))
    else:
        connection_pattern = \
            [[True for output in node.outputs]
                    for ipt in node.inputs]
    assert isinstance(connection_pattern, list)
    assert len(connection_pattern) == len(node.inputs)
    for ii in xrange(len(node.inputs)):
        assert isinstance(connection_pattern[ii], list)
        assert len(connection_pattern[ii]) == \
                len(node.outputs)
    return connection_pattern


def _populate_var_to_node_to_idx(outputs, wrt):
    """
    Common code shared between grad and grad_sources_inputs

    outputs: a list of variables we want to take gradients of

    wrt: a list of variables we want to take the gradient with
        respect to.

    returns:
        var_to_node_to_idx: a dictionary mapping a variable to
            a second dictionary.
            the second dictionary maps apply nodes acting on
            this variable to the variable's index in the apply
            node's input list
            This dictionary will only contain variables that
            meet two criteria:
                1) The elements of at least one output are a
                   function of the elements of the variable
                2) The elements of the variable are a function
                   of the elements of at least one member of
                   wrt
            This set is exactly the set of variables that
            connect the variables in wrt to the cost being
            differentiated.

    """

    # var_to_node_to_idx[var][node] = [i,j] means node has
    # var as input at positions i and j
    var_to_node_to_idx = {}
    # set of variables or nodes that have been added to their true parents
    # ('true' here means that the elements of the variable are a function
    #  of the elements of the parent, according to the op's
    #  connection_pattern)
    accounted_for = set([])

    def account_for(var):
        if var in accounted_for:
            return
        accounted_for.add(var)
        if var.owner is not None:
            node = var.owner
            if node not in accounted_for:
                accounted_for.add(node)

                connection_pattern = _node_to_pattern(node)

                var_idx = node.outputs.index(var)

                for i, ipt in enumerate(node.inputs):

                    #don't process ipt if it is not a true
                    #parent of var
                    if not connection_pattern[i][var_idx]:
                        continue

                    if ipt not in var_to_node_to_idx:
                        var_to_node_to_idx[ipt] = {}
                    node_to_idx = var_to_node_to_idx[ipt]
                    if node not in node_to_idx:
                        node_to_idx[node] = []
                    idx = node_to_idx[node]
                    assert i not in idx
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
        if var not in var_to_node_to_idx:
            return
        visited.add(var)
        nodes = var_to_node_to_idx[var]
        for node in nodes:
            connection_pattern = _node_to_pattern(node)
            for idx in nodes[node]:
                for ii, output in enumerate(node.outputs):
                    if connection_pattern[idx][ii]:
                        visit(output)

    for elem in wrt:
        visit(elem)

    # Remove variables that don't have wrt as a true ancestor
    orig_vars = list(var_to_node_to_idx.keys())
    for var in orig_vars:
        if var not in visited:
            del var_to_node_to_idx[var]

    return var_to_node_to_idx


def _populate_grad_dict(var_to_node_to_idx,
        grad_dict, wrt, cost_name=None):
    """
        Common code shared between grad_sources_inputs and grad

        var_to_node_to_idx: a dictionary mapping a variable to
                a second dictionary.
                the second dictionary maps apply nodes acting on
                this variable to the variable's index in the apply
                node's input list

        grad_dict: a dictionary mapping variables to their gradients
                   should be populated by grad or grad_sources_inputs

                        grad should set gradients to DisconnectedType()() for
                        variables to be considered constant, set the
                        gradient for the cost variable to g_cost, etc.

                        both should set the gradient for disconnected
                        inputs to a variable with type DisconnectedType()

        wrt: the minimal set of variables that must be included in grad_dict

        cost_name: The name of the cost being differentiated, optional.
                    used to name the grad with respect to x as
                    (d<cost_name>/dx)

        returns: a list of gradients corresponding to wrt

    """
    # build a dict mapping node to the terms node contributes to each of
    # its inputs' gradients
    term_dict = {}

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

            if True in inputs_connected:
                # At least one input of this op is connected to the cost so we must
                # call the op's grad method

                # Each Op's grad function requires inputs and output_grads
                # If the Op destroys any input, but the grad expression uses it,
                # then chances are the resulting graph will have a dependency
                # cycle. We avoid this cycle by passing (symbolic) copies of
                # each destroyed input.
                try:
                    dinputs = [node.inputs[x[0]] for x in
                            node.op.destroy_map.values()]
                except AttributeError:
                    dinputs = []

                def try_to_copy_if_needed(var):
                    if var in dinputs and hasattr(var, 'copy'):
                        return var.copy()
                    return var

                inputs = [try_to_copy_if_needed(ipt) for ipt in inputs]

                input_grads = node.op.grad(inputs, output_grads)

                if input_grads is None:
                    raise TypeError("%s.grad returned NoneType, "
                            "expected iterable." % str(node.op))

                if len(input_grads) != len(inputs):
                    raise ValueError(("%s returned the wrong number of" +\
                            " gradient terms.") % str(node.op))
            else:
                # All outputs of this op are disconnected so we can skip
                # Calling the op's grad method and report that the inputs
                # are disconnected
                # (The op's grad method could do this too, but this saves the
                # implementer the trouble of worrying about this case)
                input_grads = [DisconnectedType()() for ipt in inputs]

            # must convert to list in case the op returns a tuple
            # we won't be able to post-process out the Nones if it does that
            input_grads = list(input_grads)

            # Do type checking on the result

            #List of bools indicating if each output is an integer dtype
            output_is_int = [hasattr(output.type, 'dtype') and
                    output.type.dtype.find('int') != -1
                    for output in node.outputs]

            #List of bools indicating if each input only has integer outputs
            only_connected_to_int = [(True not in
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
                    raise TypeError(('%s.grad returned None for' +\
                             ' a gradient term, '
                            'this is prohibited. Instead of None,'
                            'return zeros_like(input), DisconnectedType()(),'
                            ' or a NullType variable such as those made with '
                            'the grad_undefined or grad_unimplemented helper '
                            'functions.') % node.op)

                if not isinstance(term.type,
                        (NullType, DisconnectedType)):
                    if term.type.dtype.find('float') == -1:
                        raise TypeError(str(node.op) + '.grad illegally '
                                ' returned an integer-valued variable.'
                                ' (Input index %d, dtype %s)' % (i,
                                    term.type.dtype))
                    if only_connected_to_int[i]:
                        # This term has only integer outputs and we know
                        # it's not undefined or disconnected
                        # The only other valid thing it can be is 0

                        no_constant_value = True
                        try:
                            constant_value = tensor.get_constant_value(term)
                            no_constant_value = False
                        except TypeError:
                            pass

                        extra_msg = ''

                        # The above won't work if it's a sparse type, handle sparse
                        # types here
                        if no_constant_value:
                            if isinstance(term.type, theano.sparse.SparseType):
                                if term.owner is not None and isinstance(term.owner.op,
                                        theano.sparse.CSM):
                                    data = term.owner.inputs[0]
                                    try:
                                        constant_value = tensor.get_constant_value(data)
                                        no_constant_value = False
                                    except TypeError:
                                        print theano.printing.min_informative_str(data)
                                        extra_msg += " It is a CSM, but its data isn't constant."
                                        pass
                                else:
                                    extra_msg += " It is a SparseType but theano doesn't know how"
                                    extra_msg += " to turn it into a constant."
                                #end if CSM
                            else:
                                extra_msg += " It is not a SparseType."
                            #end if SparseType
                        #end if no_constant_value

                        if no_constant_value:
                            msg = "%s.grad returned %s of type %s for input"
                            msg += " %d. This input's only connections to "
                            msg += "the cost through this op are via "
                            msg += "integer-valued outputs so it should be "
                            msg += "NullType, DisconnectedType, or some form "
                            msg += "of zeros. It is not NullType or "
                            msg += "DisconnectedType and theano can't "
                            msg += "simplify it to a constant, so it's not "
                            msg += "verifiably zeros."
                            msg += extra_msg

                            msg = msg % (str(node.op), str(term),
                                    str(type(term)), i)

                            raise ValueError(msg)
                        if constant_value != 0:
                            msg = "%s.grad returned %s of type %s for input"
                            msg += " %d. Since this input is only connected "
                            msg += "to integer-valued outputs, it should "
                            msg += "evaluate to zeros, but it evaluates to"
                            msg += "%s."

                            msg % (str(node.op), str(term), str(type(term)),
                                    i, str(constant_value))

                            raise ValueError(msg)

            #Check that op.connection_pattern matches the connectivity
            #logic driving the op.grad method
            for i, packed in \
                enumerate(zip(inputs, input_grads, inputs_connected)):
                ipt, ig, connected = packed
                actually_connected = \
                    not isinstance(ig.type, DisconnectedType)

                if actually_connected and not connected:
                    msg = "%s.grad returned %s of type %s for input %d."
                    msg += " Expected DisconnectedType instance based on "
                    msg += " the output of the op's connection_pattern "
                    msg += "method."
                    msg = msg % (str(node.op), str(ig), str(ig.type), i)
                    raise TypeError(msg)

                if connected and not actually_connected:
                    msg = "%s.grad returned DisconnectedType for input"
                    msg += " %d."
                    msg = msg % (str(node.op), i)
                    if hasattr(node.op, 'connection_pattern'):
                        msg += ' Its connection_pattern method does not'
                        msg += ' allow this.'
                        raise TypeError(msg)
                    else:
                        msg += ' You may want to implement a '
                        msg += 'connection_pattern method for it.'
                        warnings.warn(msg)

            #cache the result
            term_dict[node] = input_grads

        return term_dict[node]

    # populate grad_dict[var] and return it
    def access_grad_cache(var):
        if var not in grad_dict:
            if var in var_to_node_to_idx:
                terms = []
                node_to_idx = var_to_node_to_idx[var]
                for node in node_to_idx:
                    for idx in node_to_idx[node]:

                        term = access_term_cache(node)[idx]

                        if not isinstance(term, gof.Variable):
                            raise TypeError("%s.grad returned %s, expected"
                                    " Variable instance." % (str(node.op),
                                        type(term)))

                        if isinstance(term.type, NullType):
                            raise TypeError("tensor.grad "
                                "encountered a NaN. " +\
                                    term.type.why_null)

                        #Don't try to sum up DisconnectedType placeholders
                        if isinstance(term.type, DisconnectedType):
                            continue

                        terms.append(term)

                # Add up the terms to get the total gradient on this variable
                if len(terms) > 0:
                    # the next line is like sum(terms) but doesn't add an
                    # extraneous TensorConstant(0)
                    grad_dict[var] = reduce(lambda x, y: x + y, terms)
                else:
                    grad_dict[var] = DisconnectedType()()

                if cost_name is not None and var.name is not None:
                    grad_dict[var].name = '(d%s/d%s)' % (cost_name, var.name)
            else:
                # this variable isn't connected to the cost in the computational
                # graph
                grad_dict[var] = DisconnectedType()()
        return grad_dict[var]

    rval = [access_grad_cache(elem) for elem in wrt]

    return rval


def grad_sources_inputs(sources, graph_inputs):
    """
    Used to compute the gradient of a cost with respect to all the
    variables between graph_input and cost, but in the special
    case where you don't know the cost, you only know its gradient
    on a set of intermediate values.

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

    :rtype: dictionary whose keys and values are of type Variable
    :return: mapping from each Variable encountered in the backward
        traversal to the gradient with respect to that Variable.

    It is assumed that there is some objective J shared between all members of
    sources, so that for each v, gradient-on-v is the gradient of J with
    respect to v

    """

    outputs, output_grads = zip(*sources)

    for output_grad in output_grads:
        if not hasattr(output_grad, 'type'):
            raise TypeError('output grads must be theano variables.'
                    'Ambiguous whether %s should be made into tensor'
                    ' or sparse theano variable' % str(type(output_grad)))

    if graph_inputs is None:
        graph_inputs = gof.graph.inputs(outputs)

    wrt = graph_inputs

    var_to_node_to_idx = _populate_var_to_node_to_idx(outputs, wrt)

    # build a dict mapping var to the gradient of cost with respect to var
    grad_dict = {}
    # by default, the gradient of the cost is 1
    for output, output_grad in sources:
        grad_dict[output] = output_grad

    # variables that do not influence the cost have zero gradient.
    # if wrt is such a variable, populate the grad_dict with this info
    # so that wrt not being in var_to_node_to_idx won't cause an error below
    # according to the flag, possibly raise an error if wrt is disconnected
    for elem in wrt:
        if elem not in var_to_node_to_idx and elem not in outputs:
            grad_dict[elem] = DisconnectedType()()

    _populate_grad_dict(var_to_node_to_idx,
            grad_dict, wrt)

    # post-process out the DisconnectedTypes
    for key in grad_dict:
        if isinstance(grad_dict[key].type, DisconnectedType):
            if hasattr(key, 'zeros_like'):
                grad_dict[key] = _float_zeros_like(key)

    return grad_dict


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

    rval = tensor.ones_like(x)

    if rval.type.dtype.find('float') != -1:
        return rval

    return rval.astype(theano.config.floatX)


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

        total_size = __builtin__.sum(prod(sh) for sh in shapes)

        working_dtype = __builtin__.min((self.type_eps[dt], dt)
                                        for dt in dtypes)[1]

        # create un-initialized memory
        x = numpy.ndarray((total_size,), dtype=working_dtype)
        if (not out_type is None) and (out_type.startswith('complex')):
            gx = numpy.ndarray((total_size,), dtype=out_type)
        else:
            gx = numpy.ndarray((total_size,), dtype=working_dtype)

        if eps is None:
            eps = __builtin__.max(self.type_eps[dt] for dt in dtypes)

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


def verify_grad(fun, pt, n_tests=2, rng=None, eps=None,
                out_type=None, abs_tol=None,
                rel_tol=None, mode=None, cast_to_output_type=False):
    """Test a gradient by Finite Difference Method. Raise error on failure.

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
        Raising the value of eps can raise or lower the absolute and
        relative error of the verification depending of the
        Op. Raising the eps do not lower the verification quality. It
        is better to raise eps then raising abs_tol or rel_tol.
    :param out_type: dtype of output, if complex (i.e. 'complex32' or
        'complex64')
    :param abs_tol: absolute tolerance used as threshold for gradient
        comparison
    :param rel_tol: relative tolerance used as threshold for gradient
        comparison

    :note: WARNING to unit-test writers: if `op` is a function that builds
        a graph, try to make it a SMALL graph.  Often verify grad is run
        in debug mode, which can be very slow if it has to verify a lot of
        intermediate computations.

    :note: This function does not support multiple outputs. In
        tests/test_scan.py there is an experimental verify_grad that
        covers that case as well by using random projections.

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

    _type_tol = dict(  # relative error tolerances for different types
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
                    allow_input_downcast=True, on_unused_input='ignore')
        else:
            f = compile.function(inputs, output, accept_inplace=True,
                    allow_input_downcast=True, mode=mode,
                    on_unused_input='ignore')
        return f

    tensor_pt = [TensorType(
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
    t_r.name = 'random_projection'

    # random projection of o onto t_r
    # This sum() is defined above, it's not the builtin sum.
    cost = theano.tensor.sum(t_r * o_output)

    cost_fn = function(tensor_pt, cost)

    # todo-- determine if this is actually needed
    g_cost = as_tensor_variable(1.0, name='g_cost')
    if cast_to_output_type:
        g_cost = cast(g_cost, o_output.dtype)

    symbolic_grad = grad(cost, tensor_pt, g_cost,
                         disconnected_inputs='ignore')

    grad_fn = function(tensor_pt, symbolic_grad)

    for test_num in xrange(n_tests):
        num_grad = numeric_grad(cost_fn, [p.copy() for p in pt], eps, out_type)

        analytic_grad = grad_fn(*[p.copy() for p in pt])

        # Since `tensor_pt` is a list, `analytic_grad` should be one too.
        assert isinstance(analytic_grad, list)

        max_arg, max_err_pos, max_abs_err, max_rel_err =\
                num_grad.max_err(analytic_grad, abs_tol, rel_tol)

        if max_abs_err > abs_tol and max_rel_err > rel_tol:

            raise verify_grad.E_grad(max_arg, max_err_pos,
                    max_abs_err, max_rel_err, abs_tol, rel_tol)

        # get new random projection for next test
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
        return format_as(using_list, using_tuple, grad(expression, wrt))

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
        expr = grad(cost, input)
        hess, updates = theano.scan(lambda i, y, x: grad(
                            y[i],
                            x,
                            consider_constant=consider_constant,
                            disconnected_inputs=disconnected_inputs),
                       sequences=arange(expr.shape[0]),
                       non_sequences=[expr, input])
        assert not updates, \
                ("Scan has returned a list of updates. This should not "
                 "happen! Report this to theano-users (also include the "
                 "script that generated the error)")
        hessians.append(hess)
    return format_as(using_list, using_tuple, hessians)
