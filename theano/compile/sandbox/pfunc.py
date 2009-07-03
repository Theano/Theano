"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

from theano.gof import Container, Variable, generic, graph, Constant, Value
from theano.compile import function, In
from theano.compile.sandbox.sharedvalue import SharedVariable, shared

class Param(object):
    def __init__(self, variable, default=None, name=None, mutable=False, strict=False,
            implicit=None):
        """
        :param variable: A node in an expression graph to set with each function call.

        :param default: The default value to use at call-time (can also be a Container where
        the function will find a value at call-time.)

        :param name: A string to identify this parameter from function kwargs.

        :param mutable: True -> function is allowed to modify this argument.

        :param strict: False -> function arguments may be copied or casted to match the
        type required by the parameter `variable`.  True -> function arguments must exactly match the type
        required by `variable`.

        :param implicit: see help(theano.io.In)

        """
        self.variable = variable
        self.default = default
        self.name = name
        self.mutable = mutable
        self.strict = strict
        self.implicit = implicit

def pfunc(params, outputs=None, mode=None, updates=[]):
    """Function-constructor for graphs with shared variables.

    :type params: list of either Variable or Param instances.
    :param params: function parameters, these are not allowed to be shared
    variables

    :type outputs: list of Variables or Out instances
    :param outputs: expressions to compute

    :param mode: compilation mode

    :type updates: iterable over pairs (shared_variable, new_expression). List, tuple or dict.
    :param updates: update the values for SharedVariable inputs according to these expressions

    :rtype: theano.compile.Function
    :returns: a callable object that will compute the outputs (given the inputs)
    and update the implicit function arguments according to the `updates`.

    """
    # Note: in its early design, pfunc was also meant to accept another
    # parameter, 'givens'. This was a dictionary assigning some specific
    # values to some of the Variable in the graph, so as to allow the
    # function to possibly make some optimizations at compile time.
    # In the end, this feature was not kept, because it was not obvious
    # how to implement it, nor whether it was really needed.
    # If one wants to add this feature in the future, it may be easier instead
    # to add a new parameter to 'Param' to indicate that some input of the
    # function is taking a specific constant value.

    if not isinstance(outputs, list):
        computed_list = [outputs]
    else:
        # Copy list (because it may be extended later).
        computed_list = [out for out in outputs]

    # transform params into theano.compile.In objects.
    #
    # call theano.function
    inputs = [_pfunc_param_to_in(p) for p in params]

    set_of_param_variables = set([i.variable for i in inputs])

    # It was decided, as a first step, to prevent shared variables from being
    # used as function inputs. Although it is technically possible, it is also
    # potentially ambiguous and dangerous. This restriction may be revisited in
    # the future if there is a need for such a feature.
    if any([isinstance(v, SharedVariable) for v in set_of_param_variables]):
        raise TypeError('Cannot use a shared variable (%s) as explicit input '
                % v)

    # Add update values as quantities that must be computed.
    new_updates = {}
    for (store_into, update_val) in iter_over_pairs(updates):
        if not isinstance(update_val, Variable):
            # The value for the update is not a Variable: we cast it into
            # a shared Variable so that it can be used by 'function'. Note that
            # it means the update value may change if it is mutable and its
            # value is modified after the function is created.
            update_val = shared(update_val)
        computed_list.append(update_val)
        new_updates[store_into] = update_val
    updates = new_updates

    # Obtain all inputs we need to compute what we want.
    graph_inputs = graph.inputs(computed_list,
            blockers=set([i.variable for i in inputs]))

    shared_inputs = [i for i in graph_inputs if isinstance(i, SharedVariable)]

    # Add shared variables (from shared_inputs) that were not already present in the list of
    # params.
    inputs += [In(variable=si, value=si.container, mutable=False) 
        for si in shared_inputs
        if si not in set_of_param_variables]

    # Iterate over the updates, which are either pairs
    # (shared_var, expressionvariable), or a similar dictionary.
    # For each shared_variable, find the In instance that we created for it in the inputs list.
    # Give that In instance (in_sv) an update expression.
    # 
    # I think we usually want to set these Inputs to be mutable,
    # ... are there exceptions?

    for (sv, new_val) in iter_over_pairs(updates):
        in_sv = None
        for in_sv_i in inputs:
            if in_sv_i.variable is sv:
                assert in_sv is None
                in_sv = in_sv_i
        if in_sv is None:
            # This variable was not used anywhere and thus is not in the input
            # list yet.
            inputs.append(In(variable=sv, value=sv.container, mutable=True,
                update=new_val))
        else:
            in_sv.update = new_val
            in_sv.mutable = True 

    return function(inputs, outputs, mode, accept_inplace=False)

def _pfunc_param_to_in(param):
    if isinstance(param, Constant):
        raise TypeError('Constants not allowed in param list', param)
    if isinstance(param, Value):
        raise NotImplementedError()
    if isinstance(param, Variable): #includes SharedVariable
        return In(variable=param)
    elif isinstance(param, Param):
        return In(
                variable=param.variable, 
                name=param.name,
                value=param.default,
                mutable=param.mutable,
                strict=param.strict,
                implicit = param.implicit)
    raise NotImplementedError()


def iter_over_pairs(pairs):
    """
    Return an iterator over pairs present in the 'pairs' input.

    :type pairs: dictionary or iterable
    :param pairs: The pairs to iterate upon. These may be stored either as
    (key, value) items in a dictionary, or directly as pairs in any kind of
    iterable structure

    :rtype: iterable
    :returns: an iterable yielding pairs

    """
    if isinstance(pairs, dict):
        return pairs.iteritems()
    else:
        return pairs
