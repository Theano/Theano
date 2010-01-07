"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

from theano.gof import Container, Variable, generic, graph, Constant, Value
from theano.compile import function, In
from theano.compile.sandbox.sharedvalue import SharedVariable, shared
import numpy # for backport to 2.4, to get any().

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

def pfunc(params, outputs=None, mode=None, updates=[], givens=[]):
    """Function-constructor for graphs with shared variables.

    :type params: list of either Variable or Param instances.
    :param params: function parameters, these are not allowed to be shared
    variables

    :type outputs: list of Variables or Out instances
    :param outputs: expressions to compute

    :type mode: string or `theano.compile.Mode` instance.
    :param mode: compilation mode

    :type updates: iterable over pairs (shared_variable, new_expression). List, tuple or dict.
    :param updates: update the values for SharedVariable inputs according to these expressions

    :type givens: iterable over pairs (Var1, Var2) of Variables. List, tuple or dict.  The Var1
    and Var2 in each pair must have the same Type.

    :param givens: specific substitutions to make in the computation graph (Var2 replaces
    Var1).  

    :rtype: theano.compile.Function
    :returns: a callable object that will compute the outputs (given the inputs)
    and update the implicit function arguments according to the `updates`.

    :note: Regarding givens: Be careful to make sure that these substitutions are
    independent--behaviour when Var1 of one pair appears in the graph leading to Var2 in
    another expression is undefined.  Replacements specified with givens are different from
    optimizations in that Var2 is not expected to be equivalent to Var1.

    """
    #
    # This function works by cloning the graph (except for the inputs), and then shipping it
    # off to compile.function
    # (There it will be cloned again, unnecessarily, because it doesn't know that we already
    # cloned it.)
    # 
    # First, it clones the replacements named in the givens argument, and points each Var1 to
    # the clone of Var2.
    # Then it sets the inputs in the clone dictionary.
    # After these steps, we are assuming that the clone dictionary contains all the inputs to
    # the computation graph.
    #
    # Then it clones the outputs and the update expressions.  This rebuilds a computation graph
    # from the inputs and the givens.
    #

    if not isinstance(params,(list,tuple)):
        raise Exception("in pfunc() the first argument must be a list or a tuple")

    # initialize the clone_d mapping with the `givens` argument
    clone_d = {}
    def v_clone(v):
        return _v_clone(v, clone_d)

    try:
        givens = givens.items() # converts a dictionary to the sort of list that we want.
    except:
        pass
    for v_orig, v_repl in givens:
        if not isinstance(v_orig, Variable):
            raise TypeError('given keys must be Variable', v_orig)
        if not isinstance(v_repl, Variable):
            v_repl = shared(v_repl)
        assert v_orig not in clone_d
        clone_d[v_orig] = v_clone(v_repl)

    # transform params into theano.compile.In objects.
    #
    # call theano.function
    inputs = [_pfunc_param_to_in(p) for p in params]

    #Switch inputs to cloned variables
    input_variables = [clone_d.setdefault(i.variable, i.variable) for i in inputs]
    for i, iv in zip(inputs, input_variables):
        i.variable = iv

    set_of_param_variables = set(input_variables)

    # It was decided, as a first step, to prevent shared variables from being
    # used as function inputs. Although it is technically possible, it is also
    # potentially ambiguous and dangerous. This restriction may be revisited in
    # the future if there is a need for such a feature.
    if numpy.any([isinstance(v, SharedVariable) for v in set_of_param_variables]):
        raise TypeError('Cannot use a shared variable (%s) as explicit input '
                % v)

    # computed_list is a list of output variables
    if isinstance(outputs, list):
        for v in outputs:
            if not isinstance(v, Variable):
                raise TypeError('outputs must be theano Variable instances', v)
        # Copy list (because it may be extended later).
        computed_list = [v_clone(o) for o in outputs]
        cloned_outputs = list(computed_list)
    else:
        if not isinstance(outputs, Variable):
            raise TypeError('outputs must be a theano Variable instance or list of.', outputs)
        cloned_outputs = v_clone(outputs)
        computed_list = [cloned_outputs]

    # Add update values as quantities that must be computed.
    # Here, we
    #  - extend the computed_list
    #  - replace some update expressions (but update keys remain)
    new_updates = {}
    for (store_into, update_val) in iter_over_pairs(updates):
        if not isinstance(store_into, SharedVariable):
            raise TypeError('update target must be a SharedVariable', store_into)
        update_val = v_clone(store_into.filter_update(update_val))
        if update_val.type != store_into.type:
            raise TypeError('an update must have the same type as the original shared variable', 
                    (store_into, store_into.type,
                        update_val, update_val.type))
        computed_list.append(update_val)
        new_updates[store_into] = update_val
    updates = new_updates

    # Obtain all inputs we need to compute what we want.
    graph_inputs = graph.inputs(computed_list,
            blockers=set_of_param_variables)

    shared_inputs = [i for i in graph_inputs if isinstance(i, SharedVariable)]

    # Add shared variables (from shared_inputs) that were not already present in the list of
    # params.
    inputs += [In(variable=si, value=si.container, mutable=False) 
        for si in shared_inputs
        if si not in set_of_param_variables]
    del shared_inputs

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

    return function(inputs, cloned_outputs, mode, accept_inplace=False)

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
    raise NotImplementedError('Unknown parameter type: %s' % type(param))


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

#TODO: Make these non-recursive so they can deal with larger graphs
def _a_clone(a, dct):
    if a is None:
        return None
    if a not in dct:
        for i in a.inputs:
            _v_clone(i, dct)
        dct[a] = a.clone_with_new_inputs([dct[i] for i in a.inputs])
        for old_o, new_o in zip(a.outputs, dct[a].outputs):
            dct.setdefault(old_o, new_o)
    return dct[a]

def _v_clone(v, dct):
    assert v is not None
    if v.owner:
        _a_clone(v.owner, dct)
    return dct.setdefault(v, v)


