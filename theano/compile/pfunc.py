"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

from theano.gof import Container, Variable, generic, graph, Constant, Value
from theano.compile import orig_function, In, Out
from theano.compile.sharedvalue import SharedVariable, shared
import numpy # for backport to 2.4, to get any().

class Param(object):
    def __init__(self, variable, default=None, name=None, mutable=False, strict=False,
            implicit=None):
        """
        :param variable: A variable in an expression graph to use as a compiled-function parameter

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

def pfunc(params, outputs=None, mode=None, updates=[], givens=[],
        no_default_updates=False, accept_inplace=False, name=None):
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

    :type no_default_updates: either bool or list of Variables
    :param no_default_updates: if True, do not perform any automatic update on Variables.
    If False (default), perform them all. Else, perform automatic updates on all Variables
    that are neither in "updates" nor in "no_default_updates".

    :param name: an optional name for this fct. If used, the profile mode will print the time spent in this fct.

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

    clone_d = {}
    # Updates as list and dictionary.
    # They will also store the 'default_update' expressions applicable.
    # The dictionary is used to look up the existence of the keys, and to store
    # the final (cloned) update expressions.
    # The list of pairs is used to iterate in a consistent order while adding
    # new pairs.
    update_d = {}
    update_expr = []
    # list of shared inputs that are used as inputs of the graph
    shared_inputs = []

    def clone_v_get_shared_updates(v):
        '''Clone a variable and its inputs, until all are in clone_d.
        Also appends all shared variables met along the way to shared_inputs,
        and their default_update (if applicable) to update_d and update_expr.
        '''
        assert v is not None
        if v.owner:
            clone_a(v.owner)
        elif isinstance(v, SharedVariable):
            if v not in shared_inputs:
                shared_inputs.append(v)

            if hasattr(v, 'default_update'):
                # Check that v should not be excluded from the default updates list
                if no_default_updates is False or\
                        (isinstance(no_default_updates, list) and\
                        v not in no_default_updates):
                    # Do not use default_update if a "real" update was provided
                    if v not in update_d:
                        v_update = v.filter_update(v.default_update)
                        if v_update.type != v.type:
                            raise TypeError('an update must have the same type as the original shared variable',
                                    (v, v.type, v_update, v_update.type))
                        update_d[v] = v_update
                        update_expr.append((v, v_update))

        return clone_d.setdefault(v, v)

    def clone_a(a):
        if a is None:
            return None
        if a not in clone_d:
            for i in a.inputs:
                clone_v_get_shared_updates(i)
            clone_d[a] = a.clone_with_new_inputs([clone_d[i] for i in a.inputs])
            for old_o, new_o in zip(a.outputs, clone_d[a].outputs):
                clone_d.setdefault(old_o, new_o)
        return clone_d[a]

    #def v_clone(v):
    #    return _v_clone(v, clone_d)

    # initialize the clone_d mapping with the `givens` argument
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
        clone_d[v_orig] = clone_v_get_shared_updates(v_repl)

    # transform params into theano.compile.In objects.
    inputs = [_pfunc_param_to_in(p) for p in params]

    #Switch inputs to cloned variables
    input_variables = [clone_d.setdefault(i.variable, i.variable) for i in inputs]
    for i, iv in zip(inputs, input_variables):
        i.variable = iv

    #set_of_param_variables = set(input_variables)

    # It was decided, as a first step, to prevent shared variables from being
    # used as function inputs. Although it is technically possible, it is also
    # potentially ambiguous and dangerous. This restriction may be revisited in
    # the future if there is a need for such a feature.
    if numpy.any([isinstance(v, SharedVariable) for v in input_variables]):
        raise TypeError('Cannot use a shared variable (%s) as explicit input '
                % v)

    # Fill update_d and update_expr with provided updates
    for (store_into, update_val) in iter_over_pairs(updates):
        if not isinstance(store_into, SharedVariable):
            raise TypeError('update target must be a SharedVariable', store_into)
        if store_into in update_d:
            raise ValueError('this shared variable already has an update expression',
                    (store_into, update_d[store_into]))

        update_val = store_into.filter_update(update_val)
        if update_val.type != store_into.type:
            raise TypeError('an update must have the same type as the original shared variable', 
                    (store_into, store_into.type,
                        update_val, update_val.type))
        update_d[store_into] = update_val
        update_expr.append((store_into, update_val))


    # computed_list is a list of output variables (which will be extended later)
    #computed_list = []

    # Elements of "outputs" are here cloned to "cloned_outputs"
    if isinstance(outputs, list):
        cloned_outputs = []
        for v in outputs:
            if isinstance(v, Variable):
                cloned_v = clone_v_get_shared_updates(v)
                cloned_outputs.append(cloned_v)
            elif isinstance(v, Out):
                cloned_v = clone_v_get_shared_updates(v.variable)
                cloned_outputs.append(Out(cloned_v, borrow=v.borrow))
            else:
                raise TypeError('outputs must be theano Variable or Out instances', v)
            #computed_list.append(cloned_v)
    else:
        if isinstance(outputs, Variable):
            cloned_v = clone_v_get_shared_updates(outputs)
            cloned_outputs = cloned_v
            #computed_list.append(cloned_v)
        elif isinstance(outputs, Out):
            cloned_v = clone_v_get_shared_updates(outputs.variable)
            cloned_outputs = Out(cloned_v, borrow=outputs.borrow)
            #computed_list.append(cloned_v)
        elif outputs is None:
            cloned_outputs = [] # TODO: return None
        else:
            raise TypeError('output must be a theano Variable or Out instance (or list of them)', outputs)

    # Iterate over update_expr, cloning its elements, and updating
    # shared_inputs, update_d and update_expr from the SharedVariables
    # we discover.
    # If the variable to be updated is a shared variable not already
    # in shared_inputs, add it.
    # Note: we extend update_expr while iterating over it.
    i = 0
    while i<len(update_expr):
        v, v_update = update_expr[i]
        cloned_v_update = clone_v_get_shared_updates(v_update)
        update_d[v] = cloned_v_update
        if isinstance(v, SharedVariable) and v not in shared_inputs:
            shared_inputs.append(v)
        i += 1

    #updates = update_d #?
    for sv in shared_inputs:
        if sv in update_d:
            si = In(variable=sv, value=sv.container, mutable=True,
                    update=update_d[sv])
        else:
            si = In(variable=sv, value=sv.container, mutable=False)
        inputs.append(si)

    return orig_function(inputs, cloned_outputs, mode,
            accept_inplace=accept_inplace, name=name)

    if 0:
        # Add update values as quantities that must be computed.
        # Here, we
        #  - extend the computed_list
        #  - replace some update expressions (but update keys remain)
        new_updates = {}
        for (store_into, update_val) in iter_over_pairs(updates):
            if not isinstance(store_into, SharedVariable):
                raise TypeError('update target must be a SharedVariable', store_into)
            if store_into in new_updates:
                raise ValueError('this shared variable already has an update expression',
                        (store_into, new_updates[store_into]))
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

        return orig_function(inputs, cloned_outputs, mode, accept_inplace=accept_inplace,name=name)

def _pfunc_param_to_in(param):
    if isinstance(param, Constant):
        raise TypeError('Constants not allowed in param list', param)
    #if isinstance(param, Value):
        #return In(variable=param)
        #raise NotImplementedError()
    if isinstance(param, Variable): #N.B. includes Value and SharedVariable
        return In(variable=param)
    elif isinstance(param, Param):
        return In(
                variable=param.variable, 
                name=param.name,
                value=param.default,
                mutable=param.mutable,
                strict=param.strict,
                implicit = param.implicit)
    raise TypeError('Unknown parameter type: %s' % type(param))


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


