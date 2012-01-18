"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

import numpy # for backport to 2.4, to get any().

from profiling import ProfileStats
from theano.gof import Container, Variable, generic, graph, Constant, Value
from theano.compile import orig_function, In, Out
from theano.compile.sharedvalue import SharedVariable, shared
from theano import config

import logging
_logger=logging.getLogger("theano.compile.pfunc")

def rebuild_collect_shared( outputs
                           , inputs             = None
                           , replace            = None
                           , updates            = None
                           , rebuild_strict     = True
                           , copy_inputs_over   = True
                           , no_default_updates = False
                          ):
    """
    Function that allows replacing subgraphs of a computational
    graph.

    It returns a set of dictionaries and lists which collect (partial?)
    different information about shared variables. This info is required by
    `pfunc`.


    :type outputs: list of Theano Variables ( or Theano expressions)
    :param outputs: list of Theano variables or expressions representing the
                    outputs of the computational graph

    :type inputs: list of Theano Variables ( or Theano expressions)
    :param inputs: list of Theano variables or expressions representing the
                    inputs of the computational graph (or None)
    :type replace: dict
    :param replace: dictionary describing which subgraphs should be
                    replaced by what

    :type updates: dict
    :param updates: dictionary describing updates expressions for shared
                    variables

    :type rebuild_strict: bool
    :param rebuild_strict: flag, if true the type of all inputs should be
                            the same as the for the current node

    :type copy_inputs_over: bool
    :param copy_inputs_over: flag; if False it will clone inputs

    :type no_default_updates: either bool or list of Variables
    :param no_default_updates: if True, do not perform any automatic update
                               on Variables. If False (default), perform
                               them all. Else, perform automatic updates
                               on all Variables that are neither in
                               "updates" nor in "no_default_updates".

    """

    if isinstance(outputs,tuple):
        outputs = list(outputs)

    ## This function implements similar functionality as graph.clone
    ## and it should be merged with that
    clone_d = {}
    update_d = {}
    update_expr = []
    # list of shared inputs that are used as inputs of the graph
    shared_inputs = []


    def clone_v_get_shared_updates(v, copy_inputs_over):
        '''
        Clones a variable and its inputs recursively until all are in
        clone_d. Also appends all shared variables met along the way to
        shared inputs, and their default_update (if applicable) to update_d
        and update_expr.

        v can have an env attached to it, case in which we want to clone
        constants ( to avoid having a constant belonging to two envs)
        '''
        # this co-recurses with clone_a
        assert v is not None
        if v in clone_d:
            return clone_d[v]
        if v.owner:
            clone_a(v.owner, copy_inputs_over)
            return clone_d.setdefault(v,v)
        elif isinstance(v, SharedVariable):
            if v not in shared_inputs:
                shared_inputs.append(v)
            if hasattr(v, 'default_update'):
                # Check that v should not be excluded from the default
                # updates list
                if    ( no_default_updates is False or
                        ( isinstance(no_default_updates, list) and
                          v not in no_default_updates
                        )
                      ):
                    # Do not use default_update if a "real" update was
                    # provided
                    if v not in update_d:
                        v_update = v.type.filter_variable(v.default_update)
                        if v_update.type != v.type:
                            raise TypeError(
                                ( 'an update must have the same type as '
                                  'the original shared variable'  )
                                , (v, v.type, v_update, v_update.type))
                        update_d[v] = v_update
                        update_expr.append((v, v_update))
        if not copy_inputs_over or (isinstance(v, Constant) and
                                    hasattr(v,'env')):
            ### Cloning shared variables implies copying their underlying
            ### memory buffer ?? No.
            return clone_d.setdefault(v,v.clone())
        else:
            return clone_d.setdefault(v,v)

    def clone_a(a, copy_inputs_over):
        '''
        Clones a variable and its inputs recursively until all are in
        clone_d. It occures with clone_v_get_shared_updates
        '''
        if a is None:
            return None
        if a not in clone_d:
            for i in a.inputs:
                clone_v_get_shared_updates(i, copy_inputs_over)

            clone_d[a] = a.clone_with_new_inputs([clone_d[i] for i in
                                                  a.inputs],
                                                 strict = rebuild_strict)
            for old_o, new_o in zip(a.outputs, clone_d[a].outputs):
                clone_d.setdefault(old_o,new_o)
        return clone_d[a]


    # intialize the clone_d mapping with the replace dictionary
    if replace is None:
        replace = []
    try:
        replace_pairs = replace.items()
    except Exception:
        replace_pairs = replace

    for v_orig, v_repl in replace_pairs:
        if not isinstance(v_orig,Variable):
            raise TypeError('given keys must be Variable', v_orig)
        if not isinstance(v_repl,Variable):
            v_repl = shared(v_repl)
        assert v_orig not in clone_d
        clone_d[v_orig] = clone_v_get_shared_updates(v_repl,
                                                     copy_inputs_over)

    if inputs is None:
        inputs = []

    def clone_inputs(i):
        if not copy_inputs_over:
            return clone_d.setdefault(i,i.clone())
        else:
            return clone_d.setdefault(i,i)

    input_variables = [clone_inputs(i) for i in inputs]

    # It was decided, as a first step, to prevent shared variables from
    # being used as function inputs. Although it is technically possible,
    # it is also not clear when/how to use the value of that shared
    # variable (is it a default? ignored?, if the shared variable changes,
    # does that function default also change?).
    if numpy.any([isinstance(v, SharedVariable) for v in input_variables]):
        raise TypeError(('Cannot use a shared variable (%s) as explicit '
                         'input. Consider substituting a non-shared'
                         ' variable via the `givens` parameter') % v)

    # Fill update_d and update_expr with provided updates
    if updates is None:
        updates = []
    for (store_into, update_val) in iter_over_pairs(updates):
        if not isinstance(store_into, SharedVariable):
            raise TypeError('update target must be a SharedVariable'
                            , store_into)
        if store_into in update_d:
            raise ValueError(('this shared variable already has an update '
                              'expression'),
                              (store_into, update_d[store_into]))

        # filter_variable ensure smooth conversion of cpu/gpu Types
        update_val = store_into.type.filter_variable(update_val)
        if update_val.type != store_into.type:
            err_msg  = ( 'an update must have the same type as the '
                        'original shared variable(dest, dest.type, '
                        'update_val, update_val.type)')
            err_arg = ( store_into
                       , store_into.type
                       , update_val
                       , update_val.type)

            raise TypeError(err_msg, err_arg )
        update_d[store_into] = update_val
        update_expr.append((store_into, update_val))

    # Elements of "outputs" are here cloned to "cloned_outputs"
    if isinstance(outputs, list):
        cloned_outputs = []
        for v in outputs:
            if isinstance(v, Variable):
                cloned_v = clone_v_get_shared_updates(v, copy_inputs_over)
                cloned_outputs.append(cloned_v)
            elif isinstance(v, Out):
                cloned_v = clone_v_get_shared_updates(v.variable,
                                                      copy_inputs_over)
                cloned_outputs.append(Out(cloned_v, borrow=v.borrow))
            else:
                raise TypeError( ( 'outputs must be theano Variable or '
                                  'Out instances'), v)
            #computed_list.append(cloned_v)
    else:
        if isinstance(outputs, Variable):
            cloned_v = clone_v_get_shared_updates(outputs, copy_inputs_over)
            cloned_outputs = cloned_v
            #computed_list.append(cloned_v)
        elif isinstance(outputs, Out):
            cloned_v = clone_v_get_shared_updates(outputs.variable,
                                                  copy_inputs_over)
            cloned_outputs = Out(cloned_v, borrow=outputs.borrow)
            #computed_list.append(cloned_v)
        elif outputs is None:
            cloned_outputs = [] # TODO: get Function.__call__ to return None
        else:
            raise TypeError( ('output must be a theano Variable or Out '
                              'instance (or list of them)')
                            , outputs)


    # Iterate over update_expr, cloning its elements, and updating
    # shared_inputs, update_d and update_expr from the SharedVariables
    # we discover.
    # If the variable to be updated is a shared variable not already
    # in shared_inputs, add it.
    # Note: we extend update_expr while iterating over it.

    i = 0
    while i<len(update_expr):
        v, v_update = update_expr[i]
        cloned_v_update = clone_v_get_shared_updates(v_update,
                                                     copy_inputs_over)
        update_d[v] = cloned_v_update
        if isinstance(v, SharedVariable) and v not in shared_inputs:
            shared_inputs.append(v)
        i += 1

    return ( input_variables, cloned_outputs
            , [clone_d, update_d, update_expr, shared_inputs] )

class Param(object):
    def __init__(self, variable, default=None, name=None, mutable=False,
            strict=False, allow_downcast=None, implicit=None, borrow = None):
        """
        :param variable: A variable in an expression graph to use as a compiled-function parameter

        :param default: The default value to use at call-time (can also be a Container where
        the function will find a value at call-time.)

        :param name: A string to identify this parameter from function kwargs.

        :param mutable: True -> function is allowed to modify this argument.

        :param borrow: True -> function is allowed to alias some output to
                       this input


        False: do not permit any output to be aliased to the input
        :param strict: False -> function arguments may be copied or cast to match the
        type required by the parameter `variable`.  True -> function arguments must exactly match the type
        required by `variable`.

        :param allow_downcast: Only applies if `strict` is False.
        True -> allow assigned value to lose precision when cast during assignment.
        False -> never allow precision loss.
        None -> only allow downcasting of a Python float to a scalar floatX.

        :param implicit: see help(theano.io.In)

        """
        self.variable = variable
        self.default = default
        self.name = name
        self.mutable = mutable
        # mutable implies the output can be both aliased to the input and that the input can be
        # destroyed. borrow simply implies the output can be aliased to the input. Thus
        # mutable=True should require borrow=True. Raise warning when borrow is explicitely set
        # to False with mutable=True.
        if mutable:
            if borrow==False:
                _logger.warning("Symbolic input for variable %s (name=%s) has "
                        "flags mutable=True, borrow=False. This combination is "
                        "incompatible since mutable=True implies that the "
                        "input variable may be both aliased (borrow=True) and "
                        "over-written. We set borrow=True and continue.",
                        variable, name)
            borrow = True
        self.strict = strict
        self.allow_downcast = allow_downcast
        self.implicit = implicit
        self.borrow = borrow

def pfunc(params, outputs=None, mode=None, updates=[], givens=[],
        no_default_updates=False, accept_inplace=False, name=None,
        rebuild_strict=True, allow_input_downcast=None,
        profile=None):
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

    :type name: None or string
    :param name: attaches a name to the Profiling result of this function when
    using ProfileMode (will be deprecated).

    :type allow_input_downcast: Boolean
    :param allow_input_downcast: True means that the values passed as
    inputs when calling the function can be silently downcasted to fit
    the dtype of the corresponding Variable, which may lose precision.
    False means that it will only be cast to a more general, or
    precise, type. None (default) is almost like False, but allows
    downcasting of Python float scalars to floatX.

    :type profile: None, True, str, or ProfileStats instance
    :param profile: accumulate profiling information into a given ProfileStats
    instance. None is the default, and means to use the value of
    config.profile.
    If argument is `True` then a new ProfileStats instance will be
    used.  If argument is a string, a new ProfileStats instance will be created
    with that string as its `message` attribute.  This profiling object will be
    available via self.profile.


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
    if profile is None:
        profile = config.profile
        # profile -> True or False
    if profile == True:
        profile = ProfileStats(message=name)
        # profile -> object
    if type(profile) == str:
        profile = ProfileStats(message=profile)
    # profile is typically either False or an object at this point.
    # No need to block other objects being passed through though. It might be
    # useful.

    if not isinstance(params,(list,tuple)):
        raise Exception("in pfunc() the first argument must be a list or a tuple")

    if not isinstance(no_default_updates, bool)\
            and not isinstance(no_default_updates, list):
        raise TypeError("no_default_update should be either a boolean or a list")


    # transform params into theano.compile.In objects.
    inputs = [_pfunc_param_to_in(p, allow_downcast=allow_input_downcast)
              for p in params]

    in_variables = [ input.variable for input in inputs ]
    output_vars = rebuild_collect_shared(
                              outputs
                            , in_variables
                            , replace            = givens
                            , updates            = updates
                            , rebuild_strict     = True
                            , copy_inputs_over   = True
                            , no_default_updates = no_default_updates )
    # extracting the arguments
    input_variables, cloned_outputs, other_stuff = output_vars
    clone_d, update_d, update_expr, shared_inputs = other_stuff

    for i, iv in zip(inputs, input_variables):
        i.variable = iv

    for sv in shared_inputs:
        if sv in update_d:
            si = In(variable=sv, value=sv.container, mutable=True,
                    borrow=True, update=update_d[sv])
        else:
            si = In(variable=sv, value=sv.container,
                    mutable=False, borrow=True)
        inputs.append(si)

    return orig_function(inputs, cloned_outputs, mode,
            accept_inplace=accept_inplace, name=name, profile=profile)


def _pfunc_param_to_in(param, strict=False, allow_downcast=None):
    if isinstance(param, Constant):
        raise TypeError('Constants not allowed in param list', param)
    #if isinstance(param, Value):
        #return In(variable=param)
        #raise NotImplementedError()
    if isinstance(param, Variable): #N.B. includes Value and SharedVariable
        return In(variable=param, strict=strict, allow_downcast=allow_downcast)
    elif isinstance(param, Param):
        return In(
                variable=param.variable,
                name=param.name,
                value=param.default,
                mutable=param.mutable,
                strict=param.strict,
                borrow = param.borrow,
                allow_downcast=param.allow_downcast,
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
