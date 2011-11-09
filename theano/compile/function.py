"""Define the `function` function
"""
__docformat__ = "restructuredtext en"

import sys, traceback, logging
_logger = logging.getLogger('theano.compile.function')

from io import In
from function_module import orig_function
from profiling import ProfileStats
from pfunc import pfunc
from numpy import any #for to work in python 2.4

def function(inputs, outputs=None, mode=None, updates=[], givens=[],
             no_default_updates=False, accept_inplace=False, name=None,
             rebuild_strict=True, allow_input_downcast=None, profile=None):
    """
    Return a callable object that will calculate `outputs` from `inputs`.

    :type inputs: list of either Variable or Param instances.
    :param inputs: function parameters, these are not allowed to be shared
    variables

    :type outputs: list of Variables or Out instances
    :param outputs: expressions to compute

    :type mode: string or `Mode` instance.
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

    :param name: an optional name for this function. The profile mode will print the time spent in this function.

    :rtype: Function instance
    :returns: a callable object that will compute the outputs (given the inputs)
    and update the implicit function arguments according to the `updates`.

    :param rebuild_strict: True (Default) is the safer and better tested setting, in which case
    `givens` must substitute new variables with the same Type as the variables they replace.
    False is a you-better-know-what-you-are-doing setting, that permits `givens` to replace
    variables with new variables of any Type.  The consequence of changing a Type is that all
    results depending on that variable may have a different Type too (the graph is rebuilt from
    inputs to outputs).  If one of the new types does not make sense for one of the Ops in the
    graph, an Exception will be raised.

    :type allow_input_downcast: Boolean or None
    :param allow_input_downcast: True means that the values passed as
    inputs when calling the function can be silently downcasted to fit
    the dtype of the corresponding Variable, which may lose precision.
    False means that it will only be cast to a more general, or
    precise, type. None (default) is almost like False, but allows
    downcasting of Python float scalars to floatX.

    :type profile: None, True, or ProfileStats instance
    :param profile: accumulate profiling information into a given ProfileStats
    instance. If argument is `True` then a new ProfileStats instance will be
    used.  This profiling object will be available via self.profile.

    :note: Regarding givens: Be careful to make sure that these substitutions are
    independent--behaviour when Var1 of one pair appears in the graph leading to Var2 in
    another expression is undefined.  Replacements specified with givens are different from
    optimizations in that Var2 is not expected to be equivalent to Var1.

    """
    #tuple are used in some tests, as we accepted them in the past
    #I prefer to allow it as they act the same as list for what they are used.
    if not isinstance(inputs,(list,tuple)):
        raise Exception("Inputs variable of a Theano function should be contained in a list, even when there is a single input.")

    # compute some features of the arguments:
    uses_In = any([isinstance(i, In) for i in inputs]) #N.B. the square brackets are ncessary
    uses_tuple = any([isinstance(i, (list, tuple)) for i in inputs])#N.B. the square brackets are ncessary
    uses_updates = (updates != [])
    uses_givens = (givens != [])

    # See if we have any mutable / borrow inputs
    check_for_aliased_inputs = False
    for i in inputs:
        if (isinstance(i, In) and ( (hasattr(i,'borrow') and i.borrow) or
                                   (hasattr(i,'mutable') and i.mutable)) ):
            check_for_aliased_inputs = True

    if uses_In or uses_tuple:
        # we must use old semantics in this case.
        if profile:
            raise NotImplementedError('profiling not supported in old-style function')
        if uses_updates or uses_givens:
            raise NotImplementedError("In() instances and tuple inputs triggers the old semantics, which disallow using updates and givens")
        fn =  orig_function(inputs, outputs,
                mode=mode,
                accept_inplace=accept_inplace, name=name)
    else:
        fn = pfunc(params=inputs,
                outputs=outputs,
                mode=mode,
                updates=updates,
                givens=givens,
                no_default_updates=no_default_updates,
                accept_inplace=accept_inplace,name=name,
                rebuild_strict=rebuild_strict,
                allow_input_downcast=allow_input_downcast,
                profile=profile)
    # We need to add the flag check_aliased inputs if we have any mutable or
    # borrowed used defined inputs
    fn._check_for_aliased_inputs = check_for_aliased_inputs
    return fn
