"""Define the `function` function
"""
__docformat__ = "restructuredtext en"

import logging
import sys
import traceback
_logger = logging.getLogger('theano.compile.function')

from io import In
from function_module import orig_function
from profiling import ProfileStats
from pfunc import pfunc
from numpy import any  # to work in python 2.4
import warnings
from theano import gof

def function(inputs, outputs=None, mode=None, updates=None, givens=None,
             no_default_updates=False, accept_inplace=False, name=None,
             rebuild_strict=True, allow_input_downcast=None, profile=None,
             on_unused_input=None):
    """
    Return a callable object that will calculate `outputs` from `inputs`.

    :type inputs: list of either Variable or Param instances.
    :param inputs: function parameters, these are not allowed to be shared
    variables

    :type outputs: list of Variables or Out instances
    :param outputs: expressions to compute

    :type mode: string or `Mode` instance.
    :param mode: compilation mode

    :type updates: iterable over pairs (shared_variable, new_expression). List, tuple or OrderedDict.
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

    :param on_unused_input: What to do if a variable in the 'inputs' list is
    not used in the graph. Possible values are 'raise', 'warn', 'ignore' and None.

    :rtype: Function instance
    :returns: a callable object that will compute the outputs (given the inputs)
    and update the implicit function arguments according to the `updates`.

    :note: Regarding givens: Be careful to make sure that these substitutions are
    independent--behaviour when Var1 of one pair appears in the graph leading to Var2 in
    another expression is undefined.  Replacements specified with givens are different from
    optimizations in that Var2 is not expected to be equivalent to Var1.


    Internal documentation:

        What happens when you call theano.function?
           1. RemoveShared: shared variables are just an abstraction to make
        things more convenient for the user. The shared variables are
        transformed into implicit inputs and implicit outputs. The
        optimizations don't see which variables are shared or not.
           2. FunctionGraph: determines whether a graph is valid. For example,
        suppose
        you merge the two apply nodes in our example above, ie, do the
        addition and the tanh at the same time. If you propose a merge that
        changes the resulting dtype or broadcastable pattern of V4, the fgraph
        will detect this.
                    inplace optimizations: say we have an apply node that
        does + on V1 and V2, with output V3. We can change the output to be
        V1, to use less memory. theano must be told that this optimization is
        happening though, so that other parts of the graph are given the
        correct (pre + or post + ) version of V1.
                  fgraph will raise an error if any of these types of
        modifications causes an error
                  fgraph also adds a field called "clients" to all variables.
        clients is a list of apply nodes that use the variable. this makes it
        possible to traverse the graph in both directions. this is useful for
        determining whether to do some optimizations. for example, a fusion
        operation that removes V3 is not very helpful if V3 is also needed for
        some other apply node. fusion operations result in a composite op that
        takes a minigraph of theano scalars and uses this to do elemwise
        operations on theano tensors
         3. Optimization
               How well do optimizations apply to new ops?
                 Usually there are no optimizations for new ops. In fact, new
        ops can disrupt patterns and break currently working optimizations.
        Since the Print op, for example, is not known by any optimization,
        setting a Print op in the middle of a pattern that is usually
        optimized out will block the optimization. for example, log(1+x)
        optimizes to log1p(x) but log(1+Print(x)) is unaffected by
        optimizations.
                 One exception is elemwise ops. If you implement your new op
        as a scalar op then it will automatically work with all the elemwise
        fusion machinery.

                 Local optimizations try to replace some node in the graph
        with a different node. In the case of log(1+x), we want to replace the
        log node.

                 def opt_log1p(node):
                    if not isinstance(node.op,Elemwise):
                       return
                    if not isinstance(node.op.scalar_op, log):
                       return
                    inp = node.inputs[0]
                    if not inp.owner:
                       return
                    if not isinstance(inp.owner.op, add):
                       return
                    inp2 = inp.owner.inputs
                    check that this has length 2, and that one of the inputs
        is 1. assign the other input to x
                    return log1p(x)


         4. Linker
               The linker uses a python loop to execute the code associated
               with all the Apply nodes in the graph in the correct order.
               the cvm is a linker that replaces this python loop with a c
               loop to avoid continuously changing between python and c.
               The CVM is faster for 2 reasons:
                 1) Its internal logic in C, so no Python interpreter overhead.
                 2) It makes native calls from the VM logic into thunks that
                 have been compiled using the CLinker.
               the vm is a linker that was developed to prototype the cvm. it
        was easier to develop the vm in python then translate it to c instead
        of just writing it in c from scratch
               cvm stands for c virtual machine.




    """
    if updates is None:
        updates = []

    if isinstance(updates, dict) and \
            not isinstance(updates, gof.python25.OrderedDict):
        warnings.warn(
            "The parameter 'updates' of theano.function()"
            " expects an OrderedDict,"
            " got " + str(type(updates)) + ". Using "
            "a standard dictionary here results in "
            "non-deterministic behavior. You should use an OrderedDict"
            " if you are using Python 2.7, or use a list of (shared, update)"
            " pairs. Do not just convert your dictionary to this type before"
            " the call as the conversion will still be non-deterministic.")

    if givens is None:
        givens = []
    if not isinstance(inputs, (list, tuple)):
        raise Exception("Input variables of a Theano function should be"
                        " contained in a list, even when there is a single input.")

    # compute some features of the arguments:
    uses_In = any([isinstance(i, In) for i in inputs])  # N.B. the square brackets are ncessary
    uses_tuple = any([isinstance(i, (list, tuple)) for i in inputs])  # N.B. the square brackets are ncessary
    uses_updates = (updates != [])
    uses_givens = (givens != [])

    # See if we have any mutable / borrow inputs
    check_for_aliased_inputs = False
    for i in inputs:
        if (isinstance(i, In) and ((hasattr(i, 'borrow') and i.borrow) or
                                   (hasattr(i, 'mutable') and i.mutable))):
            check_for_aliased_inputs = True

    if uses_In or uses_tuple:
        # we must use old semantics in this case.
        if profile:
            raise NotImplementedError('profiling not supported in old-style function')
        if uses_updates or uses_givens:
            raise NotImplementedError("In() instances and tuple inputs triggers the old semantics, which disallow using updates and givens")
        fn = orig_function(inputs, outputs,
                           mode=mode,
                           accept_inplace=accept_inplace, name=name)
    else:
        #note: pfunc will also call orig_function-- orig_function is a choke point
        #      that all compilation must pass through
        fn = pfunc(params=inputs,
                outputs=outputs,
                mode=mode,
                updates=updates,
                givens=givens,
                no_default_updates=no_default_updates,
                accept_inplace=accept_inplace, name=name,
                rebuild_strict=rebuild_strict,
                allow_input_downcast=allow_input_downcast,
                on_unused_input=on_unused_input,
                profile=profile)
    # We need to add the flag check_aliased inputs if we have any mutable or
    # borrowed used defined inputs
    fn._check_for_aliased_inputs = check_for_aliased_inputs
    return fn
