"""Define the `function` function
"""
__docformat__ = "restructuredtext en"

import sys, traceback, logging
_logger = logging.getLogger('theano.compile.function_module')

import theano
from function_module import orig_function
from pfunc import pfunc
from numpy import any #for to work in python 2.4

def function(inputs, outputs=None, mode=None, updates=[], givens=[], accept_inplace=False, name = None):
    """
    Return a callable object that will calculate `outputs` from `inputs`.

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

    :param name: an optional name for this fct. If used, the profile mode will print the time spent in this fct.

    :rtype: theano.compile.Function
    :returns: a callable object that will compute the outputs (given the inputs)
    and update the implicit function arguments according to the `updates`.

    :note: Regarding givens: Be careful to make sure that these substitutions are
    independent--behaviour when Var1 of one pair appears in the graph leading to Var2 in
    another expression is undefined.  Replacements specified with givens are different from
    optimizations in that Var2 is not expected to be equivalent to Var1.

    """

    # compute some features of the arguments:
    uses_In = any([isinstance(i, theano.In) for i in inputs]) #N.B. the square brackets are ncessary
    uses_tuple = any([isinstance(i, (list, tuple)) for i in inputs])#N.B. the square brackets are ncessary
    uses_updates = (updates != [])
    uses_givens = (givens != [])

    if uses_In or uses_tuple:
        # we must use old semantics in this case.
        if uses_updates or uses_givens:
            raise NotImplementedError("In() instances and tuple inputs triggers the old semantics, which disallow using updates and givens")
        return orig_function(inputs, outputs, 
                mode=mode,
                accept_inplace=accept_inplace, name=name)
    else:
        return pfunc(params=inputs, 
                outputs=outputs,
                mode=mode, 
                updates=updates, 
                givens=givens,
                accept_inplace=accept_inplace,name=name)
