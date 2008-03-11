
"""
Contains the Op class, which is the base interface for all operations
compatible with gof's graph manipulation routines.
"""

# from result import BrokenLinkError
from utils import ClsInit, all_bases, all_bases_collect, AbstractFunctionError
import graph

from copy import copy


__all__ = ['Op',
           'GuardedOp',
           ]


class Op(object):
    """
    Op represents a computation on the storage in its 'inputs' slot,
    the results of which are stored in the Result instances in the
    'outputs' slot. The owner of each Result in the outputs list must
    be set to this Op and thus any Result instance is in the outputs
    list of at most one Op, its owner. It is the responsibility of the
    Op to ensure that it owns its outputs and it is encouraged (though
    not required) that it creates them.

    After construction, self.inputs and self.outputs should only be
    modified through the set_input and set_output methods.
    """

    __slots__ = ['_inputs', '_outputs']

    _default_output_idx = 0

    def default_output(self):
        """Returns the default output of this Op instance, typically self.outputs[0]."""
        try:
            return self.outputs[self._default_output_idx]
        except (IndexError, TypeError):
            raise AttributeError("Op does not have a default output.")

    out = property(default_output, 
            doc = "Same as self.outputs[0] if this Op's has_default_output field is True.")


    def __init__(self, *inputs):
        raise AbstractFunctionError("Op is an abstract class. Its constructor does nothing, you must override it.")

    
    def get_input(self, i):
        return self._inputs[i]        
    def set_input(self, i, new):
        self._inputs[i] = new

    def get_inputs(self):
        return self._inputs
    def set_inputs(self, new):
        self._inputs = list(new)

    def get_output(self, i):
        return self._outputs[i]

    def get_outputs(self):
        return self._outputs
    def set_outputs(self, new):
        if not hasattr(self, '_outputs') or self._outputs is None:
            for i, output in enumerate(new):
                output.role = (self, i)
            self._outputs = list(new)
        else:
            raise Exception("Can only set outputs once, to initialize them.")

    #create inputs and outputs as read-only attributes
    inputs = property(get_inputs, set_inputs, doc = "The list of this Op's input Results.")
    outputs = property(get_outputs, set_outputs, doc = "The list of this Op's output Results.")


    #
    # copy
    #

    def __copy__(self):        
        """
        Shallow copy of this Op. The inputs are the exact same, but
        the outputs are recreated because of the one-owner-per-result
        policy.
        """
        return self.__class__(*self.inputs)


    #
    # String representation
    #

    def __str__(self):
        return graph.op_as_string(self.inputs, self)

    def __repr__(self):
        return str(self)


    #
    # perform
    #
    
    def perform(self):
        """
        (abstract) Performs the computation associated to this Op,
        places the result(s) in the output Results and gives them
        the Computed status.
        """
        raise AbstractFunctionError()


    #
    # C code generators
    #

    def c_var_names(self):
        """
        Returns ([list of input names], [list of output names]) for
        use as C variables.
        """
        return [["i%i" % i for i in xrange(len(self.inputs))],
                ["o%i" % i for i in xrange(len(self.outputs))]]

    def c_validate_update(self):
        """
        Returns C code that checks that the inputs to this function
        can be worked on. If a failure occurs, set an Exception
        and insert "%(fail)s".
        
        You may use the variable names defined by c_var_names()
        """
        raise AbstractFunctionError()

    def c_validate_update_cleanup(self):
        """
        Clean up things allocated by c_validate().
        """
        raise AbstractFunctionError()

    def c_code(self):
        """
        Returns C code that does the computation associated to this
        Op. You may assume that input validation and output allocation
        have already been done.
        
        You may use the variable names defined by c_var_names()
        """
        raise AbstractFunctionError()

    def c_code_cleanup(self):
        """
        Clean up things allocated by c_code().
        """
        raise AbstractFunctionError()

    def c_compile_args(self):
        """
        Return a list of compile args recommended to manipulate this Op.
        """
        raise AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this Op.
        """
        raise AbstractFunctionError()

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this Op.
        """
        raise AbstractFunctionError()

    def c_support_code(self):
        """
        Return utility code for use by this Op.
        """
        raise AbstractFunctionError()



class GuardedOp(Op):

    def set_input(self, i, new):
        old = self._inputs[i]
        if old is new:
            return
        try:
            if not old.same_properties(new):
                raise TypeError("The new input must have the same properties as the previous one.")
        except AbstractFunction:
            pass
        Op.set_input(self, i, new)

    def set_inputs(self, new):
        if not hasattr(self, '_inputs') or self_inputs is None:
            Op.set_inputs(self, new)
        else:
            if not len(new) == len(self._inputs):
                raise TypeError("The new inputs are not as many as the previous ones.")
            for i, new in enumerate(new):
                self.set_input(i, new)
