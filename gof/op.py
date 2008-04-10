
"""
Contains the Op class, which is the base interface for all operations
compatible with gof's graph manipulation routines.
"""

import utils
from utils import ClsInit, all_bases, all_bases_collect, AbstractFunctionError
import graph

from copy import copy


__all__ = ['Op',
           'GuardedOp',
           ]


def constructor(op_cls):
    """Make an Op look like a Result-valued function."""
    def f(*args, **kwargs):
        op = op_cls(*args, **kwargs)
        if len(op.outputs) > 1:
            return op.outputs
        else:
            return op.outputs[0]
    return f

class Op(object):
    """
    Op represents a computation on the storage in its 'inputs' slot,
    the results of which are stored in the Result instances in the
    'outputs' slot. The owner of each Result in the outputs list must
    be set to this Op and thus any Result instance is in the outputs
    list of at most one Op, its owner. It is the responsibility of the
    Op to ensure that it owns its outputs and it is encouraged (though
    not required) that it creates them.
    """

    __slots__ = ['_inputs', '_outputs', '_hash_id']

    _default_output_idx = 0

    def default_output(self):
        """Returns the default output of this Op instance, typically self.outputs[0]."""
        try:
            return self.outputs[self._default_output_idx]
        except (IndexError, TypeError):
            raise AttributeError("Op does not have a default output.")

    out = property(default_output, 
                   doc = "Same as self.outputs[0] if this Op's has_default_output field is True.")


    def __init__(self, **kwargs):
        self._hash_id = utils.hashgen()

    #
    # Python stdlib compatibility
    #
    # These are defined so that sets of Ops, Results will have a consistent
    # ordering

    def __cmp__(self, other):
        return cmp(id(self), id(other))

    def __eq__(self, other):
        return self is other #assuming this is faster, equiv to id(self) == id(other)

    def __ne__(self, other):
        return self is not other #assuming this is faster, equiv to id(self) != id(other)

    def __hash__(self):
        if not hasattr(self, '_hash_id'):
            self._hash_id = utils.hashgen()
        return self._hash_id

    def desc(self):
        return self.__class__

    def strdesc(self):
        return self.__class__.__name__

    #
    #
    #

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
        # the point of this function is 
        # 1. to save the subclass's __init__ function always having to set the role of the outputs
        # 2. to prevent accidentally re-setting outputs, which would probably be a bug
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
        policy. The default behavior is to call the constructor on
        this Op's inputs.

        To do a bottom-up copy of a graph, use clone_with_new_inputs.
        """
        return self.__class__(*self.inputs)

    def clone_with_new_inputs(self, *new_inputs):
        """
        Returns a clone of this Op that takes different inputs. The
        default behavior is to call the constructor on the new inputs,
        but if your Op has additional options or a different constructor
        you might want to override this.
        """
        return self.__class__(*new_inputs)

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

    def impl(self, *args):
        """Return output data [tuple], given input data
        
        If this Op has a single output (len(self.outputs)==1) then the return
        value of this function will be assigned to self.outputs[0].data.

        If this Op has multiple otuputs, then this function should return a
        tuple with the data for outputs[0], outputs[1], outputs[2], etc. 
        """
        raise AbstractFunctionError()

    
    def perform(self):
        """
        Performs the computation associated to this Op and places the
        result(s) in the output Results.

        TODO: consider moving this function to the python linker.
        """
        res = self.impl(*[input.data for input in self.inputs])
        if len(self.outputs) == 1:
            self.outputs[0].data = res
        else:
            assert len(res) == len(self.outputs)
            for output, value in zip(self.outputs, res):
                output.data = value


    #
    # C code generators
    #

#     def c_var_names(self):
#         """
#         Returns ([list of input names], [list of output names]) for
#         use as C variables.
#         """
#         return [["i%i" % i for i in xrange(len(self.inputs))],
#                 ["o%i" % i for i in xrange(len(self.outputs))]]

    def c_validate_update(self, inputs, outputs, sub):
        """
        Returns templated C code that checks that the inputs to this
        function can be worked on. If a failure occurs, set an
        Exception and insert "%(fail)s".
        
        You may use the variable names defined by c_var_names() in
        the template.
        """
        raise AbstractFunctionError()

    def c_validate_update_cleanup(self, inputs, outputs, sub):
        """
        Clean up things allocated by c_validate().
        """
        raise AbstractFunctionError()

    def c_code(self, inputs, outputs, sub):
        """
        Returns templated C code that does the computation associated
        to this Op. You may assume that input validation and output
        allocation have already been done.
        
        You may use the variable names defined by c_var_names() in
        the templates.
        """
        raise AbstractFunctionError()

    def c_code_cleanup(self, inputs, outputs, sub):
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
        Return utility code for use by this Op. It may refer to support code
        defined for its input Results.
        """
        raise AbstractFunctionError()


#TODO: consider adding a flag to the base class that toggles this behaviour
class GuardedOp(Op):
    """An Op that disallows input properties to change after construction"""

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
