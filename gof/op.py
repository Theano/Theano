
"""
Contains the L{Op} class, which is the base interface for all operations
compatible with gof's graph manipulation routines.
"""

import utils
from utils import ClsInit, all_bases, all_bases_collect, AbstractFunctionError
import graph

from copy import copy


__all__ = ['Op',
           'GuardedOp',
           ]


def constructor(op_cls, name = None):
    """
    Make an L{Op} look like a L{Result}-valued function.
    """
    def f(*args, **kwargs):
        op = op_cls(*args, **kwargs)
        if len(op.outputs) > 1:
            return op.outputs
        else:
            return op.outputs[0]
    opname = op_cls.__name__
    if name is None:
        name = "constructor{%s}" % opname
    f.__name__ = name
    doc = op_cls.__doc__
    f.__doc__ = """

    Constructor for %(opname)s:

    %(doc)s
    """ % locals()
    return f

class Op(object):
    """
    L{Op} represents a computation on the storage in its 'inputs' slot,
    the results of which are stored in the L{Result} instances in the
    'outputs' slot. The owner of each L{Result} in the outputs list must
    be set to this L{Op} and thus any L{Result} instance is in the outputs
    list of at most one L{Op}, its owner. It is the responsibility of the
    L{Op} to ensure that it owns its outputs and it is encouraged (though
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
        """
        The point of this function is:
         1. to save the subclass's __init__ function always having to set the role of the outputs
         2. to prevent accidentally re-setting outputs, which would probably be a bug
        """
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
        Shallow copy of this L{Op}. The inputs are the exact same, but
        the outputs are recreated because of the one-owner-per-result
        policy. The default behavior is to call the constructor on this
        L{Op}'s inputs.

        To do a bottom-up copy of a graph, use L{clone_with_new_inputs}.

        @attention: If your L{Op} has additional options or a different
        constructor you probably want to override this.
        """
        return self.__class__(*self.inputs)

    def clone_with_new_inputs(self, *new_inputs):
        """
        Returns a clone of this L{Op} that takes different inputs. The
        default behavior is to call the constructor on the new inputs.

        @attention: If your L{Op} has additional options or a different
        constructor you probably want to override this.
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
        
        If this L{Op} has a single output (len(self.outputs)==1) then the return
        value of this function will be assigned to self.outputs[0].data.

        If this L{Op} has multiple otuputs, then this function should return a
        tuple with the data for outputs[0], outputs[1], outputs[2], etc. 
        """
        raise AbstractFunctionError()

    
    def perform(self):
        """
        Performs the computation associated to this L{Op} and places the
        result(s) in the output L{Result}s.

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

    def c_validate_update(self, inputs, outputs, sub):
        """
        Returns templated C code that checks that the inputs to this
        function can be worked on. If a failure occurs, set an
        Exception and insert "%(fail)s".
        
        You may use the variable names defined by c_var_names() in
        the template.

        Note: deprecated!!
        @todo: Merge this with c_code.
        """
        raise AbstractFunctionError()

    def c_validate_update_cleanup(self, inputs, outputs, sub):
        """
        Clean up things allocated by L{c_validate}().

        Note: deprecated!! 
        @todo: Merge this with c_code.
        """
        raise AbstractFunctionError()
        raise AbstractFunctionError('%s.c_validate_update_cleanup ' \
                % self.__class__.__name__)

    def c_code(self, inputs, outputs, sub):
        """Return the C implementation of an Op.

        Returns templated C code that does the computation associated
        to this L{Op}. You may assume that input validation and output
        allocation have already been done.
        
        @param inputs: list of strings.  There is a string for each input
                       of the function, and the string is the name of a C
                       L{PyObject}* variable pointing to that input.

        @param outputs: list of strings.  Each string is the name of a
                        L{PyObject}* pointer where the Op should store its
                        results.  The L{CLinker} guarantees that on entry to
                        this code block, each pointer is either NULL or is
                        unchanged from the end of the previous execution.

        @param sub: extra symbols defined in L{CLinker sub symbols} (such as
                'fail').

        """
        raise AbstractFunctionError('%s.c_code' \
                % self.__class__.__name__)

    def c_code_cleanup(self, inputs, outputs, sub):
        """Code to be run after c_code, whether it failed or not.

        This is a convenient place to clean up things allocated by c_code().  
        
        """
        raise AbstractFunctionError()

    def c_compile_args(self):
        """
        Return a list of compile args recommended to manipulate this L{Op}.
        """
        raise AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this L{Op}.
        """
        raise AbstractFunctionError()

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this L{Op}.
        """
        raise AbstractFunctionError()

    def c_support_code(self):
        """
        Return utility code for use by this L{Op}. It may refer to support code
        defined for its input L{Result}s.
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
        except AbstractFunctionError:
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
