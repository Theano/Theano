
"""
Contains the L{Op} class, which is the base interface for all operations
compatible with gof's graph manipulation routines.
"""

import utils
from utils import AbstractFunctionError, object2

from copy import copy



class Op(object2):

    default_output = None
    """@todo
    """
    
    #############
    # make_node #
    #############

    def make_node(self, *inputs):
        raise AbstractFunctionError()

    def __call__(self, *inputs):
        node = self.make_node(*inputs)
        if self.default_output is not None:
            return node.outputs[self.default_output]
        else:
            if len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs


    #########################
    # Python implementation #
    #########################

    def perform(self, node, inputs, output_storage):
        """
        Calculate the function on the inputs and put the results in the
        output storage.

        - inputs: sequence of inputs (immutable)
        - output_storage: list of mutable 1-element lists (do not change
                          the length of these lists)

        The output_storage list might contain data. If an element of
        output_storage is not None, it is guaranteed that it was produced
        by a previous call to impl and impl is free to reuse it as it
        sees fit.
        """
        raise AbstractFunctionError()

    #####################
    # C code generation #
    #####################

    def c_code(self, node, name, inputs, outputs, sub):
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

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
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


class PropertiedOp(Op):

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __str__(self):
        if hasattr(self, 'name') and self.name:
            return self.name
        else:
            return "%s{%s}" % (self.__class__.__name__, ", ".join("%s=%s" % (k, v) for k, v in self.__dict__.items() if k != "name"))


