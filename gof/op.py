"""
Contains the L{Op} class, which is the base interface for all operations
compatible with gof's graph manipulation routines.
"""

import utils
import traceback


class Op(utils.object2):

    default_output = None
    """@todo
    WRITEME
    """
    
    #############
    # make_node #
    #############

    def make_node(self, *inputs):
        """
        This function should return an Apply instance representing the
        application of this Op on the provided inputs.
        """
        raise utils.AbstractFunctionError(self)

    def __call__(self, *inputs):
        """
        Shortcut for:
          self.make_node(*inputs).outputs[self.default_output] (if default_output is defined)
          self.make_node(*inputs).outputs[0] (if only one output)
          self.make_node(*inputs).outputs (if more than one output)
        """
        node = self.make_node(*inputs)
        node.tag.trace = traceback.extract_stack()[:-1]
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

        - node: Apply instance that contains the symbolic inputs and outputs
        - inputs: sequence of inputs (immutable)
        - output_storage: list of mutable 1-element lists (do not change
                          the length of these lists)

        The output_storage list might contain data. If an element of
        output_storage is not None, it is guaranteed that it was produced
        by a previous call to impl and impl is free to reuse it as it
        sees fit.
        """
        raise utils.AbstractFunctionError(self)

    #####################
    # C code generation #
    #####################

    def c_code(self, node, name, inputs, outputs, sub):
        """Return the C implementation of an Op.

        Returns C code that does the computation associated to this L{Op},
        given names for the inputs and outputs.
        
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
        raise utils.AbstractFunctionError('%s.c_code is not defined' \
                % self.__class__.__name__)

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """Code to be run after c_code, whether it failed or not.

        This is a convenient place to clean up things allocated by c_code().  
        
        """
        raise utils.AbstractFunctionError()

    def c_compile_args(self):
        """
        Return a list of compile args recommended to manipulate this L{Op}.
        """
        raise utils.AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this L{Op}.
        """
        raise utils.AbstractFunctionError()

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this L{Op}.
        """
        raise utils.AbstractFunctionError()

    def c_support_code(self):
        """
        Return utility code for use by this L{Op}. It may refer to support code
        defined for its input L{Result}s.
        """
        raise utils.AbstractFunctionError()

