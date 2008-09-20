"""
Contains the `Op` class, which is the base interface for all operations
compatible with `gof`'s :doc:`graph` manipulation routines.
"""

__docformat__ = "restructuredtext en"

import utils
import traceback


class Op(utils.object2):
    """
    An :term:`Op` is a type of operation.

    `Op` is an abstract class that documents the interface for theano's data transformations.
    It has many subclasses, such as 
    `sparse dot <http://lgcm.iro.umontreal.ca/epydoc/theano.sparse.Dot-class.html>`__,
    and `Shape <http://lgcm.iro.umontreal.ca/epydoc/theano.tensor.Shape-class.html>`__.

    These subclasses are meant to be instantiated.  
    An instance has several responsabilities:

    - making `Apply` instances, which mean "apply this type of operation to some particular inputs" (via `make_node`),

    - performing the calculation of outputs from given inputs (via the `perform`),

    - producing c code to perform calculation of outputs from inputs (via `c_code`, 
      `c_code_cleanup`, `c_support_code`, `c_headers`, `c_libraries`, `c_compile_args`)

    - [optionally] building gradient-calculating graphs (via `grad`).


    To see how `Op`, `Type`, `Result`, and `Apply` fit together see the page on :doc:`graph`.

    """

    default_output = None
    """ 
    configuration variable for `__call__`

    A subclass should not change this class variable, but instead over-ride it with a subclass
    variable or an instance variable.

    """
    
    #############
    # make_node #
    #############

    def make_node(self, *inputs):
        """
        Contract: return an Apply instance representing the
        application of this Op to the provided inputs.

        All subclasses should over-ride this function.

        :Exceptions:
         - `AbstractFunctionError`: the subclass does not override this method

        """
        raise utils.AbstractFunctionError(self)

    def __call__(self, *inputs):
        """
        Return some or all output[s] of `make_node`.  

        It is called by code such as:

        .. python::

           x = tensor.matrix()

           # tensor.exp is an Op instance, calls Op.__call__(self=<instance of exp>, inputs=(x,))
           y = tensor.exp(x)      
           
        This class implements a convenience function (for graph-building) which uses
        `default_output`, but subclasses are free to override this function and ignore
        `default_output`.

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
        Contract:  Calculate the function on the inputs and put the results in the
        output storage.  Return None.

        :Parameters:
         `node`: Apply instance 
            contains the symbolic inputs and outputs
         `inputs`: list
            sequence of inputs (immutable)
         `output_storage`: list
             list of mutable 1-element lists (do not change the length of these lists)

        The `output_storage` list might contain data. If an element of
        output_storage is not None, it is guaranteed that it was produced
        by a previous call to impl and impl is free to reuse it as it
        sees fit.

        :Exceptions:
         - `AbstractFunctionError`: the subclass does not override this method

        """
        raise utils.AbstractFunctionError(self)

    #####################
    # C code generation #
    #####################

    def c_code(self, node, name, inputs, outputs, sub):
        """Contract: Return the C implementation of an Op.

        Returns C code that does the computation associated to this L{Op},
        given names for the inputs and outputs.
        
        :Parameters:
         `node`: Apply instance
           WRITEME
         `name`: WRITEME
           WRITEME
         `inputs`: list of strings
           There is a string for each input of the function, and the string is the name of a C
           `PyObject` variable pointing to that input.
         `outputs`: list of strings
           Each string is the name of a `PyObject` pointer where the Op should store its
           results.  The `CLinker` guarantees that on entry to this code block, each pointer
           is either NULL or is unchanged from the end of the previous execution.
         `sub`: dict of strings
           extra symbols defined in `CLinker` sub symbols (such as 'fail').
           WRITEME

        """
        raise utils.AbstractFunctionError('%s.c_code is not defined' \
                % self.__class__.__name__)

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """Code to be run after c_code, whether it failed or not.

        This is a convenient place to clean up things allocated by c_code().  
        
        WRITEME
        """
        raise utils.AbstractFunctionError()

    def c_compile_args(self):
        """
        Return a list of compile args recommended to manipulate this L{Op}.

        WRITEME
        """
        raise utils.AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this L{Op}.

        WRITEME
        """
        raise utils.AbstractFunctionError()

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this L{Op}.

        WRITEME
        """
        raise utils.AbstractFunctionError()

    def c_support_code(self):
        """
        Return utility code for use by this L{Op}. It may refer to support code
        defined for its input L{Result}s.

        WRITEME
        """
        raise utils.AbstractFunctionError()

