"""Defines base classes `Op`, `PureOp`, and `CLinkerOp`

The `Op` class is the base interface for all operations
compatible with `gof`'s :doc:`graph` routines.


"""

__docformat__ = "restructuredtext en"

import utils
from theano import config



class CLinkerObject(object):
    """Standard elements of an Op or Type used with the CLinker
    """
    def c_headers(self):
        """Optional: Return a list of header files required by code returned by
        this class.

        For example: return ['<iostream>', '<math.h>', '/full/path/to/header.h']

        These strings will be prefixed with "#include " and inserted at the beginning of the c
        source code.

        Strings in this list that start neither with '<' nor '"' will be enclosed in
        double-quotes.

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_headers", type(self), self.__class__.__name__)

    def c_header_dirs(self):
        """Optional: Return a list of header search paths required by code returned by
        this class.

        For example: return ['/usr/local/include', '/opt/weirdpath/src/include'].

        Provide search paths for headers, in addition to those in any relevant environment
        variables.
        
        Hint: for unix compilers, these are the things that get '-I' prefixed in the compiler
        cmdline.
        
        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_header_dirs", type(self), self.__class__.__name__)

    def c_libraries(self):
        """Optional: Return a list of libraries required by code returned by
        this class.

        For example: return ['gsl', 'gslcblas', 'm', 'fftw3', 'g2c'].

        The compiler will search the directories specified by the environment
        variable LD_LIBRARY_PATH in addition to any returned by `c_lib_dirs`.

        Hint: for unix compilers, these are the things that get '-l' prefixed in the compiler
        cmdline.

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_libraries", type(self), self.__class__.__name__)

    def c_lib_dirs(self):
        """Optional: Return a list of library search paths required by code returned by
        this class.

        For example: return ['/usr/local/lib', '/opt/weirdpath/build/libs'].

        Provide search paths for libraries, in addition to those in any relevant environment
        variables (e.g. LD_LIBRARY_PATH).
        
        Hint: for unix compilers, these are the things that get '-L' prefixed in the compiler
        cmdline.
        
        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_lib_dirs", type(self), self.__class__.__name__)

    def c_support_code(self):
        """Optional: Return utility code for use by a `Variable` or `Op` to be
        included at global scope prior to the rest of the code for this class.

        QUESTION: How many times will this support code be emitted for a graph
        with many instances of the same type?

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_support_code", type(self), self.__class__.__name__)

    def c_code_cache_version(self):
        """Return a tuple of integers indicating the version of this Op.

        An empty tuple indicates an 'unversioned' Op that will not be cached between processes.

        The cache mechanism may erase cached modules that have been superceded by newer
        versions.  See `ModuleCache` for details.

        :note: See also `c_code_cache_version_apply()`
        """
        return ()

    def c_code_cache_version_apply(self, node):
        """Return a tuple of integers indicating the version of this Op.

        An empty tuple indicates an 'unversioned' Op that will not be cached between processes.

        The cache mechanism may erase cached modules that have been superceded by newer
        versions.  See `ModuleCache` for details.

        :note: See also `c_code_cache_version()`

        :note: This function overrides `c_code_cache_version` unless it explicitly calls
        `c_code_cache_version`.  The default implementation simply calls `c_code_cache_version`
        and ignores the `node` argument.
        """
        return self.c_code_cache_version()


    def c_compile_args(self):
        """Optional: Return a list of compile args recommended to compile the
        code returned by other methods in this class.

        Example: return ['-ffast-math']

        Compiler arguments related to headers, libraries and search paths should be provided
        via the functions `c_headers`, `c_libraries`, `c_header_dirs`, and `c_lib_dirs`.

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_compile_args", type(self), self.__class__.__name__)

    def c_no_compile_args(self):
        """Optional: Return a list of incompatible gcc compiler arguments.

        We will remove those arguments from the command line of gcc. So if 
        another Op adds a compile arg in the graph that is incompatible 
        with this Op, the incompatible arg will not be used. 
        Useful for instance to remove -ffast-math.

        EXAMPLE

        WRITEME

        :Exceptions:
         - `MethodNotDefined`: the subclass does not override this method

        """
        raise utils.MethodNotDefined("c_no_compile_args", type(self), self.__class__.__name__)

class CLinkerOp(CLinkerObject):
    """
    Interface definition for `Op` subclasses compiled by `CLinker`.

    A subclass should implement WRITEME.

    WRITEME: structure of automatically generated C code.  Put this in doc/code_structure.txt

    """

    def c_code(self, node, name, inputs, outputs, sub):
        """Required: Return the C implementation of an Op.

        Returns C code that does the computation associated to this `Op`,
        given names for the inputs and outputs.
        
        :Parameters:
         `node` : Apply instance
           WRITEME
         `name` : WRITEME
           WRITEME
         `inputs` : list of strings
           There is a string for each input of the function, and the string is the name of a C
           `PyObject` variable pointing to that input.
         `outputs` : list of strings
           Each string is the name of a `PyObject` pointer where the Op should store its
           variables.  The `CLinker` guarantees that on entry to this code block, each pointer
           is either NULL or is unchanged from the end of the previous execution.
         `sub` : dict of strings
           extra symbols defined in `CLinker` sub symbols (such as 'fail').
           WRITEME

        :Exceptions:
         - `MethodNotDefined`: the subclass does not override this method

        """
        raise utils.MethodNotDefined('%s.c_code' \
                % self.__class__.__name__)

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """Optional: Return C code to run after c_code, whether it failed or not.

        QUESTION: is this function optional?

        This is a convenient place to clean up things allocated by c_code().  
        
        :Parameters:
         `node` : Apply instance
           WRITEME
         `name` : WRITEME
           WRITEME
         `inputs` : list of strings
           There is a string for each input of the function, and the string is the name of a C
           `PyObject` variable pointing to that input.
         `outputs` : list of strings
           Each string is the name of a `PyObject` pointer where the Op should store its
           variables.  The `CLinker` guarantees that on entry to this code block, each pointer
           is either NULL or is unchanged from the end of the previous execution.
         `sub` : dict of strings
           extra symbols defined in `CLinker` sub symbols (such as 'fail').
           WRITEME
        
        WRITEME

        :Exceptions:
         - `MethodNotDefined`: the subclass does not override this method

        """
        raise utils.MethodNotDefined('%s.c_code_cleanup' \
                % self.__class__.__name__)

    def c_support_code_apply(self, node, name):
        """Optional: Return utility code for use by an `Op` that will be inserted at global
        scope, that can be specialized for the support of a particular `Apply` node.

        :param node: an Apply instance in the graph being compiled

        :param node_id: a string or number that serves to uniquely identify this node.
        Symbol names defined by this support code should include the node_id, so that they can
        be called from the c_code, and so that they do not cause name collisions.

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise utils.MethodNotDefined("c_support_code_apply", type(self), self.__class__.__name__)


class PureOp(object):
    """
    An :term:`Op` is a type of operation.

    `Op` is an abstract class that documents the interface for theano's data transformations.
    It has many subclasses, such as 
    `sparse dot <http://pylearn.org/epydoc/theano.sparse.Dot-class.html>`__,
    and `Shape <http://pylearn.org/epydoc/theano.tensor.Shape-class.html>`__.

    These subclasses are meant to be instantiated.  
    An instance has several responsabilities:

    - making `Apply` instances, which mean "apply this type of operation to some particular inputs" (via `make_node`),

    - performing the calculation of outputs from given inputs (via the `perform`),

    - [optionally] building gradient-calculating graphs (via `grad`).


    To see how `Op`, `Type`, `Variable`, and `Apply` fit together see the page on :doc:`graph`.

    For more specifications on how these methods should behave: see the `Op Contract` in the
    sphinx docs (advanced tutorial on Op-making).

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
        Required: return an Apply instance representing the
        application of this Op to the provided inputs.

        All subclasses should over-ride this function.

        :Exceptions:
         - `MethodNotDefined`: the subclass does not override this method

        """
        raise utils.MethodNotDefined("make_node", type(self), self.__class__.__name__)

    def __call__(self, *inputs, **kwargs):
        """Optional: Return some or all output[s] of `make_node`.  

        It is called by code such as:

        .. python::

           x = tensor.matrix()

           # tensor.exp is an Op instance, calls Op.__call__(self=<instance of exp>, inputs=(x,))
           y = tensor.exp(x)      
           
        This class implements a convenience function (for graph-building) which uses
        `default_output`, but subclasses are free to override this function and ignore
        `default_output`.

        """
        node = self.make_node(*inputs, **kwargs)
        self.add_tag_trace(node)
        if self.default_output is not None:
            return node.outputs[self.default_output]
        else:
            if len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs

    # Convenience so that subclass implementers don't have to import utils
    # just to self.add_tag_trace
    add_tag_trace = staticmethod(utils.add_tag_trace)

    #########################
    # Python implementation #
    #########################

    def perform(self, node, inputs, output_storage):
        """
        Required:  Calculate the function on the inputs and put the variables in the
        output storage.  Return None.

        :Parameters:
         `node` : Apply instance 
            contains the symbolic inputs and outputs
         `inputs` : list
            sequence of inputs (immutable)
         `output_storage` : list
             list of mutable 1-element lists (do not change the length of these lists)

        The `output_storage` list might contain data. If an element of
        output_storage is not None, it is guaranteed that it was produced
        by a previous call to impl and impl is free to reuse it as it
        sees fit.

        :Exceptions:
         - `MethodNotDefined`: the subclass does not override this method

        """
        raise utils.MethodNotDefined("perform", type(self), self.__class__.__name__)

class Op(utils.object2, PureOp, CLinkerOp): 
    """Convenience class to bundle `PureOp` and `CLinkerOp`"""
    pass
