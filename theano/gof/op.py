"""
Defines base classes `Op`, `PureOp`, and `CLinkerOp`.

The `Op` class is the base interface for all operations
compatible with `gof`'s :doc:`graph` routines.

"""
from __future__ import absolute_import, print_function, division

import inspect
import logging
import numpy as np
import os
import re
import sys
import warnings

import theano
from theano import config

import theano.gof.cc
from six import itervalues
from theano.gof import graph
from theano.gof import utils
from theano.gof.cmodule import GCC_compiler
from theano.gof.fg import FunctionGraph

__authors__ = "theano-dev"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"

_logger = logging.getLogger('theano.gof.op.Op')


class CLinkerObject(object):
    """
    Standard elements of an Op or Type used with the CLinker.

    """

    def c_headers(self):
        """
        Optional: Return a list of header files required by code returned by
        this class.

        Examples
        --------
        return ['<iostream>', '<math.h>', '/full/path/to/header.h']

        These strings will be prefixed with "#include " and inserted at the
        beginning of the c source code.

        Strings in this list that start neither with '<' nor '"' will be
        enclosed in double-quotes.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined(
            "c_headers", type(self), self.__class__.__name__)

    def c_header_dirs(self):
        """
        Optional: Return a list of header search paths required by code
        returned by this class.

        Examples
        --------
        return ['/usr/local/include', '/opt/weirdpath/src/include']

        Provides search paths for headers, in addition to those in any relevant
        environment variables.

        Hint: for unix compilers, these are the things that get '-I' prefixed
        in the compiler cmdline.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined(
            "c_header_dirs",
            type(self),
            self.__class__.__name__)

    def c_libraries(self):
        """
        Optional: Return a list of libraries required by code returned by
        this class.

        Examples
        --------
        return ['gsl', 'gslcblas', 'm', 'fftw3', 'g2c'].

        The compiler will search the directories specified by the environment
        variable LD_LIBRARY_PATH in addition to any returned by `c_lib_dirs`.

        Hint: for unix compilers, these are the things that get '-l' prefixed
        in the compiler cmdline.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined(
            "c_libraries", type(self), self.__class__.__name__)

    def c_lib_dirs(self):
        """
        Optional: Return a list of library search paths required by code
        returned by this class.

        Examples
        --------
        return ['/usr/local/lib', '/opt/weirdpath/build/libs'].

        Provides search paths for libraries, in addition to those in any
        relevant environment variables (e.g. LD_LIBRARY_PATH).

        Hint: for unix compilers, these are the things that get '-L' prefixed
        in the compiler cmdline.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined(
            "c_lib_dirs", type(self), self.__class__.__name__)

    def c_support_code(self):
        """
        Optional: Return utility code for use by a `Variable` or `Op` to be
        included at global scope prior to the rest of the code for this class.

        QUESTION: How many times will this support code be emitted for a graph
        with many instances of the same type?

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined(
            "c_support_code",
            type(self),
            self.__class__.__name__)

    def c_code_cache_version(self):
        """
        Return a tuple of integers indicating the version of this Op.

        An empty tuple indicates an 'unversioned' Op that will not be cached
        between processes.

        The cache mechanism may erase cached modules that have been superceded
        by newer versions. See `ModuleCache` for details.

        See Also
        --------
        c_code_cache_version_apply()

        """
        return ()

    def c_compile_args(self):
        """
        Optional: Return a list of compile args recommended to compile the
        code returned by other methods in this class.

        Example
        -------
        return ['-ffast-math']

        Compiler arguments related to headers, libraries and search paths should
        be provided via the functions `c_headers`, `c_libraries`,
        `c_header_dirs`, and `c_lib_dirs`.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined(
            "c_compile_args",
            type(self),
            self.__class__.__name__)

    def c_no_compile_args(self):
        """
        Optional: return a list of incompatible gcc compiler arguments.

        We will remove those arguments from the command line of gcc. So if
        another Op adds a compile arg in the graph that is incompatible
        with this Op, the incompatible arg will not be used.
        Useful for instance to remove -ffast-math.

        EXAMPLE

        WRITEME

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined(
            "c_no_compile_args",
            type(self),
            self.__class__.__name__)

    def c_init_code(self):
        """
        Optional: return a list of code snippets to be inserted in module
        initialization.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined("c_init_code", type(self),
                                     self.__class__.__name__)


class CLinkerOp(CLinkerObject):
    """
    Interface definition for `Op` subclasses compiled by `CLinker`.

    A subclass should implement WRITEME.

    WRITEME: structure of automatically generated C code.
    Put this in doc/code_structure.txt

    """

    def c_code(self, node, name, inputs, outputs, sub):
        """
        Required: return the C implementation of an Op.

        Returns C code that does the computation associated to this `Op`,
        given names for the inputs and outputs.

        Parameters
        ----------
        node : Apply instance
            The node for which we are compiling the current c_code.
           The same Op may be used in more than one node.
        name : str
            A name that is automatically assigned and guaranteed to be
            unique.
        inputs : list of strings
            There is a string for each input of the function, and the
            string is the name of a C variable pointing to that input.
            The type of the variable depends on the declared type of
            the input.  There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.
        outputs : list of strings
            Each string is the name of a C variable where the Op should
            store its output.  The type depends on the declared type of
            the output.  There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.  In some cases the outputs will be preallocated and
            the value of the variable may be pre-filled.  The value for
            an unallocated output is type-dependent.
        sub : dict of strings
            Extra symbols defined in `CLinker` sub symbols (such as 'fail').
            WRITEME

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined('%s.c_code' % self.__class__.__name__)

    def c_code_cache_version_apply(self, node):
        """
        Return a tuple of integers indicating the version of this Op.

        An empty tuple indicates an 'unversioned' Op that will not be
        cached between processes.

        The cache mechanism may erase cached modules that have been
        superceded by newer versions.  See `ModuleCache` for details.

        See Also
        --------
        c_code_cache_version()

        Notes
        -----
            This function overrides `c_code_cache_version` unless it explicitly
            calls `c_code_cache_version`. The default implementation simply
            calls `c_code_cache_version` and ignores the `node` argument.

        """
        return self.c_code_cache_version()

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """
        Optional: return C code to run after c_code, whether it failed or not.

        This is a convenient place to clean up things allocated by c_code().

        Parameters
        ----------
        node : Apply instance
            WRITEME
        name : str
            A name that is automatically assigned and guaranteed to be
            unique.
        inputs : list of strings
            There is a string for each input of the function, and the
            string is the name of a C variable pointing to that input.
            The type of the variable depends on the declared type of
            the input. There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.
        outputs : list of strings
            Each string is the name of a C variable correspoinding to
            one of the outputs of the Op. The type depends on the
            declared type of the output. There is a corresponding
            python variable that can be accessed by prepending "py_" to
            the name in the list.
        sub : dict of strings
            extra symbols defined in `CLinker` sub symbols (such as 'fail').
            WRITEME

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined('%s.c_code_cleanup' %
                                     self.__class__.__name__)

    def c_support_code_apply(self, node, name):
        """
        Optional: return utility code for use by an `Op` that will be
        inserted at global scope, that can be specialized for the
        support of a particular `Apply` node.

        Parameters
        ----------
        node: an Apply instance in the graph being compiled
        name: str
            A string or number that serves to uniquely identify this node.
            Symbol names defined by this support code should include the name,
            so that they can be called from the c_code, and so that they do not
            cause name collisions.

        Notes
        -----
        This function is called in addition to c_support_code and will
        supplement whatever is returned from there.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined("c_support_code_apply",
                                     type(self), self.__class__.__name__)

    def c_init_code_apply(self, node, name):
        """
        Optional: return a code string specific to the apply
        to be inserted in the module initialization code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A string or number that serves to uniquely identify this node.
            Symbol names defined by this support code should include the name,
            so that they can be called from the c_code, and so that they do not
            cause name collisions.

        Notes
        -----
        This function is called in addition to c_init_code and will supplement
        whatever is returned from there.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined("c_init_code_apply", type(self),
                                     self.__class__.__name__)

    def c_init_code_struct(self, node, name, sub):
        """
        Optional: return a code string specific to the apply
        to be inserted in the struct initialization code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A unique name to distinguish variables from those of other nodes.
        sub
            A dictionary of values to substitute in the code.
            Most notably it contains a 'fail' entry that you should place in
            your code after setting a python exception to indicate an error.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined("c_init_code_apply", type(self),
                                     self.__class__.__name__)

    def c_support_code_struct(self, node, name):
        """
        Optional: return utility code for use by an `Op` that will be
        inserted at struct scope, that can be specialized for the
        support of a particular `Apply` node.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A unique name to distinguish you variables from those of other
            nodes.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise utils.MethodNotDefined("c_support_code_struct",
                                     type(self), self.__class__.__name__)

    def c_cleanup_code_struct(self, node, name):
        """
        Optional: return a code string specific to the apply to be
        inserted in the struct cleanup code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A unique name to distinguish variables from those of other nodes.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined("c_cleanup_code_struct", type(self),
                                     self.__class__.__name__)


class PureOp(object):
    """
    An :term:`Op` is a type of operation.

    `Op` is an abstract class that documents the interface for theano's data
    transformations. It has many subclasses, such as
    `sparse dot <http://pylearn.org/epydoc/theano.sparse.Dot-class.html>`__,
    and `Shape <http://pylearn.org/epydoc/theano.tensor.Shape-class.html>`__.

    These subclasses are meant to be instantiated.
    An instance has several responsabilities:

    - making `Apply` instances, which mean "apply this type of operation to some
      particular inputs" (via `make_node`),

    - performing the calculation of outputs from given inputs
      (via the `perform`),

    - [optionally] building gradient-calculating graphs (via `grad`).

    To see how `Op`, `Type`, `Variable`, and `Apply` fit together see the page
    on :doc:`graph`.

    For more specifications on how these methods should behave: see the
    `Op Contract` in the sphinx docs (advanced tutorial on Op-making).

    """

    default_output = None
    """
    Configuration variable for `__call__`.

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

        """
        raise utils.MethodNotDefined(
            "make_node", type(self), self.__class__.__name__)

    @classmethod
    def _get_test_value(cls, v):
        """
        Extract test value from variable v.
        Raises AttributeError if there is none.

        For a Constant, the test value is v.value.
        For a Shared variable, it is the internal value.
        For another Variable, it is the content of v.tag.test_value.

        """
        # avoid circular import
        from theano.compile.sharedvalue import SharedVariable

        if isinstance(v, graph.Constant):
            return v.value
        elif isinstance(v, SharedVariable):
            return v.get_value(borrow=True, return_internal_type=True)
        elif isinstance(v, graph.Variable) and hasattr(v.tag, 'test_value'):
            # ensure that the test value is correct
            try:
                ret = v.type.filter(v.tag.test_value)
            except Exception as e:
                # Better error message.
                detailed_err_msg = (
                    "For compute_test_value, one input test value does not"
                    " have the requested type.\n")
                detailed_err_msg += utils.get_variable_trace_string(v)

                detailed_err_msg += (
                    "\nThe error when converting the test value to that"
                    " variable type:")
                # We need to only have 1 args and it should be of type
                # string.  Otherwise, it print the tuple and so the
                # new line do not get printed.
                args = (detailed_err_msg,) + tuple(str(arg) for arg in e.args)
                e.args = ("\n".join(args),)
                raise
            return ret
        detailed_err_msg = utils.get_variable_trace_string(v)
        raise AttributeError('%s has no test value %s' % (v, detailed_err_msg))

    def __call__(self, *inputs, **kwargs):
        """
        Optional: return some or all output[s] of `make_node`.

        It is called by code such as:

        .. python::

           x = tensor.matrix()

           # tensor.exp is an Op instance, calls
           # Op.__call__(self=<instance of exp>, inputs=(x,))
           y = tensor.exp(x)

        This class implements a convenience function (for graph-building) which
        uses `default_output`, but subclasses are free to override this function
        and ignore `default_output`.

        Parameters
        ----------
        inputs
            The Op's inputs, forwarded to the call to `make_node()`.
        kwargs
            Additional keyword arguments to be forwarded to
            `make_node()` *except* for optional argument `return_list` (which
            defaults to False). If `return_list` is True, then the returned
            value is always a list. Otherwise it is either a single Variable
            when the output of `make_node()` contains a single element, or this
            output (unchanged) when it contains multiple elements.

        """
        return_list = kwargs.pop('return_list', False)
        node = self.make_node(*inputs, **kwargs)

        if config.compute_test_value != 'off':
            run_perform = True

            # build test input-values
            storage_map = {}
            compute_map = {}
            for i, ins in enumerate(node.inputs):
                try:
                    storage_map[ins] = [self._get_test_value(ins)]
                    compute_map[ins] = [True]
                except AttributeError:
                    # no test-value was specified, act accordingly
                    if config.compute_test_value == 'warn':
                        warnings.warn(
                            'Warning, Cannot compute test value: input %i (%s) of Op %s missing default value' %
                            (i, ins, node), stacklevel=2)
                        run_perform = False
                    elif config.compute_test_value == 'raise':
                        detailed_err_msg = utils.get_variable_trace_string(ins)

                        raise ValueError(
                            'Cannot compute test value: input %i (%s) of Op %s missing default value. %s' %
                            (i, ins, node, detailed_err_msg))
                    elif config.compute_test_value == 'ignore':
                        # silently skip test
                        run_perform = False
                    elif config.compute_test_value == 'pdb':
                        import pdb
                        pdb.post_mortem(sys.exc_info()[2])
                    else:
                        raise ValueError(
                            '%s is invalid for option config.compute_Test_value' %
                            config.compute_test_value)

            # if all inputs have test-values, run the actual op
            if run_perform:
                # Original values should not be destroyed:
                # copy the values of the inputs in destroy_map
                destroyed_inputs_idx = set()
                if getattr(node.op, 'destroy_map', None):
                    for i_pos_list in itervalues(node.op.destroy_map):
                        destroyed_inputs_idx.update(i_pos_list)
                for inp_idx in destroyed_inputs_idx:
                    inp = node.inputs[inp_idx]
                    storage_map[inp] = [storage_map[inp][0].copy()]

                # Prepare storage_map and compute_map for the outputs
                for o in node.outputs:
                    storage_map[o] = [None]
                    compute_map[o] = [False]

                # compute output value once with test inputs to validate graph
                thunk = node.op.make_thunk(node, storage_map, compute_map,
                                           no_recycling=[])
                thunk.inputs = [storage_map[v] for v in node.inputs]
                thunk.outputs = [storage_map[v] for v in node.outputs]

                required = thunk()
                assert not required  # We provided all inputs

                for output in node.outputs:
                    # Check that the output has been computed
                    assert compute_map[output][
                        0], (output, storage_map[output][0])

                    # add 'test_value' to output tag, so that downstream ops can use these
                    # numerical values as inputs to their perform method.
                    output.tag.test_value = storage_map[output][0]

        if self.default_output is not None:
            rval = node.outputs[self.default_output]
            if return_list:
                rval = [rval]
            return rval
        else:
            if return_list:
                return list(node.outputs)
            elif len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs

    def __ne__(self, other):
        return not (self == other)

    # Convenience so that subclass implementers don't have to import utils
    # just to self.add_tag_trace
    add_tag_trace = staticmethod(utils.add_tag_trace)

    #########################
    # Python implementation #
    #########################

    def L_op(self, inputs, outputs, output_grads):
        return self.grad(inputs, output_grads)

    def R_op(self, inputs, eval_points):
        """
        This method is primarily used by tensor.Rop

        Suppose the op outputs

        [ f_1(inputs), ..., f_n(inputs) ]

        Parameters
        ----------
        inputs : a Variable or list of Variables
        eval_points
            A Variable or list of Variables with the same length as inputs.
            Each element of eval_points specifies the value of the corresponding
            input at the point where the R op is to be evaluated.

        Returns
        -------
        list of n elements
            rval[i] should be Rop(f=f_i(inputs),
                                  wrt=inputs,
                                  eval_points=eval_points)

        """
        raise NotImplementedError(
            "%s of class %s does not "
            "implement R_op. If this is a theano op, write to the "
            "theano-dev mailing list for assistance. If it is your "
            "own op, implement the R_op method." %
            (self, self.__class__.__name__))

    def perform(self, node, inputs, output_storage, params=None):
        """
        Required: Calculate the function on the inputs and put the variables in
        the output storage. Return None.

        Parameters
        ----------
        node : Apply instance
            Contains the symbolic inputs and outputs.
        inputs : list
            Sequence of inputs (immutable).
        output_storage : list
             List of mutable 1-element lists (do not change the length of
             these lists)

        Notes
        -----
        The `output_storage` list might contain data. If an element of
        output_storage is not None, it has to be of the right type,
        for instance, for a TensorVariable, it has to be a Numpy ndarray,
        with the right number of dimensions, and the correct dtype.
        Its shape and stride pattern, can be arbitrary. It not is
        guaranteed that it was produced by a previous call to impl. It
        could be allocated by another Op impl is free to reuse it as it
        sees fit, or to discard it and allocate new memory.

        Raises
        ------
        MethodNotDefined
            The subclass does not override this method.

        """
        raise utils.MethodNotDefined(
            "perform", type(self), self.__class__.__name__,
            "Did you used Theano flags mode=FAST_COMPILE?"
            " You can use optimizer=fast_compile instead.")

    def do_constant_folding(self, node):
        """
        This allows each op to determine if it wants to be constant
        folded when all its inputs are constant. This allows it to
        choose where it puts its memory/speed trade-off. Also, it
        could make things faster as constants can't be used for inplace
        operations (see *IncSubtensor).

        """
        return True


class Op(utils.object2, PureOp, CLinkerOp):
    """
    Convenience class to bundle `PureOp` and `CLinkerOp`.

    """
    def prepare_node(self, node, storage_map, compute_map, impl):
        """
        Make any special modifications that the Op needs before doing
        make_thunk().

        This can modify the node inplace and should return nothing.

        It can be called multiple time with different impl. It is the
        op responsability to don't re-prepare the node when it isn't
        good to do so.

        """
        pass

    def make_c_thunk(self, node, storage_map, compute_map, no_recycling):
        """Like make_thunk, but will only try to make a C thunk.

        """
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        # float16 gets special treatment since running
        # unprepared C code will get bad results.
        if not getattr(self, '_f16_ok', False):
            def is_f16(t):
                return getattr(t, 'dtype', '') == 'float16'

            if (any(is_f16(i.type) for i in node.inputs) or
                    any(is_f16(o.type) for o in node.outputs)):
                print("Disabling C code for %s due to unsupported "
                      "float16" % (self,))
                raise NotImplementedError("float16")
        e = FunctionGraph(node.inputs, node.outputs)
        e_no_recycling = [new_o
                          for (new_o, old_o) in zip(e.outputs, node.outputs)
                          if old_o in no_recycling]
        cl = theano.gof.cc.CLinker().accept(e,
                                            no_recycling=e_no_recycling)

        _logger.debug('Trying CLinker.make_thunk')
        outputs = cl.make_thunk(input_storage=node_input_storage,
                                output_storage=node_output_storage)
        fill_storage, node_input_filters, node_output_filters = outputs

        def rval():
            fill_storage()
            for o in node.outputs:
                compute_map[o][0] = True

        rval.cthunk = fill_storage.cthunk
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.lazy = False
        return rval

    def make_py_thunk(self, node, storage_map, compute_map, no_recycling,
                      debug=False):
        """
        Like make_thunk() but only makes python thunks.

        """
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        if debug:
            p = node.op.debug_perform
        else:
            p = node.op.perform

        params = node.run_params()

        if params is graph.NoParams:
            # default arguments are stored in the closure of `rval`
            def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
                r = p(n, [x[0] for x in i], o)
                for o in node.outputs:
                    compute_map[o][0] = True
                return r
        else:
            params_val = node.params_type.filter(params)

            def rval(p=p, i=node_input_storage, o=node_output_storage, n=node,
                     params=params_val):
                r = p(n, [x[0] for x in i], o, params)
                for o in node.outputs:
                    compute_map[o][0] = True
                return r

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval

    def make_thunk(self, node, storage_map, compute_map, no_recycling,
                   impl=None):
        """
        This function must return a thunk, that is a zero-arguments
        function that encapsulates the computation to be performed
        by this op on the arguments of the node.

        Parameters
        ----------
        node
            Something previously returned by self.make_node.
        storage_map
            dict variable -> one-element-list where a computed
            value for this variable may be found.
        compute_map
            dict variable -> one-element-list where a boolean
            value will be found. The boolean indicates whether the
            variable's storage_map container contains a valid value (True)
            or if it has not been computed yet (False).
        no_recycling
            List of variables for which it is forbidden to reuse memory
            allocated by a previous call.
        impl
            Currently, None, 'c' or 'py'. If 'c' or 'py' we will only try
            that version of the code.

        Notes
        -----
        If the thunk consults the storage_map on every call, it is safe
        for it to ignore the no_recycling argument, because elements of the
        no_recycling list will have a value of None in the storage map.  If
        the thunk can potentially cache return values (like CLinker does),
        then it must not do so for variables in the no_recycling list.

        self.prepare_node(node, ...) is always called. If we try 'c' and it
        fail and we try again 'py', prepare_node will be called twice.
        """

        if (impl is None and theano.config.cxx) or impl == 'c':
            self.prepare_node(node, storage_map=storage_map,
                              compute_map=compute_map, impl='c')
            try:
                return self.make_c_thunk(node, storage_map, compute_map,
                                         no_recycling)
            except (NotImplementedError, utils.MethodNotDefined):
                # We requested the c code, so don't catch the error.
                if impl == 'c':
                    raise
                _logger.debug('Falling back on perform')

        # condition: either there was no c_code, or it failed or
        # python code was requested.
        self.prepare_node(node, storage_map=storage_map,
                          compute_map=compute_map, impl='py')
        return self.make_py_thunk(node, storage_map, compute_map, no_recycling)

    def make_node(self, *inputs):
        """
        Create a "apply" nodes for the inputs in that order.
        """
        if not hasattr(self, 'itypes'):
            raise NotImplementedError("You can either define itypes and otypes,\
             or implement make_node")

        if not hasattr(self, 'otypes'):
            raise NotImplementedError("You can either define itypes and otypes,\
             or implement make_node")

        if len(inputs) != len(self.itypes):
            raise ValueError("We expected %d inputs but got %d." %
                             (len(self.itypes), len(inputs)))
        if not all(inp.type == it for inp, it in zip(inputs, self.itypes)):
            raise TypeError(
                "We expected inputs of types '%s' but got types '%s' " %
                (str(self.itypes), str([inp.type for inp in inputs])))
        return theano.Apply(self, inputs, [o() for o in self.otypes])


def get_test_value(v):
    """
    Extract test value from `v`. Raises AttributeError if there is none.

    If input `v` is not already a variable, it is turned into one by calling
    `as_tensor_variable(v)`, so that this function can be applied e.g.
    on numpy arrays or Python lists and scalars, considering them as constants.

    For a Constant, the test value is v.value.
    For a Shared variable, it is the internal value.
    For another Variable, it is the content of v.tag.test_value.

    """
    if not isinstance(v, graph.Variable):
        v_var = theano.tensor.as_tensor_variable(v)
    else:
        v_var = v
    return PureOp._get_test_value(v_var)


def missing_test_message(msg):
    """
    Displays msg, a message saying that some test_value is missing,
    in the appropriate form based on config.compute_test_value:

        off: The interactive debugger is off, so we do nothing.
        ignore: The interactive debugger is set to ignore missing inputs,
                so do nothing.
        warn: Display msg as a warning.

    Raises
    ------
    AttributeError
        With msg as the exception text.

    """
    action = config.compute_test_value
    if action == 'raise':
        raise AttributeError(msg)
    elif action == 'warn':
        warnings.warn(msg, stacklevel=2)
    else:
        assert action in ['ignore', 'off']


def debug_error_message(msg):
    """
    Displays a message saying that an error was found in some
    test_values. Becomes a warning or a ValueError depending on
    config.compute_test_value.

    """
    action = config.compute_test_value

    # this message should never be called when the debugger is off
    assert action != 'off'

    if action in ['raise', 'ignore']:
        raise ValueError(msg)
    else:
        assert action == 'warn'
        warnings.warn(msg, stacklevel=2)


def debug_assert(condition, msg=None):
    """
    Customized assert with options to ignore the assert
    with just a warning
    """
    if msg is None:
        msg = 'debug_assert failed'
    if not condition:
        action = config.compute_test_value
        if action in ['raise', 'ignore']:
            raise AssertionError(msg)
        else:
            assert action == 'warn'
            warnings.warn(msg, stacklevel=2)


def get_debug_values(*args):
    """
    Intended use:

        for val_1, ..., val_n in get_debug_values(var_1, ..., var_n):
            if some condition on val_1, ..., val_n is not met:
                debug_error_message("condition was not met")

    Given a list of variables, get_debug_values does one of three things:

        1. If the interactive debugger is off, returns an empty list
        2. If the interactive debugger is on, and all variables have
            debug values, returns a list containing a single element.
            This single element is either:
                a) if there is only one variable, the element is its
                   value
                b) otherwise, a tuple containing debug values of all
                   the variables.
        3. If the interactive debugger is on, and some variable does
            not have a debug value, issue a missing_test_message about
            the variable, and, if still in control of execution, return
            an empty list.

    """

    if config.compute_test_value == 'off':
        return []

    rval = []

    for i, arg in enumerate(args):
        try:
            rval.append(get_test_value(arg))
        except AttributeError:
            if hasattr(arg, 'name') and arg.name is not None:
                missing_test_message("Argument " + str(i) + "('" + arg.name +
                                     "') has no test value")
            else:
                missing_test_message("Argument " + str(i) +
                                     " has no test value")
            return []

    if len(rval) == 1:
        return rval

    return [tuple(rval)]


ops_with_inner_function = {}
"""
Registry of Ops that have an inner compiled Theano function.

The keys are Op classes (not instances), and values are the name of the
attribute that contains the function. For instance, if the function is
self.fn, the value will be 'fn'.

We need that to be able not to run debug checks a number of times that is
exponential in the nesting level of those ops.
For instance, Scan will be registered here.

"""


class OpenMPOp(Op):
    """
    All op using OpenMP code should inherit from this Op.

    This op will check that the compiler support correctly OpenMP code.
    If not, it will print a warning and disable openmp for this Op.
    Then it will generate the not OpenMP code.

    This is needed as EPD on Windows g++ version spec information tell
    it support OpenMP, but does not include the OpenMP files.

    We also add the correct compiler flags in c_compile_args.

    """

    gxx_support_openmp = None
    """
    True/False after we tested this.

    """

    def __init__(self, openmp=None):
        if openmp is None:
            openmp = theano.config.openmp
        self.openmp = openmp

    def __setstate__(self, d):
        self.__dict__.update(d)
        # If we unpickle old op
        if not hasattr(self, "openmp"):
            self.openmp = False

    def c_compile_args(self):
        """
        Return the compilation arg "fopenmp" if openMP is supported
        """
        self.update_self_openmp()
        if self.openmp:
            return ['-fopenmp']
        return []

    def c_headers(self):
        """
        Return the header file name "omp.h" if openMP is supported
        """
        self.update_self_openmp()
        if self.openmp:
            return ["omp.h"]
        return []

    @staticmethod
    def test_gxx_support():
        """
        Check if openMP is supported
        """
        code = """
        #include <omp.h>
int main( int argc, const char* argv[] )
{
        int res[10];

        for(int i=0; i < 10; i++){
            res[i] = i;
        }
}
        """
        default_openmp = GCC_compiler.try_compile_tmp(
            src_code=code,
            tmp_prefix='test_omp_',
            flags=['-fopenmp'],
            try_run=False)
        return default_openmp

    def update_self_openmp(self):
        """
        Make sure self.openmp is not True if there is no support in gxx.

        """
        if self.openmp:
            if OpenMPOp.gxx_support_openmp is None:
                OpenMPOp.gxx_support_openmp = OpenMPOp.test_gxx_support()
                if not OpenMPOp.gxx_support_openmp:
                    # We want to warn only once.
                    warnings.warn(
                        "Your g++ compiler fails to compile OpenMP code. We"
                        " know this happen with some version of the EPD mingw"
                        " compiler and LLVM compiler on Mac OS X."
                        " We disable openmp everywhere in Theano."
                        " To remove this warning set the theano flags `openmp`"
                        " to False.",
                        stacklevel=3)
            if OpenMPOp.gxx_support_openmp is False:
                self.openmp = False
                theano.config.openmp = False

    def prepare_node(self, node, storage_map, compute_map, impl):
        if impl == 'c':
            self.update_self_openmp()


def simple_meth(tag):
    def f(self):
        if tag in self.code_sections:
            return self.code_sections[tag]
        else:
            raise utils.MethodNotDefined(
                'c_' + tag, type(self), type(self).__name__)
    f.__name__ = 'c_' + tag
    return f


def apply_meth(tag):
    def f(self, node, name):
        if tag in self.code_sections:
            code = self.code_sections[tag]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return '\n'.join(['', define_macros, code,
                              undef_macros])
        else:
            raise utils.MethodNotDefined(
                'c_' + tag, type(self), type(self).__name__)
    f.__name__ = 'c_' + tag
    return f


class COp(Op):
    """
    Class to allow an op to have an external C implementation.

    An op can use this class by inheriting from it and calling its
    __init__() method, providing it with a path to an external file containing
    the C implementation and the name of the function, in that file, to call
    to perform the computations for the op.

    """

    section_re = re.compile(r'^#section ([a-zA-Z0-9_]+)$', re.MULTILINE)
    backward_re = re.compile(
        r'^THEANO_(APPLY|SUPPORT)_CODE_SECTION$',
        re.MULTILINE)
    # This is the set of allowed markers
    SECTIONS = set([
        'init_code', 'init_code_apply', 'init_code_struct',
        'support_code', 'support_code_apply', 'support_code_struct',
        'cleanup_code_struct',
        'code', 'code_cleanup'])

    @classmethod
    def get_path(cls, f):
        """
        Convert a path relative to the location of the class file into
        an aboslute path. Paths that are already absolute are passed
        through unchanged.

        """
        if not os.path.isabs(f):
            class_file = inspect.getfile(cls)
            class_dir = os.path.dirname(class_file)
            f = os.path.realpath(os.path.join(class_dir, f))
        return f

    def __init__(self, func_files, func_name=None):
        """
        Sections are loaded from files in order with sections in later
        files overriding sections in previous files.

        """
        if not isinstance(func_files, list):
            func_files = [func_files]

        self.func_name = func_name
        # Keep the original name. If we reload old pickle, we want to
        # find the new path and new version of the file in Theano.
        self.func_files = func_files
        self.load_c_code(func_files)

        if len(self.code_sections) == 0:
            raise ValueError("No sections where defined in C files")

        if self.func_name is not None:
            if 'op_code' in self.code_sections:
                # maybe a warning instead (and clearing the key)
                raise ValueError('Cannot have an "op_code" section and '
                                 'specify the func_name')
            if 'op_code_cleanup' in self.code_sections:
                # maybe a warning instead (and clearing the key)
                raise ValueError('Cannot have an "op_code_cleanup" section '
                                 'and specify the func_name')

    def load_c_code(self, func_files):
        """
        Loads the c code to perform the Op
        """
        func_files = [self.get_path(f) for f in func_files]
        self.func_codes = []
        for func_file in func_files:
            # U (universal) will convert all new lines format to \n.
            with open(func_file, 'U') as f:
                self.func_codes.append(f.read())

        # If both the old section markers and the new section markers are
        # present, raise an error because we don't know which ones to follow.
        old_markers_present = False
        new_markers_present = False
        for code in self.func_codes:
            if self.backward_re.search(code):
                old_markers_present = True
            if self.section_re.search(code):
                new_markers_present = True

        if old_markers_present and new_markers_present:
            raise ValueError('Both the new and the old syntax for '
                             'identifying code sections are present in the '
                             'provided C code. These two syntaxes should not '
                             'be used at the same time.')

        self.code_sections = dict()
        for i, code in enumerate(self.func_codes):
            if self.backward_re.search(code):
                # This is backward compat code that will go away in a while

                # Separate the code into the proper sections
                split = self.backward_re.split(code)
                n = 1
                while n < len(split):
                    if split[n] == 'APPLY':
                        self.code_sections['support_code_apply'] = split[n + 1]
                    elif split[n] == 'SUPPORT':
                        self.code_sections['support_code'] = split[n + 1]
                    n += 2
                continue

            elif self.section_re.search(code):

                # Check for code outside of the supported sections
                split = self.section_re.split(code)
                if split[0].strip() != '':
                    raise ValueError('Stray code before first #section '
                                     'statement (in file %s): %s' %
                                     (func_files[i], split[0]))

                # Separate the code into the proper sections
                n = 1
                while n < len(split):
                    if split[n] not in self.SECTIONS:
                        raise ValueError(
                            "Unknown section type (in file %s): %s" %
                            (func_files[i], split[n]))
                    if split[n] not in self.code_sections:
                        self.code_sections[split[n]] = ""
                    self.code_sections[split[n]] += split[n + 1]
                    n += 2

            else:
                raise ValueError("No valid section marker was found in file "
                                 "%s" % func_files[i])

    def get_op_params(self):
        """
        Returns a list of (name, value) pairs that will be turned into
        macros for use within the op code. This is intended to allow
        an op's properties to influence the generated C code.

        The names must be strings that are not a C keyword and the
        values must be strings of literal C representations.

        """
        return []

    def c_code_cache_version(self):
        return hash(tuple(self.func_codes))

    def c_init_code(self):
        """
        Get the code section for init_code
        """
        if 'init_code' in self.code_sections:
            return [self.code_sections['init_code']]
        else:
            raise utils.MethodNotDefined(
                'c_init_code', type(self), type(self).__name__)

    c_init_code_apply = apply_meth('init_code_apply')
    c_support_code = simple_meth('support_code')
    c_support_code_apply = apply_meth('support_code_apply')
    c_support_code_struct = apply_meth('support_code_struct')
    c_cleanup_code_struct = apply_meth('cleanup_code_struct')

    def format_c_function_args(self, inp, out):
        # Generate an string containing the arguments sent to the external C
        # function. The argstring will be of format :
        # "input0, input1, input2, &output0, &output1"
        inp = list(inp)
        numi = getattr(self, '_cop_num_inputs', len(inp))
        while len(inp) < numi:
            inp.append('NULL')
        out = ["&%s" % o for o in out]
        numo = getattr(self, '_cop_num_outputs', len(out))
        while len(out) < numo:
            out.append('NULL')
        return ", ".join(inp + out)

    def get_c_macros(self, node, name, check_input=None):
        define_template = "#define %s %s"
        undef_template = "#undef %s"
        define_macros = []
        undef_macros = []

        if check_input is None:
            check_input = getattr(self, 'check_input', True)

        if check_input:
            # Extract the various properties of the input and output variables
            variables = node.inputs + node.outputs
            variable_names = (["INPUT_%i" % i for i in range(len(node.inputs))] +
                              ["OUTPUT_%i" % i for i in range(len(node.outputs))])

            # Generate dtype macros
            for i, v in enumerate(variables):
                if not hasattr(v, 'dtype'):
                    continue
                vname = variable_names[i]

                macro_name = "DTYPE_" + vname
                macro_value = "npy_" + v.dtype

                define_macros.append(
                    define_template %
                    (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

                d = np.dtype(v.dtype)

                macro_name = "TYPENUM_" + vname
                macro_value = d.num

                define_macros.append(
                    define_template %
                    (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

                macro_name = "ITEMSIZE_" + vname
                macro_value = d.itemsize

                define_macros.append(
                    define_template %
                    (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

        # Generate a macro to mark code as being apply-specific
        define_macros.append(define_template % ("APPLY_SPECIFIC(str)",
                                                "str##_%s" % name))
        undef_macros.append(undef_template % "APPLY_SPECIFIC")

        for n, v in self.get_op_params():
            define_macros.append(define_template % (n, v))
            undef_macros.append(undef_template % (n,))

        return '\n'.join(define_macros), '\n'.join(undef_macros)

    def _lquote_macro(self, txt):
        res = []
        spl = txt.split('\n')
        for l in spl[:-1]:
            res.append(l + ' \\')
        res.append(spl[-1])
        return '\n'.join(res)

    def get_sub_macros(self, sub):
        define_macros = []
        undef_macros = []
        define_macros.append("#define FAIL %s" % (
                             self._lquote_macro(sub['fail']),))
        undef_macros.append("#undef FAIL")
        if 'params' in sub:
            define_macros.append("#define PARAMS %s" % (sub['params'],))
            undef_macros.append("#undef PARAMS")

        return '\n'.join(define_macros), '\n'.join(undef_macros)

    def get_io_macros(self, inputs, outputs):
        define_macros = []
        undef_macros = []

        for i, inp in enumerate(inputs):
            define_macros.append("#define INPUT_%d %s" (i, inp))
            undef_macros.append("#undef INPUT_%d", (i,))

        for i, out in enumerate(outputs):
            define_macros.append("#define OUTPUT_%d %s" (i, inp))
            undef_macros.append("#undef OUTPUT_%d", (i,))

    def c_init_code_struct(self, node, name, sub):
        """
        Stitches all the macros and "init_code" together

        """
        if 'init_code_struct' in self.code_sections:
            op_code = self.code_sections['init_code_struct']

            def_macros, undef_macros = self.get_c_macros(node, name)
            def_sub, undef_sub = self.get_sub_macros(sub)

            return '\n'.join(['', def_macros, def_sub,
                              op_code,
                              undef_sub, undef_macros])
        else:
            raise utils.MethodNotDefined(
                'c_init_code_struct', type(self), type(self).__name__)

    def c_code(self, node, name, inp, out, sub):
        if self.func_name is not None:
            assert 'code' not in self.code_sections

            define_macros, undef_macros = self.get_c_macros(node, name,
                                                            check_input=False)

            params = ""
            if 'params' in sub:
                params = ", %s" % (sub['params'],)

            # Generate the C code
            return """
                %(define_macros)s
                {
                  if (%(func_name)s(%(func_args)s%(params)s) != 0) {
                    %(fail)s
                  }
                }
                %(undef_macros)s
                """ % dict(func_name=self.func_name,
                           fail=sub['fail'], params=params,
                           func_args=self.format_c_function_args(inp, out),
                           define_macros=define_macros,
                           undef_macros=undef_macros)
        else:
            if 'code' in self.code_sections:
                op_code = self.code_sections['code']

                def_macros, undef_macros = self.get_c_macros(node, name)
                def_sub, undef_sub = self.get_sub_macros(sub)
                def_io, undef_io = self.get_io_macros(inp, out)

                return '\n'.join([def_macros, def_sub, def_io,
                                  op_code,
                                  undef_io, undef_sub, undef_macros])
            else:
                raise utils.MethodNotDefined(
                    'c_code', type(self), type(self).__name__)

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """
        Stitches all the macros and "code_cleanup" together
        """
        if 'code_cleanup' in self.code_sections:
            op_code = self.code_sections['code_cleanup']

            def_macros, undef_macros = self.get_c_macros(node, name)
            def_sub, undef_sub = self.get_sub_macros(sub)
            def_io, undef_io = self.get_io_macros(inputs, outputs)

            return '\n'.join([def_macros, def_sub, def_io,
                              op_code,
                              undef_io, undef_sub, undef_macros])
        else:
            raise utils.MethodNotDefined(
                'c_code_cleanup', type(self), type(self).__name__)
