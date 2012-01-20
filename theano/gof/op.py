"""Defines base classes `Op`, `PureOp`, and `CLinkerOp`

The `Op` class is the base interface for all operations
compatible with `gof`'s :doc:`graph` routines.
"""
__authors__   = "theano-dev"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "theano-dev <theano-dev@googlegroups.com>"


__docformat__ = "restructuredtext en"


import logging
import warnings

import theano
from theano import config

import cc
import graph
import utils
from env import Env


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
           Each string is the name of a `PyObject` pointer where the Op should
           store its variables.  As of version 0.4.0, this pointer could be
           NULL, or contain an object allocated during a previous call to the
           same function, unchanged from the end of the previous execution.
           In a future version, there will be no guarantee on where that
           object will be created (it could be allocated during a previous
           execution, or by another Op, by the Mode, etc.). It will still
           be of an appropriate Type (in the Theano sense) to store the output
           of the computation: for instance, for a TensorVariable, it will be a
           Numpy ndarray with the right number of dimensions, and the right dtype.
           However, its shape, or stride pattern, could not be adequate.
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
           variables.  This pointer could be NULL, or contain an object of the right
           Type (in the Theano sense) to store the output of the computation.
           For instance, for a TensorVariable, it will be a Numpy ndarray with
           the right number of dimensions, and the right dtype. However, its
           shape, or stride pattern, could not be adequate.
           It could be unchanged from the end of the previous execution, or allocated
           by another Op, or by the Mode.
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
        raise utils.MethodNotDefined("c_support_code_apply",
                type(self), self.__class__.__name__)


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

    add_stack_trace_on_call = True
    """This class variable governs whether __call__ adds a stack trace to the node it creates.

    The tag trace is meant to connect a node to the line a user typed. It is nice for
    debugging. It does not make as much sense during optimizations to store this information.
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

    @classmethod
    def _get_test_value(cls, v):
        """
        Extract test value from variable v. Raises AttributeError if there is none.

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
            return v.type.filter(v.tag.test_value)

        raise AttributeError('%s has no test value' % v)

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
        if self.add_stack_trace_on_call:
            self.add_tag_trace(node)

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
                        warnings.warn('Warning, Cannot compute test value: input %i (%s) of Op %s missing default value' % (i, ins, node), stacklevel=2)
                        run_perform = False
                    elif config.compute_test_value == 'raise':
                        raise ValueError('Cannot compute test value: input %i (%s) of Op %s missing default value' % (i, ins, node))
                    elif config.compute_test_value == 'ignore':
                        # silently skip test
                        run_perform = False
                    else:
                        raise ValueError('%s is invalid for option config.compute_Test_value' % config.compute_test_value)

            # if all inputs have test-values, run the actual op
            if run_perform:
                # Original values should not be destroyed:
                # copy the values of the inputs in destroy_map
                destroyed_inputs_idx = set()
                if getattr(node.op, 'destroy_map', None):
                    for i_pos_list in node.op.destroy_map.itervalues():
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

                required = thunk()
                assert not required  # We provided all inputs

                for output in node.outputs:
                    # Check that the output has been computed
                    assert compute_map[output][0], (output, storage_map[output][0])

                    # add 'test_value' to output tag, so that downstream ops can use these
                    # numerical values as inputs to their perform method.
                    output.tag.test_value = storage_map[output][0]

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


    def R_op(self, inputs, eval_points):
        """

        This method is primarily used by tensor.Rop

        Suppose the op outputs

        [ f_1(inputs), ..., f_n(inputs) ]

        inputs: a Variable or list of Variables
        eval_points: a Variable or list of Variables with
                    the same length as inputs. Each element
                    of eval_points specifies the value of
                    the corresponding input at the point
                    where the R op is to be evaluated.


        returns: a list of n elements
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
        output_storage is not None, it has to be of the right type,
        for instance, for a TensorVariable, it has to be a Numpy ndarray,
        with the right number of dimensions, and the correct dtype.
        Its shape and stride pattern, can be arbitrary. It not is
        guaranteed that it was produced by a previous call to impl. It
        could be allocated by another Op impl is free to reuse it as it
        sees fit, or to discard it and allocate new memory.

        :Exceptions:
         - `MethodNotDefined`: the subclass does not override this method

        """
        raise utils.MethodNotDefined("perform", type(self), self.__class__.__name__)


class Op(utils.object2, PureOp, CLinkerOp):
    """Convenience class to bundle `PureOp` and `CLinkerOp`"""
    def __new__(cls, *args, **kwargs):
        # this function exists to silently and transparently ensure that all
        # existing Ops get a _op_use_c_code attribute
        obj = object.__new__(cls)
        if not hasattr(obj, '_op_use_c_code'):
            obj._op_use_c_code = True
        return obj

    def __init__(self, use_c_code=True):
        self._op_use_c_code = use_c_code

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """
        :param node: something previously returned by self.make_node

        :param storage_map: dict variable -> one-element-list where a computed
                value for this variable may be found.

        :param compute_map: dict variable -> one-element-list where a boolean
                value will be found.  The boolean indicates whether the
                variable's storage_map container contains a valid value (True)
                or if it has not been computed yet (False).

        :param no_recycling: list of variables for which it is forbidden to
                reuse memory allocated by a previous call.

        :note: If the thunk consults the storage_map on every call, it is safe
            for it to ignore the no_recycling argument, because elements of the
            no_recycling list will have a value of None in the storage map.  If
            the thunk can potentially cache return values (like CLinker does),
            then it must not do so for variables in the no_recycling list.
        """
        logger = logging.getLogger('theano.gof.op.Op')

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        #logger.debug('Compiling node %i of graph' % node_idx)
        if self._op_use_c_code:
            try:
                e = Env(*graph.clone(node.inputs, node.outputs))

                e_no_recycling = [new_o
                        for (new_o, old_o) in zip(e.outputs, node.outputs)
                        if old_o in no_recycling]
                cl = cc.CLinker().accept(e,
                        no_recycling=e_no_recycling)

                logger.debug('Trying CLinker.make_thunk')
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
            except (NotImplementedError, utils.MethodNotDefined):
                logger.debug('Falling back on perform')

        # condition: either there was no c_code, or it failed

        p = node.op.perform
        # default arguments are stored in the closure of `rval`

        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            return r

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval


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
    v_tensor = theano.tensor.as_tensor_variable(v)
    return PureOp._get_test_value(v_tensor)


def missing_test_message(msg):
    """ Displays msg, a message saying that some test_value is missing,
    in the appropriate form based on config.compute_test_value:

        off: the interactive debugger is off, so we do nothing
        ignore: the interactive debugger is set to ignore missing inputs,
                so do nothing
        warn: display msg as a warning
        raise: raise an AttributeError with msg as the exception text
    """
    action = config.compute_test_value
    if action == 'raise':
        raise AttributeError(msg)
    elif action == 'warn':
        warnings.warn(msg, stacklevel=2)
    else:
        assert action in ['ignore', 'off']


def debug_error_message(msg):
    """ Displays a message saying that an error was found in some
    test_values. Becomes a warning or a ValueError depending on
    config.compute_test_value"""

    action = config.compute_test_value

    #this message should never be called when the debugger is off
    assert action != 'off'

    if action in ['raise', 'ignore']:
        raise ValueError(msg)
    else:
        assert action == 'warn'
        warnings.warn(msg, stacklevel=2)


def debug_assert(condition, msg=None):
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
            an empty list

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
