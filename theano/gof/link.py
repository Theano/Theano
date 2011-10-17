"""WRITEME"""
import utils
import graph
from type import Type

import sys, traceback
from copy import copy
from theano.gof.python25 import all

__excepthook = sys.excepthook

def log_thunk_trace(value, f=sys.stderr):
    """Log theano's diagnostic stack trace for an exception
    raised by raise_with_op.
    """
    # in future, consider accepting `write` as arg rather than file
    # to support writing to a logger
    def write(msg):
        print >> f, "log_thunk_trace: %s" % msg.strip()

    if hasattr(value, '__thunk_trace__'):
        trace2 = value.__thunk_trace__
        write("There was a problem executing an Op.")
        if trace2 is None:
            write("Could not find where this Op was defined.")
            write(" * You might have instantiated this Op "
                    "directly instead of using a constructor.")
            write(" * The Op you constructed might have been"
                    " optimized. Try turning off optimizations.")
        elif trace2:
            write("Definition in: ")
            for line in traceback.format_list(trace2):
                write(line)
            write("For the full definition stack trace set"
                    " the Theano flags traceback.limit to -1")


def thunk_hook(type, value, trace):
    """WRITEME
    This function is meant to replace excepthook and do some
    special work if the exception value has a __thunk_trace__
    field. In that case, it retrieves the field, which should
    contain a trace as returned by L{traceback.extract_stack},
    and prints it out on L{stderr}.

    The normal excepthook is then called.

    :note: This hook replaced by nosetests, so it does not run in nose tests.
    """
    log_thunk_trace(value)
    __excepthook(type, value, trace)
sys.excepthook = thunk_hook


def raise_with_op(op, exc_info=None):
    """
    Re-raise an exception while annotating the exception object with
    debug info.

    Parameters
    ----------
    op : object
        The Op object that resulted in the raised exception.
    exc_info : tuple, optional
        A tuple containing the exception type, exception object and
        associated traceback, as would be returned by a call to
        `sys.exc_info()` (which is done if `None` is passed).

    Notes
    -----

    This re-raises the exception described by `exc_info` (or the last
    one raised, if `exc_info` is omitted) and annotates the exception
    object with several new members which may be helpful for debugging
    Theano graphs. They are:

     * __op_instance__: The Op that is responsible for the exception
       being raised.
     * __thunk_trace__: A traceback corresponding to the code that
       actually generated the exception, if it is available.
     * __applynode_index__: The index of the Apply node corresponding
       to this op in `op.env.toposort()`.

    The exception is not annotated if it is of type `KeyboardInterrupt`.
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    exc_type, exc_value, exc_trace = exc_info
    if exc_type == KeyboardInterrupt:
        # print a simple traceback from KeyboardInterrupt
        raise exc_type, exc_value, exc_trace
    try:
        trace = op.tag.trace
    except AttributeError:
        trace = ()
    exc_value.__thunk_trace__ = trace
    exc_value.__op_instance__ = op
    if op in op.env.toposort():
        exc_value.__applynode_index__ = op.env.toposort().index(op)
    else:
        exc_value.__applynode_index__ = None

    # nose and unittest catch the exception and do not run th thunk_hook
    # so it can be useful to just blurt out errors right here
    if raise_with_op.print_thunk_trace:
        log_thunk_trace(exc_value)

    raise exc_type, exc_value, exc_trace

raise_with_op.print_thunk_trace = False

class Linker(object):
    """WRITEME"""

    def make_thunk(self):
        """
        This function must return a triplet (function, input_variables, output_variables)
        where function is a thunk that operates on the returned variables. If inplace
        is True, the input_variables and output_variables lists will be the same as the
        inputs and outputs of the graph provided to the L{Linker}. Else, independent
        variables will be returned.

        Example::
         x, y = Variable(Double), Variable(Double)
         e = x + y
         env = Env([x, y], [e])
         fn, (new_x, new_y), (new_e, ) = MyLinker(env).make_thunk(inplace)
         new_x.data = 1.0
         new_y.data = 2.0
         fn()
         print new_e.data # 3.0
         print e.data # 3.0 iff inplace == True (else unknown)
        """
        raise utils.MethodNotDefined("make_thunk", type(self), self.__class__.__name__)

    ## DELETEME ##
    def make_function(self, unpack_single = True, **kwargs):
        """
        Returns a function that takes values corresponding to the inputs of the
        env used by this L{Linker} and returns values corresponding the the outputs
        of that env. If inplace is True, the calculations will operate in the
        same storage the env uses, else independent storage will be allocated
        for the function.

        Example::
         e = x + y
         env = Env([x, y], [e])
         fn = MyLinker(env).make_function(inplace)
         print fn(1.0, 2.0) # 3.0
         print e.data # 3.0 iff inplace == True (else unknown)

        If unpack_single is True (default) and that the function has only one
        output, then that output will be returned. Else, a list or tuple of
        length 1 will be returned.
        """
        thunk, inputs, outputs = self.make_thunk(**kwargs)
        def execute(*args):
            def e_arity(takes, got):
                return 'Function call takes exactly %i %s (%i given)' \
                        % (takes, ['argument','arguments'][takes>1], got)
            if (len(args) != len(inputs)):
                raise TypeError(e_arity(len(inputs), len(args)))
            for arg, variable in zip(args, inputs):
                variable.data = arg
            thunk()
            if unpack_single:
                return utils.to_return_values([variable.data for variable in outputs])
            else:
                return [variable.data for variable in outputs]
        execute.thunk = thunk
        execute.inputs = inputs
        execute.outputs = outputs

        return execute


#TODO: Move this class to the compile module, where it is used (and for which it exists).
class Container(object):
    """This class joins a variable with its computed value.
    It is used in linkers, especially for the inputs and outputs of a Function.
    """
    def __init__(self, r, storage, readonly=False, strict=False,
            allow_downcast=None, name=None):
        """WRITEME

        :Parameters:
         `r`: a variable
         `storage`: a list of length 1, whose element is the value for `r`
         `readonly`: True indicates that this should not be setable by Function[r] = val
         `strict`: if True, we don't allow type casting.
         `allow_downcast`: if True (and `strict` is False), allow upcasting
            of type, but not downcasting. If False, prevent it. If None
            (default), allows only downcasting of float to floatX scalar.
         `name`: A string (for pretty-printing?)

        """
        if not isinstance(storage, list) or not len(storage) >= 1:
            raise TypeError("storage must be a list of length at least one")
        #self.r = r
        if isinstance(r, Type):
            self.type = r
        else:
            self.type = r.type
        if name is None:
            self.name = r.name

        self.storage = storage
        self.readonly = readonly
        self.strict = strict
        self.allow_downcast = allow_downcast

    def __get__(self):
        return self.storage[0]

    def __set__(self, value):
        if self.readonly:
            raise Exception("Cannot set readonly storage: %s" % self.name)
        try:
            if value is None:
                self.storage[0] = None
                return

            kwargs = {}
            if self.strict:
                kwargs['strict'] = True
            if self.allow_downcast is not None:
                kwargs['allow_downcast'] = self.allow_downcast
            if hasattr(self.type,'filter_inplace'):
                self.storage[0] = self.type.filter_inplace(value, self.storage[0], **kwargs)
            else:
                self.storage[0] = self.type.filter(value, **kwargs)

        except Exception, e:
            e.args = e.args + (('Container name "%s"' % self.name),)
            raise
    data = property(__get__, __set__)
    value = property(__get__, __set__)
    def __str__(self):
        return "<" + str(self.storage[0]) + ">"
    def __repr__(self):
        return "<" + repr(self.storage[0]) + ">"


def map_storage(env, order, input_storage, output_storage):
    """Ensure there is storage (a length-1 list) for inputs, outputs, and interior nodes.

    :param env: The current env.  This function uses the inputs and outputs attributes.
    :param order: an iterable over Apply instances (in program running order)
    :param input_storage: None or existing input storage (see below)
    :param output_storage: None or existing output storage (see below)

    :rtype: 3-tuple
    :returns: (list of storage for inputs, list of storage for outputs, and the `storage_map`)


    This function iterates over the nodes in `order` and ensures that for every
    input and output `Variable`, there is a unique storage container.  This is
    returned as a dictionary Variable->storage called the `storage_map`.

    This function also returns `input_storage` which is a list of storages corresponding to env.inputs.
    This function also returns `output_storage` which is a list of storages corresponding to env.outputs.

    """
    #each Apply argument's data is stored in a list of length 1 (these lists act like pointers)

    # input_storage is a list of data-containers for the inputs.
    if input_storage is None:
        input_storage = [[None] for input in env.inputs]
    else:
        assert len(env.inputs) == len(input_storage)

    storage_map = {}
    for r, storage in zip(env.inputs, input_storage):
        storage_map[r] = storage
#     for orphan in env.orphans:
#         if not isinstance(orphan, Constant):
#             raise TypeError("Cannot link a graph with non-constant orphans.", orphan)
#         storage_map[orphan] = [orphan.data]

    if output_storage is not None:
        assert len(env.outputs) == len(output_storage)
        for r, storage in zip(env.outputs, output_storage):
            storage_map[r] = storage

    thunks = []
    for node in order:
        for r in node.inputs:
            if r not in storage_map:
                assert isinstance(r, graph.Value)
                storage_map[r] = [r.data]
        for r in node.outputs:
            storage_map.setdefault(r, [None])
    for r in env.outputs:
        if isinstance(r, graph.Constant):
            storage_map.setdefault(r, [r.data])

    if output_storage is None:
        output_storage = [storage_map[r] for r in env.outputs]

    return input_storage, output_storage, storage_map

def streamline(env, thunks, order, post_thunk_old_storage = None, no_recycling = [], profiler = None, nice_errors = True):
    """WRITEME

    :param env:

    :param thunks: the list of program instructions

    :param order: the list of apply instances that gave rise to the thunks (same order as thunks)

    :param post_thunk_old_storage: a list (corresponding to thunks, order) whose elements are
    lists of storage cells, that should be cleared after running the corresponding thunk.  A
    value of None disables this functionality

    :param no_recycling: storage elements that cannot be 'recycled' by repeatedly executing the
    program.  These storage elements are cleared before re-running.

    :param profiler: deprecated

    :param nice_errors: run in such a way that the double-traceback is printed.  This costs a
    bit of performance in the inner python loop.
    """
    if profiler is not None:
        raise NotImplementedError()

    if len(thunks) != len(order):
        raise ValueError('Length of thunks and order must match',
                (len(thunks), len(order)))

    if post_thunk_old_storage:
        if len(thunks) != len(post_thunk_old_storage):
            raise ValueError('Length of thunks and post_thunk_old_storage must match',
                    (len(thunks), len(post_thunk_old_storage)))

        def streamline_default_f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node, old_storage in zip(thunks, order, post_thunk_old_storage):
                    thunk()
                    for old_s in old_storage:
                        old_s[0] = None
            except Exception:
                raise_with_op(node)
        f = streamline_default_f
    elif nice_errors:
        thunk_node_list = zip(thunks, order)
        def streamline_nice_errors_f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node in thunk_node_list:
                    thunk()
            except Exception:
                raise_with_op(node)
        f = streamline_nice_errors_f
    else:
        # don't worry about raise_with_op, just go a little faster.
        #there is a mix of python and c thunks
        def streamline_fast_f():
            for x in no_recycling:
                x[0] = None
            for thunk in thunks:
                thunk()
        f = streamline_fast_f
    return f

class LocalLinker(Linker):
    """WRITEME
    Useful base class for L{Linker}s which keep all nodes in the graph, and run a
    thunk associated with each node.
    """

    def make_thunk(self, profiler = None, input_storage = None, output_storage = None):
        return self.make_all(profiler = profiler,
                             input_storage = input_storage,
                             output_storage = output_storage)[:3]

    def make_all(self, profiler, input_storage, output_storage):
        # By convention, subclasses of LocalLinker should implement this function!
        #
        # This function should return a tuple of 5 things
        # 1. function to run the program
        # 2. input storage
        # 3. output storage
        # 4. thunks: list of nodes' functions in the order they will be run by the function in (1)
        # 5. order: list of nodes, in the order they will be run by the function in (1)
        raise utils.MethodNotDefined("make_all", type(self), self.__class__.__name__)

def gc_helper(node_list):
    """
    :param node_list: list of Apply instances in program execution order

    :rtype: a 2-tuple
    :returns: FIRST, the set of Variable instances which are computed by node_list, and SECOND a
    dictionary that maps each Variable instance to a the last node to use Variable as an input.

    This is used to allow garbage collection within graphs.
    """
    #for freeing memory
    last_user = {}
    computed = set()
    for node in node_list:
        for input in node.inputs:
            last_user[input] = node
        for output in node.outputs:
            computed.add(output)
    return computed, last_user

class PerformLinker(LocalLinker):
    """WRITEME

    Basic L{Linker} subclass that calls the perform method on each L{Op} in
    the L{Env} in the order given by L{Env.toposort}.
    """

    def __init__(self, allow_gc=True):
        #TODO: set allow_gc = True by default, when it works with the OpWiseCLinker
        self.env = None
        self.allow_gc = allow_gc

    def accept(self, env, no_recycling = []):
        """
        :param env: a PerformLinker can have accepted one Env instance at a time.

        :param no_recycling: WRITEME

        :returns: self (TODO: WHY? Who calls this function?)
        """
        if self.env is not None and self.env is not env:
            return type(self)().accept(env, no_recycling)
            #raise Exception("Cannot accept from a Linker that is already tied to another Env.")
        self.env = env
        self.no_recycling = no_recycling
        return self

    def make_all(self, profiler = None, input_storage = None, output_storage = None):
        """
        :param profiler: WRITEME
        :param input_storage: WRITEME
        :param output_storage: WRITEME

        :returns: function to run all nodes, list of input containers, list of output containers, list of thunks (for all of program), list of nodes (for all of program)

        """
        env = self.env
        order = list(env.toposort())
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = map_storage(env, order, input_storage, output_storage)


        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = []
        for node in order:
            # Maker sure we don't use C version of the code, but rather only
            # the python version
            # Note : ops that implement their own make thunk don't usually
            # have this attribute defiend !!
            old_value = getattr(node.op, '_op_use_c_code', False)
            try:
                node.op._op_use_c_code = False
                thunks += [node.op.make_thunk(node,
                                    storage_map,
                                    compute_map,
                                    no_recycling)]
            finally:
                node.op._op_use_c_code = old_value

        computed, last_user = gc_helper(order)
        if self.allow_gc:
            post_thunk_old_storage = []
        else:
            post_thunk_old_storage = None

        for node in order:
            if self.allow_gc:
                post_thunk_old_storage.append([storage_map[input]
                    for input in node.inputs
                    if (input in computed) and (input not in env.outputs) and node == last_user[input]])

        if no_recycling is True:
            # True seems like some special code for *everything*?? -JB
            # FunctionMaker always passes a list I think   -JB
            no_recycling = storage_map.values()
            no_recycling = utils.difference(no_recycling, input_storage)
        else:
            no_recycling = [storage_map[r] for r in no_recycling if r not in env.inputs]

        # The function that actually runs your program is one of the f's in streamline.
        f = streamline(env, thunks, order, post_thunk_old_storage, no_recycling = no_recycling, profiler = profiler)

        f.allow_gc = self.allow_gc #HACK: this is a way of passing an arg to Function.__call__
        add_clear_storage(f, computed, storage_map)

        return f, [Container(input, storage) for input, storage in zip(env.inputs, input_storage)], \
            [Container(output, storage, True) for output, storage in zip(env.outputs, output_storage)], \
            thunks, order

def add_clear_storage(f, computed, storage_map):
    def clear_storage():
        for c in computed:
            storage_map[c][0] = None
    f.clear_storage = clear_storage


class WrapLinker(Linker):
    """ WRITEME
    This class makes it easier to run several L{LocalLinker}s in parallel, and
    offers some control over how each thunk is run.

    A wrapper function must be provided, and it can be used to execute the
    thunks, inspect the nodes, print stuff out, etc.

    @note:
    The outputs of the first linker will be returned.

    @note:
    This linker ensures that each linker has its own storage for
    inputs and outputs and intermediate variables.  There is no interference
    between linkers.

    """

    def __init__(self, linkers, wrapper):
        """
        Initialize a WrapLinker.

        @type linkers: list of L{LocalLinker} subclasses, whose make_all()
        method returns thunks in the same order.

        @param linkers: for each node in the graph, each linker will provide a
        thunk.  This class makes it possible to iterate over each linker's
        program in parallel.

        @type wrapper: lambda (i, i_node, i_thunk1, i_thunk2, ...) : None

        @param wrapper: do some user-defined action for the i'th element of the
        program.  i_thunk<n> is the thunk returned by the n'th linker.  (If you
        want to run the program, make sure to call the necessary thunks in this
        function.)

        """
        self.env = None
        self.linkers = linkers
        self.wrapper = wrapper

    def accept(self, env, no_recycling = []):
        """
        @type env: gof.Env
        @param env: the env which we will link

        @type no_recycling: a list of Variables that belong to env.

        @param no_recycling: If a Variable is in no_recycling, L{WrapLinker} will clear
        the output storage associated to it (for each linker in linkers) during
        the computation to avoid reusing it.

        """
        if self.env is not None and self.env is not env:
            return type(self)(self.linkers, self.wrapper).accept(env, no_recycling)

        self.env = env
        self.no_recycling = no_recycling
        self.linkers = [linker.accept(env, no_recycling) for linker in self.linkers]
        return self

    def pre(self, f, inputs, order, thunk_groups):
        pass

    def make_thunk(self, **kwargs):
        no_recycling = self.no_recycling

        make_all = [self.linkers[0].make_all(**kwargs)]
        kwargs.pop('input_storage', None)
        make_all += [l.make_all(**kwargs) for l in self.linkers[1:]]

        fns, input_lists, output_lists, thunk_lists, order_lists \
                = zip(*make_all)

        order_list0 = order_lists[0]
        for order_list in order_lists[1:]:
            if not order_list0 == order_list:
                raise Exception("All linkers to WrapLinker should execute operations in the same order.")

        inputs0 = input_lists[0]
        outputs0 = output_lists[0]

        thunk_groups = zip(*thunk_lists)
        order = [x[0] for x in zip(*order_lists)]

        to_reset = []
        for thunks, node in zip(thunk_groups, order):
            for j, output in enumerate(node.outputs):
                if output in no_recycling:
                    for thunk in thunks:
                        to_reset.append(thunk.outputs[j])

        wrapper = self.wrapper
        pre = self.pre
        def f():
            for inputs in input_lists[1:]:
                for input1, input2 in zip(inputs0, inputs):
                    input2.storage[0] = copy(input1.storage[0])
            for x in to_reset:
                x[0] = None
            pre(self, [input.data for input in input_lists[0]], order, thunk_groups)
            for i, (thunks, node) in enumerate(zip(thunk_groups, order)):
                try:
                    wrapper(i, node, *thunks)
                except Exception:
                    raise_with_op(node)
        f.thunk_groups = thunk_groups

        return f, inputs0, outputs0

def WrapLinkerMany(linkers, wrappers):
    """ Variant on WrapLinker that runs a series of wrapper functions instead of
    just one.
    """
    def wrapper(*args):
        for f in wrappers:
            f(*args)
    return WrapLinker(linkers, wrapper)
