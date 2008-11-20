"""WRITEME"""
import utils
import graph
from type import Type

import sys, traceback
from copy import copy
from cutils import run_cthunk


__excepthook = sys.excepthook
def thunk_hook(type, value, trace):
    """WRITEME
    This function is meant to replace excepthook and do some
    special work if the exception value has a __thunk_trace__
    field. In that case, it retrieves the field, which should
    contain a trace as returned by L{traceback.extract_stack},
    and prints it out on L{stderr}.

    The normal excepthook is then called.
    """
    if hasattr(value, '__thunk_trace__'):
        trace2 = value.__thunk_trace__
        if trace2 is None:
            print>>sys.stderr, "Could not find where this Op was defined."
            print>>sys.stderr, " * You might have instantiated this Op directly instead of using a constructor."
            print>>sys.stderr, " * The Op you constructed might have been optimized. Try turning off optimizations."
        elif trace2:
            print>>sys.stderr, "Definition in: "
            for line in traceback.format_list(trace2):
                print>>sys.stderr, line,
    __excepthook(type, value, trace)
sys.excepthook = thunk_hook


def raise_with_op(op, exc_info = None):
    """WRITEME"""
    if exc_info is None:
        exc_info = sys.exc_info()
    exc_type, exc_value, exc_trace = exc_info
    try:
        trace = op.tag.trace
    except AttributeError:
        trace = ()
    exc_value.__thunk_trace__ = trace
    exc_value.args = exc_value.args + (op, )
    raise exc_type, exc_value, exc_trace


class Linker(object):
    """WRITEME"""

    def make_thunk(self):
        """
        This function must return a triplet (function, input_results, output_results)
        where function is a thunk that operates on the returned results. If inplace
        is True, the input_results and output_results lists will be the same as the
        inputs and outputs of the graph provided to the L{Linker}. Else, independent
        results will be returned.

        Example::
         x, y = Result(Double), Result(Double)
         e = x + y
         env = Env([x, y], [e])
         fn, (new_x, new_y), (new_e, ) = MyLinker(env).make_thunk(inplace)
         new_x.data = 1.0
         new_y.data = 2.0
         fn()
         print new_e.data # 3.0
         print e.data # 3.0 iff inplace == True (else unknown)
        """
        raise utils.AbstractFunctionError()

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
            for arg, result in zip(args, inputs):
                result.data = arg
            thunk()
            if unpack_single:
                return utils.to_return_values([result.data for result in outputs])
            else:
                return [result.data for result in outputs]
        execute.thunk = thunk
        execute.inputs = inputs
        execute.outputs = outputs
        
        return execute


class Container(object):
    """WRITEME
    """
    def __init__(self, r, storage, readonly = False, strict = False, name = None):
        """WRITEME

        :Parameters:
         `r`: 
         `storage`: 
         `readonly`: 
         `strict`: 
         `name`: 

        """
        if not isinstance(storage, list) or not len(storage) >= 1:
            raise TypeError("storage must be a list of length at least one")
        #self.r = r
        if isinstance(r, Type):
            self.type = r
        else:
            self.type = r.type
        self.name = r.name if name is None else name
        self.storage = storage
        self.readonly = readonly
        self.strict = strict
    def __get(self):
        return self.storage[0]
    def __set(self, value):
        if self.readonly:
            raise Exception("Cannot set readonly storage: %s" % self.name)
        try:
            if value is None:
                self.storage[0] = None
                return
            if self.strict:
                self.storage[0] = self.type.filter(value, strict = True)
            else:
                self.storage[0] = self.type.filter(value)
        except Exception, e:
            e.args = e.args + (('Container name "%s"' % self.name),)
            raise
    data = property(__get, __set)
    value = property(__get, __set)
    def __str__(self):
        return "<" + str(self.storage[0]) + ">"
    def __repr__(self):
        return "<" + repr(self.storage[0]) + ">"


def map_storage(env, order, input_storage, output_storage):
    """Ensure there is storage for inputs, outputs, and interior nodes.

    :param env: The current env.  This function uses the inputs and outputs attributes.
    :param order: an iterable over Apply instances (in program running order)
    :param input_storage: None or existing input storage (see below)
    :param output_storage: None or existing output storage (see below)

    :rtype: 3-tuple
    :returns: (list of storage for inputs, list of storage for outputs, and the `storage_map`)
    

    This function iterates over the nodes in `order` and ensures that for every
    input and output `Result`, there is a unique storage container.  This is
    returned as a dictionary Result->storage called the `storage_map`.

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

def clear_storage_thunk(stg):
    """This is useful for inserting thunks that zero-out storage, which allows memory to be freed by gc."""
    def thunk():
        stg[0] = None
    thunk.outputs = []
    thunk.inputs = [stg]
    return thunk

def streamline(env, thunks, order, no_recycling = [], profiler = None, nice_errors = True):
    """WRITEME

    :param env:

    :param thunks: the list of program instructions

    :param order: the list of apply instances that gave rise to the thunks (same order as thunks)

    :param no_recycling: storage elements that cannot be 'recycled' by repeatedly executing the
    program.  These storage elements are cleared before re-running.
    
    :param profiler: deprecated

    :param nice_errors: run in such a way that the double-traceback is printed.  This costs a
    bit of performance in the inner python loop.
    """
    if profiler is not None: 
        raise NotImplementedError()

    if nice_errors:
        def f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node in zip(thunks, order):
                    thunk()
            except:
                raise_with_op(node)
    else:
        # don't worry about raise_with_op, just go a little faster.
        #there is a mix of python and c thunks
        def f():
            for x in no_recycling:
                x[0] = None
            for thunk in thunks:
                thunk()
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
        raise AbstractFunctionError


class PerformLinker(LocalLinker):
    """WRITEME

    Basic L{Linker} subclass that calls the perform method on each L{Op} in
    the L{Env} in the order given by L{Env.toposort}.
    """

    def __init__(self, allow_gc=False):
        #TODO: set allow_gc = True by default, when it works with the c&py linker
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
        order = env.toposort()
        no_recycling = self.no_recycling

        thunks = []
        new_order = []

        input_storage, output_storage, storage_map = map_storage(env, order, input_storage, output_storage)

        #for freeing memory
        if self.allow_gc:
            last_user = {}
            computed = set()
            for node in order:
                for idx, input in enumerate(node.inputs):
                    last_user[input] = (node, idx)
                for output in node.outputs:
                    computed.add(output)
        
        for node in order:
            node_input_storage = tuple(storage_map[input] for input in node.inputs)
            node_output_storage = tuple(storage_map[output] for output in node.outputs)
            p = node.op.perform
            # Thunk is meant to be called without arguments.
            # The arguments are given in the lambda expression so that they are saved in the lambda expression.
            # Using the closure in a simple way didn't work.
            thunk = lambda p = p, i = node_input_storage, o = node_output_storage, n = node: p(n, [x[0] for x in i], o)
            thunk.inputs = node_input_storage
            thunk.outputs = node_output_storage
            thunk.perform = p
            thunks.append(thunk)
            new_order.append(node)

            if self.allow_gc:
                for idx, input in enumerate(node.inputs):
                    if input not in computed:
                        continue
                    if input in env.outputs:
                        continue
                    if (node, idx) == last_user[input]:
                        #print '... zeroing', id(storage_map[input])
                        thunks.append(clear_storage_thunk(storage_map[input]))
                        new_order.append(node)



        if no_recycling is True: 
            #True is like some special code for *everything*.
            #FunctionMaker always passes a list I think   -JB
            no_recycling = storage_map.values()
            no_recycling = utils.difference(no_recycling, input_storage)
        else:
            no_recycling = [storage_map[r] for r in no_recycling if r not in env.inputs]

        # The function that actually runs your program is one of the f's in streamline.
        f = streamline(env, thunks, new_order, no_recycling = no_recycling, profiler = profiler)
  
        return f, [Container(input, storage) for input, storage in zip(env.inputs, input_storage)], \
            [Container(output, storage, True) for output, storage in zip(env.outputs, output_storage)], \
            thunks, new_order



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
    inputs and outputs and intermediate results.  There is no interference
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

        @type no_recycling: a list of Results that belong to env.  

        @param no_recycling: If a Result is in no_recycling, L{WrapLinker} will clear
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
                except:
                    raise_with_op(node)

        return f, inputs0, outputs0



import time

class Stats:
    """WRITEME"""
    def __init__(self):
        self.ncalls = 0
        self.time = 0
        self.nfailures = 0
        self.time_failures = 0
    def inc_ncalls(self, v): self.ncalls += v
    def inc_time(self, v): self.time += v
    def inc_nfailures(self, v): self.nfailures += v
    def inc_time_failures(self, v): self.time_failures += v

class Profiler:
    """WRITEME
    Collects performance statistics on a function on a per-L{Op}
    or per-L{Op}-class basis.
    """
    
    def __init__(self, ignore = [], by_class = True):
        """
        Creates a L{Profiler}. If by_class is True, stats will
        be collected for each L{Op} class, adding the totals for
        each occurrence of that L{Op} in the computation. If
        by_class is False, each node will be timed individually.

        All L{Op} classes or L{Op}s (depending on the value of by_class)
        listed in ignore will not be timed.
        """
        self.ignore = ignore
        self.stats = {}
        self.by_class = by_class

    def profile_env(self, f, env):
        """WRITEME"""
        stats = self.stats.setdefault('TOTAL', Stats())
        n, t = stats.inc_ncalls, stats.inc_time
        failed = False
        
        start = time.time()
        try:
            f()
            end = time.time()
        except:
            end = time.time()
            n, t = stats.inc_nfailures, stats.inc_times_failures
            failed = True
            ety, eva, etr = sys.exc_info()
        n(1)
        t(end - start)
        if failed:
            raise ety, eva, etr

    def profile_op(self, f, op):
        """WRITEME"""
        if self.by_class:
            entry = op.__class__
        else:
            entry = op
        stats = self.stats.setdefault(entry, Stats())
        n, t = stats.inc_ncalls, stats.inc_time
        failed = False
        
        start = time.time()
        try:
            f()
            end = time.time()
        except:
            end = time.time()
            n, t = stats.inc_nfailures, stats.inc_times_failures
            failed = True
            exc = sys.exc_info()

        if entry not in self.ignore:
            n(1)
            t(end - start)
        if failed:
            raise_with_op(op, exc)


    def print_stats(self, sort_by = 'time'):
        """WRITEME"""
        
        def compare_fn((op1, stat1), (op2, stat2)):
            x1 = getattr(stat2, sort_by)
            x2 = getattr(stat1, sort_by)
            if x1 > x2:
                return 1
            elif x1 < x2:
                return -1
            else:
                return 0

        totals = self.stats['TOTAL']

        print 'CPU usage statistics' 
        print "  %-25s %9s %12s %12s %12s" % (("Op%s" % (self.by_class and ' class' or '')), 'NCALLS', 'PER_CALL', 'TOTAL', 'CPU%')

        for op, stat in sorted(self.stats.items(), compare_fn):
            if op == 'TOTAL': continue
            to_print = self.by_class and (op.__module__ + "." + op.__name__) or str(op)
            print "  %-25s %9i %12.5f %12.5f %12.5f" % (to_print, stat.ncalls, stat.time / stat.ncalls, stat.time, stat.time / totals.time)

        stat = self.stats['TOTAL']
        print "  %-25s %9i %12.5f %12.5f %12.5f" % ('TOTAL (includes overhead)', stat.ncalls, stat.time / stat.ncalls, stat.time, stat.time / totals.time)


