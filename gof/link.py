
from utils import AbstractFunctionError
import utils

import sys
import traceback


__excepthook = sys.excepthook
def thunk_hook(type, value, trace):
    """
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
    if exc_info is None:
        exc_info = sys.exc_info()
    exc_type, exc_value, exc_trace = exc_info
    try:
        trace = op.trace
    except AttributeError:
        trace = ()
    exc_value.__thunk_trace__ = trace
    exc_value.args = exc_value.args + (op, )
    raise exc_type, exc_value, exc_trace


class Linker:

    def make_thunk(self, inplace = False):
        """
        This function must return a triplet (function, input_results, output_results)
        where function is a thunk that operates on the returned results. If inplace
        is True, the input_results and output_results lists will be the same as the
        inputs and outputs of the graph provided to the L{Linker}. Else, independent
        results will be returned.

        Example::
         e = x + y
         env = Env([x, y], [e])
         fn, (new_x, new_y), (new_e, ) = MyLinker(env).make_thunk(inplace)
         new_x.data = 1.0
         new_y.data = 2.0
         fn()
         print new_e.data # 3.0
         print e.data # 3.0 iff inplace == True (else unknown)
        """
        raise AbstractFunctionError()

    def make_function(self, inplace = False, unpack_single = True, **kwargs):
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
        thunk, inputs, outputs = self.make_thunk(inplace, **kwargs)

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




class PerformLinker(Linker):
    """
    Basic L{Linker} subclass that calls the perform method on each L{Op} in
    the L{Env} in the order given by L{Env.toposort}.
    """

    def __init__(self, env):
        self.env = env

    def make_thunk(self, inplace = False, profiler = None):
        if inplace:
            env = self.env
        else:
            env = self.env.clone(True)
        order = env.toposort()
        thunks = [op.perform for op in order]
        if profiler is None:
            def f():
                try:
                    for thunk, op in zip(thunks, order):
                        thunk()
                except:
                    raise_with_op(op)
        else:
            def f():
                def g():
                    for thunk, op in zip(thunks, order):
                        profiler.profile_op(thunk, op)
                profiler.profile_env(g, env)
            f.profiler = profiler
            
                
        return f, env.inputs, env.outputs



import time

class Stats:
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
    """
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


