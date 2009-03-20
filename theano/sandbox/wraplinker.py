from __future__ import absolute_import
import time
import numpy

from ..gof.cutils import run_cthunk
from ..gof.link import WrapLinker
from ..compile.mode import Mode

class Todo(Exception): """todo"""

#WrapLinker wrappers
if 0:
    from ..gradient import numeric_grad
    def cmp_outputs(i, node, *thunks):
        """WrapLinker wrapper: raise an exception if outputs are different

        numpy.ndarrays of floating point types are compared approximately, rather
        than exactly.

        """
        class MisMatch(Exception): """Output mismatch"""

        #define a comparison function, which works for all the variables in a graph
        #TODO: consider factoring this out (and maybe passing args explicitly
        # instead of by closure)
        def my_check_equal(x, y):
            if type(x) != type(y):
                raise MisMatch("Output type mismatch", (x, y))

            if hasattr(x, 'dtype'):
                # was: isinstance(x,numpy.ndarray), which doesn't
                # catch numpy.float64
                if x.dtype != y.dtype or x.shape != y.shape:
                    raise MisMatch("ndarray type/shape.", (x,y))

                if str(x.dtype).startswith('float'):
                    assert str(x.dtype) == 'float64' #otherwise we need to adjust
                    #our constant below... but to what?
                    abs_rel_err = numeric_grad.abs_rel_err(x, y)
                    max_abs_rel_err = numpy.max(abs_rel_err)
                    if max_abs_rel_err > 1.0e-7:
                        raise MisMatch('max_abs_rel_err exceeds tolerence', (max_abs_rel_err,
                            x, y))

                elif str(x.dtype).startswith('complex'):
                    raise Todo()

                else:
                    if not numpy.all(x==y):
                        raise MisMatch

            else:
                print 'wtf??', type(x), type(y), node.op
                if x != y:
                    print 'wow!! wtf??'
                    raise MisMatch("Output mismatch.", (x, y))

        #loop over all the thunks
        # ensure that the outputs from the first thunk match the outputs from
        # all subsequent thunks
        n_thunks = len(thunks)
        if n_thunks > 1:
            th0 = thunks[0]
            for th in thunks[1:]:
                for out0, outN in zip(th0.outputs, th.outputs):
                    my_check_equal(out0[0], outN[0])

#TODO: better name for 'f'
def numpy_wrapper(f):
    def wrapper(i, node, *thunks):
        """WrapLinker wrapper: raise an exception if a NaN is found in outputs
        """
        for thunk in thunks:
            for output in thunk.outputs:
                if hasattr(output[0], 'dtype'):
                    if f(output[0]):
                        raise Exception('uh oh', (i, node, thunk, output[0]))
    return wrapper

numpy_any_isinf = numpy_wrapper(lambda a:numpy.any(numpy.isinf(a)))
numpy_any_isnan = numpy_wrapper(lambda a:numpy.any(numpy.isnan(a)))
numpy_notall_isfinite = numpy_wrapper(lambda a: not numpy.all(numpy.isfinite(a)))

def run_all(i, node, *thunks):
    """WrapLinker wrapper: run the thunks
    """
    for th in thunks:
        th()




def DualLinker(linkers):
    #still in sandbox pending ticket 247
    # when value_cmp is implemented, then cmp_outputs can be rewritten in a solid way, and the
    # DualLinker can be this simple.
    return WrapLinkerMany(linkers, [run_all, cmp_outputs])



####
#
#  The Stats and Profiler classes used to be in gof/link.
#  But Stats was not used I think, and Profiler has been implemented using the wraplinker.
#
#  -JB20090119
###
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


