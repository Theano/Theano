import time

from ..gof.link import WrapLinkerMany
from ..gof.cutils import run_cthunk
from ..compile.mode import Mode, predefined_linkers
from ..gof.cc import OpWiseCLinker

class ProfileMode(Mode):
    def __init__(self, linker=OpWiseCLinker(), optimizer=None):
        local_time = [0.0]
        apply_time = {}
        op_time = {}
        op_cimpl = {}

        def blah(i, node, th):
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                run_cthunk(th.cthunk)
                dt = time.time() - t0
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0

            local_time[0] += dt
            apply_time[(i,node.op)] = apply_time.get((i,node.op), 0.0) + dt
            op_time[node.op] = op_time.get(node.op, 0.0) + dt
            op_cimpl[node.op] = hasattr(th, 'cthunk')

        self.local_time = local_time
        self.apply_time = apply_time
        self.op_time = op_time
        self.op_cimpl = op_cimpl

        if isinstance(linker, str):
            linker = predefined_linkers[linker]

        wrap_linker = WrapLinkerMany([linker], [blah])
        if optimizer:
            super(ProfileMode, self).__init__(wrap_linker, optimizer)
        else:
            super(ProfileMode, self).__init__(wrap_linker)

    def print_summary(self, n_apply_to_print=15, n_ops_to_print=20):
        """ Print 3 summary that show where the time is spend. The first show an Apply-wise summary, the second show an Op-wise summary, the third show an type-Op-wise summary.

        The Apply-wise summary print the timing information for the worst offending Apply nodes. This corresponds to individual Op applications within your graph which take the longest to execute (so if you use dot twice, you will see two entries there). 
        The Op-wise summary print the execution time of all Apply nodes executing the same Op are grouped together and the total execution time per Op is shown (so if you use dot twice, you will see only one entry there corresponding to the sum of the time spent in each of them). If two Op have different hash value, they will be separate.
        The type-Op-wise summary group the result by type of op. So event if two Op have different hash value, they will be merged.

        param: n_apply_to_print the number of apply to print. Default 15.

        param: n_ops_to_print the number of ops to print. Default 20.
        """
        local_time = self.local_time[0]
        apply_time = self.apply_time
        op_time = self.op_time

        print ''
        print 'ProfileMode.print_summary()'
        print '---------------------------'
        print ''
        print 'local_time %fs (Time spent running thunks)'% local_time
        print 'Apply-wise summary: <% of local_time spent at this position> <total of local_time spent at this position> (<Apply position>, <Apply Op name>)'
        atimes = [(t/local_time, t, (a[0], str(a[1]))) for a, t in apply_time.items()]
        atimes.sort()
        atimes.reverse()
        tot=0
        for f,t,a in atimes[:n_apply_to_print]:
            tot+=t
            print '   %.2f%%  %.3fs  %.3fs  %i  %s' % (f*100, tot, t, a[0], a[1])
        print '   ... (remaining %i Apply instances account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(atimes)-n_apply_to_print),
                  sum(f for f, t, a in atimes[n_apply_to_print:])*100,
                  sum(t for f, t, a in atimes[n_apply_to_print:]))


        print '\nOp-wise summary: <% of local_time spent on this kind of Op> <cumulative seconds> <self seconds> <Op name>'
        otimes = [(t/local_time, t, a, self.op_cimpl[a]) for a, t in op_time.items()]
        otimes.sort()
        otimes.reverse()
        tot=0
        for f,t,a,ci in otimes[:n_ops_to_print]:
            tot+=t
            print '   %.2f%%  %.3fs  %.3fs  %s %s' % (f*100, tot, t, '*' if ci else ' ', a)
        print '   ... (remaining %i Ops account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(otimes)-n_ops_to_print),
                  sum(f for f, t, a, ci in otimes[n_ops_to_print:])*100,
                  sum(t for f, t, a, ci in otimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'


        sop_time={}
        sop_c={} #map each op class to Bool. True iff all applies were done in c.
        for a,t in op_time.items():
            sop_time.setdefault(type(a),0)
            sop_time[type(a)]+=t
            sop_c.setdefault(type(a),True)
            sop_c[type(a)]=sop_c[type(a)] and self.op_cimpl[a]
        print '\nSingle Op-wise summary: <% of local_time spent on this kind of Op> <cumulative seconds> <self seconds> <Op name>'
        sotimes = [(t/local_time, t, a, sop_c[a]) for a, t in sop_time.items()]
        sotimes.sort()
        sotimes.reverse()
        tot=0
        for f,t,a,ci in sotimes[:n_ops_to_print]:
            tot+=t
            print '   %.2f%%  %.3fs  %.3fs  %s %s' % (f*100, tot, t, '*' if ci else ' ', a)
        print '   ... (remaining %i Ops account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(sotimes)-n_ops_to_print),
                  sum(f for f, t, a in sotimes[n_ops_to_print:])*100,
                  sum(t for f, t, a in sotimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'
