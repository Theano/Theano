import time

from ..gof.link import WrapLinkerMany
from ..gof.cutils import run_cthunk
from ..compile.mode import Mode

class ProfileMode(Mode):
    def __init__(self, linker, optimizer=None):
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

        wrap_linker = WrapLinkerMany([linker], [blah])
        if optimizer:
            super(ProfileMode, self).__init__(wrap_linker, optimizer)
        else:
            super(ProfileMode, self).__init__(wrap_linker)

    def print_summary(self):
        local_time = self.local_time[0]
        apply_time = self.apply_time
        op_time = self.op_time

        print ''
        print 'ProfileMode.print_summary()'
        print '---------------------------'
        print ''
        print 'local_time', local_time, '(Time spent running thunks)'
        print 'Apply-wise summary: <fraction of local_time spent at this position> (<Apply position>, <Apply Op name>)'
        atimes = [(t/local_time, (a[0], str(a[1]))) for a, t in apply_time.items()]
        atimes.sort()
        atimes.reverse()
        for t,a in atimes[:15]:
            print '\t%.3f\t%i\t%s' % (t, a[0], a[1])
        print '   ... (remaining %i Apply instances account for %.2f of the runtime)'\
                %(max(0, len(atimes)-15), sum(t for t, a in atimes[15:]))


        n_ops_to_print = 20
        print 'Op-wise summary: <fraction of local_time spent on this kind of Op> <Op name>'
        otimes = [(t/local_time, a, self.op_cimpl[a]) for a, t in op_time.items()]
        otimes.sort()
        otimes.reverse()
        for t,a,ci in otimes[:n_ops_to_print]:
            print '\t%.3f\t%s %s' % (t, '*' if ci else ' ', a)
        print '   ... (remaining %i Ops account for %.2f of the runtime)'\
                %(max(0, len(otimes)-n_ops_to_print), sum(t for t, a, ci in
                    otimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'
