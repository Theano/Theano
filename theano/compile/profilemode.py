import time, atexit, copy

from theano.gof.link import WrapLinkerMany
from theano.gof.cutils import run_cthunk
from theano.compile.mode import Mode, predefined_linkers, register_mode, predefined_modes
from theano.gof.cc import OpWiseCLinker

class ProfileMode(Mode):
    def __init__(self, linker=OpWiseCLinker(), optimizer=None):
        local_time = [0.0]
        apply_time = {}
        apply_call = {}
        op_time = {}
        op_cimpl = {}
        op_call = {}

        def blah(i, node, th):
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                failure = run_cthunk(th.cthunk)
                dt = time.time() - t0
                if failure:
                    raise RuntimeError('A C Op raised an exception.  PerformLinker cannot tell you what it was though.  Use a standard mode such as FAST_RUN to correct the problem.')
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0

            local_time[0] += dt
            apply_time[(i,node.op, tuple(node.inputs))] = apply_time.get((i,node.op, tuple(node.inputs)), 0.0) + dt
            apply_call[(i,node.op, tuple(node.inputs))] = apply_call.get((i,node.op, tuple(node.inputs)), 0) + 1
            op_time[node.op] = op_time.get(node.op, 0.0) + dt
            op_cimpl[node.op] = hasattr(th, 'cthunk')
            op_call[node.op] = op_call.get(node.op,0) + 1

        self.local_time = local_time
        self.apply_time = apply_time
        self.apply_call = apply_call
        self.op_time = op_time
        self.op_cimpl = op_cimpl
        self.op_call = op_call
        self.compile_time = 0 #time passed in function()

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

        Their is an hack with the Op-wise summary. Go see it if you want to know more.

        param: n_apply_to_print the number of apply to print. Default 15.

        param: n_ops_to_print the number of ops to print. Default 20.
        """
        local_time = self.local_time[0]
        compile_time = self.compile_time
        apply_time = self.apply_time
        apply_call = self.apply_call
        op_time = self.op_time
        op_call = self.op_call
        op_cimpl = self.op_cimpl

        self.print_summary_("print_summary",local_time, compile_time, apply_time, apply_call, op_time, op_call, op_cimpl, n_apply_to_print, n_ops_to_print)


    def print_diff_summary(self, other, n_apply_to_print=15, n_ops_to_print=20):
        """ As print_summary, but print the absolute difference on two different profile mode.
        TODO: Also we don't print the Apply-wise summary as it don't work for now.
        TODO: make flops the difference of flops
        TODO: make comparaison with gpu code.
        
        param: other the other instance of ProfileMode that we want to be compared to.
        
        param: n_apply_to_print the number of apply to print. Default 15.

        param: n_ops_to_print the number of ops to print. Default 20.
        """

        def diff_dict(a,b_):
            r = {}
            b = copy.copy(b_)
            for a,t in a.items():
                r.setdefault(a,0)
                t2 = b.pop(a,0)
                #print t,t2,abs(t-t2),a
                r[a]+=abs(t-t2)
                
            #they are missing in a
            print "missing items",len(b)
            for a,t in b.items():
                r.setdefault(a,0)
                r[a]+=t
            return r
        
        local_time = abs(self.local_time[0]-other.local_time[0])
        compile_time = abs(self.compile_time-other.compile_time)
        apply_time = diff_dict(self.apply_time, other.apply_time)
        apply_call = diff_dict(self.apply_call, other.apply_call)
        op_time = diff_dict(self.op_time, other.op_time)
        op_call = diff_dict(self.op_call, other.op_call)
        op_cimpl = self.op_cimpl and other.op_cimpl
       
        self.print_summary_("print_diff_summary",local_time, compile_time, apply_time, apply_call, op_time, op_call, op_cimpl, n_apply_to_print, n_ops_to_print, print_apply=False)

    @staticmethod
    def print_summary_(fct_name, local_time, compile_time, apply_time, apply_call, op_time, op_call, op_cimpl,
                       n_apply_to_print=15, n_ops_to_print=20, print_apply=True):
        """
        do the actual printing of print_summary and print_diff_summary.

        param: n_apply_to_print the number of apply to print. Default 15.

        param: n_ops_to_print the number of ops to print. Default 20.
        """

        print ''
        print 'ProfileMode.%s()'%(fct_name)
        print '---------------------------'
        print ''
        
        print 'local_time %fs (Time spent running thunks)'% local_time

        if print_apply:
            print 'Apply-wise summary: <% of local_time spent at this position> <total of local_time spent at this position> <nb_call> <Apply position> <Apply Op name>'
            atimes = [(t/local_time, t, (a[0], str(a[1])), apply_call[a]) for a, t in apply_time.items()]
            atimes.sort()
            atimes.reverse()
            tot=0
            for f,t,a,nb_call in atimes[:n_apply_to_print]:
                tot+=t
                print '   %4.1f%%  %.3fs  %.3fs  %i  %i %s' % (f*100, tot, t, nb_call, a[0], a[1])
            print '   ... (remaining %i Apply instances account for %.2f%%(%.2fs) of the runtime)'\
                    %(max(0, len(atimes)-n_apply_to_print),
                      sum(f for f, t, a, nb_call in atimes[n_apply_to_print:])*100,
                      sum(t for f, t, a, nb_call in atimes[n_apply_to_print:]))

        flops=False
        flops_msg=''
        for a,t in op_time.items():
            if hasattr(a,'flops'):
                flops=True
                flops_msg=' <MFlops/s>'
                print '\nHACK WARNING: we print the flops for some OP, but the logic don\' always work. You need to know the internal of Theano to make it work correctly. Otherwise don\'t use!'
                break
            
        print '\nOp-wise summary: < of local_time spent on this kind of Op> <cumulative seconds> <self seconds>%s <nb_call> <Op name>'%(flops_msg)

        otimes = [(t/local_time, t, a, op_cimpl[a], op_call[a]) for a, t in op_time.items()]
        otimes.sort()
        otimes.reverse()
        tot=0
        for f,t,a,ci,nb_call in otimes[:n_ops_to_print]:
            tot+=t
            if ci:
              msg = '*'
            else:
              msg = ' '
            m=-1
            if hasattr(a,'flops'):
                m=a.flops*op_call[a]/t/1e6
            if flops:
                print '   %4.1f%%  %.3fs  %.3fs  %s %7.1f %d %s' % (f*100, tot, t, msg, m, nb_call, a)
            else:
                print '   %4.1f%%  %.3fs  %.3fs  %s %s' % (f*100, tot, t, msg, a)
        print '   ... (remaining %i Ops account for %6.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(otimes)-n_ops_to_print),
                  sum(f for f, t, a, ci, nb_call in otimes[n_ops_to_print:])*100,
                  sum(t for f, t, a, ci, nb_call in otimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'


        sop_time={}
        sop_call={}
        sop_c={} #map each op class to Bool. True iff all applies were done in c.
        for a,t in op_time.items():
            sop_time.setdefault(type(a),0)
            sop_time[type(a)]+=t
            sop_c.setdefault(type(a),True)
            sop_c[type(a)]=sop_c[type(a)] and op_cimpl[a]
            sop_call[type(a)]=sop_call.get(type(a),0)+op_call[a]
        print '\nSingle Op-wise summary: <% of local_time spent on this kind of Op> <cumulative seconds> <self seconds> <nb_call> <Op name>'
        sotimes = [(t/local_time, t, a, sop_c[a], sop_call[a]) for a, t in sop_time.items()]
        sotimes.sort()
        sotimes.reverse()
        tot=0
        for f,t,a,ci, nb_call in sotimes[:n_ops_to_print]:
            tot+=t
            if ci:
              msg = '*'
            else:
              msg = ' '
            print '   %4.1f%%  %.3fs  %.3fs  %s %d %s' % (f*100, tot, t, msg, nb_call, a)
        print '   ... (remaining %i Ops account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(sotimes)-n_ops_to_print),
                  sum(f for f, t, a, nb_call in sotimes[n_ops_to_print:])*100,
                  sum(t for f, t, a, nb_call in sotimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'
        print 'compile time: %.3fs'%compile_time

register_mode('PROFILE_MODE',ProfileMode())

def atexit_print_default_profile_mode():
    """Print the summary of the predefied mode PROFILE_MODE if used.
    
    This all to have the summary printed at exit when we do
    THEANO_DEFAULT_MODE=PROFILE_MODE
    """
    prof_mode=predefined_modes["PROFILE_MODE"]
    if prof_mode.local_time[0]>0: prof_mode.print_summary()

#Register atexit_print_default_profile_mode to have the summary of the
#predefined mode PROFILE_MODE if it is used printed when the program terminate.
atexit.register(atexit_print_default_profile_mode)

