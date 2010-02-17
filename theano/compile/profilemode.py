import time, atexit, copy

from theano.gof.link import WrapLinker
from theano.gof.cutils import run_cthunk
from theano.compile.mode import Mode, register_mode, predefined_modes, predefined_linkers, predefined_optimizers
from theano.gof.cc import OpWiseCLinker
from theano.gof.python25 import any
from theano import gof
from theano.configparser import config, AddConfigVar, IntParam
from theano.compile.function_module import FunctionMaker

import_time = time.time()

AddConfigVar('ProfileMode.n_apply_to_print',
        "Number of apply instances to print by default",
        IntParam(15, lambda i: i > 0))

AddConfigVar('ProfileMode.n_ops_to_print',
        "Number of ops to print by default",
        IntParam(20, lambda i: i > 0))

class Profile_Maker(FunctionMaker):
    def create(self, input_storage=None, trustme=False):
        ret = super(Profile_Maker,self).create(input_storage, trustme)
        for i, node in enumerate(ret.maker.env.toposort()):
            self.mode.apply_time[(i,node.op)]=0.0
            self.mode.apply_call[(i,node.op)]=0
#            self.mode.op_cimpl[node.op] = 

        return ret

class ProfileMode(Mode):
    def __init__(self, linker=config.linker, optimizer=config.optimizer):
        local_time = [0.0]
        apply_time = {}
        apply_call = {}
        op_cimpl = {}
        compile_time = 0 #time passed in theano.function()
        fct_call_time = {}#time passed inside theano fct call including op time.
        fct_call = {}

        self.__setstate__((linker, optimizer, local_time,
                           apply_time, apply_call,
                           op_cimpl,
                           compile_time, fct_call_time, fct_call))

    def function_maker(self, i,o,m, *args, **kwargs):
        """Return an instance of `Profiler_Maker` which init the count"""

        assert m is self
        return Profile_Maker(i, o, self, *args, **kwargs)

    def __getstate__(self):
        #print "__getstate__",self.provided_linker,self.provided_optimizer
        return (self.provided_linker, self.provided_optimizer, self.local_time,
                self.apply_time, self.apply_call,
                self.op_cimpl, self.compile_time, self.fct_call_time, self.fct_call)

    def __setstate__(self, (linker, optimizer, local_time,
                            apply_time, apply_call,
                            op_cimpl,
                            compile_time, fct_call_time, fct_call)):
        
        self.local_time = local_time
        self.apply_time = apply_time
        self.apply_call = apply_call
        self.op_cimpl = op_cimpl
        self.compile_time = compile_time
        self.fct_call_time = fct_call_time
        self.fct_call = fct_call
        self.call_time = 0
        self.fn_time = 0

        def blah(i, node, th):
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                failure = run_cthunk(th.cthunk)
                dt = time.time() - t0
                if failure:
                    raise RuntimeError(('A C Op raised an exception.  PROFILE_MODE cannot' 
                        ' tell you what it was though.  Use a standard mode such as'
                        ' FAST_RUN_NOGC to correct the problem.'))
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0

            local_time[0] += dt
            apply_time[(i,node.op)] += dt
            apply_call[(i,node.op)] += 1
            op_cimpl[node.op] = hasattr(th, 'cthunk')

        
        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, str) or linker is None:
            linker = predefined_linkers[linker]

        linker = WrapLinker([linker], blah)
            
        self.linker = linker
        if isinstance(optimizer, str) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        self._optimizer = optimizer

    def print_summary(self, 
                      n_apply_to_print=config.ProfileMode.n_apply_to_print,
                      n_ops_to_print=config.ProfileMode.n_ops_to_print):
        """ Print 3 summary that show where the time is spend. The first show an Apply-wise summary, the second show an Op-wise summary, the third show an type-Op-wise summary.

        The Apply-wise summary print the timing information for the worst offending Apply nodes. This corresponds to individual Op applications within your graph which take the longest to execute (so if you use dot twice, you will see two entries there). 
        The Op-wise summary print the execution time of all Apply nodes executing the same Op are grouped together and the total execution time per Op is shown (so if you use dot twice, you will see only one entry there corresponding to the sum of the time spent in each of them). If two Op have different hash value, they will be separate.
        The type-Op-wise summary group the result by type of op. So event if two Op have different hash value, they will be merged.

        Their is an hack with the Op-wise summary. Go see it if you want to know more.

        :param n_apply_to_print: the number of apply to print. Default 15, or n_ops_to_print flag.

        :param n_ops_to_print: the number of ops to print. Default 20, or n_apply_to_print flag.
        """

        local_time = self.local_time[0]
        compile_time = self.compile_time
        fct_call_time = self.fct_call_time
        fct_call = self.fct_call
        apply_time = self.apply_time
        apply_call = self.apply_call
        op_cimpl = self.op_cimpl

        self.print_summary_("print_summary",local_time, compile_time, fct_call_time, fct_call,
                            apply_time, apply_call, op_cimpl,
                            n_apply_to_print, n_ops_to_print)


    def print_diff_summary(self, other, n_apply_to_print=15, n_ops_to_print=20):
        """ As print_summary, but print the difference on two different profile mode.
        TODO: Also we don't print the Apply-wise summary as it don't work for now.
        TODO: make comparaison with gpu code.
        
        :param other: the other instance of ProfileMode that we want to be compared to.
        
        :param n_apply_to_print: the number of apply to print. Default 15.

        :param n_ops_to_print: the number of ops to print. Default 20.
        """

        def diff_dict(a_time,b_time_):
            r = {}
            b_time = copy.copy(b_time_)
            for a,ta in a_time.items():
                r.setdefault(a,0)
                tb = b_time.pop(a,0)
                r[a]+=ta-tb
                
            #they are missing in a
            for a,t in b_time.items():
                r.setdefault(a,0)
                r[a]+=t
            return r
        
        local_time = self.local_time[0]-other.local_time[0]
        compile_time = self.compile_time-other.compile_time
        fct_call_time = diff_dict(self.fct_call_time,other.fct_call_time)
        fct_call = diff_dict(self.fct_call,other.fct_call)
        apply_time = diff_dict(self.apply_time, other.apply_time)
        apply_call = diff_dict(self.apply_call, other.apply_call)
        op_cimpl = self.op_cimpl and other.op_cimpl

        self.print_summary_("print_diff_summary",local_time, compile_time, fct_call_time, fct_call,
                            apply_time, apply_call, op_cimpl,
                            n_apply_to_print=n_apply_to_print,
                            n_ops_to_print=n_ops_to_print, print_apply=False)

    @staticmethod
    def print_summary_(fct_name, local_time, compile_time, fct_call_time, fct_call,
                       apply_time, apply_call, op_cimpl,
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
        
        print 'local_time %.3fs (Time spent running thunks)'% local_time

        if print_apply:
            print 'Apply-wise summary: <% of local_time spent at this position> <cumulative seconds> <apply time> <time per call> <nb_call> <Apply position> <Apply Op name>'
            atimes = [(t/local_time, t, (a[0], str(a[1])), apply_call[a]) for a, t in apply_time.items()]
            atimes.sort()
            atimes.reverse()
            tot=0
            for f,t,a,nb_call in atimes[:n_apply_to_print]:
                tot+=t
                print '   %4.1f%%  %.3fs  %.3fs %.2es  %i  %i %s' % (f*100, tot, t, t/nb_call,nb_call, a[0], a[1])
            print '   ... (remaining %i Apply instances account for %.2f%%(%.2fs) of the runtime)'\
                    %(max(0, len(atimes)-n_apply_to_print),
                      sum(f for f, t, a, nb_call in atimes[n_apply_to_print:])*100,
                      sum(t for f, t, a, nb_call in atimes[n_apply_to_print:]))

        op_time = {}
        op_call = {}
        op_apply = {}
        for (i,a),t in apply_time.items():
            op_time.setdefault(a,0)
            op_call.setdefault(a,0)
            op_apply.setdefault(a,0)
            op_time[a]+=t
            op_call[a]+=apply_call[(i,a)]
            op_apply[a]+=1

        op_flops = {}
        for a,t in op_time.items():
            if hasattr(a,'flops'):
                op_flops[a]=a.flops*op_call[a]/t/1e6

        flops_msg=''
        if op_flops:
            flops_msg=' <MFlops/s>'
            print '\nHACK WARNING: we print the flops for some OP, but the logic don\' always work. You need to know the internal of Theano to make it work correctly. Otherwise don\'t use!'
        print '\nOp-wise summary: <%% of local_time spent on this kind of Op> <cumulative seconds> <self seconds> <time per call> %s <nb_call> <nb apply> <Op name>'%(flops_msg)

        otimes = [(t/local_time, t, a, op_cimpl.get(a, 0), op_call.get(a, 0), op_apply.get(a,0)) 
                for a, t in op_time.items()]
        otimes.sort()
        otimes.reverse()
        tot=0
        for f,t,a,ci,nb_call,nb_apply in otimes[:n_ops_to_print]:
            if nb_call == 0: 
                assert t == 0
                continue
            tot+=t
            if ci:
              msg = '*'
            else:
              msg = ' '
            if op_flops:
                print '   %4.1f%%  %.3fs  %.3fs  %.2es %s %7.1f %d %d %s' % (f*100, tot, t, t/nb_call, msg, op_flops.get(a,-1), nb_call, nb_apply, a)
            else:
                print '   %4.1f%%  %.3fs  %.3fs  %.2es  %s %d %d %s' % (f*100, tot, t, t/nb_call, msg, nb_call, nb_apply, a)
        print '   ... (remaining %i Ops account for %6.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(otimes)-n_ops_to_print),
                  sum(f for f, t, a, ci, nb_call, nb_op in otimes[n_ops_to_print:])*100,
                  sum(t for f, t, a, ci, nb_call, nb_op in otimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'


        sop_time={}
        sop_call={}
        sop_op = {}
        sop_c={} #map each op class to Bool. True iff all applies were done in c.
        for a,t in op_time.items():
            sop_time.setdefault(type(a),0)
            sop_time[type(a)]+=t
            sop_op.setdefault(type(a),0)
            sop_op[type(a)]+=1
            sop_c.setdefault(type(a),True)
            sop_c[type(a)]=sop_c[type(a)] and op_cimpl.get(a, False)
            sop_call[type(a)]=sop_call.get(type(a),0)+op_call[a]
        print '\nSingle Op-wise summary: <% of local_time spent on this kind of Op> <cumulative seconds> <self seconds> <time per call> <nb_call> <nb_op> <nb_op> <Op name>'
        sotimes = [(t/local_time, t, a, sop_c[a], sop_call[a], sop_op[a]) for a, t in sop_time.items()]
        sotimes.sort()
        sotimes.reverse()
        tot=0
        for f,t,a,ci, nb_call, nb_op in sotimes[:n_ops_to_print]:
            if nb_call == 0: 
                assert t == 0
                continue
            tot+=t
            if ci:
              msg = '*'
            else:
              msg = ' '
            print '   %4.1f%%  %.3fs  %.3fs  %.2es %s %d %d %s' % (f*100, tot, t, t/nb_call, msg, nb_call, nb_op, a)
        print '   ... (remaining %i Ops account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(sotimes)-n_ops_to_print),
                  sum(f for f, t, a, ci, nb_call, nb_op in sotimes[n_ops_to_print:])*100,
                  sum(t for f, t, a, ci, nb_call, nb_op in sotimes[n_ops_to_print:]))

        print '(*) Op is running a c implementation'
            

        total_time = time.time() - import_time
        total_fct_time = sum(fct_call_time.values())
        total_fct_call = sum(fct_call.values())
        other_time = total_time - local_time - compile_time
        print
        print 'Theano fct summary: <% total fct time> <total time> <time per call> <nb call> <fct name>'
        for key in fct_call.keys():
            print '   %4.1f%% %.3fs %.2es %d %s'%(fct_call_time[key]/total_fct_time*100 ,fct_call_time[key],
                                                  fct_call_time[key]/fct_call[key], fct_call[key],key)
        print
        print 'Time since import %.3fs'%(total_time)
        print 'Compile time: %.3fs %.1f%%'%(compile_time, compile_time/total_time*100)
        print 'Theano fct call %.3fs %.1f%%'%(total_fct_time,total_fct_time/total_time*100)
        print '   Theano Op time (included in fct call, Time spent running thunks) %.3fs %.1f%%(of total) %.1f%%(of fct call)'% (local_time,local_time/total_time*100,local_time/total_fct_time*100)
        print 'Other time since import %.3fs %.1f%%'%(other_time,other_time/total_time*100)
        print '%i Theano fct call, %.3fs per call'%(total_fct_call, total_fct_time/total_fct_call)
        
        if any([x[2].__name__.startswith("Gpu") for x in sotimes]):
            cpu=[]
            gpu=[]
            trans=[]
            for so in sotimes:
                if so[2].__name__ in ["HostFromGpu", "GpuFromHost"]:
                    trans.append(so)
                elif so[2].__name__.startswith("Gpu"):
                    gpu.append(so)
                else:
                    cpu.append(so)
            sum_cpu=sum(so[1] for so in cpu)
            sum_gpu=sum(so[1] for so in gpu)
            sum_trans=sum(so[1] for so in trans)
            print sum_cpu+sum_gpu+sum_trans,local_time

            print "Spent %.3fs(%.3f%%) in cpu Op, %.3fs(%.3f%%) in gpu Op and %.3fs(%.3f%%) transfert Op"%(
                sum_cpu, sum_cpu/local_time*100, sum_gpu, sum_gpu/local_time*100, sum_trans, sum_trans/local_time*100)

register_mode('PROFILE_MODE',ProfileMode())

def atexit_print_default_profile_mode():
    """Print the summary of the predefined mode PROFILE_MODE if used.
    
    This all to have the summary printed at exit when
    config.mode=PROFILE_MODE
    """
    prof_mode=predefined_modes["PROFILE_MODE"]
    if prof_mode.local_time[0]>0:
        prof_mode.print_summary()

#Register atexit_print_default_profile_mode to have the summary of the
#predefined mode PROFILE_MODE if it is used printed when the program terminate.
atexit.register(atexit_print_default_profile_mode)

