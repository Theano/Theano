"""ProfileStats object for runtime and memory profiling.
"""
#
# TODO: measure memory usage like ProfileMode did
# TODO: put the optimization tips into a tips section??
# TODO: add tip to use specify_shape (is specify_shape even in library doc?)
# TODO: ensure field width for string fields makes columns line up
# TODO: what to do about 'diff summary'? (ask Fred?)
#
__authors__   = "James Bergstra"
__reviewer__  = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"
import atexit
import sys
import theano
from theano.configparser import AddConfigVar, StrParam, BoolParam
import time
import numpy
import_time = time.time()
config = theano.config

_atexit_print_list = []
_atexit_print_file = sys.stderr

AddConfigVar('profiling.time_thunks',
             """Time individual thunks when profiling""",
        BoolParam(True))


def _atexit_print_fn():
    """Print ProfileStat objects in _atexit_print_list to _atexit_print_file
    """
    for ps in _atexit_print_list:
        if ps.fct_callcount or ps.compile_time > 0:
            ps.summary(file=_atexit_print_file)
        else:
            print 'Skipping empty Profile'
atexit.register(_atexit_print_fn)

class ProfileStats(object):
    """
    Object to store runtime and memory profiling information for all of
    Theano's operations: compilation, optimization, execution.
    """

    #
    # Note on implementation:
    # Class variables are used here so that each one can be
    # documented and initialized together.
    # dictionary variables are initialized with None.
    #

    compile_time = 0.0
    # Total time spent in body of orig_function,
    # dominated by graph optimization and compilation of C
    #

    fct_call_time = 0.0
    # The total time spent in Function.__call__
    #

    fct_callcount = 0
    # Number of calls to Function.__call__
    #

    vm_call_time = 0.0
    # Total time spent in Function.fn.__call__
    #

    apply_time = None
    # dict from node -> float runtime
    #

    apply_callcount = None
    # dict from node -> number of executions
    #

    apply_cimpl = None
    # dict from node -> bool (1 if c, 0 if py)
    #

    message = None
    # pretty string to print in summary, to identify this output
    #

    outputs_size = None
    # node -> size of allocated output
    #

    optimizer_time = 0.0
    # time spent optimizing graph (FunctionMaker.__init__)

    linker_time = 0.0
    # time spent linking graph (FunctionMaker.create)

    line_width = 140
    # param is called flag_time_thunks because most other attributes with time
    # in the name are times *of* something, rather than configuration flags.
    def __init__(self, atexit_print=True, flag_time_thunks=None, **kwargs):
        """
        atexit_print - bool. True means that this object will be printed to
                       stderr (using .summary()) at the end of the program.
        **kwargs - misc initializers. These should (but need not) match the
                   names of the class vars declared in this class.
        """
        self.apply_callcount = {}
        self.output_size = {}
        self.apply_time = {}
        self.apply_cimpl = {}
        self.outputs_size = {}
        if flag_time_thunks is None:
            self.flag_time_thunks = config.profiling.time_thunks
        else:
            self.flag_time_thunks = flag_time_thunks
        self.__dict__.update(kwargs)
        #print >> sys.stderr, "self.message", self.message
        if atexit_print:
            global _atexit_print_list
            _atexit_print_list.append(self)

    def op_time(self):
        """dict op -> total time on thunks"""
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for node, t in self.apply_time.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += t
        return rval

    def op_callcount(self):
        """dict op -> total number of thunk calls"""
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for node, count in self.apply_callcount.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += count
        return rval

    def op_nodes(self):
        """dict op -> total number of nodes"""
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for node, count in self.apply_callcount.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += 1
        return rval

    def op_impl(self):
        """dict op -> total number of nodes"""
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for node in self.apply_callcount:
            if self.apply_cimpl[node]:
                rval[node.op] = 'C '
            else:
                rval[node.op] = 'Py'
        return rval

    def op_flops(self):
        """dict op -> total number of flops"""
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        return rval #TODO: continue here
        for node, count in self.apply_callcount.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += 1
        return rval
        for a,t in op_time.items():
            if hasattr(a,'flops'):
                op_flops[a]=a.flops*op_call[a]/t/1e6

        flops_msg=''
        if op_flops:
            flops_msg=' <MFlops/s>'
            print '\nHACK WARNING: we print the flops for some OP, but the logic don\' always work. You need to know the internal of Theano to make it work correctly. Otherwise don\'t use!'
        print '\nOp-wise summary: <%% of local_time spent on this kind of Op> <cumulative %%> <self seconds> <cumulative seconds> <time per call> %s <nb_call> <nb apply> <Op name>'%(flops_msg)

    def summary_ops(self, file=sys.stderr, N=None):
        if self.apply_time:
            local_time = sum(self.apply_time.values())
        else:
            local_time = 0
        if local_time == 0:
            print >> file, ('ProfileMode.summary_ops: total time 0'
                    ' (did you forget to enable counters?)')
            return
        op_time = self.op_time()
        op_call = self.op_callcount()
        op_apply = self.op_nodes()
        op_flops = self.op_flops()
        op_impl = self.op_impl()
        if N is None:
            N = len(self.op_flops)
        otimes = [(t*100/local_time,
                    t,
                    op,
                    op_impl.get(op, '  '),
                    op_call.get(op, 0),
                    op_apply.get(op,0))
                for op, t in op_time.items()]
        otimes.sort()
        otimes.reverse()
        tot=0
        print >> file, 'Ops'
        print >> file, '---'
        #print >> file, '<% time> <cumulative %%> <apply time> <cumulative seconds> <time per call> <nb_call> <Op name>'
        hs = []
        # formatting string
        es = []

        hs += ['<% time>']
        es += ['  %4.1f%% ']

        hs += ['<sum %>']
        es += [' %5.1f%% ']

        hs += ['<apply time>']
        es += ['   %7.3fs ']

        hs += ['<time per call>']
        es += ['     %8.2es ']

        hs += ['<type>']
        es += ['   %2s ']

        hs += ['<#call>']
        es += ['  %4d  ']

        hs += ['<#apply>']
        es += ['  %4d  ']

        upto_length = numpy.sum([len(x) for x in hs]) + len(hs)
        maxlen = self.line_width - upto_length
        hs += ['<Op name>']
        es += ['%s']
        header_str = ' '.join(hs)
        format_str = ' '.join(es)

        print >> file, header_str

        for f,t,a,impl,nb_call,nb_apply in otimes[:N]:
            if nb_call == 0:
                assert t == 0
                continue
            tot+=t
            ftot=tot*100/local_time
            print >> file, format_str%(f,ftot,t,t/nb_call, impl, nb_call,
                                       nb_apply, str(a)[:maxlen])
            # While this carries over less information, it is arranged such
            # that it way more readeable that the previous output of the
            # profiler
            #if op_flops:
            #    print >>file, '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %7.1f %5d %2d %s' % (
            #            f, ftot, t, tot, t/nb_call, impl, op_flops.get(a,-1), nb_call, nb_apply, a)
            #else:
            #    print >>file, '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %5d %2d %s' % (
            #            f, ftot, t, tot, t/nb_call, impl, nb_call, nb_apply, a)
        print >>file, '   ... (remaining %i Ops account for %6.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(otimes)-N),
                  sum(f for f, t, a, ci, nb_call, nb_op in otimes[N:]),
                  sum(t for f, t, a, ci, nb_call, nb_op in otimes[N:]))
        print >> file, ''

    def summary_nodes(self, file=sys.stderr, N=None):
        if self.apply_time:
            local_time = sum(self.apply_time.values())
        else:
            local_time = 0
        if local_time == 0:
            print >> file, ('ProfileMode.summary_nodes: total time 0'
                    ' (did you forget to enable counters?)')
            return

        print >> file, 'Apply'
        print >> file, '------'
        #print >> file, '<% time> <cumulative %%> <apply time> <cumulative seconds> <time per call> <nb_call> <Apply Op name>'
        # headers
        hs = []
        # formatting string
        es = []

        hs += ['<% time>']
        es += ['  %4.1f%% ']

        hs += ['<sum %>']
        es += [' %5.1f%% ']

        hs += ['<apply time>']
        es += ['   %7.3fs ']

        hs += ['<time per call>']
        es += ['     %8.2es ']

        hs += ['<#call>']
        es += [' %4d  ']

        hs += ['<id>']
        es += ['%3d']

        upto_length = numpy.sum([len(x) for x in hs]) + len(hs)
        maxlen = self.line_width - upto_length
        hs += ['<Apply name>']
        es += ['%s']

        header_str = ' '.join(hs)
        format_str = ' '.join(es)

        print >> file, header_str

        atimes = [(
                t*100/local_time,
                t,
                a,
                a.env.toposort().index(a),
                self.apply_callcount[a])
            for a, t in self.apply_time.items()]
        atimes.sort()
        atimes.reverse()
        tot=0
        for (f, t, a, nd_id, nb_call) in atimes[:N]:
            tot+=t
            ftot=tot*100/local_time
            if nb_call==0:
                continue
            print >> file, format_str %(f,ftot, t, t/nb_call, nb_call,
                                        nd_id,
                                        str(a)[:maxlen])
            # Same as before, this I've sacrificied some information making
            # the output more readable
            #print >> file, '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs %.2es  %i  %s'%(
            #        f, ftot, t, tot, t/nb_call,nb_call, str(a))
        print >> file, '   ... (remaining %i Apply instances account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(atimes)-N),
                  sum(f for f, t, a, nd_id, nb_call in atimes[N:]),
                  sum(t for f, t, a, nd_id, nb_call in atimes[N:]))
        print >> file, ''

    def summary_function(self, file):
        print >> file, 'Function profiling'
        print >> file, '=================='
        print >> file, '  Message: %s'%self.message
        print >> file, '  Time in %i calls to Function.__call__: %es' % (
                self.fct_callcount, self.fct_call_time)
        if self.fct_call_time>0:
            print >> file, '  Time in Function.fn.__call__: %es (%.3f%%)' %(
                    self.vm_call_time, 100*self.vm_call_time / self.fct_call_time)
            local_time = sum(self.apply_time.values())
            if local_time > 0:
                print >> file, '  Time in thunks: %es (%.3f%%)' %(
                        local_time, 100*local_time / self.fct_call_time)
        print >> file, ''


    def summary(self, file=sys.stderr, n_ops_to_print=20, n_applies_to_print=20):
        self.summary_function(file)
        local_time = sum(self.apply_time.values())
        if local_time > 0:
            self.summary_ops(file, n_ops_to_print)
            self.summary_nodes(file, n_applies_to_print)
        else:
            print >> file, "  No node time accumulated (hint: try config profiling.time_thunks=1)"



if 0: # old code still to be ported from ProfileMode
    def long_print(self, file=sys.stderr, fct_name=None, message=None,
            n_apply_to_print=15, n_ops_to_print=20, print_apply=False):
        """
        Print a readable summary of the stats.

        param: n_apply_to_print the number of apply to print. Default 15.

        param: n_ops_to_print the number of ops to print. Default 20.
        """
        local_time = sum(self.apply_time.values())

        print ''
        print 'ProfileMode.long_print()'
        print 'name = %s'%fct_name
        print 'msg = %s'%message
        print '---------------------------'
        print ''

        print 'Total time spent running thunks: %.3fs'% local_time

        sop_time={}
        sop_call={}
        sop_op = {}
        sop_c={} #map each op class to Bool. True iff all applies were done in c.
        for a,t in op_time.items():
            typ = type(a)
            sop_time.setdefault(typ,0)
            sop_time[typ]+=t
            sop_op.setdefault(typ,0)
            sop_op[typ]+=1
            sop_c.setdefault(typ,True)
            sop_c[typ]=sop_c[typ] and op_cimpl.get(a, False)
            sop_call[typ]=sop_call.get(typ,0)+op_call[a]
        print '\nSingle Op-wise summary: <% of local_time spent on this kind of Op> <cumulative %%> <self seconds> <cumulative seconds> <time per call> <nb_call> <nb_op> <nb_op> <Op name>'
        sotimes = [(t*100/local_time, t, a, sop_c[a], sop_call[a], sop_op[a]) for a, t in sop_time.items()]
        sotimes.sort()
        sotimes.reverse()
        tot=0
        for f,t,a,ci, nb_call, nb_op in sotimes[:n_ops_to_print]:
            if nb_call == 0:
                assert t == 0
                continue
            tot+=t
            ftot=tot*100/local_time
            if ci:
                msg = '*'
            else:
                msg = ' '
            print '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %5d %2d %s' % (f, ftot, t, tot, t/nb_call, msg, nb_call, nb_op, a)
        print '   ... (remaining %i Ops account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(sotimes)-n_ops_to_print),
                  sum(f for f, t, a, ci, nb_call, nb_op in sotimes[n_ops_to_print:]),
                  sum(t for f, t, a, ci, nb_call, nb_op in sotimes[n_ops_to_print:]))


        total_time = time.time() - import_time
        total_fct_time = sum(fct_call_time.values())
        total_fct_call = sum(fct_call.values())
        other_time = total_time - total_fct_time - compile_time
        print
        print 'Theano fct summary: <% total fct time> <total time> <time per call> <nb call> <fct name>'
        for key in fct_call.keys():
            if fct_call[key]>0:
                print '   %4.1f%% %.3fs %.2es %d %s'%(fct_call_time[key]/total_fct_time*100 ,fct_call_time[key],
                                                      fct_call_time[key]/fct_call[key], fct_call[key], key.name)
            else:
                print '   NOT CALLED',key.name

        if total_fct_time>0:
            time_pr_in_fct=local_time/total_fct_time*100
            time_per_call=total_fct_time/total_fct_call
        else:
            time_pr_in_fct=0
            time_per_call=0

        print
        print 'Time since import %.3fs'%(total_time)
        print 'Compile time: %.3fs %.1f%%'%(compile_time, compile_time/total_time*100)
        print 'Theano fct call %.3fs %.1f%%'%(total_fct_time,total_fct_time/total_time*100)
        print '   Theano Op time (included in fct call, Time spent running thunks) %.3fs %.1f%%(of total) %.1f%%(of fct call)'% (local_time,local_time/total_time*100, time_pr_in_fct)
        print 'Other time since import %.3fs %.1f%%'%(other_time,other_time/total_time*100)
        print '%i Theano fct call, %.3fs per call'%(total_fct_call, time_per_call)

        print
        print "List of apply that don't have float64 as input but have float64 in outputs. Usefull to know if we forgot some cast when using floatX=float32 or gpu code."
        print '<Apply> <Apply position> <fct name> <inputs type> <outputs type>'
        for fct in fct_call.keys():
            for idx, node in enumerate(fct.maker.env.toposort()):
                if any(hasattr(i,'dtype') and i.dtype=='float64' for i in node.outputs) and not any(hasattr(i,'dtype') and i.dtype=='float64' for i in node.inputs):
                    print str(node), idx, fct.name, str([getattr(i,'dtype',None) for i in node.inputs]),str([getattr(i,'dtype',None) for i in node.outputs])

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
            print

            print "Spent %.3fs(%.3f%%) in cpu Op, %.3fs(%.3f%%) in gpu Op and %.3fs(%.3f%%) transfert Op"%(
                sum_cpu, sum_cpu/local_time*100, sum_gpu, sum_gpu/local_time*100, sum_trans, sum_trans/local_time*100)

            print "Theano function input that are float64"
            print "<fct name> <input name> <input type> <str input>"
            for fct in fct_call.keys():
                for i in fct.input_storage:
                    if hasattr(i.type, 'dtype') and i.type.dtype=='float64':
                        print fct.name, i.name, i.type, i

        if outputs_size:
            fct_memory={}#env->dict(node->(outputs size))
            var_mem = {}
            for node,val in outputs_size.items():
                fct_memory.setdefault(node.env,{})
                fct_memory[node.env][node]=val
                for out,v in zip(node.outputs,val):
                    var_mem[out]=v
            print
            print "Profile of Theano functions memory:"
            for env,nodes_mem in fct_memory.iteritems():
                print "Theano fct:", [fct for fct in fct_call.keys() if fct.maker.env is env][0].name
                size_sum=sum([sum(val) for key,val in nodes_mem.iteritems()])
                print "    Max without gc, inplace and view (KB)",size_sum/1024

                node_memory_size = 0
                node_memory_saved_by_view = 0
                node_memory_saved_by_inplace = 0
                running_memory_size = 0
                running_max_memory_size = 0
                post_thunk_old_storage = []
                items = nodes_mem.items()
                items.sort(key=lambda a: a[1])
                items.reverse()

                order = env.toposort()
                computed, last_user = gc_helper(order)
                for node in order:
                    post_thunk_old_storage.append([ input_idx
                                                    for input_idx,input in enumerate(node.inputs)
                                                    if (input in computed) and (input not in env.outputs) and node == last_user[input]])
                for node,val in items[:n_apply_to_print]:
                    dmap = getattr(node.op,'destroy_map',None)
                    vmap = getattr(node.op,'view_map',None)

                    for idx,v in enumerate(val):
                        if dmap and idx in dmap:#TODO check the op returned a view
                            node_memory_saved_by_inplace += v
                        elif vmap and idx in vmap:#TODO check the op returned a view
                            node_memory_saved_by_view += v
                        else:
                            node_memory_size += v
                            running_memory_size += v
                            if running_memory_size > running_max_memory_size:
                                running_max_memory_size = running_memory_size
                            old_storage = post_thunk_old_storage[order.index(node)]
                            for old_s in old_storage:
                                running_memory_size -= var_mem[node.inputs[old_s]]
                                pass
                    pass

                print "    Max FAST_RUN_NO_GC (KB)", node_memory_size/1024
                print "    Max FAST_RUN (KB)", running_max_memory_size/1024
                print "    Memory saved by view (KB)", node_memory_saved_by_view/1024
                print "    Memory saved by inplace (KB)", node_memory_saved_by_inplace/1024
                print "    Memory saved by GC (KB)", (node_memory_size-running_max_memory_size)/1024

                n_apply_to_print+=10#TODO remove this line
                print "    <Sum apply outputs (bytes)> <Apply outputs memory size(bytes)> <created/inplace/view> <Apply node>"
                print "    <created/inplace/view> is taked from the op declaration, not the op exeuction. Use DebugMode to have warning about inplace/view declaration being respected."
                for key,val in items[:n_apply_to_print]:
                    code = ['c']*len(node.outputs)
                    for out,inp in getattr(key.op,'destroy_map',{}).iteritems():
                        code[out] = "i"
                    for out,inp in getattr(key.op,'view_map',{}).iteritems():
                        code[out] = "v"
                    print '       %9dB  %s %s %s' % (sum(val), str(val), ' '.join(code), key)

                print '   ... (remaining %i Apply account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(nodes_mem)-n_ops_to_print),
                  sum(sum(val) for key, val in items[n_ops_to_print:]),
                  sum(sum(val) for key, val in items[n_ops_to_print:])/size_sum)


        print
        print "Here are tips to potentially make your code run faster (if you think of new ones, suggest them on the mailing list). Test them first as they are not guaranteed to always provide a speedup."
        from theano import tensor as T
        from theano.tensor.raw_random import RandomFunction
        import theano
        import theano.scalar as scal
        scalar_op_amdlibm_no_speed_up = [scal.LT, scal.GT, scal.LE, scal.GE, scal.EQ, scal.NEQ, scal.InRange, scal.Switch, scal.OR, scal.XOR, scal.AND, scal.Invert, scal.Maximum, scal.Minimum, scal.Add, scal.Mul, scal.Sub, scal.TrueDiv, scal.IntDiv, scal.Clip, scal.First, scal.Second, scal.Identity, scal.Cast, scal.Sgn, scal.Neg, scal.Inv, scal.Sqr ]
        scalar_op_amdlibm_speed_up = [scal.Mod, scal.Pow, scal.Ceil, scal.Floor, scal.RoundHalfToEven, scal.RoundHalfAwayFromZero, scal.Log, scal.Log2, scal.Log10, scal.Log1p, scal.Exp, scal.Sqrt, scal.Abs, scal.Cos,  scal.Sin,  scal.Tan,  scal.Tanh,  scal.Cosh,  scal.Sinh, T.nnet.sigm.ScalarSigmoid, T.nnet.sigm.ScalarSoftplus ]#Abs, Mod in float{32,64} only

        def get_scalar_ops(s):
            if isinstance(s, theano.scalar.Composite):
                l = []
                for node in s.env.toposort():
                    l+=get_scalar_ops(node.op)
                return l
            else: return [s]
        def list_scalar_op(op):
            if isinstance(op.scalar_op, theano.scalar.Composite):
                return get_scalar_ops(op.scalar_op)
            else: return [op.scalar_op]

        def amdlibm_speed_up(op):
            if not isinstance(op, T.Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                for s_op in l:
                    if s_op.__class__ in scalar_op_amdlibm_speed_up:
                        return True
                    elif s_op.__class__ not in scalar_op_amdlibm_no_speed_up:
                        import pdb;pdb.set_trace()
                        print "We don't know if amdlibm will accelerate this scalar op.", s_op
                return False
        def exp_float32_op(op):
            if not isinstance(op, T.Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                return any([s_op.__class__ in [scal.Exp] for s_op in l])

        #tip 1
        if config.floatX=='float64':
            print "  - Try the Theano flag floatX=float32"

        #tip 2
        if not config.lib.amdlibm and any([amdlibm_speed_up(a.op) for i,a in apply_time]):
            print "  - Try installing amdlibm and set the Theano flag lib.amdlibm=True. This speed up only some Elemwise operation."

        #tip 3
        if not config.lib.amdlibm and any([exp_float32_op(a.op) and a.inputs[0].dtype=='float32' for i,a in apply_time]):
            print "  - With the default gcc libm, exp in float32 is slower then in float64! Try Theano flags floatX=float64 or install amdlibm and set the theano flags lib.amdlibm=True"

        #tip 4
        for a, t in apply_time.iteritems():
            node = a
            if isinstance(node.op, T.Dot) and all([ len(i.type.broadcastable)==2 for i in node.inputs]):
                print "  - You have a dot operation that was not optimized to dot22 that is faster. Make sure the inputs are float32 or 64 and are the same for both input. Currently they are:",[i.type for i in node.inputs]

        #tip 5
        for a, t in apply_time.iteritems():
            node = a
            if isinstance(node.op, RandomFunction):
                print "  - Replace the default random number generator by 'from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams' as this is is faster. It is still experimental, but seam to work correctly."
                if config.device.startswith("gpu"):
                    print "     - MRG_RandomStreams is the only random number supported on the GPU."
                break

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
        fct_call_time = self.mode.fct_call_time
        fct_call = self.mode.fct_call
        apply_time = self.apply_time
        op_cimpl = self.op_cimpl
        message = self.message
        outputs_size = self.outputs_size

        self.print_summary_("print_summary",
                None,
                None,
                None,
                apply_time,
                op_cimpl,
                message,
                outputs_size,
                n_apply_to_print,
                n_ops_to_print)


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

        compile_time = self.compile_time-other.compile_time
        fct_call_time = diff_dict(self.fct_call_time,other.fct_call_time)
        fct_call = diff_dict(self.fct_call,other.fct_call)
        apply_time = diff_dict(self.apply_time, other.apply_time)
        op_cimpl = self.op_cimpl and other.op_cimpl
        message = self.message
        outputs_size = diff_dict(self.outputs_size,other.outputs_size)

        self.print_summary_("print_diff_summary", compile_time, fct_call_time, fct_call,
                            apply_time, op_cimpl, message, outputs_size,
                            n_apply_to_print=n_apply_to_print,
                            n_ops_to_print=n_ops_to_print, print_apply=False)


class ScanProfileStats(ProfileStats):
    callcount = 0.0
    nbsteps   = 0.0
    call_time = 0.0
    def __init__(self, atexit_print = True, name = None,  **kwargs):
        super(ScanProfileStats, self).__init__(atexit_print, **kwargs)
        self.name = name


    def summary_function(self, file):
        # RP: everytime we compile a function a ProfileStats is created for
        # that function. This means that everytime a optimization replaces
        # some scan op, some orphane ProfileStats remains in the air ..
        # also even without any optimization, scan compiles a dummy function
        # that will produce a ProfileStats that will correspond to a
        # function that will never be called. Printing several empty
        # Function profiling is just extremely confusing
        if self.callcount == 0:
            return
        print >> file, ''

        if self.name is not None:
            print >> file, 'Scan Op profiling (', self.name, ')'
        else:
            print >> file, 'Scan Op profiling'
        print >> file, '=================='
        print >> file, '  Message: %s'%self.message

        print >> file, '  Time in %i calls of the op (for a total of %i steps) %es' % (
                self.callcount, self.nbsteps, self.call_time)
        print >> file, ''
        val = 0
        if self.call_time > 0:
            val = self.vm_call_time*100/self.call_time
        print >> file, '  Total time spent in calling the VM %es (%.3f%%)'%(
            self.vm_call_time, val)
        val = 100
        if self.call_time > 0:
            val = 100.-self.vm_call_time*100/self.call_time
        print >> file, '  Total overhead (computing slices ..) %es (%.3f%%)'%(
            self.call_time - self.vm_call_time, val)
        print >> file, ''

