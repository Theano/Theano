import time, atexit, copy

from theano.gof.link import WrapLinker
from theano.compile.mode import Mode, register_mode, predefined_modes, predefined_linkers, predefined_optimizers
from theano.gof.python25 import any
from theano import gof
from theano.configparser import config, AddConfigVar, IntParam, BoolParam
from theano.compile.function_module import FunctionMaker
run_cthunk = None # Will be imported only when needed.

from profiling import ProfileStats

import_time = time.time()

AddConfigVar('ProfileMode.n_apply_to_print',
        "Number of apply instances to print by default",
        IntParam(15, lambda i: i > 0),
        in_c_key=False)

AddConfigVar('ProfileMode.n_ops_to_print',
        "Number of ops to print by default",
        IntParam(20, lambda i: i > 0),
        in_c_key=False)

AddConfigVar('ProfileMode.min_memory_size',
             """For the memory profile, do not print apply nodes if the size
 of their outputs (in bytes) is lower then this threshold""",
        IntParam(1024, lambda i: i >= 0),
        in_c_key=False)

AddConfigVar('ProfileMode.profile_memory',
             """Enable profiling of memory used by Theano functions""",
        BoolParam(False),
        in_c_key=False)

class Profile_Maker(FunctionMaker):
    def create(self, input_storage=None, trustme=False):
        ret = super(Profile_Maker,self).create(input_storage, trustme)

        # create a function-specific storage container for profiling info
        profile = ProfileStats(atexit_print=False)
        self.mode.profile_stats[ret] = profile
        ret.profile = profile

        #initialize the timers
        for i, node in enumerate(ret.maker.env.toposort()):
            profile.apply_time[node]=0.0
            profile.outputs_size[node]=[0.0] * len(node.outputs)

            # a thunk_group is a list of the thunks from each linker
            # corresponding to the i'th position in the toposort.
            assert len(ret.fn.thunk_groups[i])==1
            profile.apply_cimpl[node] = hasattr(
                    ret.fn.thunk_groups[i][0],
                    'cthunk')

        # Here we replace the linker function.
        # This ugliness makes WrapLinker (an object that *generates*
        # functions and is not function-specific)  work with ProfileStats
        # objects which are function-specific.

        #capture old fn in closure. This is important since new_fn is about to
        #take its place as ret.fn.
        ret_fn = ret.fn
        def new_fn():
            self.mode.apply_time = self.mode.profile_stats[ret].apply_time
            self.mode.outputs_size = self.mode.profile_stats[ret].outputs_size
            ret_fn()
            # delete the old apply_time variable
            # because it doesn't mean the same thing anymore.
            # This prevents old code from looking like it still works.
            del self.mode.apply_time
            del self.mode.outputs_size

        ret.fn = new_fn

        global run_cthunk
        if run_cthunk is None and any(profile.apply_cimpl.values()):
            # Lazy import to avoid compilation when importing theano.
            from theano.gof.cutils import run_cthunk

        return ret

class ProfileMode(Mode):
    def __init__(self, linker=config.linker, optimizer=config.optimizer):
        message=""
        profile_stats={}
        self.__setstate__((linker,
            optimizer,
            message,
            profile_stats))

    def function_maker(self, i,o,m, *args, **kwargs):
        """Return an instance of `Profiler_Maker` which init the count"""

        assert m is self
        return Profile_Maker(i, o, self, *args, **kwargs)

    def __get_local_time(self):
        rval = 0
        for ps in self.profile_stats.values():
            rval += sum(ps.apply_time.values())
        return rval
    local_time = property(__get_local_time)

    def __getstate__(self):
        #print "__getstate__",self.provided_linker,self.provided_optimizer
        return (self.provided_linker,
                self.provided_optimizer,
                self.message,
                self.profile_stats)

    def __setstate__(self, state):
        linker, optimizer, message, profile_stats = state
        self.message = message
        self.profile_stats = profile_stats

        def profile_thunk(i, node, th):
            """ Profile only the execution time
            """
            global run_cthunk
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                failure = run_cthunk(th.cthunk)
                dt = time.time() - t0
                if failure:
                    raise RuntimeError(('A C Op raised an exception.  ProfileMode cannot'
                        ' tell you what it was though.  Use a standard mode such as'
                        ' FAST_RUN to correct the problem.'))
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0

            # Some Op are so fast that the time.time() resolution is
            # insufficient to measure it.  So we add an epsilon.
            self.apply_time[node] += max(dt, 1e-14)


        def profile_thunk2(i, node, th):
            """ Profile the execution time and the memory size.
            """
            global run_cthunk
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                failure = run_cthunk(th.cthunk)
                dt = time.time() - t0
                if failure:
                    raise RuntimeError(('A C Op raised an exception.  ProfileMode cannot'
                        ' tell you what it was though.  Use a standard mode such as'
                        ' FAST_RUN to correct the problem.'))
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0
            size=[]
            for o in th.outputs:
                if not hasattr(o[0],'size'):
                    #if the output type don't have a size attribute, set -1
                    #to signify we can't evaluate it.
                    #This happen at least for mtrand.RandomState type(in numpy)
                    size.append(-1)
                    continue
                s=o[0].size
                #can't use o[0].dtype.itemsize as dtype is a str for CudaNdarray
                dtype = str(o[0].dtype)
                dtype2=dtype[-2:]
                if dtype2 == '32':
                    s *= 4
                elif dtype2 == '64':
                    s *= 8
                elif dtype2 == '16':
                    s *= 2
                elif dtype[-1] == '8':
                    s *= 1
                elif dtype[-3:] == '128':
                    s *= 16
                else:
                    raise Exception("Can't determine the memory size of dtype",o[0].dtype)
                size.append(s)
            self.outputs_size[node]=size
            self.apply_time[node] += max(dt, 1e-14)


        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, basestring) or linker is None:
            linker = predefined_linkers[linker]

        if not config.ProfileMode.profile_memory:
            p_thunk = profile_thunk
        else:
            p_thunk = profile_thunk2
        linker = WrapLinker([linker], p_thunk)

        self.linker = linker
        if isinstance(optimizer, basestring) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        self._optimizer = optimizer

        self.call_time = 0
        self.fn_time = 0
        self.optimizer_time = 0
        self.linker_time = 0

    def print_summary(self,**kwargs):
        """ Print 3 summary that show where the time is spend. The first show an Apply-wise summary, the second show an Op-wise summary, the third show an type-Op-wise summary.

        The Apply-wise summary print the timing information for the worst offending Apply nodes. This corresponds to individual Op applications within your graph which take the longest to execute (so if you use dot twice, you will see two entries there).
        The Op-wise summary print the execution time of all Apply nodes executing the same Op are grouped together and the total execution time per Op is shown (so if you use dot twice, you will see only one entry there corresponding to the sum of the time spent in each of them). If two Op have different hash value, they will be separate.
        The type-Op-wise summary group the result by type of op. So event if two Op have different hash value, they will be merged.

        Their is an hack with the Op-wise summary. Go see it if you want to know more.

        :param kwargs: They are passed to print_summary_ expanded.
                       Currently there is n_apply_to_print, n_ops_to_print and min_memory_size
                       that are accepted.
        """
        compile_time = sum([ps.compile_time for ps in self.profile_stats.values()])

        fct_call = dict([(fn, ps.fct_callcount)
            for (fn, ps) in self.profile_stats.items()])

        fct_call_time = dict([(fn, ps.fct_call_time)
            for (fn, ps) in self.profile_stats.items()])

        apply_time = {}
        for fn, ps in self.profile_stats.items():
            for (i, node) in enumerate(fn.maker.env.toposort()):
                apply_time[(i, node)] = ps.apply_time[node]
        for (i,n),t in apply_time.items():
            if t == 0:
                print i, n

        apply_cimpl = {}
        outputs_size = {}
        for fn, ps in self.profile_stats.items():
            apply_cimpl.update(ps.apply_cimpl)

        message = self.message

        outputs_size = {}
        for fn, ps in self.profile_stats.items():
            outputs_size.update(ps.outputs_size)

        other_time = dict(
                linker_time = sum(
                    [ps.linker_time for ps in self.profile_stats.values()]),
                optimizer_time = sum(
                    [ps.optimizer_time for ps in self.profile_stats.values()]))

        self.print_summary_("print_summary", compile_time, fct_call_time, fct_call,
                        apply_time, apply_cimpl, message, outputs_size,
                        self.local_time, other_time,
                        **kwargs)

    def print_diff_summary(self, other, **kwargs):
        """ As print_summary, but print the difference on two different profile mode.
        TODO: Also we don't print the Apply-wise summary as it don't work for now.
        TODO: make comparaison with gpu code.

        :param other: the other instance of ProfileMode that we want to be compared to.
        :param kwargs: They are passed to print_summary_ expanded.
                       Currently there is n_apply_to_print, n_ops_to_print and min_memory_size
                       that are accepted.
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
        apply_cimpl = self.apply_cimpl and other.apply_cimpl
        message = self.message
        outputs_size = diff_dict(self.outputs_size,other.outputs_size)
        other_time = {'linker_time':self.linker_time-other.linker_time,
                      'optimizer_time':self.optimizer_time-other.optimizer_time}
        self.print_summary_("print_diff_summary", compile_time, fct_call_time, fct_call,
                            apply_time, apply_cimpl, message, outputs_size,
                            print_apply=False, other_time=other_time,
                            **kwargs)

    @staticmethod
    def print_summary_(fct_name, compile_time, fct_call_time, fct_call,
                       apply_time, apply_cimpl, message, outputs_size,
                       local_time, other_time,
                       n_apply_to_print=config.ProfileMode.n_apply_to_print,
                       n_ops_to_print=config.ProfileMode.n_ops_to_print,
                       print_apply=True,
                       min_memory_size=config.ProfileMode.min_memory_size,
                      ):
        """
        do the actual printing of print_summary and print_diff_summary.

        :param n_apply_to_print: the number of apply to print. Default 15.

        :param n_ops_to_print: the number of ops to print. Default 20.
        :param min_memory_size: Don't print memory profile of apply
                                whose outputs memory size is lower then that.
        """

        total_time = time.time() - import_time
        total_fct_time = sum(fct_call_time.values())
        total_fct_call = sum(fct_call.values())
        unknown_time = total_time - total_fct_time - compile_time
        overhead_time = total_fct_time - local_time
        if total_fct_time>0:
            time_pr_in_fct = local_time/total_fct_time*100
            overhead_time_pourcent_fct_time = overhead_time/total_fct_time*100
            time_per_call = total_fct_time/total_fct_call
        else:
            time_pr_in_fct = 0
            overhead_time_pourcent_fct_time = 0
            time_per_call = 0

        print
        print 'ProfileMode.%s(%s)'%(fct_name,message)
        print '---------------------------'
        print
        print 'Time since import %.3fs'%(total_time)
        print 'Theano compile time: %.3fs (%.1f%% since import)'%(compile_time, compile_time/total_time*100)
        print '    Optimization time: %.3fs'%(other_time['optimizer_time'])
        print '    Linker time: %.3fs'%(other_time['linker_time'])
        print 'Theano fct call %.3fs (%.1f%% since import)'%(total_fct_time, total_fct_time/total_time*100)
        print '   Theano Op time %.3fs %.1f%%(since import) %.1f%%(of fct call)'% (
            local_time, local_time/total_time*100, time_pr_in_fct)
        print '   Theano function overhead in ProfileMode %.3fs %.1f%%(since import) %.1f%%(of fct call)'% (
            overhead_time, overhead_time/total_time*100, overhead_time_pourcent_fct_time)
        print '%i Theano fct call, %.3fs per call'%(total_fct_call, time_per_call)
        print 'Rest of the time since import %.3fs %.1f%%'%(unknown_time, unknown_time/total_time*100)

        print
        print 'Theano fct summary:'
        print '<% total fct time> <total time> <time per call> <nb call> <fct name>'
        for key in fct_call.keys():
            if fct_call[key]>0:
                print '   %4.1f%% %.3fs %.2es %d %s'%(fct_call_time[key]/total_fct_time*100 ,fct_call_time[key],
                                                      fct_call_time[key]/fct_call[key], fct_call[key], key.name)
            else:
                print '   NOT CALLED',key.name


        # Compute stats per op.
        op_time = {}
        op_call = {}
        op_apply = {}
        op_cimpl = {}
        sop_apply = {}
        for (i,a),t in apply_time.items():
            op=a.op
            op_time.setdefault(op,0)
            op_call.setdefault(op,0)
            op_apply.setdefault(op,0)
            sop_apply.setdefault(type(a.op),0)
            op_time[op]+=t
            nb_call = [v for k,v in fct_call.items() if k.maker.env is a.env][0]
            op_cimpl.setdefault(a.op, True)
            op_cimpl[a.op] = op_cimpl[a.op] and apply_cimpl.get(a, False)
            if t==0:
                assert nb_call == 0, nb_call
            else:
                op_call[op] += nb_call
                op_apply[op] += 1
                sop_apply[type(a.op)] += 1

        # Compute stats per op class
        sop_time={}
        sop_call={}
        sop_op = {}
        sop_cimpl={} #map each op class to Bool. True iff all applies were done in c.
        for a,t in op_time.items():
            typ = type(a)
            sop_time.setdefault(typ,0)
            sop_time[typ]+=t
            sop_op.setdefault(typ,0)
            sop_op[typ]+=1
            sop_cimpl.setdefault(typ,True)
            sop_cimpl[typ]=sop_cimpl[typ] and op_cimpl.get(a, False)
            sop_call[typ]=sop_call.get(typ,0)+op_call[a]


        # Print the summary per op class.
        print
        print 'Single Op-wise summary:'
        print '<% of local_time spent on this kind of Op> <cumulative %> <self seconds> <cumulative seconds> <time per call> [*] <nb_call> <nb_op> <nb_apply> <Op name>'
        sotimes = [(t*100/local_time, t, a, sop_cimpl[a], sop_call[a], sop_op[a], sop_apply[a]) for a, t in sop_time.items()]
        sotimes.sort()
        sotimes.reverse()
        tot=0
        for f,t,a,ci, nb_call, nb_op, nb_apply in sotimes[:n_ops_to_print]:
            if nb_call == 0:
                assert t == 0
                continue
            tot+=t
            ftot=tot*100/local_time
            if ci:
                msg = '*'
            else:
                msg = ' '
            print '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %5d %2d %2d %s' % (f, ftot, t, tot, t/nb_call, msg, nb_call, nb_op, nb_apply, a)
        print '   ... (remaining %i single Op account for %.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(sotimes)-n_ops_to_print),
                  sum(soinfo[0] for soinfo in sotimes[n_ops_to_print:]),
                  sum(soinfo[1] for soinfo in sotimes[n_ops_to_print:]))

        print '(*) Op is running a c implementation'


        # The summary per op
        op_flops = {}
        for a,t in op_time.items():
            if hasattr(a,'flops'):
                op_flops[a]=a.flops*op_call[a]/t/1e6
        flops_msg=''
        if op_flops:
            flops_msg=' <MFlops/s>'
            print '\nHACK WARNING: we print the flops for some OP, but the logic don\'t always work. You need to know the internal of Theano to make it work correctly. Otherwise don\'t use!'
        print
        print 'Op-wise summary:'
        print '<%% of local_time spent on this kind of Op> <cumulative %%> <self seconds> <cumulative seconds> <time per call> [*] %s <nb_call> <nb apply> <Op name>'%(flops_msg)

        otimes = [(t*100/local_time, t, a, op_cimpl.get(a, 0), op_call.get(a, 0), op_apply.get(a,0))
                for a, t in op_time.items()]
        otimes.sort()
        otimes.reverse()
        tot=0
        for f,t,a,ci,nb_call,nb_apply in otimes[:n_ops_to_print]:
            if nb_call == 0:
                assert t == 0
                continue
            tot+=t
            ftot=tot*100/local_time
            if ci:
                msg = '*'
            else:
                msg = ' '
            if op_flops:
                print '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %7.1f %5d %2d %s' % (f, ftot, t, tot, t/nb_call, msg, op_flops.get(a,-1), nb_call, nb_apply, a)
            else:
                print '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %5d %2d %s' % (f, ftot, t, tot, t/nb_call, msg, nb_call, nb_apply, a)
        print '   ... (remaining %i Op account for %6.2f%%(%.2fs) of the runtime)'\
                %(max(0, len(otimes)-n_ops_to_print),
                  sum(f for f, t, a, ci, nb_call, nb_op in otimes[n_ops_to_print:]),
                  sum(t for f, t, a, ci, nb_call, nb_op in otimes[n_ops_to_print:]))
        print '(*) Op is running a c implementation'


        if print_apply:
            print
            print 'Apply-wise summary:'
            print '<% of local_time spent at this position> <cumulative %%> <apply time> <cumulative seconds> <time per call> [*] <nb_call> <Apply position> <Apply Op name>'
            atimes = [(t*100/local_time, t, a, [v for k,v in fct_call.items() if k.maker.env is a[1].env][0]) for a, t in apply_time.items()]
            atimes.sort()
            atimes.reverse()
            tot=0
            for f,t,a,nb_call in atimes[:n_apply_to_print]:
                tot+=t
                ftot=tot*100/local_time
                if nb_call==0:
                    continue
                if a[1] in apply_cimpl:
                    msg = '*'
                else:
                    msg = ' '
                print '   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs %.2es  %s %i  %2i %s' % (
                    f, ftot, t, tot, t/nb_call, msg, nb_call, a[0], str(a[1]))
            print '   ... (remaining %i Apply instances account for %.2f%%(%.2fs) of the runtime)'\
                    %(max(0, len(atimes)-n_apply_to_print),
                      sum(f for f, t, a, nb_call in atimes[n_apply_to_print:]),
                      sum(t for f, t, a, nb_call in atimes[n_apply_to_print:]))
            print '(*) Op is running a c implementation'
        for printer in profiler_printers:
            printer(fct_name, compile_time, fct_call_time, fct_call,
                    apply_time, apply_cimpl, message, outputs_size,
                    other_time)

        if not outputs_size:
            print """\nProfile of Theano intermediate memory disabled.
 To enabled, put the Theano flag ProfileMode.profile_memory to True."""
        else:
            fct_memory={}#env->dict(node->(outputs size))
            var_mem = {}
            for node, val in outputs_size.items():
                fct_memory.setdefault(node.env, {})
                fct_memory[node.env][node]=val
                for out,v in zip(node.outputs,val):
                    var_mem[out]=v
            print
            print "Profile of Theano functions memory:"
            print "(This check only the output of each apply node. It don't check the temporary memory used by the op in the apply node.)"
            nb_skipped = 0
            for env,nodes_mem in fct_memory.iteritems():
                size_sum=sum([sum(val) for key,val in nodes_mem.iteritems()])
                if size_sum < min_memory_size:
                    nb_skipped += 1
                    continue
                print "Theano fct:", [fct for fct in fct_call.keys() if fct.maker.env is env][0].name
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
                computed, last_user = gof.link.gc_helper(order)
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

                n_apply_printed = 0
                for key,val in items[:n_apply_to_print]:
                    if sum(val) < min_memory_size:
                        break
                    code = ['c']*len(node.outputs)
                    for out,inp in getattr(key.op,'destroy_map',{}).iteritems():
                        code[out] = "i"
                    for out,inp in getattr(key.op,'view_map',{}).iteritems():
                        code[out] = "v"
                    print '       %9dB  %s %s %s' % (sum(val), str(val), ' '.join(code), key)
                    n_apply_printed += 1

                print '   ... (remaining %i Apply account for %.2f%%(%d bytes) of the total intermediate memory used)'\
                %(max(0, len(nodes_mem)-n_apply_printed),
                  sum(sum(val) for key, val in items[n_apply_printed:])/float(size_sum),
                  sum(sum(val) for key, val in items[n_apply_printed:]))
            if nb_skipped > 0:
                print '   We skipped %d theano function(s). Each of them used less then %dB(theano flags ProfileMode.min_memory_size) of total intermediate memory size'%(nb_skipped, min_memory_size)

        print
        print """Here are tips to potentially make your code run faster
(if you think of new ones, suggest them on the mailing list).
Test them first, as they are not guaranteed to always provide a speedup."""
        from theano import tensor as T
        from theano.tensor.raw_random import RandomFunction
        import theano
        import theano.scalar as scal
        scalar_op_amdlibm_no_speed_up = [scal.LT, scal.GT, scal.LE, scal.GE, scal.EQ, scal.NEQ, scal.InRange, scal.Switch, scal.OR, scal.XOR, scal.AND, scal.Invert, scal.Maximum, scal.Minimum, scal.Add, scal.Mul, scal.Sub, scal.TrueDiv, scal.IntDiv, scal.Clip, scal.Second, scal.Identity, scal.Cast, scal.Sgn, scal.Neg, scal.Inv, scal.Sqr ]
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
                        print "We don't know if amdlibm will accelerate this scalar op.", s_op
                return False
        def exp_float32_op(op):
            if not isinstance(op, T.Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                return any([s_op.__class__ in [scal.Exp] for s_op in l])

        printed_tip = False
        #tip 1
        if config.floatX=='float64':
            print "  - Try the Theano flag floatX=float32"
            printed_tip = True

        #tip 2
        if not config.lib.amdlibm and any([amdlibm_speed_up(a.op) for i,a in apply_time]):
            print "  - Try installing amdlibm and set the Theano flag lib.amdlibm=True. This speeds up only some Elemwise operation."
            printed_tip = True

        #tip 3
        if not config.lib.amdlibm and any([exp_float32_op(a.op) and a.inputs[0].dtype=='float32' for i,a in apply_time]):
            print "  - With the default gcc libm, exp in float32 is slower then in float64! Try Theano flag floatX=float64, or install amdlibm and set the theano flags lib.amdlibm=True"
            printed_tip = True

        #tip 4
        for a, t in apply_time.iteritems():
            node = a[1]
            if isinstance(node.op, T.Dot) and all([ len(i.type.broadcastable)==2 for i in node.inputs]):
                print "  - You have a dot operation that was not optimized to dot22 (which is faster). Make sure the inputs are float32 or 64, and are the same for both inputs. Currently they are:",[i.type for i in node.inputs]
                printed_tip = True

        #tip 5
        for a, t in apply_time.iteritems():
            node = a[1]
            if isinstance(node.op, RandomFunction):
                printed_tip = True
                print "  - Replace the default random number generator by 'from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams', as this is is faster. It is still experimental, but seems to work correctly."
                if config.device.startswith("gpu"):
                    print "     - MRG_RandomStreams is the only random number generator supported on the GPU."
                break

        if not printed_tip:
            print "  Sorry, no tip for today."

register_mode('PROFILE_MODE',ProfileMode())

#needed to print the profile at the end automatically
prof_mode_instance_to_print=[predefined_modes["PROFILE_MODE"]]

def atexit_print_default_profile_mode():
    """Print the summary of the predefined mode PROFILE_MODE if used.

    This all to have the summary printed at exit when
    config.mode=PROFILE_MODE
    """
    for prof_mode in prof_mode_instance_to_print:
        if prof_mode.local_time>0:
            prof_mode.print_summary()

#Register atexit_print_default_profile_mode to have the summary of the
#predefined mode PROFILE_MODE if it is used printed when the program terminate.
atexit.register(atexit_print_default_profile_mode)


# Here we define an hook that allow to print extra profiling information
profiler_printers = []
def register_profiler_printer(fct):
    profiler_printers.append(fct)
    return fct
