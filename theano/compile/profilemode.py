from __future__ import absolute_import, print_function, division
import atexit
import copy
import os
import time
import warnings

import theano
from theano.gof.link import WrapLinker
from six import string_types, iteritems, itervalues
from theano.compile.mode import (Mode, register_mode,
                                 predefined_modes, predefined_linkers,
                                 predefined_optimizers)
from theano.configparser import config
from theano.compile.function_module import FunctionMaker

from .profiling import ProfileStats

run_cthunk = None  # Will be imported only when needed.
import_time = time.time()


class Profile_Maker(FunctionMaker):
    def create(self, input_storage=None, trustme=False, storage_map=None):
        ret = super(Profile_Maker, self).create(input_storage, trustme,
                                                storage_map)

        if (hasattr(theano, 'sandbox') and
                hasattr(theano.sandbox, 'cuda') and
                theano.sandbox.cuda.cuda_enabled):
            if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '1':
                raise Exception(
                    "You are running the Theano profiler with CUDA enabled."
                    " Theano GPU ops execution is asynchronous by default."
                    " So by default, the profile is useless."
                    " You must set the environment variable"
                    " CUDA_LAUNCH_BLOCKING to 1 to tell the CUDA driver to"
                    " synchronize the execution to get a meaningful profile.")

        # create a function-specific storage container for profiling info
        profile = ProfileStats(atexit_print=False)
        self.mode.profile_stats[ret] = profile
        ret.profile = profile

        # initialize the timers
        for i, node in enumerate(ret.maker.fgraph.toposort()):
            profile.apply_time[node] = 0.0

            # a thunk_group is a list of the thunks from each linker
            # corresponding to the i'th position in the toposort.
            assert len(ret.fn.thunk_groups[i]) == 1
            profile.apply_cimpl[node] = hasattr(
                ret.fn.thunk_groups[i][0],
                'cthunk')

        # Here we replace the linker function.
        # This ugliness makes WrapLinker (an object that *generates*
        # functions and is not function-specific)  work with ProfileStats
        # objects which are function-specific.

        # capture old fn in closure. This is important since new_fn is about to
        # take its place as ret.fn.
        ret_fn = ret.fn

        def new_fn():
            self.mode.apply_time = self.mode.profile_stats[ret].apply_time
            self.mode.variable_shape = \
                self.mode.profile_stats[ret].variable_shape
            ret_fn()
            # delete the old apply_time variable
            # because it doesn't mean the same thing anymore.
            # This prevents old code from looking like it still works.
            del self.mode.apply_time
            del self.mode.variable_shape

        ret.fn = new_fn

        global run_cthunk
        if run_cthunk is None and any(profile.apply_cimpl.values()):
            # Lazy import to avoid compilation when importing theano.
            from theano.gof.cutils import run_cthunk  # noqa

        warnings.warn(
            "DEPRECATION WARNING: The ProfileMode is deprecated. "
            "Use the Theano flags/parameter to theano.function "
            "'profile=True' instead of 'mode=ProfileMode'")
        return ret


class ProfileMode(Mode):
    def __init__(self, linker=None, optimizer='default'):
        if linker is None:
            linker = config.linker
        if optimizer is 'default':
            optimizer = config.optimizer
        message = ""
        profile_stats = {}
        self.__setstate__((linker,
                           optimizer,
                           message,
                           profile_stats))

    def function_maker(self, i, o, m, *args, **kwargs):
        """
        Return an instance of `Profiler_Maker` which init the count.

        """

        assert m is self
        return Profile_Maker(i, o, self, *args, **kwargs)

    def __get_local_time(self):
        rval = 0
        for ps in itervalues(self.profile_stats):
            rval += sum(ps.apply_time.values())
        return rval
    local_time = property(__get_local_time)

    def __getstate__(self):
        # print "__getstate__",self.provided_linker,self.provided_optimizer
        return (self.provided_linker,
                self.provided_optimizer,
                self.message,
                self.profile_stats)

    def __setstate__(self, state):
        linker, optimizer, message, profile_stats = state
        self.message = message
        self.profile_stats = profile_stats

        def profile_thunk(i, node, th):
            """
            Profile only the execution time.

            """
            global run_cthunk
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                failure = run_cthunk(th.cthunk)
                dt = time.time() - t0
                if failure:
                    raise RuntimeError(
                        ('A C Op raised an exception.  ProfileMode cannot'
                         ' tell you what it was though.  Use a standard mode'
                         ' such as FAST_RUN to correct the problem.'))
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0

            # Some Op are so fast that the time.time() resolution is
            # insufficient to measure it.  So we add an epsilon.
            self.apply_time[node] += max(dt, 1e-14)

        def profile_thunk2(i, node, th):
            """
            Profile the execution time and the memory size.

            """
            global run_cthunk
            if hasattr(th, 'cthunk'):
                t0 = time.time()
                failure = run_cthunk(th.cthunk)
                dt = time.time() - t0
                if failure:
                    raise RuntimeError(
                        ('A C Op raised an exception.  ProfileMode cannot'
                         ' tell you what it was though.  Use a standard mode'
                         ' such as FAST_RUN to correct the problem.'))
            else:
                t0 = time.time()
                th()
                dt = time.time() - t0
            for var, data in zip(node.outputs, th.outputs):
                sh = getattr(data[0], 'shape', 'input no shape')
                self.variable_shape[var] = sh

            self.apply_time[node] += max(dt, 1e-14)

        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, string_types) or linker is None:
            linker = predefined_linkers[linker]

        if not config.ProfileMode.profile_memory:
            p_thunk = profile_thunk
        else:
            p_thunk = profile_thunk2
        linker = WrapLinker([linker], p_thunk)

        self.linker = linker
        if isinstance(optimizer, string_types) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        self._optimizer = optimizer

        self.call_time = 0
        self.fn_time = 0

    def print_summary(self, **kwargs):
        """
        Print 3 summaries that show where time is spent. The first shows
        an Apply-wise summary, the second an Op-wise summary and the
        third a type-Op-wise summary.

        The Apply-wise summary prints the timing information for the
        worst offending Apply nodes. This corresponds to individual Op
        applications within your graph which take the longest to
        execute (so if you use dot twice, you will see two entries
        there).

        The Op-wise summary prints the execution time of all Apply
        nodes executing the same Op grouped together and the total
        execution time per Op is shown (so if you use dot twice, you
        will see only one entry there corresponding to the sum of the
        time spent in each of them). If two Ops have different hash
        value, they will be separate.

        The type-Op-wise summary group the result by type of op. So
        event if two Op have different hash value, they will be
        merged.

        There is an hack with the Op-wise summary. Go see it if you
        want to know more.

        Parameters
        ----------
        kwargs
            They are passed to print_summary_ expanded. Currently there is
            n_apply_to_print, n_ops_to_print and min_memory_size that are
            accepted.

        """
        compile_time = sum([ps.compile_time for ps
                            in self.profile_stats.values()])

        fct_call = dict([(fn, ps.fct_callcount)
                         for (fn, ps) in iteritems(self.profile_stats)])

        fct_call_time = dict([(fn, ps.fct_call_time)
                              for (fn, ps) in iteritems(self.profile_stats)])

        apply_time = {}
        for fn, ps in iteritems(self.profile_stats):
            for (i, node) in enumerate(fn.maker.fgraph.toposort()):
                apply_time[(i, node)] = ps.apply_time[node]
        for (i, n), t in iteritems(apply_time):
            if t == 0:
                print(i, n)

        apply_cimpl = {}
        for ps in itervalues(self.profile_stats):
            apply_cimpl.update(ps.apply_cimpl)

        message = self.message

        variable_shape = {}
        for ps in itervalues(self.profile_stats):
            variable_shape.update(ps.variable_shape)

        other_time = dict(
            linker_time=sum(
                [ps.linker_time for ps in self.profile_stats.values()]),
            optimizer_time=sum(
                [ps.optimizer_time for ps in self.profile_stats.values()]))

        self.print_summary_("print_summary",
                            compile_time, fct_call_time, fct_call,
                            apply_time, apply_cimpl, message, variable_shape,
                            self.local_time, other_time,
                            **kwargs)

    def print_diff_summary(self, other, **kwargs):
        """
        As print_summary, but print the difference on two different
        profile mode.

        TODO: Also we don't print the Apply-wise summary as it don't
        work for now.
        TODO: make comparaison with gpu code.

        Parameters
        ----------
        other
            The other instance of ProfileMode that we want to be compared to.
        kwargs
            They are passed to print_summary_ expanded.
            Currently there is n_apply_to_print, n_ops_to_print and
            min_memory_size that are accepted.

        """

        def diff_dict(a_time, b_time_):
            r = {}
            b_time = copy.copy(b_time_)
            for a, ta in iteritems(a_time):
                r.setdefault(a, 0)
                tb = b_time.pop(a, 0)
                r[a] += ta - tb

            # they are missing in a
            for a, t in iteritems(b_time):
                r.setdefault(a, 0)
                r[a] += t
            return r

        compile_time = self.compile_time - other.compile_time
        fct_call_time = diff_dict(self.fct_call_time, other.fct_call_time)
        fct_call = diff_dict(self.fct_call, other.fct_call)
        apply_time = diff_dict(self.apply_time, other.apply_time)
        apply_cimpl = self.apply_cimpl and other.apply_cimpl
        message = self.message
        variable_shape = diff_dict(self.variable_shape, other.variable_shape)
        self_linker_time = sum([ps.linker_time for ps
                                in self.profile_stats.values()])
        other_linker_time = sum([ps.linker_time for ps
                                 in other.profile_stats.values()])
        self_optimizer_time = sum([ps.optimizer_time for ps
                                   in self.profile_stats.values()])
        other_optimizer_time = sum([ps.optimizer_time for ps
                                    in other.profile_stats.values()])

        other_time = {'linker_time': self_linker_time - other_linker_time,
                      'optimizer_time': self_optimizer_time -
                      other_optimizer_time}
        self.print_summary_("print_diff_summary", compile_time,
                            fct_call_time, fct_call,
                            apply_time, apply_cimpl, message, variable_shape,
                            print_apply=False, other_time=other_time,
                            **kwargs)

    @staticmethod
    def print_summary_(fct_name, compile_time, fct_call_time, fct_call,
                       apply_time, apply_cimpl, message, variable_shape,
                       local_time, other_time,
                       n_apply_to_print=config.ProfileMode.n_apply_to_print,
                       n_ops_to_print=config.ProfileMode.n_ops_to_print,
                       print_apply=True,
                       min_memory_size=config.ProfileMode.min_memory_size,
                       ):
        """
        Do the actual printing of print_summary and print_diff_summary.

        Parameters
        ----------
        n_apply_to_print
            The number of apply to print. Default 15.
        n_ops_to_print
            The number of ops to print. Default 20.
        min_memory_size
            Don't print memory profile of apply whose outputs memory size is
            lower than that.

        """

        print("ProfileMode is deprecated! Use the new profiler.")
        print(" The Theano flags to enable it ise: profile=True")
        print(" The Theano flags for the memory profile to it is: "
              "profile_memory=True")

        total_time = time.time() - import_time
        total_fct_time = sum(fct_call_time.values())
        total_fct_call = sum(fct_call.values())
        unknown_time = total_time - total_fct_time - compile_time
        overhead_time = total_fct_time - local_time
        if total_fct_time > 0:
            time_pr_in_fct = local_time / total_fct_time * 100
            overhead_time_pourcent_fct_time = (overhead_time / total_fct_time *
                                               100)
            time_per_call = total_fct_time / total_fct_call
        else:
            time_pr_in_fct = 0
            overhead_time_pourcent_fct_time = 0
            time_per_call = 0

        print()
        print('ProfileMode.%s(%s)' % (fct_name, message))
        print('---------------------------')
        print()
        print('Time since import %.3fs' % (total_time))
        print('Theano compile time: %.3fs (%.1f%% since import)' %
              (compile_time, compile_time / total_time * 100))
        print('    Optimization time: %.3fs' % (other_time['optimizer_time']))
        print('    Linker time: %.3fs' % (other_time['linker_time']))
        print('Theano fct call %.3fs (%.1f%% since import)' %
              (total_fct_time, total_fct_time / total_time * 100))
        print('   Theano Op time %.3fs %.1f%%(since import) %.1f%%'
              '(of fct call)' % (local_time, local_time / total_time * 100,
                                 time_pr_in_fct))
        print('   Theano function overhead in ProfileMode %.3fs %.1f%%'
              '(since import) %.1f%%(of fct call)' % (
                  overhead_time, overhead_time / total_time * 100,
                  overhead_time_pourcent_fct_time))
        print('%i Theano fct call, %.3fs per call' %
              (total_fct_call, time_per_call))
        print('Rest of the time since import %.3fs %.1f%%' %
              (unknown_time, unknown_time / total_time * 100))

        print()
        print('Theano fct summary:')
        print('<% total fct time> <total time> <time per call> <nb call> '
              '<fct name>')
        for key in fct_call:
            if fct_call[key] > 0:
                print('   %4.1f%% %.3fs %.2es %d %s' %
                      (fct_call_time[key] / total_fct_time * 100,
                       fct_call_time[key],
                       fct_call_time[key] / fct_call[key],
                       fct_call[key],
                       key.name))
            else:
                print('   NOT CALLED', key.name)

        # Compute stats per op.
        op_time = {}
        op_call = {}
        op_apply = {}
        op_cimpl = {}
        sop_apply = {}
        for (i, a), t in iteritems(apply_time):
            op = a.op
            op_time.setdefault(op, 0)
            op_call.setdefault(op, 0)
            op_apply.setdefault(op, 0)
            sop_apply.setdefault(type(a.op), 0)
            op_time[op] += t
            nb_call = [v for k, v in iteritems(fct_call)
                       if k.maker.fgraph is a.fgraph][0]
            op_cimpl.setdefault(a.op, True)
            op_cimpl[a.op] = op_cimpl[a.op] and apply_cimpl.get(a, False)
            if t == 0:
                assert nb_call == 0, nb_call
            else:
                op_call[op] += nb_call
                op_apply[op] += 1
                sop_apply[type(a.op)] += 1

        # Compute stats per op class
        sop_time = {}
        sop_call = {}
        sop_op = {}
        # map each op class to Bool. True iff all applies were done in c.
        sop_cimpl = {}
        for a, t in iteritems(op_time):
            typ = type(a)
            sop_time.setdefault(typ, 0)
            sop_time[typ] += t
            sop_op.setdefault(typ, 0)
            sop_op[typ] += 1
            sop_cimpl.setdefault(typ, True)
            sop_cimpl[typ] = sop_cimpl[typ] and op_cimpl.get(a, False)
            sop_call[typ] = sop_call.get(typ, 0) + op_call[a]

        # Print the summary per op class.
        print()
        print('Single Op-wise summary:')
        print('<% of local_time spent on this kind of Op> <cumulative %> '
              '<self seconds> <cumulative seconds> <time per call> [*] '
              '<nb_call> <nb_op> <nb_apply> <Op name>')
        sotimes = [(t * 100 / local_time, t, a, sop_cimpl[a], sop_call[a],
                    sop_op[a], sop_apply[a]) for a, t in iteritems(sop_time)]
        sotimes.sort()
        sotimes.reverse()
        tot = 0
        for f, t, a, ci, nb_call, nb_op, nb_apply in sotimes[:n_ops_to_print]:
            if nb_call == 0:
                assert t == 0
                continue
            tot += t
            ftot = tot * 100 / local_time
            if ci:
                msg = '*'
            else:
                msg = ' '
            print('   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %5d %2d '
                  '%2d %s' % (f, ftot, t, tot, t / nb_call, msg, nb_call,
                              nb_op, nb_apply, a))
        print('   ... (remaining %i single Op account for %.2f%%(%.2fs) of '
              'the runtime)' %
              (max(0, len(sotimes) - n_ops_to_print),
               sum(soinfo[0] for soinfo in sotimes[n_ops_to_print:]),
               sum(soinfo[1] for soinfo in sotimes[n_ops_to_print:])))

        print('(*) Op is running a c implementation')

        # The summary per op
        op_flops = {}
        for a, t in iteritems(op_time):
            if hasattr(a, 'flops'):
                op_flops[a] = a.flops * op_call[a] / t / 1e6
        flops_msg = ''
        if op_flops:
            flops_msg = ' <MFlops/s>'
            print("\nHACK WARNING: we print the flops for some OP, but the "
                  "logic doesn't always work. You need to know the "
                  "internals of Theano to make it work correctly. "
                  "Otherwise don't use it!")
        print()
        print('Op-wise summary:')
        print('<%% of local_time spent on this kind of Op> <cumulative %%> '
              '<self seconds> <cumulative seconds> <time per call> [*] %s '
              '<nb_call> <nb apply> <Op name>' % (flops_msg))

        otimes = [(t * 100 / local_time, t, a, op_cimpl.get(a, 0),
                   op_call.get(a, 0), op_apply.get(a, 0))
                  for a, t in iteritems(op_time)]
        otimes.sort()
        otimes.reverse()
        tot = 0
        for f, t, a, ci, nb_call, nb_apply in otimes[:n_ops_to_print]:
            if nb_call == 0:
                assert t == 0
                continue
            tot += t
            ftot = tot * 100 / local_time
            if ci:
                msg = '*'
            else:
                msg = ' '
            if op_flops:
                print('   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %7.1f '
                      '%5d %2d %s' % (f, ftot, t, tot, t / nb_call, msg,
                                      op_flops.get(a, -1), nb_call, nb_apply,
                                      a))
            else:
                print('   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs  %.2es %s %5d %2d '
                      '%s' % (f, ftot, t, tot, t / nb_call, msg, nb_call,
                              nb_apply, a))
        print('   ... (remaining %i Op account for %6.2f%%(%.2fs) of the '
              'runtime)' %
              (max(0, len(otimes) - n_ops_to_print),
               sum(f for f, t, a, ci, nb_call, nb_op in
                   otimes[n_ops_to_print:]),
               sum(t for f, t, a, ci, nb_call, nb_op in
                   otimes[n_ops_to_print:])))
        print('(*) Op is running a c implementation')

        if print_apply:
            print()
            print('Apply-wise summary:')
            print('<% of local_time spent at this position> <cumulative %%> '
                  '<apply time> <cumulative seconds> <time per call> [*] '
                  '<nb_call> <Apply position> <Apply Op name>')
            atimes = [(t * 100 / local_time, t, a,
                       [v for k, v in iteritems(fct_call)
                        if k.maker.fgraph is a[1].fgraph][0])
                      for a, t in iteritems(apply_time)]
            atimes.sort()
            atimes.reverse()
            tot = 0
            for f, t, a, nb_call in atimes[:n_apply_to_print]:
                tot += t
                ftot = tot * 100 / local_time
                if nb_call == 0:
                    continue
                if apply_cimpl.get(a[1], False):
                    msg = '*'
                else:
                    msg = ' '
                print('   %4.1f%%  %5.1f%%  %5.3fs  %5.3fs %.2es  %s %i  '
                      '%2i %s' %
                      (f, ftot, t, tot, t / nb_call, msg, nb_call, a[0],
                       str(a[1])))
            print('   ... (remaining %i Apply instances account for '
                  '%.2f%%(%.2fs) of the runtime)' %
                  (max(0, len(atimes) - n_apply_to_print),
                   sum(f for f, t, a, nb_call in atimes[n_apply_to_print:]),
                   sum(t for f, t, a, nb_call in atimes[n_apply_to_print:])))
            print('(*) Op is running a c implementation')
        for printer in profiler_printers:
            printer(fct_name, compile_time, fct_call_time, fct_call,
                    apply_time, apply_cimpl, message, variable_shape,
                    other_time)

        if not variable_shape:
            print("\nProfile of Theano intermediate memory disabled. "
                  "To enable, set the Theano flag ProfileMode.profile_memory "
                  "to True.")
        else:
            print("""
            The memory profile in ProfileMode is removed!
            Use the new profiler. Use the Theano flags
            profile=True,profile_memory=True to enable it.""")

        print()
        print("""Here are tips to potentially make your code run faster
(if you think of new ones, suggest them on the mailing list).
Test them first, as they are not guaranteed to always provide a speedup.""")
        from theano import tensor as T
        from theano.tensor.raw_random import RandomFunction
        import theano
        import theano.scalar as scal
        scalar_op_amdlibm_no_speed_up = [scal.LT, scal.GT, scal.LE, scal.GE,
                                         scal.EQ, scal.NEQ, scal.InRange,
                                         scal.Switch, scal.OR, scal.XOR,
                                         scal.AND, scal.Invert, scal.Maximum,
                                         scal.Minimum, scal.Add, scal.Mul,
                                         scal.Sub, scal.TrueDiv, scal.IntDiv,
                                         scal.Clip, scal.Second, scal.Identity,
                                         scal.Cast, scal.Sgn, scal.Neg,
                                         scal.Inv, scal.Sqr]
        scalar_op_amdlibm_speed_up = [scal.Mod, scal.Pow, scal.Ceil,
                                      scal.Floor, scal.RoundHalfToEven,
                                      scal.RoundHalfAwayFromZero, scal.Log,
                                      scal.Log2, scal.Log10, scal.Log1p,
                                      scal.Exp, scal.Sqrt, scal.Abs, scal.Cos,
                                      scal.Sin, scal.Tan, scal.Tanh,
                                      scal.Cosh, scal.Sinh,
                                      T.nnet.sigm.ScalarSigmoid,
                                      T.nnet.sigm.ScalarSoftplus]

        def get_scalar_ops(s):
            if isinstance(s, theano.scalar.Composite):
                l = []
                for node in s.fgraph.toposort():
                    l += get_scalar_ops(node.op)
                return l
            else:
                return [s]

        def list_scalar_op(op):
            if isinstance(op.scalar_op, theano.scalar.Composite):
                return get_scalar_ops(op.scalar_op)
            else:
                return [op.scalar_op]

        def amdlibm_speed_up(op):
            if not isinstance(op, T.Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                for s_op in l:
                    if s_op.__class__ in scalar_op_amdlibm_speed_up:
                        return True
                    elif s_op.__class__ not in scalar_op_amdlibm_no_speed_up:
                        print("We don't know if amdlibm will accelerate "
                              "this scalar op.", s_op)
                return False

        def exp_float32_op(op):
            if not isinstance(op, T.Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                return any([s_op.__class__ in [scal.Exp] for s_op in l])

        printed_tip = False
        # tip 1
        if config.floatX == 'float64':
            print("  - Try the Theano flag floatX=float32")
            printed_tip = True

        # tip 2
        if not config.lib.amdlibm and any([amdlibm_speed_up(a.op) for i, a
                                           in apply_time]):
            print("  - Try installing amdlibm and set the Theano flag "
                  "lib.amdlibm=True. This speeds up only some Elemwise "
                  "operation.")
            printed_tip = True

        # tip 3
        if not config.lib.amdlibm and any([exp_float32_op(a.op) and
                                           a.inputs[0].dtype == 'float32'
                                           for i, a in apply_time]):
            print("  - With the default gcc libm, exp in float32 is slower "
                  "than in float64! Try Theano flag floatX=float64, or "
                  "install amdlibm and set the theano flags lib.amdlibm=True")
            printed_tip = True

        # tip 4
        for a, t in iteritems(apply_time):
            node = a[1]
            if (isinstance(node.op, T.Dot) and
                    all([len(i.type.broadcastable) == 2
                         for i in node.inputs])):
                print("  - You have a dot operation that was not optimized to"
                      " dot22 (which is faster). Make sure the inputs are "
                      "float32 or float64, and are the same for both inputs. "
                      "Currently they are: %s" %
                      [i.type for i in node.inputs])
                printed_tip = True

        # tip 5
        for a, t in iteritems(apply_time):
            node = a[1]
            if isinstance(node.op, RandomFunction):
                printed_tip = True
                print("  - Replace the default random number generator by "
                      "'from theano.sandbox.rng_mrg import MRG_RandomStreams "
                      "as RandomStreams', as this is is faster. It is still "
                      "experimental, but seems to work correctly.")
                if config.device.startswith("gpu"):
                    print("     - MRG_RandomStreams is the only random number"
                          " generator supported on the GPU.")
                break

        if not printed_tip:
            print("  Sorry, no tip for today.")

    def clone(self, link_kwargs=None, optimizer="", message=None):
        """
        Create a new instance of this Mode.

        Keyword arguments can be provided for the linker, in which case its
        `clone` method will be called with these arguments.

        """
        new_linker = self.linker.clone(**link_kwargs)
        new_optimizer = optimizer
        if optimizer == "":
            new_optimizer = self.provided_optimizer
        new_mode = type(self)(linker=new_linker,
                              optimizer=new_optimizer)
        # If self is in the list or profiles to print, then add the
        # new one as well
        if self in prof_mode_instance_to_print:
            prof_mode_instance_to_print.append(new_mode)

        if message:
            new_mode.message = message

        return new_mode


register_mode('PROFILE_MODE', ProfileMode())


# needed to print the profile at the end automatically
prof_mode_instance_to_print = [predefined_modes["PROFILE_MODE"]]


def atexit_print_default_profile_mode():
    """
    Print the summary of the predefined mode ProfileMode if used.

    This all to have the summary printed at exit when config.mode=ProfileMode.

    """
    for prof_mode in prof_mode_instance_to_print:
        if prof_mode.local_time > 0:
            prof_mode.print_summary()

# Register atexit_print_default_profile_mode to have the summary of the
# predefined mode ProfileMode if it is used printed when the program terminate.
atexit.register(atexit_print_default_profile_mode)


# Here we define an hook that allow to print extra profiling information
profiler_printers = []


def register_profiler_printer(fct):
    profiler_printers.append(fct)
    return fct
