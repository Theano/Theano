"""
VMs that run Theano graph computations.

A VM is not actually different from a Linker, we just decided
VM was a better name at some point.

"""
from __future__ import absolute_import, print_function, division

from . import link
from collections import defaultdict
import logging
import sys
import time
import warnings

from theano.configparser import (config, _config_var_list)

import theano.gof.cmodule

from six import iteritems, itervalues
from six.moves import xrange

logger = logging.getLogger(__name__)


def calculate_reallocate_info(order, fgraph, storage_map, compute_map_re,
                              dependencies):
    """
    WRITEME : explain the parameters
    """
    reallocated_info = {}
    viewed_by = {}
    for var in fgraph.variables:
        viewed_by[var] = []
    view_of = {}
    pre_allocated = set([])
    allocated = set([])

    for idx in range(len(order)):
        node = order[idx]
        dmap = getattr(node.op, 'destroy_map', None)
        vmap = getattr(node.op, 'view_map', None)

        idx_o = 0
        for out in node.outputs:
            for var in node.outputs:
                compute_map_re[var][0] = 1
            ins = None
            if dmap and idx_o in dmap:
                idx_v = dmap[idx_o]
                assert len(idx_v) == 1, ("Here we only support the possibility"
                                         " to destroy one input")
                ins = node.inputs[idx_v[0]]
            if vmap and idx_o in vmap:
                assert ins is None
                idx_v = vmap[idx_o]
                assert len(idx_v) == 1, ("Here we only support the possibility"
                                         " to view one input")
                ins = node.inputs[idx_v[0]]
            if ins is not None:
                assert isinstance(ins, theano.Variable)
                origin = view_of.get(ins, ins)
                view_of[out] = origin
                viewed_by[origin].append(out)
            idx_o += 1

        for ins in node.inputs:
            assert not (ins in view_of and viewed_by[ins])
            if (getattr(ins, 'ndim', None) == 0 and not storage_map[ins][0] and
                    ins not in fgraph.outputs and ins.owner and
                    all([compute_map_re[v][0]
                         for v in dependencies.get(ins, [])]) and
                    ins not in allocated):
                # Constant Memory cannot be changed
                # Constant and shared variables' storage_map value is not empty
                reuse_out = None
                if ins not in view_of and not viewed_by.get(ins, []):
                    # where gc
                    for i in range(idx + 1, len(order)):
                        if reuse_out is not None:
                            break
                        for out in order[i].outputs:
                            if (getattr(out, 'ndim', None) == 0 and
                                    out not in pre_allocated and
                                    ins.type == out.type):
                                reuse_out = out
                                pre_allocated.add(out)
                                allocated.add(ins)
                                break
                elif ins in view_of:
                    origin = view_of[ins]
                    if ins in viewed_by[origin]:
                        viewed_by[origin].remove(ins)
                    if (not viewed_by[origin] and
                            origin not in fgraph.inputs and
                            not isinstance(origin, theano.Constant)):
                        # where gc
                        for i in range(idx + 1, len(order)):
                            if reuse_out is not None:
                                break
                            for out in order[i].outputs:
                                if (getattr(out, 'ndim', None) == 0 and
                                        out not in pre_allocated and
                                        ins.type == out.type):
                                    reuse_out = out
                                    pre_allocated.add(out)
                                    allocated.add(ins)
                                    break
                if reuse_out is not None:
                    reallocated_info[ins] = [ins, reuse_out]

    return reallocated_info


class VM(object):
    """
    A VM object's __call__ method evaluates a Theano program.

    The Stack should be considered the reference VM/Linker implementation.
    It can correctly evaluate all graphs and is the easiest to read. The CVM
    is a port of Stack, and should have the same behavior, but run faster.
    The CVM's code is harder to read though.

    The other python VMs are maybe not necessary anymore, and don't take
    advantage of lazy computation, though they still produce the correct
    output for lazy nodes.

    Parameters
    ----------
    nodes
        A list of nodes in toposort order.
    thunks
        A list of thunks to execute those nodes, in toposort order.
    pre_call_clear
        A list of containers to empty at the beginning of each call.

    Attributes
    ----------
    call_counts
        List of integers, one for each thunk. call_count[i] is the number of
        times thunks[i] was called in the course of computations performed by
        call_with_timers().
    call_times
        List of floats, one for each thunk. call_times[i] is the amount of
        runtime spent on thunks[i] in the course of computations performed by
        call_with_timers().

    need_update_inputs : bool
        True indicates that Function.__call__ must implement the feedback from
        output storage to input storage. False means it *must not* repeat that
        feedback.

    """

    def __init__(self, nodes, thunks, pre_call_clear):

        if len(nodes) != len(thunks):
            raise ValueError()
        self.nodes = nodes
        self.thunks = thunks
        self.pre_call_clear = pre_call_clear
        self.call_counts = [0] * len(nodes)
        self.call_times = [0] * len(nodes)
        self.time_thunks = False

        # This variable (self.need_update_inputs) is overshadowed by
        # CLazyLinker in CVM which has an attribute of the same name that
        # defaults to 0 (aka False).
        self.need_update_inputs = True

    def __call__(self):
        """
        Run the machine.

        Postcondition - all output variables have been computed.  VMs vary in
        what exactly this means and how it is done.

        """
        raise NotImplementedError('override me')

    def clear_storage(self):
        """
        Free any internal references to temporary variables.

        Free internal variables and outputs.  Essentially, free as much memory
        as possible without intefering with the ability to evaluate subsequent
        calls.

        """
        raise NotImplementedError('override me')

    def update_profile(self, profile):
        """
        Accumulate into the profile object
        """
        for node, thunk, t, c in zip(self.nodes, self.thunks,
                                     self.call_times, self.call_counts):
            profile.apply_time.setdefault(node, 0.0)
            profile.apply_time[node] += t

            profile.apply_callcount.setdefault(node, 0)
            profile.apply_callcount[node] += c

            profile.apply_cimpl[node] = hasattr(thunk, 'cthunk')

        if hasattr(self, 'variable_shape'):
            profile.variable_shape = self.variable_shape.copy()
            profile.variable_strides = self.variable_strides.copy()
            profile.variable_offset = self.variable_offset.copy()

        if hasattr(self, 'node_executed_order'):
            profile.node_executed_order = self.node_executed_order[:]

        if hasattr(self, 'node_cleared_order'):
            profile.node_cleared_order = self.node_cleared_order[:]

        if hasattr(self, 'dependencies'):
            profile.dependencies = self.dependencies

        # clear the timer info out of the buffers
        for i in xrange(len(self.call_times)):
            self.call_times[i] = 0.0
            self.call_counts[i] = 0


class Loop(VM):
    """
    Unconditional start-to-finish program execution in Python.
    No garbage collection is allowed on intermediate results.

    """
    # Some other part of Theano query that information
    allow_gc = False

    def __call__(self):
        if self.time_thunks:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                for i, (thunk, node) in enumerate(zip(self.thunks,
                                                      self.nodes)):
                    t0 = time.time()
                    thunk()
                    t1 = time.time()
                    self.call_counts[i] += 1
                    self.call_times[i] += t1 - t0
            except:
                link.raise_with_op(node, thunk)
        else:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                for thunk, node in zip(self.thunks, self.nodes):
                    thunk()
            except:
                link.raise_with_op(node, thunk)


class LoopGC(VM):
    """
    Unconditional start-to-finish program execution in Python.
    Garbage collection is possible on intermediate results.

    """

    def __init__(self, nodes, thunks, pre_call_clear, post_thunk_clear):
        super(LoopGC, self).__init__(nodes, thunks, pre_call_clear)
        self.post_thunk_clear = post_thunk_clear
        # Some other part of Theano query that information
        self.allow_gc = True
        if not (len(nodes) == len(thunks) == len(post_thunk_clear)):
            raise ValueError()

    def __call__(self):
        if self.time_thunks:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                i = 0
                for thunk, node, old_storage in zip(self.thunks,
                                                    self.nodes,
                                                    self.post_thunk_clear):
                    t0 = time.time()
                    thunk()
                    t1 = time.time()
                    self.call_counts[i] += 1
                    self.call_times[i] += t1 - t0
                    for old_s in old_storage:
                        old_s[0] = None
                    i += 1
            except:
                link.raise_with_op(node, thunk)
        else:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                for thunk, node, old_storage in zip(self.thunks, self.nodes,
                                                    self.post_thunk_clear):
                    thunk()
                    for old_s in old_storage:
                        old_s[0] = None
            except:
                link.raise_with_op(node, thunk)


class Stack(VM):
    """
    Finish-to-start evalution order of thunks.

    This supports lazy evaluation of subtrees and partial
    computations of graphs when only some inputs have changed.

    At a pseudo-code level, the basic idea is the following:

    def recursively_evaluate(var):
        if var is up to date:
            return
        if var.owner.inputs are up to date:
            update var
            return
        for input in var.owner.unputs:
            recursively_evaluate(var)

    for output in outputs:
        recursively_evaluate(output)

    The actual logic is more complex to support intermediate
    garbage collection, lazily-evaluated nodes, and better speed.

    """

    def __init__(self, nodes, thunks, pre_call_clear,
                 storage_map, compute_map, fgraph, allow_gc,
                 n_updates, dependencies=None, callback=None,
                 callback_input=None):
        super(Stack, self).__init__(nodes, thunks, pre_call_clear)

        self.allow_gc = allow_gc
        self.message = ""
        self.base_apply_stack = [o.owner for o in fgraph.outputs if o.owner]
        self.outputs = fgraph.outputs
        self.storage_map = storage_map
        self.variable_shape = {}  # Variable -> shape
        self.variable_strides = {}  # Variable -> strides
        self.variable_offset = {}  # Variable -> offset
        self.compute_map = compute_map
        self.node_idx = node_idx = {}
        self.callback = callback
        self.callback_input = callback_input
        self.n_updates = n_updates

        ords = fgraph.orderings()

        for i, node in enumerate(self.nodes):
            node_idx[node] = i

            # XXX: inconsistent style - why modify node here rather
            #      than track destroy_dependencies with dictionary like
            #      storage_map?
            #
            # destroy_dependencies
            # --------------------
            # The destroy_dependencies is a list of variables that are implicit
            # dependencies induced by destroy_map and view_map (compared to
            # node.inputs which are *explicit* dependencies). The variables in
            # destroy_dependencies would be impossible to compute after the
            # current `node` runs, because node.thunk() is going to destroy a
            # common input variable needed by whatever node owns each variable
            # in destroy_depenencies.

            node.destroy_dependencies = []
            if node in ords:
                for prereq in ords[node]:
                    node.destroy_dependencies += prereq.outputs

        self.dependencies = dependencies

        if self.allow_gc and self.dependencies is None:
            raise ValueError("Must set dependencies when using GC")

    def run_thunk_of_node(self, node):
        """
        Run the thunk corresponding to Apply instance `node`.

        Calls self.callback if it is defined.

        """
        idx = self.node_idx[node]
        t0 = time.time()
        rval = self.thunks[idx]()
        self.node_executed_order.append(node)

        # Some thunks on some computers run faster than the granularity
        # of the time.time clock.
        # Profile output looks buggy if a node has run but takes 0 time.
        # (and profile code might hide real bugs if it rounds up 0)
        dt = max(time.time() - t0, 1e-10)
        if self.callback is not None:
            self.callback(
                node=node,
                thunk=self.thunks[idx],
                storage_map=self.storage_map,
                compute_map=self.compute_map,
            )
        return rval, dt

    def __call__(self, output_subset=None):
        storage_map = self.storage_map
        compute_map = self.compute_map
        thunks = self.thunks
        dependencies = self.dependencies
        self.node_executed_order = []
        self.node_cleared_order = []

        for cont in self.pre_call_clear:
            cont[0] = None

        for k in self.storage_map:
            compute_map[k][0] = (k.owner is None)
            if self.callback_input and compute_map[k][0]:
                self.callback_input(k, self.storage_map[k][0])

        # apply_stack contains nodes
        if output_subset is not None:
            first_updated = len(self.outputs) - self.n_updates
            output_subset = output_subset + list(range(first_updated,
                                                       len(self.outputs)))
            apply_stack =\
                [self.outputs[i].owner for i in output_subset
                    if self.outputs[i].owner]
        else:
            apply_stack = list(self.base_apply_stack)

        last_apply_stack_len = -1

        # This record all function inputs/shared variables and constants
        for var, data in iteritems(self.storage_map):
            if data[0] is None:
                continue
            if hasattr(var.type, 'get_shape_info'):
                sh = var.type.get_shape_info(data[0])
            else:
                sh = 'no shape'
            self.variable_shape[var] = sh
            st = getattr(data[0], 'strides', 'no strides')
            if getattr(data[0], 'flags', False) and data[0].flags.c_contiguous:
                st = 'c'
            elif (hasattr(data[0], 'is_c_contiguous') and
                  data[0].is_c_contiguous()):
                st = "c"
            self.variable_strides[var] = st
            off = getattr(data[0], 'offset', '')
            self.variable_offset[var] = off

        while apply_stack:
            # Make sure something happened last time round.  This is
            # just a safety check to make sure the op is written
            # correctly apply_stack should either decrease in length
            # by one (a thunk successfully applied), or increase in
            # length (added dependencies over and above the original).
            # NB: this doesn't catch cycles (would be too expensive/slow),
            #     just stalls.
            apply_stack_len = len(apply_stack)
            assert apply_stack_len != last_apply_stack_len
            last_apply_stack_len = apply_stack_len

            current_apply = apply_stack.pop()
            current_inputs = current_apply.inputs
            current_outputs = current_apply.outputs
            current_deps = current_inputs + current_apply.destroy_dependencies

            computed_ins = all(compute_map[v][0] for v in current_deps)
            computed_outs = all(compute_map[v][0] for v in current_outputs)

            if not thunks[self.node_idx[current_apply]].lazy:
                #
                # stack loop: Normal Non-Lazy Case
                # ================================
                #
                # Check if all inputs are in place
                # If so compute thunk and remove it from the apply_stack
                # If not leave it in, and add to the apply_stack those
                # that will produce you those inputs

                if computed_ins and not computed_outs:
                    # -- Non-lazy case: have inputs, time to compute outputs
                    try:
                        _, dt = self.run_thunk_of_node(current_apply)
                        del _
                        if config.profile or config.print_global_stats:
                            current_idx = self.node_idx[current_apply]
                            self.call_counts[current_idx] += 1
                            self.call_times[current_idx] += dt
                            # Computing the memory footprint of the the op
                            # ?? What about inplace .. if the op is inplace
                            # you don't actually ask for more memory!
                            for (idx, o) in enumerate(
                                    thunks[self.node_idx[
                                        current_apply]].outputs):
                                var = self.nodes[current_idx].outputs[idx]
                                if hasattr(var.type, 'get_shape_info'):
                                    sh = var.type.get_shape_info(o[0])
                                else:
                                    sh = 'no shape'
                                self.variable_shape[var] = sh
                                st = getattr(o[0], 'strides',
                                             'no strides')
                                if (getattr(o[0], 'flags', False) and
                                        o[0].flags.c_contiguous):
                                    st = 'c'
                                elif (hasattr(o[0], 'is_c_contiguous') and
                                      o[0].is_c_contiguous()):
                                    st = "c"
                                self.variable_strides[var] = st
                                off = getattr(o[0], 'offset', '')
                                self.variable_offset[var] = off
                    except Exception:
                        link.raise_with_op(
                            current_apply,
                            self.thunks[self.node_idx[current_apply]],
                            storage_map=storage_map)
                    for o in current_apply.outputs:
                        compute_map[o][0] = 1

                    input_index = []
                    # A list store the index of inputs variables

                    if self.allow_gc:
                        for i in current_apply.inputs:
                            # Garbage Collection -> check if anybody else uses
                            # this input
                            if (dependencies[i] and
                                    i.owner and
                                    i not in self.outputs):
                                if all(compute_map[v][0]
                                        for v in dependencies[i]):
                                    storage_map[i][0] = None
                                    input_index.append(
                                        current_apply.inputs.index(i))

                                    # DO NOT set compute_map to 0

                                    # If values become False and the
                                    # current_apply is still in the
                                    # stack, this will cause it to be
                                    # recomputed! This can cause wrong value
                                    # with some combination of inplace op.
                                    compute_map[i][0] = 2
                                    if (config.warn.vm_gc_bug and
                                        current_apply in apply_stack and
                                        getattr(current_apply.op,
                                                'destroy_map',
                                                False)):
                                        warnings.warn(
                                            "There was a bug that existed in "
                                            "the default Theano configuration,"
                                            " only in the development version "
                                            "between July 5th 2012 and "
                                            "July 30th 2012. This was not in "
                                            "a released version. The bug was "
                                            "affecting this script.",
                                            # The stack level is not good when
                                            # inside a Scan.
                                            stacklevel=3
                                        )
                    self.node_cleared_order.append(input_index)

                elif not computed_ins:
                    # -- Non-lazy case, need inputs
                    apply_stack.append(current_apply)
                    apply_stack.extend(inp.owner
                                       for inp in current_deps
                                       if inp.owner)

            elif not computed_outs:
                #
                # stack loop: Lazy Evaluation Case
                # ================================
                #
                # Lazy evaluation protocol is to run the thunk with the
                # current storage_map and compute_map accessed via closure,
                # and the thunk will return a list of variables from its input
                # list that it requires.

                try:
                    requires, dt = self.run_thunk_of_node(current_apply)
                    current_idx = self.node_idx[current_apply]
                    self.call_counts[current_idx] += 1
                    self.call_times[current_idx] += dt

                except Exception:
                    link.raise_with_op(
                        current_apply,
                        self.thunks[self.node_idx[current_apply]],
                        storage_map=storage_map)

                if requires:
                    for r in requires:
                        # We are not done with this op ..  so we added
                        # back and see to get the inputs we are
                        # missing
                        apply_stack.append(current_apply)
                        if current_apply.inputs[r].owner:
                            apply_stack.append(current_apply.inputs[r].owner)
                else:
                    if config.profile or config.print_global_stats:
                        for (idx, o) in enumerate(thunks[
                                self.node_idx[current_apply]].outputs):
                            var = self.nodes[
                                self.node_idx[current_apply]].outputs[idx]

                            if hasattr(var.type, 'get_shape_info'):
                                sh = var.type.get_shape_info(o[0])
                            else:
                                sh = 'no shape'
                            self.variable_shape[var] = sh
                            st = getattr(o[0], 'strides', 'no strides')
                            if (getattr(o[0], 'flags', False) and
                                    o[0].flags.c_contiguous):
                                st = 'c'
                            elif (hasattr(o[0], 'is_c_contiguous') and
                                  o[0].is_c_contiguous()):
                                st = "c"
                            self.variable_strides[var] = st
                            off = getattr(o[0], 'offset', '')
                            self.variable_offset[var] = off

                    input_index = []

                    if self.allow_gc:
                        for i in current_apply.inputs:
                            if (dependencies[i] and i.owner and
                                    i not in self.outputs):
                                empty_storage_map = True
                                for x in dependencies[i]:
                                    if not compute_map[x][0]:
                                        empty_storage_map = False
                                        break
                                if empty_storage_map:
                                    storage_map[i][0] = None
                                    input_index.append(
                                        current_apply.inputs.index(i))
                                    # See the not lazy gc code for explanations
                                    # of compute_map change
                                    compute_map[i][0] = 2

                    self.node_cleared_order.append(input_index)

        # Hacky coarse gc final pass
        # This is required until we have a proper gc algorithm for graphs with
        # lazy evaluation. See discussion on theano-dev June 19 2012.
        final_index = []

        if self.allow_gc:
            for v in storage_map:
                if v.owner and v not in self.outputs:
                    if compute_map[v][0] == 2:
                        continue
                    else:
                        storage_map[v][0] = None
                        final_index.append(v)
                        compute_map[v][0] = 2

        self.node_cleared_order.append(final_index)


try:
    # If cxx is explicitely set to an empty string, we do not want to import neither lazylinker C code
    # nor lazylinker compiled C code from cache.
    if not theano.config.cxx:
        raise theano.gof.cmodule.MissingGXX('lazylinker will not be imported if theano.config.cxx is not set.')
    from . import lazylinker_c

    class CVM(lazylinker_c.CLazyLinker, VM):

        def __init__(self, *args, **kwargs):
            lazylinker_c.CLazyLinker.__init__(self, *args, **kwargs)
            # skip VM.__init__
except ImportError:
    pass
except (OSError, theano.gof.cmodule.MissingGXX) as e:
    # OSError happens when g++ is not installed.  In that case, we
    # already changed the default linker to something else then CVM.
    # Currently this is the py linker.
    # Here we assert that the default linker is not cvm.
    assert not [x for x in _config_var_list
                if x.fullname == 'linker'][0].default.startswith('cvm'), e
    pass


class VM_Linker(link.LocalLinker):
    """
    Class that satisfies the Linker interface by acting as a VM factory.

    Parameters
    ----------
    allow_gc
        Force the virtual machine to clean up unnecessary
        references, in order to allow garbage collection on
        intermediate values during computation of a function.
        If None use as default the value of the Theano flag allow_gc.
    use_cloop
        Use the C-based virtual machine if possible
    callback
        A callable object to call after each call to a thunk within
        the virtual machine.  It will be called with four arguments called
        'node', 'thunk', 'storage_map', and 'compute_map'.
    callback_input
        A callable object to call on each input to the graph
        (variables with no owner).  This includes constants and shared
        variables values.  It will be called with two arguments:
        'var', 'value'.
    lazy
        Useful only when use_cloop is False. When lazy is None, use the
        theano flag vm.lazy value. Then if we have a None (default) we auto
        detect if lazy evaluation is needed and use the appropriate
        version. If lazy is True or False, we force the version used
        between Loop/LoopGC and Stack.
    c_thunks
        If None or True, don't change the default. If False,
        don't compile c code for the thunks.
    allow_partial_eval
        If True, enforces usage of Stack or CVM, to allow for partial
        evaluation of functions (calculating a subset of outputs).

    """

    def __init__(self, allow_gc=None, use_cloop=False, callback=None,
                 callback_input=None, lazy=None, schedule=None,
                 c_thunks=None, allow_partial_eval=None):
        # Note: if more parameters are added to __init__, make sure to forward
        # them in the "type(self)(...)" call in the "accept" method below.
        if allow_gc is None:
            allow_gc = config.allow_gc
        self.fgraph = None
        self.allow_gc = allow_gc
        self.use_cloop = use_cloop
        self.callback = callback
        self.callback_input = callback_input
        self.lazy = lazy
        if c_thunks is None:
            c_thunks = bool(theano.config.cxx)
        self.c_thunks = c_thunks
        self.allow_partial_eval = allow_partial_eval
        self.updated_vars = {}
        if schedule:
            self.schedule = schedule

    def accept(self, fgraph, no_recycling=None, profile=None):
        """Check if fgraph is the first FunctionGraph that has ever been
        associated to self, else, create a new VM_Linker
        associated to fgraph

        Parameters
        ----------
        fgraph
            A PerformLinker can have accepted one FunctionGraph instance
            at a time.

        no_recycling

            no_recycling is a list of storage (list of 1 element, the
            value corresponding to one variable). Those variable
            storage should not be reused after the call that created
            them.

            This happen for example for output of the graph that we
            give to the user. We don't want to reuse those object in
            case the user have kept it.

            VM_Linker make sure this happen by setting the list
            element to None at the start of each call.

            Older Linker use not exactly the same mechanism. They will
            also modify the c code to don't look up the value in the
            storage. This cause duplicate c code compilation for the
            same op if they are in the middle of the graph or in the
            no_recycling. We don't want that, so compile all c code
            the same (middle of the graph vs output).

            TODO: change the logic to remove the reference at the end
            of the call instead of the start. This will request all VM
            implementation (Loop, LoopGC, Stack, CVM).__call__ to
            return the user outputs as Function.__call__ won't be able
            to find them anymore.

        Returns
        -------
        Self if fgraph is the first FunctionGraph that has ever been
        associated to self, else, a new VM_Linker associated to fgraph.

        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            # Build a new VM_Linker, and call accept on that one.
            # Warning: make sure to forward the correct values of
            # all parameters to __init__ here.
            return type(self)(
                allow_gc=self.allow_gc,
                use_cloop=self.use_cloop,
                callback=self.callback,
                callback_input=self.callback_input,
                lazy=self.lazy,
                schedule=self.schedule,
                c_thunks=self.c_thunks,
                allow_partial_eval=self.allow_partial_eval
            ).accept(fgraph, no_recycling, profile)
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        self.profile = profile

        return self

    def accept_var_updates(self, updated_vars):
        self.updated_vars = updated_vars
        # This method simply records in the linker which variables have update
        # expressions.  It does not imply that the linker will actually
        # implement these updates (see need_update_inputs).  This mechanism is
        # admittedly confusing, and it could use some cleaning up. The base
        # Linker object should probably go away completely.

    def compute_gc_dependencies(self, variables):
        """
        Returns dict: variable K -> list of variables [v1, v2, v3, ...]
        for each K in variables.

        The variables v1, v2, ... are the full set of variables that depend
        directly on K. When we know that none of them will need to be
        computed, we know that:
        * K will not need to be computed.
        * If K is already computed, it can be released for garbage collection.

        Parameters
        ----------
        variables
            Iterable over the variables used in a graph computation.

        Notes
        -----
        It doesn't take care of the view_map/destroy_map. So it means it relies
        on Python gc no to free the object real storage.

        N.B. gc means garbage collection

        """
        dependencies = {}
        for k in variables:
            dependencies[k] = []
            # If k has no owner, it is an input / constant and its value
            # should not be removed from the storage_map because we have no
            # way of getting it back.
            #
            # XXX if k has no clients... what is it doing in the computation?
            # Fred guess: it could happen for node with multiple outputs when
            # we don't use all outputs.

            if k.owner and k.clients:
                ls = []
                for cl in k.clients:
                    if cl[0] != 'output':
                        ls += cl[0].outputs
                dependencies[k] += ls
        return dependencies

    def make_vm(self, nodes, thunks,
                input_storage, output_storage, storage_map,
                post_thunk_clear,
                computed,
                compute_map,
                updated_vars,
                ):

        pre_call_clear = [storage_map[v] for v in self.no_recycling]

        if (self.callback is not None or self.callback_input is not None or
                ((config.profile or config.print_global_stats) and config.profile_memory) or
                (self.allow_partial_eval and not self.use_cloop)):

            if self.use_cloop and (self.callback is not None or
                                   self.callback_input is not None):
                logger.warn('CVM does not support callback, using Stack VM.')
            if self.use_cloop and config.profile_memory:
                warnings.warn(
                    'CVM does not support memory profile, using Stack VM.')
            if not self.use_cloop and self.allow_partial_eval:
                warnings.warn(
                    'LoopGC does not support partial evaluation, '
                    'using Stack VM.')
            # Needed for allow_gc=True, profiling and storage_map reuse
            deps = self.compute_gc_dependencies(storage_map)
            vm = Stack(
                nodes, thunks, pre_call_clear,
                storage_map, compute_map,
                self.fgraph, self.allow_gc,
                len(updated_vars),
                dependencies=deps,
                callback=self.callback,
                callback_input=self.callback_input)
        elif self.use_cloop:
            # create a map from nodes to ints and vars to ints
            nodes_idx = {}
            vars_idx = {}
            for i, node in enumerate(nodes):
                nodes_idx[node] = i
                for v in node.inputs + node.outputs:
                    vars_idx.setdefault(v, len(vars_idx))
            for v in self.fgraph.inputs + self.fgraph.outputs:
                vars_idx.setdefault(v, len(vars_idx))

            nodes_idx_inv = {}
            vars_idx_inv = {}
            for (node, i) in iteritems(nodes_idx):
                nodes_idx_inv[i] = node
            for (var, i) in iteritems(vars_idx):
                vars_idx_inv[i] = var

            # put storage_map and compute_map into a int-based scheme
            storage_map_list = [storage_map[vars_idx_inv[i]]
                                for i in xrange(len(vars_idx_inv))]
            compute_map_list = [compute_map[vars_idx_inv[i]]
                                for i in xrange(len(vars_idx_inv))]
            if nodes:
                assert type(storage_map_list[0]) is list
                assert type(compute_map_list[0]) is list

            # Needed for allow_gc=True, profiling and storage_map reuse
            dependency_map = self.compute_gc_dependencies(storage_map)
            dependency_map_list = [
                [vars_idx[d] for d in dependency_map[vars_idx_inv[i]]]
                for i in xrange(len(vars_idx_inv))]

            # build the pointers to node inputs and offsets
            base_input_output_list = []
            node_n_inputs = []
            node_n_outputs = []
            node_input_offset = []
            node_output_offset = []
            for node in nodes:
                inputs_idx = [vars_idx[v] for v in node.inputs]
                outputs_idx = [vars_idx[v] for v in node.outputs]
                node_n_inputs.append(len(inputs_idx))
                node_n_outputs.append(len(outputs_idx))
                node_input_offset.append(len(base_input_output_list))
                base_input_output_list.extend(inputs_idx)
                node_output_offset.append(len(base_input_output_list))
                base_input_output_list.extend(outputs_idx)

            # build the var owner array
            var_owner = [None] * len(vars_idx)
            for (var, i) in iteritems(vars_idx):
                if var.owner:
                    var_owner[i] = nodes_idx[var.owner]

            is_lazy_list = [int(th.lazy) for th in thunks]
            output_vars = [vars_idx[v] for v in self.fgraph.outputs]

            # builds the list of prereqs induced by e.g. destroy_handler
            ords = self.fgraph.orderings()
            node_prereqs = []
            node_output_size = []
            for i, node in enumerate(nodes):
                node_output_size.append(0)
                prereq_var_idxs = []
                for prereq_node in ords.get(node, []):
                    prereq_var_idxs.extend(
                        [vars_idx[v] for v in prereq_node.outputs])
                prereq_var_idxs = list(set(prereq_var_idxs))
                prereq_var_idxs.sort()  # TODO: why sort?
                node_prereqs.append(prereq_var_idxs)

            # Builds the list of input storage to update (according to update
            # rules) when the outputs are computed.
            # They are in the same order as the second part of output_vars
            # (output_vars contains first the returned outputs, then the
            # values of the update expressions).
            update_storage = []
            update_in_from_out = {}
            for (ivar, ovar) in iteritems(updated_vars):
                update_in_from_out[vars_idx[ovar]] = vars_idx[ivar]
            for oidx in output_vars:
                if oidx in update_in_from_out:
                    update_storage.append(update_in_from_out[oidx])

            c0 = sys.getrefcount(node_n_inputs)
            vm = CVM(
                nodes,
                thunks,
                pre_call_clear,
                allow_gc=self.allow_gc,
                call_counts=[0] * len(nodes),
                call_times=[0.0] * len(nodes),
                compute_map_list=compute_map_list,
                storage_map_list=storage_map_list,
                base_input_output_list=base_input_output_list,
                node_n_inputs=node_n_inputs,
                node_n_outputs=node_n_outputs,
                node_input_offset=node_input_offset,
                node_output_offset=node_output_offset,
                var_owner=var_owner,
                is_lazy_list=is_lazy_list,
                output_vars=output_vars,
                node_prereqs=node_prereqs,
                node_output_size=node_output_size,
                update_storage=update_storage,
                dependencies=dependency_map_list,
            )
            assert c0 == sys.getrefcount(node_n_inputs)
        else:
            lazy = self.lazy
            if lazy is None:
                lazy = config.vm.lazy
            if lazy is None:
                lazy = not all([(not th.lazy) for th in thunks])
            if not lazy:
                # there is no conditional in the graph
                if self.allow_gc:
                    vm = LoopGC(
                        nodes,
                        thunks,
                        pre_call_clear,
                        post_thunk_clear,
                    )
                else:
                    vm = Loop(
                        nodes,
                        thunks,
                        pre_call_clear,
                    )
            else:
                # Needed when allow_gc=True and profiling
                deps = self.compute_gc_dependencies(storage_map)
                vm = Stack(
                    nodes, thunks, pre_call_clear,
                    storage_map, compute_map,
                    self.fgraph, self.allow_gc,
                    len(updated_vars),
                    dependencies=deps,
                )
        return vm

    def make_all(self, profiler=None, input_storage=None,
                 output_storage=None, storage_map=None,
                 ):
        fgraph = self.fgraph
        order = self.schedule(fgraph)

        input_storage, output_storage, storage_map = link.map_storage(
            fgraph, order, input_storage, output_storage, storage_map)
        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = []

        # Collect Reallocation Info
        compute_map_re = defaultdict(lambda: [0])
        for var in fgraph.inputs:
            compute_map_re[var][0] = 1

        if getattr(fgraph.profile, 'dependencies', None):
            dependencies = getattr(fgraph.profile, 'dependencies')
        else:
            dependencies = self.compute_gc_dependencies(storage_map)

        reallocated_info = calculate_reallocate_info(
            order, fgraph, storage_map, compute_map_re, dependencies)
        t0 = time.time()
        linker_make_thunk_time = {}
        impl = None
        if self.c_thunks is False:
            impl = 'py'
        for node in order:
            try:
                thunk_start = time.time()
                # no-recycling is done at each VM.__call__ So there is
                # no need to cause duplicate c code by passing
                # no_recycling here.
                thunks.append(node.op.make_thunk(node,
                                                 storage_map,
                                                 compute_map,
                                                 [],
                                                 impl=impl))
                linker_make_thunk_time[node] = time.time() - thunk_start
                if not hasattr(thunks[-1], 'lazy'):
                    # We don't want all ops maker to think about lazy Ops.
                    # So if they didn't specify that its lazy or not, it isn't.
                    # If this member isn't present, it will crash later.
                    thunks[-1].lazy = False
            except Exception as e:
                e.args = ("The following error happened while"
                          " compiling the node", node, "\n") + e.args
                raise
        t1 = time.time()

        if self.profile:
            self.profile.linker_node_make_thunks += t1 - t0
            self.profile.linker_make_thunk_time = linker_make_thunk_time

        for node, thunk in zip(order, thunks):
            thunk.inputs = [storage_map[v] for v in node.inputs]
            thunk.outputs = [storage_map[v] for v in node.outputs]

        lazy = self.lazy
        if lazy is None:
            lazy = config.vm.lazy
        if lazy is None:
            lazy = not all([(not th.lazy) for th in thunks])
        if not (lazy or ((config.profile or config.print_global_stats) and config.profile_memory) or
                self.use_cloop or self.callback or self.callback_input):
            for pair in itervalues(reallocated_info):
                storage_map[pair[1]] = storage_map[pair[0]]

        computed, last_user = link.gc_helper(order)
        if self.allow_gc:
            post_thunk_clear = []
            for node in order:
                clear_after_this_thunk = []
                for input in node.inputs:
                    if (input in computed and
                            input not in fgraph.outputs and
                            node == last_user[input] and
                            input not in reallocated_info):
                        clear_after_this_thunk.append(storage_map[input])
                post_thunk_clear.append(clear_after_this_thunk)
        else:
            post_thunk_clear = None

        vm = self.make_vm(order, thunks,
                          input_storage, output_storage, storage_map,
                          post_thunk_clear,
                          computed,
                          compute_map,
                          self.updated_vars,
                          )

        vm.storage_map = storage_map
        vm.compute_map = compute_map

        return (vm,
                [link.Container(input, storage)
                 for input, storage in zip(fgraph.inputs, input_storage)],
                [link.Container(output, storage, True)
                 for output, storage in zip(fgraph.outputs, output_storage)],
                thunks,
                order)

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'c_thunks'):
            self.c_thunks = True
        if not hasattr(self, 'allow_partial_eval'):
            self.allow_partial_eval = None
        if not hasattr(self, 'callback_input'):
            self.callback_input = None
