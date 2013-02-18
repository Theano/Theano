"""
VMs that run Theano graph computations.
A VM is not actually different from a Linker, we just decided
VM was a better name at some point
"""
import link
import logging
import sys
import time
import warnings

from theano.gof.python25 import all

import theano
config = theano.config

from theano.configparser import config, AddConfigVar, BoolParam, ConfigParam

logger = logging.getLogger(__name__)

AddConfigVar('profile',
        "If VM should collect profile information",
        BoolParam(False))
AddConfigVar('profile_optimizer',
        "If VM should collect optimizer profile information",
        BoolParam(False))


def filter_vm_lazy(val):
    if val == 'False' or val is False:
        return False
    elif val == 'True' or val is True:
        return True
    elif val == 'None':
        return None
    else:
        raise ValueError('Valid values for an vm.lazy parameter '
                        'should be None, False or True, not `%s`.' % val)

AddConfigVar('vm.lazy',
             "Useful only for the vm linkers. When lazy is None,"
             " auto detect if lazy evaluation is needed and use the apropriate"
             " version. If lazy is True/False, force the version used between"
             " Loop/LoopGC and Stack.",
         ConfigParam('None', filter_vm_lazy))

raise_with_op = link.raise_with_op


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

    Attributes:

    call_counts - list of integers, one for each thunk. call_count[i] is the
        number of times thunks[i] was called in the course of computations
        performed by call_with_timers().

    call_times - list of floats, one for each thunk. call_times[i] is
        the amount of runtime spent on thunks[i] in the course of
        computations performed by call_with_timers().

    need_update_inputs - bool. True indicates that Function.__call__
        must implement the feedback from output storage to input
        storage. False means it *must not* repeat that feedback.

    """
    def __init__(self, nodes, thunks, pre_call_clear):
        """
        Allocate a virtual machine.

        nodes - a list of nodes in toposort order

        thunks - a list of thunks to execute those nodes, in toposort order

        pre_call_clear - a list of containers to empty at the beginning of each
                         call.
        """
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
        # accumulate into the profile object
        for node, thunk, t, c in zip(self.nodes, self.thunks,
                                     self.call_times, self.call_counts):
            profile.apply_time.setdefault(node, 0.0)
            profile.apply_time[node] += t

            profile.apply_callcount.setdefault(node, 0)
            profile.apply_callcount[node] += c

            profile.apply_cimpl[node] = hasattr(thunk, 'cthunk')

        # clear the timer info out of the buffers
        for i in xrange(len(self.call_times)):
            self.call_times[i] = 0.0
            self.call_counts[i] = 0


class Loop(VM):
    """
    Unconditional start-to-finish program execution in Python.
    No garbage collection is allowed on intermediate results.
    """
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
                raise_with_op(node)
        else:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                for thunk, node in zip(self.thunks, self.nodes):
                    thunk()
            except:
                raise_with_op(node)


class LoopGC(VM):
    """
    Unconditional start-to-finish program execution in Python.
    Garbage collection is possible on intermediate results.
    """
    def __init__(self, nodes, thunks, pre_call_clear, post_thunk_clear):
        super(LoopGC, self).__init__(nodes, thunks, pre_call_clear)
        self.post_thunk_clear = post_thunk_clear
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
                raise_with_op(node)
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
                raise_with_op(node)


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
                 dependencies=None, callback=None):
        super(Stack, self).__init__(nodes, thunks, pre_call_clear)

        self.allow_gc = allow_gc
        self.message = ""
        self.base_apply_stack = [o.owner for o in fgraph.outputs if o.owner]
        self.outputs = fgraph.outputs
        self.storage_map = storage_map
        self.apply_time = {}
        self.outputs_size = {}
        self.compute_map = compute_map
        self.node_idx = node_idx = {}
        self.callback = callback

        ords = fgraph.orderings()

        for i, node in enumerate(self.nodes):
            node_idx[node] = i
            self.apply_time[node] = 0
            self.outputs_size[node] = []
            # XXX: inconsistent style - why modify node here rather
            #      than track destroy_dependencies with dictionary like
            #      storage_map?
            #
            # destroy_dependencies
            # --------------------
            # The destroy_dependencies is a list of variables that are implicit
            # dependencies induced by a destroy_map (compare node.inputs which
            # are *explicit* dependencies). The variables in
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

        if config.profile:
            self.memory_size_map = {"nt8": 1, "t16": 2, "t32": 4,
                                    "t64": 8, "128": 16}
            atexit.register(self.atexit_print_all)

    def run_thunk_of_node(self, node):
        """Run the thunk corresponding to Apply instance `node`

        Calls self.callback if it is defined.
        """
        idx = self.node_idx[node]
        t0 = time.time()
        rval = self.thunks[idx]()

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

    def __call__(self):
        storage_map = self.storage_map
        compute_map = self.compute_map
        thunks = self.thunks
        dependencies = self.dependencies
        for k in self.storage_map:
            compute_map[k][0] = (k.owner is None)

        # apply_stack contains nodes
        apply_stack = list(self.base_apply_stack)
        last_apply_stack_len = -1
        ls = []
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
                        if config.profile:
                            self.apply_time[current_apply] += dt
                            ## Computing the memory footprint of the the op
                            # ?? What about inplace .. if the op is inplace
                            # you don't actually ask for more memory!
                            size = []
                            for (idx, o) in enumerate(
                                    thunks[self.node_idx[
                                        current_apply]].outputs):
                                if not hasattr(o[0], 'size'):
                                    size.append(-1)
                                    continue
                                s = o[0].size
                                dtype = str(o[0].dtype)
                                dtype2 = dtype[-3:]
                                # KeyError here: couldn't determine
                                # the dtype memory size
                                s *= self.memory_size_map[dtype2]
                                size.append(s)
                            self.outputs_size[current_apply] = size
                    except Exception:
                        raise_with_op(current_apply)
                    for o in current_apply.outputs:
                        compute_map[o][0] = 1
                    if self.allow_gc:
                        for i in current_apply.inputs:
                            # Garbage Collection -> check if anybody else uses
                            # this input
                            if (dependencies[i]
                                    and i.owner
                                    and i not in self.outputs):
                                if all(compute_map[v][0]
                                        for v in dependencies[i]):
                                    storage_map[i][0] = None
                                    #DO NOT set compute_map to 0

                                    #If values become False and the
                                    #current_apply is still in the
                                    #stack, this will cause it to be
                                    #recomputed! This can cause wrong value
                                    #with some combination of inplace op.
                                    compute_map[i][0] = 2
                                    if (config.warn.vm_gc_bug and
                                        current_apply in apply_stack and
                                        getattr(current_apply.op,
                                                'destroy_map',
                                                False)):
                                        warnings.warn(
        "There was a bug that existed in the default Theano configuration,"
        " only in the development version between July 5th 2012"
        " and July 30th 2012. This was not in a released version."
        " The bug was affecting this script.",
        #The stack level is not good when inside a Scan.
        stacklevel=3
                                        )
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
                    self.apply_time[current_apply] += dt

                except Exception:
                    raise_with_op(current_apply)

                if requires:
                    for r in requires:
                        # We are not done with this op ..  so we added
                        # back and see to get the inputs we are
                        # missing
                        apply_stack.append(current_apply)
                        if current_apply.inputs[r].owner:
                            apply_stack.append(current_apply.inputs[r].owner)
                else:
                    if config.profile:
                        size = []
                        for (idx, o) in enumerate(thunks[
                                self.node_idx[current_apply]].outputs):
                            if not hasattr(o[0], 'size'):
                                size.append(-1)
                                continue
                            s = o[0].size
                            dtype = str(o[0].dtype)
                            dtype2 = dtype[-2:]
                            # KeyError here: couldn't determine the
                            # dtype memory size
                            s *= self.memory_size_map[dtype2]
                            size.append(s)
                        self.outputs_size[current_apply] = size
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
                                    #See the not lazy gc code for explanations
                                    #of compute_map change
                                    compute_map[i][0] = 2

        # Hacky coarse gc final pass
        # This is required until we have a proper gc algorithm for graphs with
        # lazy evaluation. See discussion on theano-dev June 19 2012.
        if self.allow_gc:
            for v in storage_map:
                if v.owner and not v in self.outputs:
                    storage_map[v][0] = None


try:
    import lazylinker_c

    class CVM(lazylinker_c.CLazyLinker, VM):
        def __init__(self, *args, **kwargs):
            lazylinker_c.CLazyLinker.__init__(self, *args, **kwargs)
            # skip VM.__init__
except ImportError:
    pass
except (OSError, theano.gof.cmodule.MissingGXX):
    # OSError happens when g++ is not installed.  In that case, we
    # already changed the default linker to something else then CVM.
    # Currently this is the py linker.
    # Here we assert that the default linker is not cvm.
    assert not [x for x in theano.configparser._config_var_list
                if x.fullname == 'linker'][0].default.startswith('cvm')
    pass


class VM_Linker(link.LocalLinker):
    """
    Class that satisfies the Linker interface by acting as a VM factory.
    """

    def __init__(self, allow_gc=None, use_cloop=False, callback=None,
                 lazy=None, schedule=None):
        """
        allow_gc - force the virtual machine to clean up unnecessary
            references, in order to allow garbage collection on
            intermediate values during computation of a function.
            If None use as default the value of the Theano flag allow_gc.

        use_cloop - use the C-based virtual machine if possible

        callback - a callable object to call after each call to a thunk within
            the virtual machine.  It will be called with four arguments called
            'node', 'thunk', 'storage_map', and 'compute_map'.

        lazy - Useful only when use_cloop is False. When lazy is None, use the
            theano flag vm.lazy value. Then if we have a None (default) we auto
            detect if lazy evaluation is needed and use the apropriate
            version. If lazy is True or False, we force the version used
            between Loop/LoopGC and Stack.

        """
        # Note: if more parameters are added to __init__, make sure to forward
        # them in the "type(self)(...)" call in the "accept" method below.
        if allow_gc is None:
            allow_gc = config.allow_gc
        self.fgraph = None
        self.allow_gc = allow_gc
        self.use_cloop = use_cloop
        self.callback = callback
        self.lazy = lazy
        self.updated_vars = {}
        self.schedule = schedule or self.schedule

    def accept(self, fgraph, no_recycling=None):
        """
        :param fgraph: a PerformLinker can have accepted one FunctionGraph
            instance at a time.

        :param no_recycling: WRITEME

        :returns: self if fgraph is the first FunctionGraph that has ever been
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
                    lazy=self.lazy,
                    schedule=self.schedule
                    ).accept(fgraph, no_recycling)
        self.fgraph = fgraph
        self.no_recycling = no_recycling
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
        * K will not need to be computed
        * if K is already computed, it can be released for garbage collection


        Parameters
        ----------
        variables - iterable over the variables used in a graph computation.


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
            if k.owner and k.clients:
                ls = []
                is_output = 0
                for cl in k.clients:
                    if cl[0] is not 'output':
                        ls += cl[0].outputs
                dependencies[k] += ls
        return dependencies

    def make_vm(self, nodes, thunks,
            input_storage, output_storage, storage_map,
            post_thunk_clear,
            computed,
            compute_map,
            updated_vars
            ):

        pre_call_clear = [storage_map[v] for v in self.no_recycling]

        if self.callback is not None:
            if self.use_cloop:
                logger.warn('CLoop does not support callback, using Stack VM.')
            deps = None
            if self.allow_gc:
                deps = self.compute_gc_dependencies(storage_map)
            vm = Stack(
                    nodes, thunks, pre_call_clear,
                    storage_map, compute_map,
                    self.fgraph, self.allow_gc,
                    dependencies=deps,
                    callback=self.callback)
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
            for (node, i) in nodes_idx.items():
                nodes_idx_inv[i] = node
            for (var, i) in vars_idx.items():
                vars_idx_inv[i] = var

            # put storage_map and compute_map into a int-based scheme
            n_applies = len(nodes)
            storage_map_list = [storage_map[vars_idx_inv[i]]
                    for i in xrange(len(vars_idx_inv))]
            compute_map_list = [compute_map[vars_idx_inv[i]]
                    for i in xrange(len(vars_idx_inv))]
            if nodes:
                assert type(storage_map_list[0]) is list
                assert type(compute_map_list[0]) is list

            if self.allow_gc:
                dependency_map = self.compute_gc_dependencies(storage_map)
                dependency_map_list = [
                    [vars_idx[d] for d in dependency_map[vars_idx_inv[i]]]
                    for i in xrange(len(vars_idx_inv))]
            else:
                dependency_map_list = None

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
            for (var, i) in vars_idx.items():
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
            for (ivar, ovar) in updated_vars.items():
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
                            post_thunk_clear)
                else:
                    vm = Loop(
                            nodes,
                            thunks,
                            pre_call_clear)
            else:
                deps = None
                if self.allow_gc:
                    deps = self.compute_gc_dependencies(storage_map)
                vm = Stack(
                        nodes, thunks, pre_call_clear,
                        storage_map, compute_map,
                        self.fgraph, self.allow_gc,
                        dependencies=deps
                        )
        return vm

    def make_all(self, profiler=None, input_storage=None,
                 output_storage=None,
                ):
        fgraph = self.fgraph
        order = self.schedule(fgraph)
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = link.map_storage(
                fgraph, order, input_storage, output_storage)
        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = [node.op.make_thunk(node,
                    storage_map,
                    compute_map,
                    no_recycling)
                        for node in order]

        computed, last_user = link.gc_helper(order)
        if self.allow_gc:
            post_thunk_clear = []
            for node in order:
                clear_after_this_thunk = []
                for input in node.inputs:
                    if ((input in computed)
                            and (input not in fgraph.outputs)
                            and (node == last_user[input])):
                        clear_after_this_thunk.append(storage_map[input])
                post_thunk_clear.append(clear_after_this_thunk)
        else:
            post_thunk_clear = None

        vm = self.make_vm(order, thunks,
                input_storage, output_storage, storage_map,
                post_thunk_clear,
                computed,
                compute_map,
                self.updated_vars
                )

        return (vm,
                [link.Container(input, storage)
                 for input, storage in zip(fgraph.inputs, input_storage)],
                [link.Container(output, storage, True)
                 for output, storage in zip(fgraph.outputs, output_storage)],
                thunks,
                order)
