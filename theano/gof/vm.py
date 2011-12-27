"""
VMs that run Theano graph computations.
"""
import logging
import sys
import time
import link
import traceback
from theano.gof.python25 import all

import theano
config = theano.config

from theano.configparser import config, AddConfigVar, BoolParam
from theano import config

logger = logging.getLogger(__name__)

AddConfigVar('profile',
        "If VM should collect profile information",
        BoolParam(False))

raise_with_op = link.raise_with_op


class VM(object):
    """
    A VM object evaluates a Theano program with its __call__ method.

    Attributes:

    call_counts - list of integers, one for each thunk. call_count[i] is the
        number of times thunks[i] was called in the course of computations
        performed by call_with_timers().

    call_times - list of floats, one for each thunk. call_times[i] is the amount
        of runtime spent on thunks[i] in the course of computations performed by
        call_with_timers().

    need_update_inputs - bool. True indicates that Function.__call__ must
        implement the feedback from output storage  to input storage. False means
        it *must not* repeat that feedback.

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
        self.call_counts = [0]*len(nodes)
        self.call_times = [0]*len(nodes)
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
        for node, thunk, t, c in zip(self.nodes, self.thunks, self.call_times, self.call_counts):
            profile.apply_time.setdefault(node,0.0)
            profile.apply_time[node] += t

            profile.apply_callcount.setdefault(node,0)
            profile.apply_callcount[node] += c

            profile.apply_cimpl[node] = hasattr(thunk,'cthunk')

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
                for i, (thunk, node) in enumerate(zip(self.thunks, self.nodes)):
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
                for thunk, node, old_storage in zip(self.thunks, self.nodes, self.post_thunk_clear):
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
                for thunk, node, old_storage in zip(self.thunks, self.nodes, self.post_thunk_clear):
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

    """

    def __init__(self, nodes, thunks, pre_call_clear,
            storage_map, compute_map,
            env, allow_gc,
            callback=None):
        super(Stack, self).__init__(nodes, thunks, pre_call_clear)

        self.allow_gc = allow_gc
        self.message = ""
        self.base_apply_stack = [o.owner for o in env.outputs if o.owner]
        self.outputs = env.outputs
        self.storage_map = storage_map
        self.apply_time = {}
        self.outputs_size = {}
        self.compute_map = compute_map
        self.node_idx = node_idx = {}
        self.callback = callback

        ords = env.orderings()

        for i, node in enumerate(self.nodes):
            node_idx[node] = i
            self.apply_time[node]     = 0
            self.outputs_size[node]   = []
            node.destroy_dependencies = []
            if node in ords:
                for prereq in ords[node]:
                    node.destroy_dependencies += prereq.outputs

        dependencies = self.dependencies = {}
        for k in storage_map:
            dependencies[k] = []
            if k.owner and k.clients:
                ls = []
                is_output = 0
                for cl in k.clients:
                    if cl[0] is not 'output':
                        ls += cl[0].outputs
                dependencies[k] += ls

        if config.profile:
            self.memory_size_map = {"nt8": 1, "t16": 2, "t32": 4, "t64": 8, "128": 16}
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
            # Make sure something happened last time round.
            # This is just a safety check to make sure the op is written correctly
            # apply_stack should either decrease in length by one (a thunk successfully applied), or
            # increase in length (added dependencies over and above the original).
            # NB: this doesn't catch cycles (would be too expensive/slow), just stalls.
            apply_stack_len = len(apply_stack)
            assert apply_stack_len != last_apply_stack_len
            last_apply_stack_len = apply_stack_len

            current_apply = apply_stack.pop()

            # Use these for loops + breaks to short circuit evaluation
            # This is a significant performance point
            computed_ins = True
            for i in current_apply.inputs:
                if not compute_map[i][0]:
                    computed_ins = False
                    break
            computed_outs = True
            for o in current_apply.outputs:
                if not compute_map[o][0]:
                    computed_outs = False
                    break
            if computed_ins:
                for d in current_apply.destroy_dependencies:
                    if not compute_map[d][0]:
                        computed_ins = False
                        break

            if not thunks[self.node_idx[current_apply]].lazy:
                # Check if all inputs are in place
                # If so compute thunk and remove it from the apply_stack
                # If not leave it in, and add to the apply_stack those that will
                # produce you those inputs

                if computed_ins and not computed_outs:
                    try:
                        _, dt = self.run_thunk_of_node(current_apply)
                        del _
                        if config.profile:
                            self.apply_time[current_apply] += dt
                            ## Computing the memory footprint of the the op
                            # ?? What about inplace .. if the op is inplace
                            # you don't actually ask for more memory!
                            size = []
                            for (idx,o) in enumerate(
                                    thunks[self.node_idx[current_apply]].outputs):
                                if not hasattr(o[0],'size'):
                                    size.append(-1)
                                    continue
                                s=o[0].size
                                dtype = str(o[0].dtype)
                                dtype2 = dtype[-3:]
                                s *= self.memory_size_map[dtype2] # KeyError here: couldn't determine the dtype memory size
                                size.append(s)
                            self.outputs_size[current_apply] = size
                    except Exception:
                        raise_with_op(current_apply)
                    for o in current_apply.outputs:
                        compute_map[o][0] = 1
                    # Garbage Collection -> check if anybody else uses this input
                    if self.allow_gc:
                        for i in current_apply.inputs:
                            if (dependencies[i] and i.owner
                                and i not in self.outputs):
                                empty_storage_map = True
                                for x in dependencies[i]:
                                    if not compute_map[x][0]:
                                        empty_storage_map = False
                                        break
                                if empty_storage_map:
                                    storage_map[i][0] = None
                elif not computed_ins:

                    apply_stack.append(current_apply)
                    apply_stack.extend(inp.owner for inp in current_apply.inputs if inp.owner)
                    apply_stack.extend(inp.owner for inp in current_apply.destroy_dependencies if inp.owner)

            elif not computed_outs:
                # Try and run it to see if it works
                try:
                    requires, dt = self.run_thunk_of_node(current_apply)
                    self.apply_time[current_apply] += dt

                except Exception:
                    raise_with_op(current_apply)

                if requires:
                    for r in requires:
                        # We are not done with this op ..
                        # so we added back and see to get the inputs we are missing
                        apply_stack.append(current_apply)
                        if current_apply.inputs[r].owner:
                            apply_stack.append(current_apply.inputs[r].owner)
                else:
                    if config.profile:
                        size = []
                        for (idx,o) in enumerate(thunks[self.node_idx[current_apply]].outputs):
                            if not hasattr(o[0], 'size'):
                                size.append(-1)
                                continue
                            s=o[0].size
                            dtype = str(o[0].dtype)
                            dtype2 = dtype[-2:]
                            s *= self.memory_size_map[dtype2] # KeyError here: couldn't determine the dtype memory size
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


try:
    import lazylinker_c
    class CVM(lazylinker_c.CLazyLinker, VM):
        def __init__(self, *args, **kwargs):
            lazylinker_c.CLazyLinker.__init__(self, *args, **kwargs)
            # skip VM.__init__
except ImportError:
    pass


class VM_Linker(link.LocalLinker):
    """
    Class that satisfies the Linker interface by acting as a VM factory.
    """

    def __init__(self, allow_gc=True, use_cloop=False, callback=None):
        """
        allow_gc - force the virtual machine to clean up unnecessary references,
            in order to allow garbage collection on intermediate values during
            computation of a function.

        use_cloop - use the C-based virtual machine if possible

        callback - a callable object to call after each call to a thunk within
            the virtual machine.  It will be called with four arguments called
            'node', 'thunk', 'storage_map', and 'compute_map'.

        """
        self.env = None
        self.allow_gc = allow_gc
        self.use_cloop = use_cloop
        self.callback = callback
        self.updated_vars = {}

    def accept(self, env, no_recycling = []):
        """
        :param env: a PerformLinker can have accepted one Env instance at a time.

        :param no_recycling: WRITEME

        :returns: self (TODO: WHY? Who calls this function?)
        """
        if self.env is not None and self.env is not env:
            return type(self)().accept(env, no_recycling)
        self.env = env
        self.no_recycling = no_recycling
        return self

    def accept_var_updates(self, updated_vars):
        self.updated_vars = updated_vars
        # This method simply records in the linker which variables have update
        # expressions.  It does not imply that the linker will actually
        # implement these updates (see need_update_inputs).  This mechanism is
        # admittedly confusing, and it could use some cleaning up. The base
        # Linker object should probably go away completely.

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
            vm = Stack(
                    nodes, thunks, pre_call_clear,
                    storage_map, compute_map,
                    self.env, self.allow_gc,
                    callback=self.callback)
        elif self.use_cloop:
            # create a map from nodes to ints and vars to ints
            nodes_idx = {}
            vars_idx = {}
            for i, node in enumerate(nodes):
                nodes_idx[node] = i
                for v in node.inputs + node.outputs:
                    vars_idx.setdefault(v, len(vars_idx))
            for v in self.env.inputs + self.env.outputs:
                vars_idx.setdefault(v, len(vars_idx))

            nodes_idx_inv = {}
            vars_idx_inv = {}
            for (node,i) in nodes_idx.items():
                nodes_idx_inv[i] = node
            for (var,i) in vars_idx.items():
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
            var_owner = [None]*len(vars_idx)
            for (var,i) in vars_idx.items():
                if var.owner:
                    var_owner[i] = nodes_idx[var.owner]

            is_lazy_list = [int(th.lazy) for th in thunks]
            output_vars = [vars_idx[v] for v in self.env.outputs]

            # builds the list of prereqs induced by e.g. destroy_handler
            ords = self.env.orderings()
            node_prereqs = []
            node_output_size = []
            for i, node in enumerate(nodes):
                node_output_size.append(0)
                prereq_var_idxs = []
                for prereq_node in ords.get(node,[]):
                    prereq_var_idxs.extend(
                            [vars_idx[v] for v in prereq_node.outputs])
                prereq_var_idxs = list(set(prereq_var_idxs))
                prereq_var_idxs.sort() # TODO: why sort?
                node_prereqs.append(prereq_var_idxs)

            update_storage = []
            for (ivar, ovar) in updated_vars.items():
                if ivar != ovar:
                    update_storage.append(vars_idx[ivar]) #dst
                    update_storage.append(vars_idx[ovar]) #src

            c0 = sys.getrefcount(node_n_inputs)
            vm = CVM(
                    nodes,
                    thunks,
                    pre_call_clear,
                    allow_gc=self.allow_gc,
                    call_counts=[0]*len(nodes),
                    call_times=[0.0]*len(nodes),
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
                    )
            assert c0 == sys.getrefcount(node_n_inputs)
        else:
            if all([(not th.lazy) for th in thunks]):
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
                vm = Stack(
                        nodes, thunks, pre_call_clear,
                        storage_map, compute_map,
                        self.env, self.allow_gc
                        )
        return vm

    def make_all(self, profiler = None, input_storage = None,
            output_storage = None,
            ):
        env = self.env
        order = list(env.toposort())
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = link.map_storage(
                env, order, input_storage, output_storage)
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
                            and (input not in env.outputs)
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
                    for input, storage in zip(env.inputs, input_storage)],
                [link.Container(output, storage, True)
                    for output, storage in zip(env.outputs, output_storage)],
                thunks,
                order)

