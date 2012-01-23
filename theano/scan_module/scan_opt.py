"""
This module provides optimizations for scan
"""


__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin "
               "Arnaud Bergeron ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import logging
import numpy
import sys

import theano
from theano import tensor
from theano.tensor import opt, get_constant_value
from theano import gof
from theano.compile import optdb
from theano import config
from theano.compile.function_module import deep_copy_op

import scan_op
import scan_utils
from scan_utils import equal_computations, find_up, scan_args
from theano.gof.opt import pre_constant_merge, pre_greedy_local_optimizer

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_opt')

list_opt_slice = [tensor.opt.local_abs_merge,
                  tensor.opt.local_mul_switch_sink,
                  tensor.opt.local_upcast_elemwise_constant_inputs,
                  tensor.opt.local_remove_switch_const_cond,
                  tensor.opt.constant_folding]


def warning(*msg):
    _logger.warning('WARNING theano.scan: ' + ' '.join(msg))


def info(*msg):
    _logger.info('INFO theano.scan: ' + ' '.join(msg))


@gof.local_optimizer([None])
def remove_constants_and_unused_inputs_scan(node):
    '''
    Move constants into the inner graph, and remove unused inputs.

    Constants that are in the outer graph are represented by a free symbolic
    variable in the inner graph. If we move them into the inner graph,
    constant-folding can happen in the inner graph.
    This is applied only on sequences and non-sequences,
    not on initial states.
    '''
    if not isinstance(node.op, scan_op.Scan):
        return False
    op = node.op
    # We only need to take care of sequences and other arguments
    st = op.n_seqs
    st += int(numpy.sum([len(x) for x in
                     op.tap_array[:(op.n_mit_mot + op.n_mit_sot)]]))
    st += op.n_sit_sot
    st += op.n_shared_outs
    op_ins, op_outs = scan_utils.reconstruct_graph(op.inputs, op.outputs)

    # Corresponds to the initial states, which should stay untouched.
    # We put those variables aside, and put them back at the end.
    out_stuff_inner = op_ins[op.n_seqs:st]

    non_seqs = op_ins[st:]
    st = (op.n_seqs +
          op.n_mit_mot +
          op.n_mit_sot +
          op.n_sit_sot +
          op.n_nit_sot +
          op.n_shared_outs + 1)
    outer_non_seqs = node.inputs[st:]
    out_stuff_outer = node.inputs[1 + op.n_seqs:st]

    # To replace constants in the outer graph by clones in the inner graph
    givens = {}
    # All the inputs of the inner graph of the new scan
    nw_inner = []
    # Same for the outer graph, initialized w/ number of steps
    nw_outer = [node.inputs[0]]

    all_ins = gof.graph.inputs(op_outs)
    for idx in xrange(op.n_seqs):
        if (isinstance(node.inputs[idx + 1], tensor.TensorConstant) and
            node.inputs[idx + 1].tag.unique_value is not None):
            try:
                # This works if input is a constant that has all entries
                # equal
                val = tensor.get_constant_value(node.inputs[idx + 1])
                givens[op_ins[idx]] = node.inputs[idx + 1].clone()[0]
            except TypeError:
                pass
        elif op_ins[idx] in all_ins:
            nw_inner += [op_ins[idx]]
            nw_outer += [node.inputs[idx + 1]]

    nw_n_seqs = len(nw_inner)
    # Add outputs stuff
    nw_inner += out_stuff_inner
    nw_outer += out_stuff_outer
    # Look through non sequences
    for nw_in, nw_out in zip(non_seqs, outer_non_seqs):
        if isinstance(nw_out, tensor.Constant):
            givens[nw_in] = nw_out.clone()
        elif nw_in in all_ins:
            nw_inner += [nw_in]
            nw_outer += [nw_out]

    if len(nw_inner) != len(op_ins):
        op_outs = scan_utils.clone(op_outs, replace=givens)
        nw_info = op.info.copy()
        nw_info['n_seqs'] = nw_n_seqs
        # DEBUG CHECK
        nwScan = scan_op.Scan(nw_inner, op_outs, nw_info)
        nw_outs = nwScan.make_node(*nw_outer).outputs
        return nw_outs
    else:
        return False

scan_seqopt = theano.gof.SequenceDB()
# We run before blas opt at 1.7 and specialize 2.0
# but after stabilize at 1.5. Should we put it before stabilize?
optdb.register('scan_seqopt', scan_seqopt, 1.6, 'fast_run', 'scan')
scan_seqopt.register('scanOp_remove_constants_and_unused_inputs',
                     opt.in2out(remove_constants_and_unused_inputs_scan,
                                ignore_newtrees=True),
                     5,
                     'fast_run',
                     'scan')


# This is a global opt for historical reason
# It should be possible to change it to a local opt.
class PushOutNonSeqScan(gof.Optimizer):

    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(gof.toolbox.ReplaceValidate())

    def apply(self, env):
        nodelist = [x for x in env.toposort() if isinstance(x.op,
                                                           scan_op.Scan)]
        for node in nodelist:
            self.process_node(env, node)

    def process_node(self, env, node):
        # this flag tells if there was any change during the last iterations
        changed = True
        clean_inputs, clean_outputs = scan_utils.reconstruct_graph(
                        node.op.inputs, node.op.outputs)

        local_env = gof.Env(clean_inputs, clean_outputs)
        max_iterations = 2 * len(local_env.toposort()) + 3
        counts = 0
        to_remove = []
        to_replace = []
        replace_with_in = []
        replace_with_out = []
        op = node.op
        # Construct the list of non_sequences to simplify a few things
        st = op.n_seqs
        st += int(numpy.sum([len(x) for x in
                             op.tap_array[:(op.n_mit_mot + op.n_mit_sot)]]))
        st += op.n_sit_sot
        st += op.n_shared_outs
        non_seqs = clean_inputs[st:]
        st = (op.n_seqs +
              op.n_mit_mot +
              op.n_mit_sot +
              op.n_sit_sot +
              op.n_nit_sot +
              op.n_shared_outs + 1)
        outer_non_seqs = node.inputs[st:]
        assert len(non_seqs) == len(outer_non_seqs)
        while changed and counts < max_iterations:
            counts += 1
            changed = False

            for nd in local_env.toposort():
                if (numpy.all([(x in non_seqs) or
                               (x.owner in to_remove) or
                               isinstance(x, tensor.Constant)
                                 for x in nd.inputs]) and
                        # we can do this because the assumption is that a
                        # viewOp or deepCopyOp will be just at the end of the
                        # function and not somewhere in the middle ..
                        not isinstance(nd.op, theano.compile.ViewOp) and
                        not isinstance(nd.op, theano.compile.DeepCopyOp) and
                        # and we didn't already looked at this node
                        not nd in to_remove):

                    # We have a candidate node to removable
                    # Step 1. Reconstruct it on outside
                    to_remove.append(nd)
                    outside_ins = []
                    for x in nd.inputs:
                        if x in non_seqs:
                            outside_ins += [outer_non_seqs[non_seqs.index(x)]]
                        elif x in to_replace:
                            outside_ins += [
                                replace_with_out[to_replace.index(x)]]
                        elif isinstance(x, theano.Constant):
                            outside_ins += [x.clone()]
                        else:
                            raise Exception(
                                ('Error in the `scan_pushout_non_seq_'
                                 'operations`. The optimization tries '
                                 'to move some computation fron scan '
                                 'which is not allowed to move. Report '
                                 'this on theano-users list'), x)
                    nw_outer_node = nd.op.make_node(*outside_ins)
                    # Step 2. Create variables for replacements
                    for idx, y in enumerate(nd.outputs):

                        y_place_holder = scan_utils.safe_new(y, '_replace')
                        to_replace += [y]
                        replace_with_in += [y_place_holder]
                        assert type(y) == type(nw_outer_node.outputs[idx])
                        replace_with_out += [nw_outer_node.outputs[idx]]
                    changed = True
        if counts >= max_iterations:
            raise Exception('Error in the `scan_pushout_non_seq_operations`.'
                            ' The optimization exhausted the maximal number '
                            'of iterations allowed!')
        # We need to check all candidate replacements and choose those that
        # make sense for us

        # Step 1. which elements of `to_replace` are used by remaining
        # components of the inner function
        clean_to_replace = []
        clean_replace_with_in = []
        clean_replace_with_out = []
        existent_nodes = [nd for nd in local_env.toposort()
                            if nd not in to_remove]
        to_keep = []
        for nd in existent_nodes:
            to_keep += nd.inputs
        for idx, out in enumerate(to_replace):
            if out in to_keep and out.owner not in existent_nodes:
                clean_to_replace += [out]
                clean_replace_with_in += [replace_with_in[idx]]
                clean_replace_with_out += [replace_with_out[idx]]

        if len(clean_to_replace) > 0:
            # We can finally put an end to all this madness
            givens = {}
            nw_outer = []
            nw_inner = []
            for to_repl, repl_in, repl_out in zip(clean_to_replace,
                                              clean_replace_with_in,
                                              clean_replace_with_out):
                if isinstance(repl_out, theano.Constant):
                    repl_in = repl_out.clone()
                else:
                    nw_inner += [repl_in]
                    nw_outer += [repl_out]
                givens[to_repl] = repl_in

            _op_outs = scan_utils.clone(clean_outputs,
                                        replace=givens)
            _op_ins = clean_inputs + nw_inner
            op_ins, op_outs = scan_utils.reconstruct_graph(_op_ins, _op_outs)
            # Reconstruct node
            nwScan = scan_op.Scan(op_ins, op_outs, op.info)
            nw_node = nwScan.make_node(* (node.inputs + nw_outer))
            env.replace_all_validate(zip(node.outputs, nw_node.outputs),
                                     reason='scan_push_computation_out')
            return True
        elif to_keep == []:
            # Nothing in the inner graph should be kept
            replace_with = {}
            for idx, out in enumerate(to_replace):
                if out in local_env.outputs:
                    x = node.outputs[local_env.outputs.index(out)]
                    y = replace_with_out[idx]
                    shape = [y.shape[idx] for idx in xrange(y.ndim)]
                    replace_with[x] = tensor.alloc(y,
                                                   node.inputs[0],
                                                   *shape)

            # We need to add one extra dimension to the outputs
            env.replace_all_validate(replace_with.items(),
                                     reason='scan_push_computation_out')

        else:
            return False


scan_seqopt.register('scanOp_pushout_nonseqs_ops',
                     PushOutNonSeqScan(),
                     1,
                     'fast_run',
                     'scan')


@gof.local_optimizer([None])
def scan_make_inplace(node):
    op = node.op
    if (isinstance(op, scan_op.Scan) and
        (not op.info['inplace']) and
        (not op.info['gpu'])):
        info = op.info.copy()
        info['inplace'] = True
        # inputs corresponding to sequences and n_steps
        ls_begin = node.inputs[:1 + op.n_seqs]
        ls = op.outer_mitmot(node.inputs)
        ls += op.outer_mitsot(node.inputs)
        ls += op.outer_sitsot(node.inputs)
        ls_end = op.outer_shared(node.inputs)
        ls_end += op.outer_nitsot(node.inputs)
        ls_end += op.outer_non_seqs(node.inputs)
        n_outs = len(ls)
        for idx in xrange(n_outs):
            if ls[idx] in ls[:idx]:
                ls[idx] = deep_copy_op(ls[idx])

        inputs = ls_begin + ls + ls_end
        new_op = scan_op.Scan(op.inputs,
                              op.outputs,
                              info)
        return new_op.make_node(*inputs).outputs
    return False

optdb.register('scanOp_make_inplace',
               opt.in2out(scan_make_inplace, ignore_newtrees=True),
               75,
               'fast_run',
               'inplace',
               'scan')


class ScanSaveMem(gof.Optimizer):
    """ Graph Optimizer that reduces scan memory consumption """
    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(gof.toolbox.ReplaceValidate())

    def process_node(self, env, node):

        # helpful functions
        def select_min(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return tensor.minimum(x, y)

        def select_max(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return tensor.maximum(x, y)

        def sanitize(x):
            if x is None:
                return None
            else:
                return tensor.as_tensor_variable(x)

        if hasattr(env, 'shape_feature'):
            shape_of = node.env.shape_feature.shape_of
        else:
            # Each access to shape_of is in a try..except block in order to
            # use a default version when the variable is not in the shape_of
            # dictionary.
            shape_of = {}
        # 1. Initialization of variables
        # Note 1) We do not actually care about outputs representing shared
        # variables (those have no intermediate values) so it is safer to
        # ignore them and not change them in any way. To simplify the
        # optimizations I construct the variable ``c_outs`` ( that counts
        # outputs up to those we care) and the list ``init_l`` which for any
        # output we care says the length of its initial state. Note that
        # defining ``init_l`` for mit_mot sequences is a bit trickier but
        # it is safe to set it to 0
        op = node.op
        c_outs = op.n_mit_mot + op.n_mit_sot + op.n_sit_sot + op.n_nit_sot

        init_l = [0 for x in xrange(op.n_mit_mot)]
        init_l += [abs(numpy.min(v)) for v in op.tap_array[op.n_mit_mot:]]
        init_l += [0 for x in xrange(op.n_nit_sot)]
        # 2. Check the clients of each output and see for how many steps
        # does scan need to run

        # This comparison checks if there is any uncounted output, which
        # can only be an output corresponding to a shared variable

        # 2.1 Initialize
        # global_nsteps is a dictionary having two fields ( 'real' deals
        # with int values, 'sym' with symbolic ones) or None
        # given that a scan op has k outputs o_1, .. o_k and each
        # output has n_j clients c_1^1, c_1^2, .. c_1^{n_1}, c_2^1, ..,
        # global_nsteps is None if any of the clients is different
        # from a subtensor or its real and sym field equal to
        # max(c_i_j.idx_list[0].stop), meaning store up to which maximal
        # index(step) for any output scan actually needs to compute
        # In other words n_steps should be equal to this maximal !
        # Note: if we have a shared variable that gets updated at every step
        # of the loop, reducing the number of steps will affect the the
        # value of the shared variable after the loop so we need not to
        # change the number of steps in that case. To do this we set
        # global_nsteps to None which is seen as a flag that nothing needs
        # to be done
        assert len(node.outputs) >= c_outs
        if len(node.outputs) == c_outs:
            global_nsteps = {'real': -1, 'sym': []}
        else:
            global_nsteps = None

        # Keeps track of the original slices that each client represent
        slices = [None for o in node.outputs]

        # A list for each output indicating how many intermediate values
        # should be stored. If negative it means none of the intermediate
        # values (i.e. the output can be removed since it is not used
        # afterwards in the computations), if 0 it means that all
        # intermediate values are required, otherwise is up to that number
        # of intermediate values
        # Note that for mit_mot outputs and shared outputs we can not change
        # the number of intermediate steps stored without affecting the
        # result of the op
        store_steps = [0 for o in xrange(op.n_mit_mot)]
        store_steps += [-1 for o in node.outputs[op.n_mit_mot:c_outs]]
        # Flag that says if an input has changed and we need to do something
        # or not
        flag_store = False

        # 2.2 Loop over the clients
        for i, out in enumerate(node.outputs[:c_outs]):
            # look at all its clients
            slices[i] = []
            for cl, _ in out.clients:

                # 2.1 outputs of the function
                #=> output needs all its intermediate values
                if type(cl) == str:
                    # if the node is actually an output, then
                    # we need to store the entire thing
                    global_nsteps = None
                    slices[i] = None
                    break
                # 2.2 non-subtensor nodes
                #=> output needs all its intermediate values
                elif not isinstance(cl.op, tensor.basic.Subtensor):
                    global_nsteps = None
                    slices[i] = None
                    break
                # 2.3 subtensor nodes
                #=> output might need to store just a subset of its values
                else:
                    # 2.3.1 extract idx list of subtensor
                    this_slice = tensor.basic.get_idx_list(cl.inputs,
                                                     cl.op.idx_list)
                    if this_slice == None:
                        # if unable to extract idx_list
                        #=> outputs needs all its intermediate values
                        global_nsteps = None
                        slices[i] = None
                        break

                    # 2.3.2 extract the begin/end of the first dimension
                    if i >= op.n_mit_mot:
                        try:
                            length = shape_of[out][0]
                        except KeyError:
                            length = node.inputs[0] + init_l[i]
                    else:
                        try:
                            length = shape_of[out][0]
                        except KeyError:
                            length = out.shape[0]
                    cf_slice = tensor.basic.get_canonical_form_slice(
                                                    this_slice[0], length)
                    slices[i] += [(cf_slice, this_slice)]

                    if (isinstance(this_slice[0], slice) and
                        this_slice[0].stop is None):
                        global_nsteps = None
                        break
                    if isinstance(cf_slice[0], slice):
                        stop = tensor.basic.extract_constant(cf_slice[0].stop)
                    else:
                        stop = tensor.basic.extract_constant(cf_slice[0]) + 1
                    if stop == sys.maxint or stop == length:
                        stop = None
                    else:
                        # there is a **gotcha** here ! Namely, scan returns an
                        # array that contains the initial state of the output
                        # as well. Which means that if have a initial state of
                        # length 3, and you look for 5 steps you get an output
                        # y of length 8. If you only use y[:5], this does not
                        # mean that you only need to loop for 5 steps but
                        # actually only for 2 steps ( the first 3 are the
                        # initial state)
                        stop = stop - init_l[i]

                    # 2.3.3 we might get away with less number of steps
                    if stop is not None and global_nsteps is not None:
                        # yes if it is a tensor
                        if isinstance(stop, tensor.Variable):
                            global_nsteps['sym'] += [stop]
                        # not if it is maxint
                        elif (type(stop) is int and stop == sys.maxint):
                            global_nsteps = None
                        # yes if it is a int k, 0 < k < maxint
                        elif (type(stop) is int and
                              global_nsteps['real'] < stop):
                            global_nsteps['real'] = stop
                        # yes if it is a int k, 0 < k < maxint
                        elif (type(stop) is int and stop > 0):
                            pass
                        # not otherwise
                        else:
                            global_nsteps = None

        # 2.3. Analyze global_nsteps to figure out for how many steps scan
        # needs to iterate
        if global_nsteps is not None:
            nw_steps = node.inputs[0]

            # there are some symbolic tensors that limit the number of
            # steps
            if len(global_nsteps['sym']) == 0:
                sym_steps = None
            else:
                sym_steps = global_nsteps['sym'][0]
                for c in global_nsteps['sym'][1:]:
                    sym_steps = tensor.maximum(sym_steps, c)

            if global_nsteps['real'] >= 0:
                real_steps = global_nsteps['real']
            else:
                real_steps = None
            nw_steps = select_min(select_max(sym_steps, real_steps),
                                  node.inputs[0])
        else:
            nw_steps = node.inputs[0]
            global_nsteps = None

        # 2.4 Loop over the clients again now looking just to see how many
        # intermediate steps to store
        for i, out in enumerate(node.outputs[:c_outs]):
            # look at all its clients
            for cl, _ in out.clients:
                if type(cl) == str:
                    store_steps[i] = 0
                    break
                elif not isinstance(cl.op, tensor.basic.Subtensor):
                    store_steps[i] = 0
                    break
                else:
                    this_slice = tensor.basic.get_idx_list(cl.inputs,
                                                         cl.op.idx_list)
                    if this_slice == None:
                        store_steps[i] = 0
                        break

                    if (isinstance(this_slice[0], slice) and
                        this_slice[0].start is None):
                        store_steps[i] = 0
                        break

                    if i > op.n_mit_mot:
                        length = node.inputs[0] + init_l[i]
                    else:
                        try:
                            length = shape_of[out][0]
                        except KeyError:
                            length = out.shape[0]
                    cf_slice = tensor.basic.get_canonical_form_slice(
                                                    this_slice[0], length)

                    if isinstance(cf_slice[0], slice):
                        start = tensor.basic.extract_constant(
                            cf_slice[0].start)
                    else:
                        start = tensor.basic.extract_constant(cf_slice[0])
                    if start == 0 or store_steps[i] == 0:
                        store_steps[i] = 0
                    else:
                        pval = select_max(nw_steps - start + init_l[i],
                                          init_l[i])
                        if store_steps[i] != -1:
                            pval = select_max(pval, store_steps[i])

                        store_steps[i] = pval
                        flag_store = True

        orphane_outs = [i for i, x in enumerate(store_steps)
                        if (type(x) is int) and (x < 0)]
        flag_store = flag_store or (len(orphane_outs) > 0)
        # 3. is there anything to change ?
        if (flag_store or global_nsteps is not None):
            # 3.1 initialize inputs for the new scan
            old_outputs = []
            nw_inputs = list(node.inputs)
            nw_inputs[0] = nw_steps

            # 3.2 check orphane outputs to see if we can eliminate any
            required, not_required = \
                    scan_utils.scan_can_remove_outs(node.op,
                                                    orphane_outs)
            # 3.3. compose replace pairs for those nodes that need not
            # to store everything in memory ( or ar orphane and required
            # by the inner function .. )
            replaced_outs = []
            offset = 1 + op.n_seqs + op.n_mit_mot
            for idx, _val in enumerate(store_steps[op.n_mit_mot:]):
                i = idx + op.n_mit_mot
                if not(type(_val) is int and _val <= 0 and i not in required):

                    if idx + op.n_mit_mot in required:
                        val = 1
                    else:
                        val = _val
                    # If the memory for this output has been pre-allocated
                    # before going into the scan op (by an alloc node)
                    if idx < op.n_mit_sot + op.n_sit_sot:
                        # In case the input is still an alloc node, we
                        # actually have two options:
                        #   a) the input is a set_subtensor, in that case we
                        #      can replace the initial tensor by a slice,
                        #   b) it is not, and we simply take a slice of it.
                        if (nw_inputs[offset + idx].owner and
                            isinstance(nw_inputs[offset + idx].owner.op,
                                       tensor.IncSubtensor)):
                            _nw_input = nw_inputs[offset + idx].owner.inputs[1]
                            val = tensor.as_tensor_variable(val)
                            initl = tensor.as_tensor_variable(init_l[i])
                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                    tensor.maximum(val - initl, 0))
                            tmp = pre_constant_merge([tmp])[0]
                            nw_input = scan_utils.expand(_nw_input, tmp)
                        else:
                            tmp = tensor.as_tensor_variable(val)
                            initl = tensor.as_tensor_variable(init_l[i])
                            tmp = tensor.maximum(tmp, initl)
                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                                             tmp)
                            tmp = pre_constant_merge([tmp])[0]
                            nw_input = nw_inputs[offset + idx][:tmp]

                        nw_inputs[offset + idx] = nw_input
                        replaced_outs.append(op.n_mit_mot + idx)
                        odx = op.n_mit_mot + idx
                        old_outputs += [(odx, [x[0].outputs[0] for x in
                                        node.outputs[odx].clients])]
                    # If there is no memory pre-allocated for this output
                    elif idx < op.n_mit_sot + op.n_sit_sot + op.n_nit_sot:

                        pos = (op.n_mit_mot + idx + op.n_seqs +
                               1 + op.n_shared_outs)
                        if nw_inputs[pos] == node.inputs[0]:
                            nw_inputs[pos] = val
                        odx = op.n_mit_mot + idx
                        replaced_outs.append(odx)
                        old_outputs += [(odx, [x[0].outputs[0] for x in
                                        node.outputs[odx].clients])]
            # 3.4. Recompute inputs for everything else based on the new
            # number of steps
            if global_nsteps is not None:
                for idx, val in enumerate(store_steps[op.n_mit_mot:]):
                    if val == 0:
                        if idx < op.n_mit_sot + op.n_sit_sot:
                            _nw_input = nw_inputs[offset + idx].owner.inputs[1]
                            odx = op.n_mit_mot + idx
                            nw_input = scan_utils.expand(_nw_input, nw_steps)
                            nw_inputs[offset + idx] = nw_input
                        elif idx < (op.n_mit_sot + op.n_sit_sot +
                                    op.n_nit_sot):
                            in_idx = offset + idx + op.n_shared_outs
                            if nw_inputs[in_idx] == node.inputs[0]:
                                nw_inputs[in_idx] = nw_steps
                            odx = op.n_mit_mot + idx

            # 3.5 Remove unwanted orphane outputs
            (inps, outs, info, node_ins, compress_map) = \
                    scan_utils.compress_outs(op, not_required, nw_inputs)
            inv_compress_map = {}
            for k, v in compress_map.items():
                inv_compress_map[v] = k

            node_ins = [pre_greedy_local_optimizer(list_opt_slice, x) for x in
                        node_ins]
            node_ins = pre_constant_merge(node_ins)
            # 3.6 Compose the new scan
            # I need to make sure I'm not reapplying the same optimization
            # twice since bad things usually happen if I do that
            info['_scan_merge_visited'] = True
            new_outs = scan_op.Scan(inps,
                                    outs,
                                    info).make_node(*node_ins).outputs

            old_new = []
            # 3.7 Get replace pairs for those outputs that do not change
            # the number of intermediate steps stored
            for idx, sl in enumerate(slices):
                if global_nsteps and sl is not None and store_steps[idx] == 0:
                    for hdx, cl in enumerate(node.outputs[idx].clients):
                        cnf_slice, old_slices = sl[hdx]
                        # Sanitize the nw_slice by converting ints back into
                        # constants :) I only need to do this for the first
                        # slice since that is the only slice

                        if isinstance(cnf_slice[0], slice):
                            fslice = slice(
                                sanitize(cnf_slice[0].start),
                                sanitize(cnf_slice[0].stop),
                                sanitize(cnf_slice[0].step))
                        else:
                            fslice = sanitize(cnf_slice[0])

                        nw_slice = (fslice,) + tuple(old_slices[1:])
                        nw_pos = inv_compress_map[idx]
                        nw_out = new_outs[nw_pos]

                        subtens = tensor.basic.Subtensor(nw_slice)
                        # slice inputs
                        sl_ins = tensor.basic.Subtensor.collapse(
                            nw_slice,
                            lambda entry: isinstance(entry,
                                                    tensor.Variable))
                        new_o = subtens.make_node(new_outs[nw_pos],
                                                  *sl_ins).outputs[0]
                        if new_o.ndim > 0:
                            new_o = new_o[::cnf_slice[1]]
                        replaced_outs.append(idx)
                        old_new += [(cl[0].outputs[0], new_o)]
            # 3.8. Get replace pairs for those outputs that change
            # the number of stored intermediate steps
            for pos, old_outs in old_outputs:
                if len(old_outs) > 0:
                    nw_pos = compress_map[pos]
                    nw_out = new_outs[nw_pos]
                    for k, old in enumerate(old_outs):
                        # Get the correct slice
                        cnf_slice, old_slices = slices[pos][k]
                        if type(cnf_slice[0]) is slice:
                            start = (cnf_slice[0].start - nw_steps -
                                     init_l[pos] + store_steps[pos])
                            if (cnf_slice[0].stop is not None and
                                cnf_slice[0].stop != sys.maxint):
                                stop = (cnf_slice[0].stop - nw_steps -
                                        init_l[pos] + store_steps[pos])
                            else:
                                stop = None
                            nw_slice = ((slice(sanitize(start),
                                               sanitize(stop),
                                               sanitize(cnf_slice[0].step)),)
                                        + tuple(old_slices[1:]))

                        else:
                            position = (cnf_slice[0] - nw_steps -
                                         init_l[pos] + store_steps[pos])

                            nw_slice = (sanitize(position),) + \
                                    tuple(old_slices[1:])

                        subtens = tensor.basic.Subtensor(nw_slice)
                        sl_ins = tensor.basic.Subtensor.collapse(
                            nw_slice,
                            lambda entry: isinstance(entry,
                                                     tensor.Variable))
                        new_o = subtens.make_node(new_outs[nw_pos],
                                                  *sl_ins).outputs[0]
                        if new_o.ndim > 0:
                            new_o = new_o[::cnf_slice[1]]
                        old_new += [(old, new_o)]

            # 3.9. Get replace pairs for all other nodes
            if flag_store or global_nsteps is not None:
                for idx, o in enumerate(node.outputs):
                    if not (idx in replaced_outs) and not idx in not_required:
                        nw_pos = compress_map[idx]
                        old_new += [(o, new_outs[nw_pos])]

                env.replace_all_validate(old_new, reason='scan_save_mem')

    def apply(self, env):

        nodelist = [x for x in env.toposort() if isinstance(x.op,
                                                           scan_op.Scan)]
        for node in nodelist:
            if not hasattr(node.op, '_scan_merge_visited'):
                self.process_node(env, node)

# Just before specialize to have the other optimization
# like constant folding being applied
# This don't introduce inplace.
scan_seqopt.register('scanOp_save_mem',
                     ScanSaveMem(),
                     4,
                     'fast_run',
                     'scan')


class ScanMerge(gof.Optimizer):
    """ Graph Optimizer that merges different scan ops """
    def add_requirements(self, env):
        env.extend(gof.toolbox.ReplaceValidate())

    def merge(self, nodes):

        if nodes[0].op.as_while:
            as_while = True
            condition = nodes[0].op.outputs[-1]
        else:
            as_while = False

        info = {}
        info['tap_array'] = []
        info['n_seqs'] = sum([nd.op.n_seqs for nd in nodes])
        info['n_mit_mot'] = sum([nd.op.n_mit_mot for nd in nodes])
        info['n_mit_mot_outs'] = sum([nd.op.n_mit_mot_outs for nd in nodes])
        info['mit_mot_out_slices'] = []
        info['n_mit_sot'] = sum([nd.op.n_mit_sot for nd in nodes])
        info['n_sit_sot'] = sum([nd.op.n_sit_sot for nd in nodes])
        info['n_shared_outs'] = sum([nd.op.n_shared_outs for nd in nodes])
        info['n_nit_sot'] = sum([nd.op.n_nit_sot for nd in nodes])
        info['truncate_gradient'] = nodes[0].op.truncate_gradient
        info['name'] = '&'.join([nd.op.name for nd in nodes])
        info['mode'] = nodes[0].op.mode
        info['inplace'] = False
        info['gpu'] = False
        info['as_while'] = as_while
        info['profile'] = nodes[0].op.profile

        inner_ins = []
        outer_ins = []
        inner_outs = []
        outer_outs = []

        def rename(ls, suffix):
            for k in ls:
                if k.name:
                    k.name += str(suffix)
            return ls

        for idx, nd in enumerate(nodes):
            # Seq
            inner_ins += rename(nd.op.inner_seqs(nd.op.inputs), idx)
            outer_ins += rename(nd.op.outer_seqs(nd.inputs), idx)

        for idx, nd in enumerate(nodes):
            # MitMot
            inner_ins += rename(nd.op.inner_mitmot(nd.op.inputs), idx)
            inner_outs += nd.op.inner_mitmot_outs(nd.op.outputs)
            info['tap_array'] += nd.op.mitmot_taps()
            info['mit_mot_out_slices'] += nd.op.mitmot_out_taps()
            outer_ins += rename(nd.op.outer_mitmot(nd.inputs), idx)
            outer_outs += nd.op.outer_mitmot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # MitSot
            inner_ins += rename(nd.op.inner_mitsot(nd.op.inputs), idx)
            inner_outs += nd.op.inner_mitsot_outs(nd.op.outputs)
            info['tap_array'] += nd.op.mitsot_taps()
            outer_ins += rename(nd.op.outer_mitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_mitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # SitSot
            inner_ins += rename(nd.op.inner_sitsot(nd.op.inputs), idx)
            info['tap_array'] += [[-1] for x in xrange(nd.op.n_sit_sot)]
            inner_outs += nd.op.inner_sitsot_outs(nd.op.outputs)
            outer_ins += rename(nd.op.outer_sitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_sitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            inner_ins += rename(nd.op.inner_shared(nd.op.inputs), idx)
            outer_ins += rename(nd.op.outer_shared(nd.inputs), idx)

        for idx, nd in enumerate(nodes):
            # NitSot
            inner_outs += nd.op.inner_nitsot_outs(nd.op.outputs)
            outer_ins += rename(nd.op.outer_nitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_nitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            outer_outs += nd.op.outer_shared_outs(nd.outputs)
            inner_outs += nd.op.inner_shared_outs(nd.op.outputs)

        for idx, nd in enumerate(nodes):
            # Non Seqs
            inner_ins += rename(nd.op.inner_non_seqs(nd.op.inputs), idx)
            outer_ins += rename(nd.op.outer_non_seqs(nd.inputs), idx)

        # Add back the number of steps
        outer_ins = [nodes[0].inputs[0]] + outer_ins

        if as_while:
            # add the condition
            inner_outs.append(condition)
        inner_ins, inner_outs = scan_utils.reconstruct_graph(inner_ins,
                                                             inner_outs)

        new_op = scan_op.Scan(inner_ins, inner_outs, info)
        new_outs = new_op(*outer_ins)

        if not isinstance(new_outs, (list, tuple)):
            new_outs = [new_outs]

        return zip(outer_outs, new_outs)

    def belongs_to_set(self, node, set_nodes):
        """
        This function checks if node `node` belongs to `set_nodes`, in the
        sense that it can be merged together with every other node in
        `set_nodes`. In order for two nodes to be mergeable, they have to go
        over the same number of steps, have the same condition (if any),
        have the same value for truncate_gradient, and have the same mode.
        Questionable, we should also consider profile ?
        """
        rep = set_nodes[0]
        if not rep.op.as_while and node.op.as_while:
            return False

        nsteps = node.inputs[0]
        try:
            nsteps = int(get_constant_value(nsteps))
        except TypeError:
            pass

        rep_nsteps = rep.inputs[0]
        try:
            rep_nsteps = int(get_constant_value(rep_nsteps))
        except TypeError:
            pass

        # Check to see if it is an input of a different node
        can_add = True
        for nd in set_nodes:
            if find_up(node, nd) or find_up(nd, node):
                can_add = False

        can_add = can_add and (node.op.truncate_gradient ==
                               rep.op.truncate_gradient)
        can_add = can_add and (node.op.mode == rep.op.mode)
        if not node.op.as_while:
            return nsteps == rep_nsteps and can_add
        cond = node.op.outputs[-1]
        rep_cond = rep.op.outputs[-1]
        same_cond = scan_utils.equal_computations([cond], [rep_cond],
                                                  node.op.inputs,
                                                  rep.op.inputs)
        return same_cond and (nsteps == rep_nsteps) and can_add

    def apply(self, env):
        # Collect all scan nodes ordered according to toposort
        scan_nodes = [nd for nd in env.toposort()
                      if isinstance(nd.op, scan_op.Scan)]

        # All sets of possibly mergeable nodes
        all_sets = []

        for nd in scan_nodes:
            belongs_to_set_idx = -1
            for pos, subset in enumerate(all_sets):
                if self.belongs_to_set(nd, subset):
                    assert belongs_to_set_idx == -1
                    belongs_to_set_idx = pos

            if belongs_to_set_idx == -1:
                all_sets.append([nd])
            else:
                all_sets[belongs_to_set_idx].append(nd)

        for subset in all_sets:
            if len(subset) > 1:
                proposal = self.merge(subset)
                env.replace_all_validate(proposal, reason='scan_merge')


# after const merge but before stabilize so that we can have identity
# for equivalent nodes but we still have the chance to hoist stuff out
# of the scan later.
scan_seqopt.register('scanOp_merge',
                     ScanMerge(),
                     2,
                     'fast_run',
                     'scan')


def has_duplicates(l):
    """returns true if l has any duplicates (according to __eq__)."""
    return len(set(l)) < len(l)


def make_equiv(lo, li):
    """builds a dictionary of equivalences between inner inputs based on
    the equivalence of their corresponding outer inputs."""
    seeno = {}
    left = []
    right = []
    for o, i in zip(lo, li):
        if o in seeno:
            left += [i]
            right += [o]
        else:
            seeno[o] = i
    return left, right


@gof.local_optimizer([None])
def scan_merge_inouts(node):
    if not isinstance(node.op, scan_op.Scan):
        return False

    a = scan_args(node.inputs, node.outputs,
                  node.op.inputs, node.op.outputs, node.op.info)

    inp_equiv = {}

    if has_duplicates(a.outer_in_seqs):
        new_outer_seqs = []
        new_inner_seqs = []
        for out_seq, in_seq in zip(a.outer_in_seqs, a.inner_in_seqs):
            if out_seq in new_outer_seqs:
                i = new_outer_seqs.index(out_seq)
                inp_equiv[in_seq] = new_inner_seqs[i]
            else:
                new_outer_seqs.append(out_seq)
                new_inner_seqs.append(in_seq)
        a.outer_in_seqs = new_outer_seqs
        a.inner_in_seqs = new_inner_seqs

    if has_duplicates(a.outer_in_non_seqs):
        new_outer_nseqs = []
        new_inner_nseqs = []
        for out_nseq, in_nseq in zip(a.outer_in_non_seqs, a.inner_in_non_seqs):
            if out_nseq in new_outer_nseqs:
                i = new_outer_nseqs.index(out_nseq)
                inp_equiv[in_nseq] = new_inner_nseqs[i]
            else:
                new_outer_nseqs.append(out_nseq)
                new_inner_nseqs.append(in_nseq)
        a.outer_in_non_seqs = new_outer_nseqs
        a.inner_in_non_seqs = new_inner_nseqs

    if len(inp_equiv) > 0:
        # do the replacement now. The rest will be left to ScanSaveMem
        inner_inputs = a.inner_inputs
        outer_inputs = a.outer_inputs
        info = a.info
        if info['as_while']:
            a_inner_outs = a.inner_outputs + a.cond
        else:
            a_inner_outs = a.inner_outputs
        inner_outputs = scan_utils.clone(a_inner_outs, replace=inp_equiv)
        orig_outputs = a.outer_outputs

        op = scan_op.Scan(inner_inputs, inner_outputs, info)
        outputs = op(*outer_inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        na = scan_args(outer_inputs, outputs, op.inputs, op.outputs, op.info)
    else:
        na = a

    # start again
    left = []
    right = []

    if has_duplicates(na.outer_in_shared):
        _left, _right = make_equiv(na.outer_in_shared, na.inner_in_shared)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_sit_sot):
        _left, _right = make_equiv(na.outer_in_sit_sot, na.inner_in_sit_sot)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_mit_mot):
        seen = {}
        for omm, imm, _sl in zip(na.outer_in_mit_mot,
                                 na.inner_in_mit_mot, na.mit_mot_in_slices):
            sl = tuple(_sl)
            if (omm, sl) in seen:
                simm = seen[(omm, sl)]
                left += imm
                right += simm
            else:
                seen[(omm, sl)] = imm

    if has_duplicates(na.outer_in_mit_sot):
        seen = {}
        for oms, ims, _sl in zip(na.outer_in_mit_sot,
                                 na.inner_in_mit_sot,
                                 na.mit_sot_in_slices):
            sl = tuple(_sl)
            if (oms, sl) in seen:
                sims = seen[(oms, sl)]
                left += ims
                right += sims
            else:
                seen[(oms, sl)] = ims

    def map_out(i, o, seen):
        for si, so in seen:
            if equal_computations([i], [si], left, right):
                return so
        seen.append((i, o))
        return o

    seen = []
    na.outer_out_nit_sot = [map_out(i, o, seen)
                            for i, o in zip(na.inner_out_nit_sot,
                                            na.outer_out_nit_sot)]

    seen = []
    na.outer_out_sit_sot = [map_out(i, o, seen)
                            for i, o in zip(na.inner_out_sit_sot,
                                            na.outer_out_sit_sot)]

    seen = []
    na.outer_out_mit_sot = [map_out(i, o, seen)
                            for i, o in zip(na.inner_out_mit_sot,
                                            na.outer_out_mit_sot)]

    seen = []
    new_outer_out_mit_mot = []
    for imm, omm, osl in zip(na.inner_out_mit_mot,
                             na.outer_out_mit_mot, na.mit_mot_out_slices):
        for simm, somm, sosl in seen:
            if osl == sosl and equal_computations(imm, simm, left, right):
                new_outer_out_mit_mot.append(somm)
                break
        else:
            seen.append((imm, omm, osl))
            new_outer_out_mit_mot.append(omm)
    na.outer_out_mit_mot = new_outer_out_mit_mot

    return na.outer_outputs

scan_seqopt.register('scanOp_merge_inouts',
                     opt.in2out(scan_merge_inouts, ignore_newtrees=True),
                     3,
                     'fast_run',
                     'scan')
