"""
This module provides optimizations for scan
"""


__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
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
from theano.tensor import opt, TensorType, get_constant_value
from theano import gof
from theano.compile import optdb
from theano.gof.opt import EquilibriumOptimizer
from theano import config

import scan_op
import scan_utils
from scan_utils import clone, equal_computations
from theano.gof.opt import pre_constant_merge, pre_greedy_local_optimizer

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_opt')

list_opt_slice = [ tensor.opt.local_abs_merge,
                tensor.opt.local_mul_switch_sink,
                tensor.opt.local_upcast_elemwise_constant_inputs,
                tensor.opt.local_remove_switch_const_cond,
                tensor.opt.constant_folding ]
def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))

def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))

@gof.local_optimizer([None])
def remove_constants_and_unused_inputs_scan(node):
    if not isinstance(node.op, scan_op.Scan):
        return False
    op = node.op
    # We only need to take care of sequences and other arguments
    st  = op.n_seqs
    st += int(numpy.sum([len(x) for x in
                     op.tap_array[:(op.n_mit_mot+op.n_mit_sot)] ]))
    st += op.n_sit_sot
    st += op.n_shared_outs
    op_ins, op_outs = scan_utils.reconstruct_graph(op.inputs, op.outputs,
                                                   '')
    out_stuff_inner = op_ins[op.n_seqs:st]
    non_seqs = op_ins[st:]
    st  = ( op.n_seqs +
           op.n_mit_mot +
           op.n_mit_sot +
           op.n_sit_sot +
           op.n_nit_sot +
           op.n_shared_outs +1 )
    outer_non_seqs = node.inputs[st:]
    out_stuff_outer = node.inputs[1+op.n_seqs:st]

    givens   = {}
    nw_inner = []
    nw_outer = [node.inputs[0]]
    all_ins = gof.graph.inputs(op_outs)
    for idx in xrange(op.n_seqs):
        if (isinstance(node.inputs[idx+1], tensor.TensorConstant) and
            node.inputs[idx+1].tag.unique_value is not None):
            try:
                val = tensor.get_constant_value(node.inputs[idx+1],
                                                return_ndarray = True)
                givens[op_ins[idx]] = tensor.constant(val[0])
            except TypeError:
                pass
        elif op_ins[idx] in all_ins:
            nw_inner += [op_ins[idx]]
            nw_outer += [node.inputs[idx+1]]
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
        op_outs = scan_utils.clone(op_outs, replace = givens)
        nw_info = op.info.copy()
        nw_info['n_seqs'] = nw_n_seqs
        nwScan = scan_op.Scan(nw_inner, op_outs, nw_info)
        nw_outs = nwScan.make_node(*nw_outer).outputs
        return nw_outs
    else:
        return False

optdb.register( 'scanOp_remove_constants_and_unused_inputs'
               , opt.in2out(remove_constants_and_unused_inputs_scan,
                            ignore_newtrees = True)
               , 1.995
               , 'fast_run'
               , 'scan')


@gof.local_optimizer([None])
def scan_pushout_non_seq_operation(node):
    if not isinstance(node.op, scan_op.Scan):
        return False
    # this flag tells if there was any change during the last iterations
    changed   = True
    try:
        clean_inputs, clean_outputs = scan_utils.reconstruct_graph(
                        node.op.inputs, node.op.outputs)
        local_env = gof.Env(clean_inputs, clean_outputs)
    except:
        import ipdb; ipdb.set_trace()

    max_iterations = 2*len(local_env.toposort()) + 3
    counts = 0
    to_remove        = []
    to_replace       = []
    replace_with_in  = []
    replace_with_out = []
    op = node.op
    # Construct the list of non_sequences to simplify a few things
    st  = op.n_seqs
    st += int(numpy.sum([len(x) for x in
                         op.tap_array[:(op.n_mit_mot+op.n_mit_sot)] ]))
    st += op.n_sit_sot
    st += op.n_shared_outs
    non_seqs = clean_inputs[st:]
    st  = ( op.n_seqs +
           op.n_mit_mot +
           op.n_mit_sot +
           op.n_sit_sot +
           op.n_nit_sot +
           op.n_shared_outs +1 )
    outer_non_seqs = node.inputs[st:]

    while changed and counts < max_iterations:
        counts += 1
        changed = False
        for nd in local_env.toposort():
            if (    numpy.all([ (x in non_seqs) or
                                (x.owner in to_remove) or
                                isinstance(x, tensor.Constant)
                               for x in nd.inputs]) and
                    # we can do this because the assumption is that a
                    # viewOp or deepCopyOp will be just at the end of the
                    # function and not somewhere in the middle ..
                    not isinstance(nd.op,theano.compile.ViewOp) and
                    not isinstance(nd.op,theano.compile.DeepCopyOp) and
                    # and we didn't already looked at this node
                    not nd in to_remove
               ):

                # We have a candidate node to removable
                # Step 1. Reconstruct it on outside
                to_remove += [nd]
                outside_ins = []
                for x in nd.inputs:
                    if x in non_seqs:
                        outside_ins +=[ outer_non_seqs[non_seqs.index(x)]]
                    elif x in to_replace:
                        outside_ins +=[replace_with_out[to_replace.index(x)]]
                    elif isinstance(x, theano.Constant):
                        outside_ins +=[x.clone()]
                    else:
                        raise Exception(
                            ('Error in the `scan_pushout_non_seq_operations`'
                             '. The optimization tries to move some '
                             'computation fron scan which is not allowed '
                             'to move. Report this on theano-users list'),x )
                nw_outer_node = nd.op.make_node(*outside_ins)
                # Step 2. Create variables for replacements
                for idx,y in enumerate(nd.outputs):
                    y_place_holder = scan_utils.safe_new(y,'_replace')
                    to_replace       += [y]
                    replace_with_in  += [y_place_holder]
                    if (cuda.cuda_available and
                        isinstance(nw_outer_node.outputs[idx],
                                   CudaNdarrayType)):
                        nw_out = nw_outer_node.outputs[idx]
                        replace_with_out += [host_from_gpu(nw_out)]
                    else:
                        replace_with_out += [nw_outer_node.outputs[idx]]
                changed = True

    if counts >= max_iterations:
        raise Exception( ('Error in the `scan_pushout_non_seq_operations`.'
                          ' The optimization exhausted the maximal number '
                          'of iterations allowed!'))
    # We need to check all candidate replacements and choose those that
    # make sense for us

    # Step 1. which elements of `to_replace` are used by remaining
    # components of the inner function
    clean_to_replace       = []
    clean_replace_with_in  = []
    clean_replace_with_out = []
    existent_nodes = [ nd for nd in local_env.toposort()
                        if nd not in to_remove]
    to_keep = []
    for nd in existent_nodes:
        to_keep += nd.inputs
    for idx,out in enumerate(to_replace):
        if out in to_keep and out.owner not in existent_nodes:
            clean_to_replace += [out]
            clean_replace_with_in  += [replace_with_in[idx]]
            clean_replace_with_out += [replace_with_out[idx]]

    if len(clean_to_replace) > 0:
        # We can finally put an end to all this madness
        givens = {}
        nw_outer = []
        nw_inner = []
        for to_repl, repl_in, repl_out in zip( clean_to_replace,
                                              clean_replace_with_in,
                                              clean_replace_with_out):
            if isinstance(repl_out, theano.Constant):
                # Is this even possible !?
                repl_in = repl_out.clone()
            else:
                nw_inner += [repl_in]
                nw_outer += [repl_out]
            givens[to_repl] = repl_in


        _op_outs = scan_utils.clone(clean_outputs,
                                    replace=givens)

        _op_ins = clean_inputs + nw_inner
        op_ins, op_outs = scan_utils.reconstruct_graph(_op_ins, _op_outs, '')
        # Reconstruct node
        nwScan = scan_op.Scan(op_ins, op_outs, op.info)
        node = nwScan.make_node(* (node.inputs + nw_outer))
        return node.outputs
    else:
        return False


optdb.register('scanOp_pushout_nonseqs_ops',
               opt.in2out( scan_pushout_non_seq_operation,
                          ignore_newtrees=True),
               1.90,
               'fast_run',
               'scan')


@gof.local_optimizer([None])
def scan_make_inplace(node):
    op = node.op
    if ( isinstance(op, scan_op.Scan) and
        (not op.info['inplace']) ):
        info = op.info.copy()
        info['inplace'] = True
        new_op = scan_op.Scan( op.inputs
                              , op.outputs
                              , info)
        return new_op.make_node(*node.inputs).outputs
    return False

optdb.register( 'scanOp_make_inplace'
               , opt.in2out(scan_make_inplace,ignore_newtrees=True)
               , 75
               , 'fast_run'
               , 'inplace'
               , 'scan')



class ScanSaveMem(gof.Optimizer):
    """ Graph Optimizer that reduces scan memory consumption """
    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self,env):
        env.extend(gof.toolbox.ReplaceValidate())

    def process_node(self, env, node):

        # helpful functions
        def select_min(x,y):
            if x is None:
                return y
            if y is None:
                return x
            return tensor.minimum(x,y)
        def select_max(x,y):
            if x is None:
                return y
            if y is None:
                return x
            return tensor.maximum(x,y)

        def sanitize(x):
            if x is None:
                return None
            else:
                return tensor.as_tensor_variable(x)

        shape_of = node.env.shape_feature.shape_of
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

        init_l  = [ 0 for x in xrange(op.n_mit_mot)]
        init_l += [ abs(numpy.min(v)) for v in op.tap_array[op.n_mit_mot:] ]
        init_l += [ 0 for x in xrange(op.n_nit_sot)]
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
        if len(node.outputs) <= c_outs :
            global_nsteps = {'real' :-1, 'sym': []}
        else:
            global_nsteps = None

        # Keeps track of the original slices that each client represent
        slices = [ None for o in node.outputs]

        # A list for each output indicating how many intermediate values
        # should be stored. If negative it means none of the intermediate
        # values (i.e. the output can be removed since it is not used
        # afterwards in the computations), if 0 it means that all
        # intermediate values are required, otherwise is up to that number
        # of intermediate values
        # Note that for mit_mot outputs and shared outputs we can not change
        # the number of intermediate steps stored without affecting the
        # result of the op
        store_steps  = [ 0 for o in xrange(op.n_mit_mot)]
        store_steps += [-1 for o in node.outputs[op.n_mit_mot:c_outs]]
        # Flag that says if an input has changed and we need to do something
        # or not
        flag_store = False

        # 2.2 Loop over the clients
        for i,out in enumerate(node.outputs[:c_outs]):
            # look at all its clients
            slices[i] = []
            for cl,_ in out.clients:

                # 2.1 outputs of the function
                #=> output needs all its intermediate values
                if type(cl) == str:
                    # if the node is actually an output, then
                    # we need to store the entire thing
                    global_nsteps  = None
                    slices[i]      = None
                    break
                # 2.2 non-subtensor nodes
                #=> output needs all its intermediate values
                elif not isinstance(cl.op, tensor.basic.Subtensor):
                    global_nsteps  = None
                    slices[i]      = None
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
                        global_nsteps  = None
                        slices[i]      = None
                        break


                    # 2.3.2 extract the begin/end of the first dimension

                    if i > op.n_mit_mot:
                        try:
                            length = shape_of[out][0]
                        except:
                            length = node.inputs[0] + init_l[i]
                    else:
                        try:
                            length = shape_of[out][0]
                        except:
                            length = out.shape[0]
                    cf_slice = tensor.basic.get_canonical_form_slice(
                                                    this_slice[0], length)
                    slices[i] += [(cf_slice,this_slice)]

                    if ( isinstance(this_slice[0],slice) and
                        this_slice[0].stop is None ):
                        global_nsteps = None
                        break
                    if isinstance(cf_slice[0], slice):
                        stop  = tensor.basic.extract_constant(cf_slice[0].stop)
                    else:
                        stop  = tensor.basic.extract_constant(cf_slice[0]) + 1
                    if stop == sys.maxint or stop == length:
                        stop = None
                    else:
                        # there is a **gotcha** here ! Namely, scan returns an
                        # array that contains the initial state of the output as
                        # well. Which means that if have a initial state of
                        # length 3, and you look for 5 steps you get an output y
                        # of length 8. If you only use y[:5], this does not mean
                        # that you only need to loop for 5 steps but actually
                        # only for 2 steps ( the first 3 are the initial state)
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
                        elif (type(stop) is int and global_nsteps['real'] < stop):
                            global_nsteps['real'] = stop
                        # yes if it is a int k, 0 < k < maxint
                        elif (type(stop) is int and stop > 0 ):
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
            if len(global_nsteps['sym']) == 0 :
                sym_steps = None
            else:
                sym_steps =global_nsteps['sym'][0]
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
        for i,out in enumerate(node.outputs[:c_outs]):
            # look at all its clients
            for cl,_ in out.clients:
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

                    if ( isinstance(this_slice[0],slice) and
                        this_slice[0].start is None):
                        store_steps[i] = 0
                        break

                    if i > op.n_mit_mot:
                        length = node.inputs[0] + init_l[i]
                    else:
                        try:
                            length = shape_of[out][0]
                        except:
                            length = out.shape[0]
                    cf_slice = tensor.basic.get_canonical_form_slice(
                                                    this_slice[0],length)

                    if isinstance(cf_slice[0], slice):
                        start = tensor.basic.extract_constant(cf_slice[0].start)
                    else:
                        start = tensor.basic.extract_constant(cf_slice[0])
                    if start == 0 or store_steps[i] == 0:
                        store_steps[i] = 0
                    else:
                        pval = select_max(nw_steps -start + init_l[i], init_l[i])
                        if store_steps[i] != -1:
                            pval = select_max(pval, store_steps[i])

                        store_steps[i] = pval
                        flag_store = True

        orphane_outs = [ i for i,x in enumerate(store_steps)
                        if (type(x) is int) and (x<0) ]
        flag_store = flag_store or (len(orphane_outs) > 0 )
        # 3. is there anything to change ?
        if (flag_store or global_nsteps is not None):
            # 3.1 initialize inputs for the new scan
            old_outputs  = []
            nw_inputs    = list(node.inputs)
            nw_inputs[0] = nw_steps

            # 3.2 check orphane outputs to see if we can eliminate any
            required,not_required = \
                    scan_utils.scan_can_remove_outs(node.op
                                                    , orphane_outs)
            # 3.3. compose replace pairs for those nodes that need not
            # to store everything in memory ( or ar orphane and required
            # by the inner function .. )
            replaced_outs = []
            offset = 1 + op.n_seqs + op.n_mit_mot
            for idx,_val in enumerate(store_steps[op.n_mit_mot:]):
                i = idx + op.n_mit_mot
                if not( type(_val) is int and _val <=0 and i not in required):

                    if idx+op.n_mit_mot in required:
                        val = 1
                    else:
                        val = _val
                    # If the memory for this output has been pre-allocated
                    # before going into the scan op (by an alloc node)
                    if idx < op.n_mit_sot + op.n_sit_sot:
                        # In case the input is still an alloc node, we
                        # actually have two options:
                        #   a) the input is an alloc (due to an optimization
                        #   that converts set_subtensor(0,0) in 0
                        #   b) the input is an set subtensor
                        if ( nw_inputs[offset+idx].owner and
                            isinstance(nw_inputs[offset+idx].owner.op,
                                       tensor.IncSubtensor)):
                            _nw_input = nw_inputs[offset+idx].owner.inputs[1]
                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                tensor.as_tensor_variable(val - init_l[i]))
                            tmp = pre_constant_merge([tmp])[0]
                            nw_input = scan_utils.expand( _nw_input,tmp )
                        # If it is an alloc
                        elif ( nw_inputs[offset+idx].owner and
                          isinstance(nw_inputs[offset+idx].owner.op,
                                     tensor.Alloc)):

                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                tensor.as_tensor_variable(val))
                            tmp = pre_constant_merge([tmp])[0]
                            nw_input = nw_inputs[offset+idx][:tmp]
                        # Else, if it was constant folded to a single value
                        elif isinstance(nw_inputs[offset+idx], tensor.Constant):
                            # The hope is that constant folding will fold
                            # this as well

                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                            tensor.as_tensor_variable(val))
                            tmp = pre_constant_merge([tmp])[0]
                            nw_input = nw_inputs[offset+idx][:tmp]
                        else:
                            raise Exception(('Unforseen case. Please report'
                                            ' to theano-dev with an example'
                                            ' script for this case to be'
                                            ' debuged'))

                        nw_inputs[offset+idx] = nw_input
                        replaced_outs.append(op.n_mit_mot + idx)
                        odx = op.n_mit_mot + idx
                        old_outputs += [(odx, [x[0].outputs[0] for x in
                                        node.outputs[odx].clients])]
                    # If there is no memory pre-allocated for this output
                    elif idx < op.n_mit_sot + op.n_sit_sot + op.n_nit_sot:

                        pos = ( op.n_mit_mot + idx + op.n_seqs
                                   + 1 + op.n_shared_outs )
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
                            _nw_input = nw_inputs[offset+idx].owner.inputs[1]
                            odx = op.n_mit_mot + idx
                            nw_input = scan_utils.expand(_nw_input, nw_steps)
                            nw_inputs[offset+idx] = nw_input
                        elif idx < (op.n_mit_sot + op.n_sit_sot +
                                     +  op.n_nit_sot):
                            in_idx = offset+idx+op.n_shared_outs
                            if nw_inputs[in_idx] == node.inputs[0]:
                                nw_inputs[in_idx] =nw_steps
                            odx = op.n_mit_mot + idx


            # 3.5 Remove unwanted orphane outputs
            (inps, outs, info, node_ins, compress_map) = \
                    scan_utils.compress_outs(op, not_required, nw_inputs)
            inv_compress_map = {}
            for k,v in compress_map.items():
                inv_compress_map[v] = k

            node_ins = [ pre_greedy_local_optimizer(list_opt_slice, x) for x in
                        node_ins]
            node_ins = pre_constant_merge(node_ins)
            # 3.6 Compose the new scan
            # I need to make sure I'm not reapplying the same optimization
            # twice since bad things usually happen if I do that
            info['_scan_merge_visited'] = True
            new_outs = scan_op.Scan(inps
                                    , outs
                                    , info).make_node(*node_ins).outputs


            old_new = []
            # 3.7 Get replace pairs for those outputs that do not change
            # the number of intermediate steps stored
            for idx,sl in enumerate(slices):
                if global_nsteps and sl is not None and store_steps[idx] == 0:
                    for hdx,cl in enumerate(node.outputs[idx].clients):
                        cnf_slice, old_slices = sl[hdx]
                        # Sanitize the nw_slice by converting ints back into
                        # constants :) I only need to do this for the first
                        # slice since that is the only slice

                        if isinstance(cnf_slice[0], slice):
                            fslice = slice(
                                sanitize(cnf_slice[0].start),
                                sanitize(cnf_slice[0].stop),
                                sanitize(cnf_slice[0].step)
                                )
                        else:
                            fslice = sanitize(cnf_slice[0])


                        nw_slice = (fslice,) + tuple(old_slices[1:])
                        nw_pos = inv_compress_map[idx]
                        nw_out = new_outs[nw_pos]


                        subtens = tensor.basic.Subtensor(nw_slice)
                        # slice inputs
                        sl_ins = tensor.basic.Subtensor.collapse(
                            nw_slice
                            , lambda entry: isinstance(entry
                                                    , tensor.Variable))
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
                    for k,old in enumerate(old_outs):
                        # Get the correct slice
                        cnf_slice, old_slices = slices[pos][k]
                        if type(cnf_slice[0]) is slice:
                            start = ( cnf_slice[0].start - nw_steps -
                                     init_l[pos] + store_steps[pos] )
                            if ( cnf_slice[0].stop is not None and
                                cnf_slice[0].stop != sys.maxint ):
                                stop = ( cnf_slice[0].stop - nw_steps -
                                        init_l[pos] + store_steps[pos])
                            else:
                                stop = None
                            nw_slice = ( (slice(sanitize(start),
                                                sanitize(stop),
                                                sanitize(cnf_slice[0].step)),) +
                                        tuple(old_slices[1:]) )

                        else:
                            position = (cnf_slice[0] - nw_steps -
                                         init_l[pos] +  store_steps[pos] )

                            nw_slice = (sanitize(position),) + tuple(old_slices[1:])

                        subtens = tensor.basic.Subtensor(nw_slice)
                        sl_ins = tensor.basic.Subtensor.collapse(
                            nw_slice
                            , lambda entry: isinstance(entry
                                                    , tensor.Variable))
                        new_o = subtens.make_node(new_outs[nw_pos],
                                                  *sl_ins).outputs[0]
                        if new_o.ndim > 0:
                            new_o = new_o[::cnf_slice[1]]
                        old_new += [(old, new_o)]

            # 3.9. Get replace pairs for all other nodes
            if flag_store or global_nsteps is not None:
                for idx,o in enumerate(node.outputs):
                    if not (idx in replaced_outs) and not idx in not_required:
                        nw_pos = compress_map[idx]
                        old_new += [(o,new_outs[nw_pos])]

                env.replace_all_validate(old_new, reason = 'scan_save_mem')


    def apply(self, env):

        nodelist = [x for x in env.toposort() if isinstance(x.op,
                                                           scan_op.Scan)]
        for node in nodelist:
            if not hasattr(node.op, '_scan_merge_visited'):
                self.process_node(env, node)

# Just before specialize to have the other optimization
# like constant folding being applied
# This don't introduce inplace.
optdb.register( 'scanOp_save_mem'
               , ScanSaveMem()
               , 1.99
               , 'fast_run'
               , 'scan')


from theano.sandbox import cuda

if cuda.cuda_available:

    from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
    from theano.sandbox.cuda.type import CudaNdarrayType
    from theano.sandbox.cuda.opt import register_opt, local_optimizer

    def safe_to_gpu(x):
        if (isinstance(x.type, TensorType) and
            x.type.dtype == 'float32'):
            return gpu_from_host(x)
        else:
            return x

    def safe_to_cpu(x):
        if isinstance(x.type, CudaNdarrayType):
            return host_from_gpu(x)
        else:
            return x

    def tensor_to_cuda(x):
        if (isinstance(x.type, TensorType) and
            x.type.dtype == 'float32'):
            y = CudaNdarrayType( broadcastable = x.type.broadcastable)()
            if x.name :
                y.name = x.name +'[cuda]'
            return y
        else:
            return x


    @register_opt('scan')
    @local_optimizer([])
    def gpuScanOptimization(node):
        """
        scan(host_from_gpu) -> host_from_gpu(GPUscan)
        gpu_from_host(scan) -> GPUscan(gpu_from_host)
        """

        #gpu_from_host(scan) -> GPUscan(gpu_from_host)
        if node.op == gpu_from_host:
            host_input = node.inputs[0]
            if (host_input.owner and
                isinstance(host_input.owner.op, scan_op.Scan) and
                not host_input.owner.op.info['gpu'] and
                len(host_input.owner.outputs) == 1 ):
                # Note that we are not doing the right thing here !!
                # This is because the local optimizer expects only one
                # output that corresponds to the input of ``node``
                # If we do this for each output seperately we will have
                # multiple scan ops in the graph ( as many as outputs )
                # and I'm not sure they will get merged into one again
                # So for now I will just cover a limited case when there
                # is only one output and the local optimizer can be used
                # TODO (fix) : either make sure the different scans get
                # merged or implement this optimization as a global
                # optimization
                thescan = host_input.owner.op
                info = thescan.info.copy()
                info['gpu'] = True
                inputs = host_input.owner.inputs
                nw_ins = [ inputs[0]]
                e = ( 1+ thescan.n_seqs
                     + thescan.n_mit_mot
                     + thescan.n_mit_sot
                     + thescan.n_sit_sot
                     + thescan.n_shared_outs)
                nw_ins += [safe_to_gpu(x) for x in inputs[1:e] ]
                b = e
                e = e + thescan.n_nit_sot
                nw_ins += inputs[b:e]
                nw_ins += [safe_to_gpu(x) for x in inputs[e:] ]
                scan_ins = [ tensor_to_cuda(x) for x in thescan.inputs]
                scan_outs = [ safe_to_gpu(x) for x in thescan.outputs ]
                scan_outs = scan_utils.clone(
                    scan_outs
                    , replace = zip(thescan.inputs,
                                    [safe_to_cpu(x) for x in  scan_ins]))
                nw_op = scan_op.Scan( scan_ins
                                     , scan_outs
                                     , info).make_node(*nw_ins)
                _outputs = nw_op.outputs
                return _outputs

        #scan(host_from_gpu) -> host_from_gpu(GPUscan)
        if (type(node.op) == scan_op.Scan
            and not node.op.info['gpu']):
            if numpy.any([(i.owner and i.owner.op == host_from_gpu)
                          for i in node.inputs]):
                thescan = node.op
                info = thescan.info.copy()
                info['gpu'] = True
                inputs = node.inputs
                nw_ins = [ inputs[0]]
                e = ( 1+ thescan.n_seqs
                     + thescan.n_mit_mot
                     + thescan.n_mit_sot
                     + thescan.n_sit_sot
                     + thescan.n_shared_outs)
                nw_ins += [safe_to_gpu(x) for x in inputs[1:e] ]
                b = e
                e = e + thescan.n_nit_sot
                nw_ins += inputs[b:e]
                nw_ins += [safe_to_gpu(x) for x in inputs[e:] ]

                scan_ins = [ tensor_to_cuda(x) for x in thescan.inputs]
                scan_outs = [ safe_to_gpu(x) for x in thescan.outputs ]
                scan_outs = scan_utils.clone(
                    scan_outs
                    , replace = zip(thescan.inputs
                                    ,[safe_to_cpu(x) for x in  scan_ins]))
                _outputs = scan_op.Scan(
                        scan_ins
                        , scan_outs
                        , info).make_node(*nw_ins).outputs
                outputs = [safe_to_cpu(x) for x in _outputs]
                return outputs
        return False
