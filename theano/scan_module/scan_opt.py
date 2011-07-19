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
                        # In case the input is still an alloc node
                        if nw_inputs[offset+idx].owner:
                            _nw_input = nw_inputs[offset+idx].owner.inputs[1]
                            nw_input = scan_utils.expand( _nw_input, val - init_l[i] )
                        # Else, if it was constant folded to a single value
                        elif isinstance(nw_inputs[offset+idx], tensor.Constant):
                            # The hope is that constant folding will fold
                            # this as well
                            nw_input = nw_inputs[offset+idx][:val]
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
            # 3.6 Compose the new scan
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
                        nw_pos = compress_map[idx]
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
        nodelist = list(env.toposort())
        old_new = []
        for node in nodelist:
            op = node.op
            if isinstance(op, scan_op.Scan):
                self.process_node(env, node)

# Just before specialize to have the other optimization
# like constant folding being applied
# This don't introduce inplace.
optdb.register( 'scanOp_save_mem'
               , ScanSaveMem()
               , 1.99
               , 'fast_run'
               , 'scan')

'''
class ScanMerge(gof.Optimizer):
    """ Graph Optimizer that reduces scan memory consumption """
    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self,env):
        env.extend(gof.toolbox.ReplaceValidate())

    def merge(self, A,B):
        # Step 1. Identify common inputs
        equal_ins = []
        for Aidx, Ainp in enumerate(A.inputs):
            if Ainp in B.inputs:
                equal_ins += [ (Aidx, B.inputs.index(Ainp) ) ]

        # Step 2. Get their slices together with taps
        Cslices = {}
        for Aidx,Bidx in equal_ins:
            Aslices = self.get_slice(A, Aidx)
            Bslices = self.get_slice(B, Bidx)
            Cslices = Aslices.copy()
            for tap, var in Bslices.iteritems():
                if tap in Cslices :
                    cvar = Clisces[tap]
                    replace = {var: cvar}
                else:
                    Cslices[tap] = var



        #   two outputs are equal if they implement same computations
        #   and start from the same inputs
        # Step 2. Get their corresponding slices in the input
        # Step 3.

    def apply(self, env):
        nodelist = list(env.toposort())
        cond_nodes = [ x for x in nodelist if x.op.__class__.__name__=='IfElse']
        scan_nodes = [ x for x in nodelist if x.op.__class__.__name__=='Scan']

        # Having lazy ifs in the graph complicates a bit things, and for
        # now I will not treat that case
        if len(cond_nodes) > 0:
            return False

        tomerge_nodes = []
        for try_node in scan_nodes:
            can_merge = False
            for idx in xrange(len(tomerge_nodes)):
                node = tomerge_nodes[idx]
                if scan_utils.equal_computations(
                    node.inputs[0], try_node.inputs[0], strict = True):
                    can_merge = True
                    try:
                        new_node = self.merge(try_node, node)
                        position = idx
                    except NotImplementedError:
                        can_merge = False

            if not can_merge:
                tomerge_nodes += [try_node]
            else:
                tomerge_nodes[position] = new_node



optdb.register( 'scanOp_merge'
               , ScanMerge()
               , 2.39
               , 'fast_run'
               , 'scan')
'''


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
