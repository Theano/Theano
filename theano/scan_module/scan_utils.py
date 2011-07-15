"""
This module provides utility functions for the Scan Op

See scan.py for details on scan
"""
__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
                "Frederic Bastien "
                "James Bergstra "
                "Pascal Lamblin "  )
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import logging
import numpy

from theano import config
from theano.compile.pfunc import rebuild_collect_shared
from theano import gof
from theano import tensor
from theano.tensor.basic import get_constant_value

from theano.sandbox import cuda


################ Utility Functions and Classes #######################

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_utils')


def safe_new(x):
    if isinstance(x, numpy.ndarray):
        x = tensor.as_tensor_variable(x)
    if cuda.cuda_available and isinstance(x.type, cuda.CudaNdarrayType):
        return tensor.TensorType(
            broadcastable = x.type.broadcastable
            , dtype = config.floatX)()
    else:
        return x.type()

def safe_to_cpu(x):
    if isinstance(x, numpy.ndarray):
        x = tensor.as_tensor_variable(x)
    if cuda.cuda_available and isinstance(x.type, cuda.CudaNdarrayType):
        return cuda.basic_ops.host_from_gpu(x)
    else:
        return x


def traverse(out, x,x_copy, d):
    ''' Function used by scan to parse the tree and figure out which nodes
    it needs to replace. There are two options :
        1) x and x_copy or on host, then you would replace x with x_copy
        2) x is on gpu, x_copy on host, then you need to replace
        host_from_gpu(x) with x_copy
    This happens because initially shared variables are on GPU .. which is
    fine for the main computational graph but confuses things a bit for the
    inner graph of scan '''
    if out == x:
        d[out] = cuda.gpu_from_host(x_copy)
        return d
    elif out.owner is None:
        return d
    elif (out.owner.op == cuda.host_from_gpu
          and out.owner.inputs == [x] ):
        d[out] = x_copy
        return d
    else:
        for inp in out.owner.inputs:
            d = traverse(inp, x, x_copy, d)
        return d


# Hashing a dictionary/list/tuple by xoring the hash of each element
def hash_listsDictsTuples(x):
    hash_value = 0
    if isinstance(x, dict):
        for k,v in x.iteritems():
            hash_value ^= hash_listsDictsTuples(k)
            hash_value ^= hash_listsDictsTuples(v)
    elif isinstance(x, (list,tuple)):
        for v in x:
            hash_value ^= hash_listsDictsTuples(v)
    else:
        try:
            hash_value ^= hash(x)
        except:
            pass
    return hash_value


def clone( output
            , replace = None
            , strict = True
            , copy_inputs = True):
    """
    Function that allows replacing subgraphs of a computational
    graph. It returns a copy of the initial subgraph with the corresponding
    substitutions.


    :type output: Theano Variables ( or Theano expressions)
    :param outputs: Theano expression that represents the computational
                    graph

    :type replace: dict
    :param replace: dictionary describing which subgraphs should be
                    replaced by what
    """

    inps, outs, other_stuff = rebuild_collect_shared( output
                                                   , []
                                                   , replace
                                                   , []
                                                   , strict
                                                   , copy_inputs
                                                   )
    return outs



def get_updates_and_outputs(outputs_updates):
    """
    This function tries to recognize the updates dictionary and the
    list of outputs from the input argument and return them in a
    predefined order


    The code that follows tries to be as flexible as possible allowing the
    user to return the output and updates in any order, and giving the
    updates however (s)he wants ( as a dictionary or a list o pairs ..)
    Is there a way to compress all this by writing it in a more
    pythonic/functional way?
    """
    outputs = []
    updates = {}

    # we will try now to separate the outputs from the updates
    if not isinstance(outputs_updates, (list,tuple)):
        if isinstance(outputs_updates, dict) :
            # we have just an update dictionary
            updates = outputs_updates
        else:
            outputs = [outputs_updates]
    elif len(outputs_updates) == 1:
        if isinstance(outputs_updates[0], (dict, tuple)):
            updates = dict(otuputs_updates[1])
        else:
            outputs = outputs_updates
    else:
        elem0 = outputs_updates[0]
        elem1 = outputs_updates[1]
        t_el0 = type(elem0)
        t_el1 = type(elem1)
        if ( t_el0 == dict or
                ( t_el0 in (list,tuple) and
                    isinstance(elem0[0], (list,tuple)))):
            # elem0 is the updates dictionary / list
            updates = elem0
            outputs = elem1
            if not isinstance(outputs, (list,tuple)):
                outputs = [outputs]
        elif ( isinstance(elem1, dict) or
                ( isinstance(elem1, (list,tuple)) and
                    isinstance(elem1[0], (list,tuple))) ):
            # elem1 is the updates dictionary / list
            updates = elem1
            outputs = elem0
            if not isinstance(outputs, (list,tuple)):
                outputs = [outputs]
        else :
            if ( isinstance(outputs_updates, (list,tuple)) and
                    isinstance(outputs_updates[0], (list,tuple))):
                outputs = []
                updates = outputs_updates
            else:
                outputs = outputs_updates
                updates = {}

    # in case you return a tuple .. convert it to a list (there are certain
    # operation that are not permited on tuples, like element assignment)
    outputs = list(outputs)

    # If you return numbers (highly unlikely) this will not go well for
    # theano. We need to convert them to Theano constants:
    for i,out in enumerate(outputs):
        outputs[i] = tensor.as_tensor(out)

    return outputs, updates


def isNaN_or_Inf_or_None(x):
    isNone = x is None
    try:
        isNaN = numpy.isnan(x)
        isInf = numpy.isinf(x)
        isStr = isinstance(x, str)
    except:
        isNaN = False
        isInf = False
        isStr = False
    if not isNaN and not isInf:
        try:
            val   = get_constant_value(x)
            isInf = numpy.isinf(val)
            isNaN = numpy.isnan(val)
        except:
            isNaN = False
            isInf = False
    if isinstance(x, gof.Constant) and isinstance(x.data, str):
        isStr = True
    else:
        isStr = False
    return isNone or isNaN or isInf or isStr


def expand( tensor_var, size):
    '''
    Transoforms the shape of a tensor from (d1, d2 ... ) to ( d1+size, d2, ..)
    by adding 0s at the end of the tensor.
    '''
    # Corner case that I might use in an optimization
    if size == 0:
        return tensor_var
    shapes      = [ tensor_var.shape[x] for x in xrange(tensor_var.ndim) ]
    zeros_shape = [size+shapes[0]] + shapes[1:]
    empty       = tensor.zeros( zeros_shape
                              , dtype = tensor_var.dtype)
    return tensor.set_subtensor(empty[:shapes[0]], tensor_var)






def equal_computations(x,y, strict=False):
    '''
     Checks if to theano graphs represent the same computations (applied to
     different inputs).
    '''
    if not x.type == y.type:
        return False
    elif not x.owner and not y.owner:
        if not strict:
            return True
        else:
            if isinstance(x, tensor.Constant):
                # not they both have the same type
                return x.data == y.data
            else:
                return x == y
    elif x.owner and not y.owner:
        return False
    elif not x.owner and y.owner:
        return False
    elif not x.owner.op == y.owner.op:
        return False
    elif not len(x.owner.inputs) == len(y.owner.inputs):
        return False
    else:
        for xx,yy in zip(x.owner.inputs,y.owner.inputs):
            if not equal_computations(xx,yy):
                return False
        return True

def infer_shape(outs, inputs, input_shapes):
    '''
    Compute the shape of the outputs given the shape of the inputs
    of a theano graph.
    '''
    # We use a ShapeFeature because it has all the necessary logic inside.
    # We don't use the Feature interface, so we need to initialize some
    # things by hand.
    shape_feature = tensor.opt.ShapeFeature()

    # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
    # All keys of shape_of should be either in valid or in invalid
    shape_feature.shape_of = {}

    # To avoid merging lots of ones together.
    shape_feature.lscalar_one = tensor.constant(1, dtype='int64')

    # Initialize shape_of with the input shapes
    for inp, inp_shp in zip(inputs, input_shapes):
        shape_feature.set_shape(inp, inp_shp)

    def local_traverse(out):
        '''
        Go back in the graph, from out, adding computable shapes to shape_of.
        '''

        if out in shape_feature.shape_of:
            # Its shape is already known
            return
        elif out.owner is None:
            # This is an input of the graph
            shape_feature.init_r(out)
        else:
            # Recurse over inputs
            for inp in out.owner.inputs:
                if not inp in shape_feature.shape_of:
                    local_traverse(inp)

            # shape_feature.on_import does not actually use an env
            # It will call infer_shape and set_shape appropriately
            dummy_env = None
            shape_feature.on_import(dummy_env, out.owner)

    ret = []
    for o in outs:
        local_traverse(o)
        ret.append(shape_feature.shape_of[o])
    return ret

class Validator(object):
    def __init__(self, valid=[], invalid=[], valid_equivalent={}):
        '''
        Check if variables can be expressed without using variables in invalid.

        init_valid_equivalent provides a dictionary mapping some invalid
        variables to valid ones that can be used instead.
        '''

        # Nodes that are valid to have in the graph computing outputs
        self.valid = set(valid)

        # Nodes that are NOT valid to have in the graph computing outputs
        self.invalid = set(invalid)

        # Mapping from invalid variables to equivalent valid ones.
        self.valid_equivalent = valid_equivalent.copy()
        self.valid.update(valid_equivalent.values())
        self.invalid.update(valid_equivalent.keys())

    def check(self, out):
        '''
        Go backwards in the graph, from out, and check if out is valid.

        If out is a valid node, (out, True) is returned.
        If out is not valid, but has an equivalent e, (e, False) is returned.
        If out is not valid and has no equivalent, None is returned.
        '''
        if out in self.valid:
            return out, True
        elif out in self.valid_equivalent:
            return self.valid_equivalent[out], False
        elif out in self.invalid:
            return None

        if out.owner is None:
            # This is an unknown input node, so it is invalid.
            self.invalid.add(out)
            if isinstance(out, tensor.TensorConstant):
                # We can clone it to get a valid constant
                cloned_out = out.clone()
                self.valid.add(cloned_out)
                self.valid_equivalent[out] = cloned_out
                return cloned_out, False

            return None

        # Recurse over inputs
        inputs = [self.check(i) for i in out.owner.inputs]

        # If some inputs are invalid without equivalent, so is out
        if None in inputs:
            self.invalid.add(out)
            return None

        # If some inputs are invalid with equivalent,
        # an equivalent out should be built and returned
        all_inputs = [inp for (inp, is_valid) in inputs]
        equiv_inputs = [inp for (inp, is_valid) in inputs if not is_valid]
        if equiv_inputs:
            cloned_node = out.owner.clone_with_new_inputs(all_inputs)
            cloned_out = cloned_node.outputs[out.index]
            self.invalid.add(out)
            self.valid.add(cloned_out)
            self.valid_equivalent[out] = cloned_out
            return cloned_out, False

        # All inputs are valid, so is out
        return out, True


def scan_can_remove_outs(op, out_idxs):
    '''
    Looks at all outputs defined by indices ``out_idxs`` and see whom can be
    removed from the scan op without affecting the rest. Return two lists,
    the first one with the indices of outs that can be removed, the second
    with the outputs that can not be removed.
    '''
    non_removable = [ o for i,o in enumerate(op.outputs) if i not in
                     out_idxs]
    required_inputs = gof.graph.inputs(non_removable)

    out_ins = []
    offset  = op.n_seqs
    lim = op.n_mit_mot + op.n_mit_sot + op.n_sit_sot
    for idx in range(lim):
        n_ins    = len(op.info['tap_array'][idx])
        out_ins += [op.inputs[offset:offset+n_ins]]
        offset  += n_ins
    out_ins += [ [] for k in xrange(op.n_nit_sot) ]
    out_ins += [ [op.inputs[offset+k]] for k in xrange(op.n_shared_outs)]

    added = True
    out_idxs_mask = [1 for idx in out_idxs]
    while added:
        added = False
        for pos,idx in enumerate(out_idxs):
            if ( out_idxs_mask[pos] and
                 numpy.any([x in required_inputs for x in out_ins[idx]]) ):
                # This output is required ..
                out_idxs_mask[pos] = 0
                required_inputs += gof.graph.inputs([op.outputs[idx]])
                added = True

    required_outs = [x for i,x in enumerate(out_idxs)
                        if out_idxs_mask[i] == 0]
    not_required = [x for i,x in enumerate(out_idxs) if out_idxs_mask[i]==1]
    return (required_outs, not_required)


def compress_outs(op, not_required, inputs):
    '''
    Helpful function that gets a Scan op, a list of indices indicating
    which outputs are not required anymore and should be removed, and
    a list of inputs to the apply node corresponding to the scan op and
    produces the list of inputs and outputs and the info dictionary where
    the indicated outputs are eliminated. Note that eliminating an output
    means removing its inputs from the inner funciton and from the
    node inputs, and changing the dictionary.
    '''
    info = {}
    info['tap_array']          = []
    info['n_seqs']             = op.info['n_seqs']
    info['n_mit_mot']          = 0
    info['n_mit_mot_outs']     = 0
    info['mit_mot_out_slices'] = []
    info['n_mit_sot']          = 0
    info['n_sit_sot']          = 0
    info['n_shared_outs']      = 0
    info['n_nit_sot']          = 0
    info['truncate_gradient']  = op.info['truncate_gradient']
    info['name']               = op.info['name']
    info['inplace']            = op.info['inplace']
    info['gpu']                = op.info['gpu']
    info['mode']               = op.info['mode']

    op_inputs   = op.inputs[:op.n_seqs]
    op_outputs  = []
    node_inputs = inputs[:op.n_seqs + 1]
    map_old_new = {}

    offset = 0
    ni_offset = op.n_seqs+1
    i_offset  = op.n_seqs
    o_offset  = 0
    curr_pos  = 0
    for idx in xrange(op.info['n_mit_mot']):
        if offset + idx not in not_required:
            map_old_new[offset+idx] = curr_pos
            curr_pos += 1
            info['n_mit_mot'] += 1
            info['tap_array'] += [op.tap_array[offset+idx]]
            info['mit_mot_out_slices'] += [op.mit_mot_out_slices[offset+idx]]
            # input taps
            for jdx in op.tap_array[offset+idx]:
                op_inputs += [op.inputs[i_offset]]
                i_offset += 1
            # output taps
            for jdx in op.mit_mot_out_slices[offset+idx]:
                op_outputs += [op.outputs[o_offset]]
                o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset+idx]]
        else:
            o_offset += len(op.mit_mot_out_slices[offset+idx])
            i_offset += len(op.tap_array[offset+idx])
    info['n_mit_mot_outs'] = len(op_outputs)
    offset    += op.n_mit_mot
    ni_offset += op.n_mit_mot

    for idx in xrange(op.info['n_mit_sot']):
        if offset + idx not in not_required:
            map_old_new[offset+idx] = curr_pos
            curr_pos += 1
            info['n_mit_sot'] += 1
            info['tap_array'] += [op.tap_array[offset+idx]]
            #input taps
            for jdx in op.tap_array[offset+idx]:
                op_inputs += [op.inputs[i_offset]]
                i_offset += 1
            #output taps
            op_outputs += [op.outputs[o_offset]]
            o_offset+=1
            #node inputs
            node_inputs += [inputs[ni_offset+idx]]
        else:
            o_offset+=1
            i_offset+=len(op.tap_array[offset+idx])

    offset    += op.n_mit_sot
    ni_offset += op.n_mit_sot
    for idx in xrange(op.info['n_sit_sot']):
        if offset + idx not in not_required:
            map_old_new[offset+idx] = curr_pos
            curr_pos += 1
            info['n_sit_sot'] += 1
            info['tap_array'] += [op.tap_array[offset+idx]]
            #input taps
            op_inputs += [op.inputs[i_offset]]
            i_offset += 1
            #output taps
            op_outputs += [op.outputs[o_offset]]
            o_offset+=1
            #node inputs
            node_inputs += [inputs[ni_offset+idx]]
        else:
            o_offset+=1
            i_offset+=1

    offset    += op.n_sit_sot
    ni_offset += op.n_sit_sot
    nit_sot_ins = []
    for idx in xrange(op.info['n_nit_sot']):
        if offset + idx not in not_required:
            map_old_new[offset+idx] = curr_pos
            curr_pos += 1
            info['n_nit_sot'] += 1
            op_outputs += [op.outputs[o_offset]]
            o_offset+=1
            nit_sot_ins += [inputs[ni_offset+idx+op.n_shared_outs]]
        else:
            o_offset += 1

    offset += op.n_nit_sot
    shared_ins = []
    for idx in xrange(op.info['n_shared_outs']):
        if offset + idx not in not_required:
            map_old_new[offset+idx] = curr_pos
            curr_pos += 1
            info['n_shared_outs'] += 1
            op_outputs += [ op.outputs[o_offset]]
            o_offset +=1
            op_inputs += [ op.inputs[i_offset]]
            i_offset += 1
            shared_ins += [inputs[ni_offset+idx]]
        else:
            o_offset += 1
            i_offset += 1
    node_inputs += shared_ins
    node_inputs += nit_sot_ins
    # other stuff
    op_inputs += op.inputs[i_offset:]
    node_inputs += inputs[ni_offset+op.n_shared_outs+op.n_nit_sot:]

    return (op_inputs, op_outputs, info, node_inputs, map_old_new)
