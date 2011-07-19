"""
This module provides utility functions for the Scan Op

See scan.py for details on scan
"""
__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
                "Frederic Bastien "
                "James Bergstra "
                "Pascal Lamblin "
                "Arnaud Bergeron")
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

import theano

################ Utility Functions and Classes #######################

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_utils')

def safe_new(x, tag = ''):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name ( old name + tag). This function is used
    by gradient, or the R-op to construct new variables for the inputs of
    the inner graph such that there is no interference between the original
    graph and the newly constructed graph.
    """
    if hasattr(x, 'name') and x.name is not None:
        nw_name = x.name + tag
    else:
        nw_name = None
    if isinstance(x.type, tensor.Constant):
        return x.clone()
    else:
        try:
            x = tensor.as_tensor_variable(x)
        except TypeError:
            # This could happend for example for random states, and I really
            # want to avoid the convoluted logic that checks for cuda
            # ndarrays
            pass
    nw_x = x.type()
    nw_x.name = nw_name
    return nw_x


class until(object):
    """
    Class used to encode the different things the inner function of scan can
    (or needs) to return.

    This class has to be used when scan needs to halt when a condition is
    met, otherwise the list of outputs and dictionary can directly be return
    as a tuple. The reason is that otherwise scan has no way to distinguish
    between the condition and the list of outputs ( unless we enforce and
    order, but since this was not impose up to know it can make quite a bit
    of code to fail).
    """
    def __init__(self, condition, outputs = None, updates = None):
        self.condition = tensor.as_tensor_variable(condition)
        assert self.condition.ndim == 0
        if outputs is None:
            self.outputs = []
        elif type(outputs) in (list, tuple):
            self.outputs = list(outputs)
        else:
            self.outptus = [outputs]
        if updates is None:
            self.updates = {}
        elif type(updates) is dict:
            self.updates = updates
        elif type(udpates) is (list, tuple):
            self.updates = dict(updates)
        else:
            raise Exception( ('Scan could not parse the returned values by'
                              ' the lambda function describing the inner'
                              ' operations of scan '))


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
    cond = None

    def pick_from2(elem0, elem1):
        lupd = {}
        lout = []
        if ( isinstance(elem0,dict) or
                ( isinstance(elem0, (list,tuple)) and
                    isinstance(elem0[0], (list,tuple)))):
            # elem0 is the updates dictionary / list
            lupd = dict(elem0)
            lout = elem1
            if not isinstance(outputs, (list,tuple)):
                lout = [outputs]
        elif ( isinstance(elem1, dict) or
                ( isinstance(elem1, (list,tuple)) and
                    isinstance(elem1[0], (list,tuple))) ):
            # elem1 is the updates dictionary / list
            lupd = dict(elem1)
            lout = elem0
            if not isinstance(outputs, (list,tuple)):
                lout = [outputs]
        else :
            if ( isinstance(outputs_updates, (list,tuple)) and
                    isinstance(outputs_updates[0], (list,tuple))):
                lout = []
                lupd = dict(outputs_updates)
            else:
                lout = outputs_updates
                lupd = {}
        return lupd, lout

    def pick_from1(elem0):
        lupd = {}
        lout = []
        if ( isinstance(elem0, dict) or
            (isinstance(elem0, (list,tuple)) and
             isinstance(elem0[0], (list, tuple)))):
            lupd = dict(elem0)
        else:
            if not isinstance(elem0, (list, tuple)):
                lout = [elem0]
            else:
                lout = elem0
        return lupd, lout

    # we will try now to separate the outputs from the updates
    if not isinstance(outputs_updates, (list,tuple)):
        if isinstance(outputs_updates, dict) :
            # we have just an update dictionary
            updates = outputs_updates
        elif isinstance(outputs_updates, until):
            updates = outputs_updates.updates
            outputs = outputs_updates.outputs
            cond    = outputs_updates.condition
        else:
            outputs = [outputs_updates]
    elif len(outputs_updates) == 1:
            rval = pick_from1(outputs_updates)
            updates = rval[0]
            outputs = rval[1]
    elif len(outputs_updates) == 2:
        elem0 = outputs_updates[0]
        elem1 = outputs_updates[1]
        if isinstance(elem0,until):
            cond = elem0.condition
            rval = pick_from1(elem1)
            updates = rval[0].updates(elem0.updates)
            outputs = rval[1] + elem0.outputs
        elif isinstance(elem1, until):
            cond = elem1.condition
            rval = pick_from1(elem0)
            updates = rval[0].update(elem1.updates)
            outputs = rval[1] + elem1.outputs
        else:
            rval = pick_from2(elem0, elem1)
            updates = rval[0]
            outputs = rval[1]
    elif len(outputs_updates) == 3:
        elem0 = outputs_updates[0]
        elem1 = outputs_updates[1]
        elem2 = outputs_updates[2]
        if isinstance(elem0, until):
            cond = elem0.condition
            rval = pick_from2(elem1, elem2)
            updates = rval[0].update(elem0.updates)
            outputs = rval[1] + elem0.outputs
        elif isinstance(elem1, until):
            cond = elem1.condition
            rval = pick_from2(elem0, elem2)
            updates = rval[0].update(elem1.updates)
            outputs = rval[1] + elem1.outputs
        elif isinstance(elem2, until):
            cond = elem2.condition
            rval = pick_from2(elem0, elem1)
            updates = rval[0].update(elem2.updates)
            outputs = rval[1] + elem2.outputs
        else:
            outputs = outputs_updates
    else:
        outputs = outputs_updates

    # in case you return a tuple .. convert it to a list (there are certain
    # operation that are not permited on tuples, like element assignment)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    # If you return numbers (highly unlikely) this will not go well for
    # theano. We need to convert them to Theano constants:
    for i,out in enumerate(outputs):
        outputs[i] = tensor.as_tensor(out)

    #return cond, outputs, updates
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

def equal_computations(xs,ys, in_xs = None, in_ys = None, strict=True):
    '''
     Checks if to theano graphs represent the same computations (with
     equivalence of inputs defined by map).  Inputs are always assumed
     equal if strict is set to False.
    '''
    import time
    t00 = time.time()

    if in_xs is None:
        in_xs = []
    if in_ys is None:
        in_ys = []


    for x,y in zip(xs,ys):
        if x.owner and not y.owner:
            return False
        if y.owner and not x.owner:
            return False
        if x.owner and y.owner:
            if x.owner.outputs.index(x) != y.owner.outputs.index(y):
                return False

    nds_x = gof.graph.io_toposort(in_xs, xs)
    nds_y = gof.graph.io_toposort(in_ys, ys)
    if len(nds_x) != len(nds_y):
        return False
    common = set(zip(in_xs,in_ys))
    n_nodes = len(nds_x)
    cont = True
    idx = 0
    for dx,dy in zip(xs,ys):
        if not dx.owner or not dy.owner:
            if dy.owner or dx.owner:
                return False
            elif (isinstance(dx, tensor.Constant) and
                isinstance(dy, tensor.Constant) and
                dx.data == dy.data):
                pass
            elif strict:
                if dx != dy:
                    return False
            else:
                if dx.type != dy.type:
                    return False

    while cont and idx < n_nodes:
        nd_x = nds_x[idx]
        nd_y = nds_y[idx]
        if nd_x.op != nd_y.op:
            cont = False
        elif len(nd_x.inputs) != len(nd_y.inputs):
            cont = False
        elif len(nd_x.outputs) != len(nd_y.outputs):
            cont = False
        else:
            for dx,dy in zip(nd_x.inputs, nd_y.inputs):
                if (dx,dy) not in common:
                    if strict and dx!= dy:
                        if (isinstance(dx, tensor.Constant) and
                            isinstance(dy, tensor.Constant) and
                            dx.data == dy.data):
                            pass
                        else:
                            cont = False
                    else:
                        cont = cont and (dx.type == dy.type)

        if cont:
            for dx,dy in zip(nd_x.outputs, nd_y.outputs):
                common.add((dx,dy))
        idx += 1

    return cont





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

def find_up(l_node, f_node):
    r"""
    Goes up in the graph and returns True if a node in nodes is found.
    """
    if isinstance(l_node, gof.Apply):
        l_outs = l_node.outputs
    else:
        l_outs = l_node
    l_ins  = graph.inputs(l_outs)
    nodes = graph.io_toposort(l_ins, l_outs)
    return f_node in nodes


def flatten(l):
    """flattens a list by one level only"""
    return sum(l , [])


def reconstruct_graph(inputs, outputs, tag = None):
    """
    Different interface to clone, that allows you to pass inputs.
    Compared to clone, this method always replaces the inputs with
    new variables of the same type, and returns those ( in the same
    order as the original inputs).
    """
    if tag is None:
        tag = ''
    nw_inputs = [safe_new(x,tag) for x in inputs]
    givens = {}
    for nw_x, x in zip(nw_inputs, inputs):
        givens[x] = nw_x
    nw_outputs = clone( outputs, replace=givens)
    return (nw_inputs, nw_outputs)

class scan_args(object):
    """Parses the inputs and outputs of scan in an easy to manipulate format"""
    def __init__(self, outer_inputs, outer_outputs,
                 _inner_inputs, _inner_outputs, info):
        self.n_steps = outer_inputs[0]
        rval = reconstruct_graph(_inner_inputs, _inner_outputs, '_merge')
        #if info['as_while']:
        #    self.cond = [rval[1][-1]]
        #    inner_outputs = rval[1][:-1]
        #else:
        inner_outputs = rval[1]
        inner_inputs  = rval[0]

        p = 1
        q = 0

        n_seqs = info['n_seqs']
        self.outer_in_seqs = outer_inputs[p:p+n_seqs]
        self.inner_in_seqs = inner_inputs[q:q+n_seqs]
        p += n_seqs
        q += n_seqs

        n_mit_mot = info['n_mit_mot']
        n_mit_sot = info['n_mit_sot']

        self.mit_mot_in_slices = info['tap_array'][:n_mit_mot]
        self.mit_sot_in_slices = info['tap_array'][n_mit_mot:n_mit_mot+n_mit_sot]

        n_mit_mot_ins = sum(len(s) for s in self.mit_mot_in_slices)
        n_mit_sot_ins = sum(len(s) for s in self.mit_sot_in_slices)

        iimm = inner_inputs[q:q+n_mit_mot_ins]
        self.inner_in_mit_mot = []
        qq = 0
        for sl in self.mit_mot_in_slices:
            self.inner_in_mit_mot.append(iimm[qq:qq+len(sl)])
            qq += len(sl)
        q += n_mit_mot_ins

        iims = inner_inputs[q:q+n_mit_sot_ins]
        self.inner_in_mit_sot = []
        qq = 0
        for sl in self.mit_sot_in_slices:
            self.inner_in_mit_sot.append(iims[qq:qq+len(sl)])
            qq += len(sl)
        q += n_mit_sot_ins

        self.outer_in_mit_mot = outer_inputs[p:p+n_mit_mot]
        p += n_mit_mot
        self.outer_in_mit_sot = outer_inputs[p:p+n_mit_sot]
        p += n_mit_sot

        n_sit_sot = info['n_sit_sot']
        self.outer_in_sit_sot = outer_inputs[p:p+n_sit_sot]
        self.inner_in_sit_sot = inner_inputs[q:q+n_sit_sot]
        p += n_sit_sot
        q += n_sit_sot

        n_shared_outs = info['n_shared_outs']
        self.outer_in_shared = outer_inputs[p:p+n_shared_outs]
        self.inner_in_shared = inner_inputs[q:q+n_shared_outs]
        p += n_shared_outs
        q += n_shared_outs

        n_nit_sot = info['n_nit_sot']
        self.outer_in_nit_sot = outer_inputs[p:p+n_nit_sot]
        p += n_nit_sot

        self.outer_in_non_seqs = outer_inputs[p:]
        self.inner_in_non_seqs = inner_inputs[q:]

        # now for the outputs
        p = 0
        q = 0

        self.mit_mot_out_slices = info['mit_mot_out_slices']
        n_mit_mot_outs = info['n_mit_mot_outs']
        self.outer_out_mit_mot = outer_outputs[p:p+n_mit_mot]
        iomm = inner_outputs[q:q+n_mit_mot_outs]
        self.inner_out_mit_mot = []
        qq = 0
        for sl in self.mit_mot_out_slices:
            self.inner_out_mit_mot.append(iomm[qq:qq+len(sl)])
            qq += len(sl)
        p += n_mit_mot
        q += n_mit_mot_outs

        self.outer_out_mit_sot = outer_outputs[p:p+n_mit_sot]
        self.inner_out_mit_sot = inner_outputs[q:q+n_mit_sot]
        p += n_mit_sot
        q += n_mit_sot

        self.outer_out_sit_sot = outer_outputs[p:p+n_sit_sot]
        self.inner_out_sit_sot = inner_outputs[q:q+n_sit_sot]
        p += n_sit_sot
        q += n_sit_sot

        self.outer_out_nit_sot = outer_outputs[p:p+n_nit_sot]
        self.inner_out_nit_sot = inner_outputs[q:q+n_nit_sot]
        p += n_nit_sot
        q += n_nit_sot

        self.outer_out_shared = outer_outputs[p:p+n_shared_outs]
        self.inner_out_shared = inner_outputs[q:q+n_shared_outs]
        p += n_shared_outs
        q += n_shared_outs


        self.other_info = dict()
        for k in ('truncate_gradient', 'name', 'mode', 'inplace',
                  'gpu', 'profile'):
            self.other_info[k] = info[k]

    inner_inputs = property(lambda self: (self.inner_in_seqs +
                                          flatten(self.inner_in_mit_mot) +
                                          flatten(self.inner_in_mit_sot) +
                                          self.inner_in_sit_sot +
                                          self.inner_in_shared +
                                          self.inner_in_non_seqs))

    outer_inputs = property(lambda self: ([self.n_steps] +
                                          self.outer_in_seqs +
                                          self.outer_in_mit_mot +
                                          self.outer_in_mit_sot +
                                          self.outer_in_sit_sot +
                                          self.outer_in_shared +
                                          self.outer_in_nit_sot +
                                          self.outer_in_non_seqs))

    inner_outputs = property(lambda self: (flatten(self.inner_out_mit_mot) +
                                           self.inner_out_mit_sot +
                                           self.inner_out_sit_sot +
                                           self.inner_out_nit_sot +
                                           self.inner_out_shared))

    outer_outputs = property(lambda self: (self.outer_out_mit_mot +
                                           self.outer_out_mit_sot +
                                           self.outer_out_sit_sot +
                                           self.outer_out_nit_sot +
                                           self.outer_out_shared))

    info = property(lambda self: dict(n_seqs=len(self.outer_in_seqs),
                                      n_mit_mot=len(self.outer_in_mit_mot),
                                      n_mit_sot=len(self.outer_in_mit_sot),
                                      tap_array=(self.mit_mot_in_slices +
                                                 self.mit_sot_in_slices +
                                                 [[-1]] * len(self.inner_in_sit_sot)),
                                      n_sit_sot=len(self.outer_in_sit_sot),
                                      n_nit_sot=len(self.outer_in_nit_sot),
                                      n_shared_outs=len(self.outer_in_shared),
                                      n_mit_mot_outs=sum(len(s) for s in self.mit_mot_out_slices),
                                      mit_mot_out_slices=self.mit_mot_out_slices,
                                      **self.other_info))

    def __copy__(self):
        res = object.__new__(type(self))
        res.__dict__.update(self.__dict__)
        # also copy mutable attrs
        for attr in self.__dict__:
            if (attr.startswith('inner_in') or attr.startswith('inner_out') or
                attr.startswith('outer_in') or attr.startswith('outer_out') or
                attr in ('mit_mot_out_slices', 'mit_mot_in_slices',
                         'mit_sot_in_slices', 'other_info')):
                setattr(res, attr, copy.copy(getattr(self, attr)))
        return res

    def merge(self, other):
        res = copy.copy(self)
        for attr in self.__dict__:
            if (attr.startswith('inner_in') or attr.startswith('inner_out') or
                attr.startswith('outer_in') or attr.startswith('outer_out') or
                attr in ('mit_mot_out_slices', 'mit_mot_in_slices',
                         'mit_sot_in_slices')):
                getattr(res, attr).extend(getattr(other, attr))
        return res
