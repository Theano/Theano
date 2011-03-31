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

import copy_reg
import cPickle
import itertools
import logging
import numpy

import sys, time, copy

from theano import config
from theano.gof.python25 import partial
from theano.compile.pfunc import rebuild_collect_shared
from theano import gof
from theano import tensor
from theano.tensor.basic import get_constant_value
from theano.gof import Op, Apply
from theano.compile.io import *
from theano.compile.function_module import Supervisor, view_tree_set, alias_root
from theano.misc.safe_asarray import _asarray
import theano.compile.mode as mode_module
from theano.scalar import Scalar, ScalarVariable, ScalarConstant

from theano.sandbox import cuda

import theano

################ Utility Functions and Classes #######################

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_utils')

def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))

def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))


def safe_new(x):
    if cuda.cuda_available and isinstance(x.type, cuda.CudaNdarrayType):
        return tensor.TensorType(
            broadcastable = x.type.broadcastable
            , dtype = config.floatX)()
    else:
        return x.type()

def safe_to_cpu(x):
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

class EmptyObject(object):
    def __init__(self):
        pass

class ScanInnerFunction(object):
    """
    Stripped down, simplified version of theano.function class that has a
    low overhead at calling a function.
    """
    def __init__( self
                 , fn
                 , input_storage
                 , output_storage
                 , env
                 , inputs
                 , outputs
                 , nonmutable_indices
                 , mode
                 , name
                ):

        self.fn                       = fn
        self.input_storage           = input_storage
        self.n_ins                    = len(input_storage)
        self.n_outs                   = len(output_storage)
        self.outputs_storage          = output_storage
        self.maker                    = EmptyObject()
        self.maker.env                = env
        self.maker.inputs             = inputs
        for i in inputs:
            i.update = None
        self.maker.expanded_inputs    = inputs
        self.maker.outputs            = outputs
        self.maker.nonmutable_indices = nonmutable_indices
        self.maker.mode               = mode
        self.name                     = name


    def __call__(self, inputs, outputs):
        t0 = time.time()
        # put data into the storage
        for idx in xrange(self.n_ins):
            self.input_storage[idx][0] = inputs[idx]
        for idx in xrange(self.n_outs):
            self.outputs_storage[idx][0] = outputs[idx][0]
        _t0 = time.time()
        self.fn()
        dt_fn = time.time() - _t0
        for idx in xrange(self.n_outs):
            if outputs[idx][0] is not None:
                if outputs[idx][0] is not self.outputs_storage[idx][0]:
                    if outputs[idx][0].shape:
                        outputs[idx][0][:] = self.outputs_storage[idx][0]
                    else:
                        outputs[idx][0].itemset(self.outputs_storage[idx][0])
        dt_call = time.time() - t0
        if hasattr(self.maker.mode,'fct_call_time'):
            self.maker.mode.fct_call_time[self] += dt_call
            self.maker.mode.fct_call[self]      += 1
        self.maker.mode.fn_time   += dt_fn
        self.maker.mode.call_time += dt_call
        return self.outputs_storage




    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fn']
        del state['input_storage']
        del state['outputs_storage']
        del state['maker'].env
        return state

    def __setstate__(self):
        self.__dict__ = state
        name               = self.name
        mode               = self.maker.mode
        inputs             = self.maker.inputs
        outputs            = self.maker.outputs
        nonmutable_indices = self.maker.nonmutable_indices

        new_inputs, new_outputs = gof.graph.clone( inputs, ouputs )
        env                     = gof.env.Env(new_inputs, new_outputs)
        nonmutable  = []
        for idx in nonmutable_indices :
            nonmutable.append( new_inputs[idx] )

        env.extend(
            Supervisor( inp for inp in nonmutable if
                       not (hasattr(env,'destroyers') and
                            env.destroyers(inp))))

        # If named nodes are replaced, keep the name
        env.extend(gof.toolbox.PreserveNames())
        optimizer, linker = mode.optimizer, copy.copy(mode.linker)
        # optimize the env
        t0 = time.time()
        optimizer(env)
        _logger.debug('Optimizing took %f seconds' %(time.time() - t0))

        if not hasattr(linker, 'accept'):
                raise ValueError( ( "'linker' parameter of FunctionFactory "
                                 "should be a Linker with an accept method "
                                 "or one of %s") %
                                        mode_module.predefined_linkers.keys())

        my_linker = linker.accept ( env )

        input_storage  = []
        output_storage = []
        for input in inputs:
            input_storage += [[ None ]]

        for output in outputs:
            output_storage += [[ None ]]
        t0 = time.time()

        _fn, _i,_o = my_linker.make_thunk( input_storage = input_storage,
                                    output_storage = output_storage)
        _logger.debug('Linking took %f seconds' %(time.time() - t0))
        fn = ScanInnerFunction( _fn
                               , input_storage
                               , output_storage
                               , env)

        t2 = time.time()
        self.fn = _fn
        self.input_storage = input_storage
        self.outputs_storage = output_storage
        if hasattr(mode, 'fct_call_time'):
            mode.fct_call_time.setdefault(fn, 0)
        if hasattr(mode, 'fct_call'):
            mode.fct_call.set_default(fn,0)


def scan_function( inputs
                  , outputs
                  , nonmutable_indices = None
                  , mode               = None
                  , name               = None
                  , slices             = 0
                 ):
    """
    ``Constructor`` of the ScanInnerFunction ( a simplified version of
    theano.function ). This should only be used internally by Scan.

    :param inputs: theano variable that represent the input of the function
    :param outputs: theano expression that represents the outputs of the
                    function
    :param nonmutable_indices: the subset of indices corresponding to
                            nonmutable inputs
    :param mode: compilation mode for the function
    :param name: name of the function
    """
    t1   = time.time()
    mode = mode_module.get_mode(mode)
    if isinstance(mode, (list, tuple)): # "mode comparison" semantics
        _logger.warning('Passing multiple modes is deprecated (20091019)')
        if not mode:
            raise ValueError("Please provide at least one mode.")
        else:
            mode = mode[0]



    ## Replacing the Function Maker
    if not isinstance(outputs, (list, tuple)):
        outputs       = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    new_inputs, new_outputs = gof.graph.clone( inputs, outputs )
    env                     = gof.env.Env(new_inputs, new_outputs)
    nonmutable  = []
    for idx in nonmutable_indices :
        nonmutable.append( new_inputs[idx] )

    env.extend(
        Supervisor( inp for inp in nonmutable if
                   not (hasattr(env,'destroyers') and env.destroyers(inp))))

    # If named nodes are replaced, keep the name
    env.extend(gof.toolbox.PreserveNames())
    optimizer, linker = mode.optimizer, copy.copy(mode.linker)
    # optimize the env
    t0 = time.time()
    optimizer(env)
    _logger.debug('Optimizing took %f seconds' %(time.time() - t0))
    mask = [ 0 for x in env.outputs[slices:] ]


    for i,out in enumerate(env.outputs):
        if (out in env.inputs or
            isinstance(out, tensor.Constant)):
                env.change_input('output', i, Clone()(out) )


    for i in xrange(len(env.outputs[slices:])):
        views_of_output_i = set()
        view_tree_set(alias_root(env.outputs[i]), views_of_output_i)
        copied = False
        # do not allow outputs to be aliased
        for j in xrange(i+1, len(env.outputs)):
            if env.outputs[j] in views_of_output_i:
                mask[i] = 1
                copied = True
                break

        if not copied:
            for input_j in env.inputs:
                # do not allow outputs to be aliased to an inputs (j), unless
                # a) that j'th input has been 'destroyed' by e.g. in-place computations
                if hasattr(env,'get_destroyers_of') and env.get_destroyers_of(input_j):
                    continue
                if input_j in views_of_output_i:
                    mask[i] = 1
                    break


    if not hasattr(linker, 'accept'):
            raise ValueError( ( "'linker' parameter of FunctionFactory "
                             "should be a Linker with an accept method "
                             "or one of %s") %
                                    mode_module.predefined_linkers.keys())


    my_linker = linker.accept ( env )
    input_storage  = []
    output_storage = []
    for input in inputs:
        input_storage += [[ None ]]

    for output in outputs:
        output_storage += [[ None ]]
    t0 = time.time()


    _fn, _i,_o = my_linker.make_thunk( input_storage = input_storage,
                                output_storage = output_storage)

    _logger.debug('Linking took %f seconds' %(time.time() - t0))
    if hasattr(mode, 'apply_time'):
        for i, node in enumerate(env.toposort()):
           mode.apply_time[(i,node)] = 0.0
           assert len(_fn.thunk_groups[i])==1
           mode.op_cimpl[node.op] = hasattr(_fn.thunk_groups[i][0],'cthunk')


    fn = ScanInnerFunction( _fn
                           , input_storage
                           , output_storage
                           , env
                           , inputs
                           , outputs
                           , nonmutable_indices
                           , mode
                           , name
                          )

    t2 = time.time()

    if hasattr(mode, 'compile_time'):
        mode.compile_time += t2-t1
    if hasattr(mode, 'fct_call_time'):
        mode.fct_call_time.setdefault(fn, 0)
    if hasattr(mode, 'fct_call'):
        mode.fct_call.setdefault(fn,0)

    return mask, fn


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


def check_NaN_Inf_None(x):
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
    #shapes      = [ tensor_var.shape[x] for x in xrange(tensor_var.ndim) ]
    #zeros_shape = [size] + shapes[1:]
    #empty       = tensor.zeros( zeros_shape
    #                          , dtype = tensor_var.dtype)
    #return tensor.join(0, tensor_var, empty)
    # V2:
    shapes      = [ tensor_var.shape[x] for x in xrange(tensor_var.ndim) ]
    zeros_shape = [size+shapes[0]] + shapes[1:]
    empty       = tensor.zeros( zeros_shape
                              , dtype = tensor_var.dtype)
    return tensor.set_subtensor(empty[:shapes[0]], tensor_var)



class Clone(Op):
    def __init__(self):
        self.view_map = {0:[0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'clone[as_view]'

    def make_node(self, *inputs):
        x = inputs[0]
        return Apply(self, inputs, [x.type()] )

    def perform( self, node, args, outs):
        outs[0][0] = args[0]

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, args, g_outs):
        return g_outs

cloneOp = Clone()

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

def infer_shape( outs, inputs, input_shapes):
    '''
     Compute the shape of the outputs given the shape of the inputs
     of a theano graph ( assuming that all ops on the way have infer_shape
     implemented).
    '''
    shape_dict = {}
    for inp, inp_shp in zip(inputs, input_shapes):
        shape_dict[inp] = inp_shp

    def local_traverse(out, shape_dict):
        if out in shape_dict:
            return shape_dict
        elif not out.owner:
            if isinstance(out, tensor.TensorConstant):
                shape_dict[out] = out.data.shape
                return shape_dict
            elif isinstance(out, tensor.sharedvar.TensorSharedVariable):
                shape_dict[out] = out.value.shape
                return shape_dict
            else:
                raise ValueError('Could not figure shape of', out)
        else:
            for inp in out.owner.inputs:
                if not inp in shape_dict:
                    shape_dict = local_traverse(inp,shape_dict)
            try:
                self = out.owner.op
                node = out.owner
                input_shapes = [ shape_dict[i] for i in out.owner.inputs]
                shapes = self.infer_shape(node, input_shapes)
                out_idx = node.outputs.index(out)
                shape_dict[out] = shapes[out_idx]
            except:
                shape_dict[out] = None
            return shape_dict
    for out in outs:
        shape_dict = local_traverse(out, shape_dict)
    return [ shape_dict[o] for o in outs]


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
    info['n_other_ignore']     = op.info['n_other_ignore']
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

