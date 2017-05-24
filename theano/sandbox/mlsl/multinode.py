"""
Module contains transport independent implementation of communication Ops for Theano
Supported transports:
    - mlsl (requires mlsl.py)
"""
from ctypes import c_void_p

import numpy as np

import theano
import theano.tensor as T
from theano.gof import local_optimizer, Variable
from theano.tensor.opt import in2out
from theano.compile.mode import optdb

import theano.sandbox.mlsl as mlsl


class Distribution(object):
    """
    Distribution class
    Initializes environment, own rank and number of ranks (size)

    Dependency (must be defined in the transport layer):
    dist_init()
    """
    init = 0
    rank = 0
    size = 1
    if not init:
        init = 1
        rank, size = mlsl.dist_init()

    def destroy(self):
        mlsl.dist_finalize()


def set_global_batch_size(size):
    print ('global_batch_size = ' + str(size))
    mlsl.set_global_batch_size(size)


def set_param_count(param_count):
    print ('param_count = ' + str(param_count))
    mlsl.set_param_count(param_count)
"""
Collective parameters:

dict coll_params:
    key = collective kind
    value = (input/output shape layout, derivative kind, color)

collectives which support inplace optimization
"""
coll_params = {
    'Allgather'     : ('1',   'x',   'Reduce_scatter', 1), 
    'Allreduce'     : ('1',   '1',   'Allreduce',      2), 
    'Alltoall'      : ('1,x', 'x,1', 'Alltoall',       3), 
    'Bcast'         : ('1',   '1',   'Reduce',         4), 
    'Gather'        : ('x',   '1',   'Scatter',        5), 
    'Reduce'        : ('1',   '1',   'Bcast',          6), 
    'Reduce_scatter': ('x',   '1',   'Allgather',      7), 
    'Scatter'       : ('1',   'x',   'Gather',         8) }
coll_supported_inplace = [ 'Allreduce' ]
seqn = 0

"""
Utilities
"""
def addr(x):
    xaddr, offset = x.ctypes.data_as(c_void_p), 0
    for i in range(len(x.shape)):
        if x.strides[i] < 0: offset += (x.shape[i]-1)*x.strides[i]
    xaddr.value += offset
    return xaddr

def new_shape(shape,layout):
    out_shape = [dim for dim in shape]
    for i in range(min(len(layout[0]),len(out_shape))):
        il, ol = layout[0][i],layout[1][i]
        out_shape[i] = out_shape[i]*ol/il
    return out_shape

class Coll(Distribution,theano.Op):
    """
    Class Coll

    Dependency (must be defined in the transport layer):
    ctxt_init()
    coll_perform()
    """
    __props__ = ('inplace','blocking','kind','root','priority','seqn','color')
    def __init__(self, kind='Allreduce', inplace=False, blocking=True, root=0, layout=None,
                 priority=None, seqn=0, color=0):
        self.kind = kind
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}
        self.blocking = blocking
        self.root = root
        if layout == None: layout = coll_params[kind][0:2]
        self.layout = layout
        self.real_layout =  \
            [[int(s) for s in l.replace('x',str(self.size)).split(',')] for l in layout]
        self.priority = priority
        self.seqn = seqn
        self.color = color
        if (seqn > 0):
            self.ctxt = mlsl.ctxt_init(self.seqn)

    def make_node(self,x):
        x = theano.tensor.as_tensor_variable(x)
        outputs = [x.type(),]
        # Non-blocking call return an extra int64 output as a placeholder for additional info
        if not self.blocking:
            y = theano.tensor.lscalar()
            outputs.append(y.type())
        return theano.Apply(self,[x],outputs)
    def perform(self,node,inputs,outputs):
        # Prepare outputs and call arguments
        x, = inputs
        # Non-blocking call return an extra output - integer reqeust
        if self.blocking:
            y, = outputs
            req_addr = 0
        else:
            y, req = outputs
            req[0] = np.empty((),dtype=np.int64)
            req_addr = addr(req[0])
        inbuf = addr(x)
        if self.inplace or self.kind == 'Bcast':
            y[0] = x
            outbuf = addr(x)
        else:
            y[0] = np.empty(new_shape(x.shape,self.real_layout),dtype=x.dtype)
            outbuf = addr(y[0])
        mlsl.coll_perform(inbuf, x.dtype, [x.shape,y[0].shape], self.seqn)

    def infer_shape(self, node, shapes):
        out_shapes = [new_shape(shapes[0],self.real_layout),]
        if not self.blocking:
            out_shapes.append(())
        return out_shapes
    def grad(self,inputs,grads):
        derivative = coll_params[self.kind][2]
        layout = [self.layout[1],self.layout[0]]
        return [coll(grads[0],kind=derivative,blocking=self.blocking,root=self.root,layout=layout,priority=self.priority),]


class Wait(Distribution,theano.Op):
    """
    Class Wait

    Dependency (must be defined in the transport layer):
    wait_perform()
    """
    __props__ = ('inplace','priority','seqn', 'color')
    def __init__(self, inplace=False, priority=None, seqn=0, color=0):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}
        self.priority = priority
        self.seqn = seqn
        self.color = color
    def make_node(self,x,req):
        x = theano.tensor.as_tensor_variable(x)
        req = theano.tensor.as_tensor_variable(req)
        return theano.Apply(self,[x,req],[x.type()])
    def perform(self,node,inputs,outputs):
        # TODO: Here's MPI specific yet, need to abstract
        x, req = inputs
        y, = outputs
        mlsl.wait_perform(self.seqn)
        y[0] = x
    def connection_pattern(self, node):
        return [[True], [False]]
    def grad(self,inputs,grads):
        return [grads[0],theano.gradient.DisconnectedType()()]


def coll(x,kind='Allreduce',blocking=True,inplace=False,root=0,layout=None,priority=None):
    """
    Aggregate collective operation - blocking or non-blocking
    """
    global seqn
    seqn = seqn+1
    if blocking:
        return Coll(kind=kind,blocking=blocking,inplace=inplace,root=root,layout=layout,priority=None)(x)
    else:
        color = coll_params[kind][3]
        y, z = Coll(kind=kind,blocking=blocking,inplace=inplace,root=root,layout=layout,
                    priority=None,seqn=seqn,color=color)(x)
        return Wait(inplace=inplace,priority=priority,seqn=seqn,color=-color)(y, z)

"""
Optimizations:

inplace
"""
@local_optimizer([Coll()], inplace=True)
def local_coll_inplace(node):
    op = node.op
    if isinstance(op, Coll) and op.inplace is False and op.kind in coll_supported_inplace :
        outputs = Coll(kind=op.kind,blocking=op.blocking,inplace=True,root=op.root,
                       priority=op.priority,seqn=op.seqn,color=op.color)(*node.inputs)
        if isinstance(outputs,list):
            return outputs
        else:
            return [outputs]
coll_inplace = in2out(local_coll_inplace,name="coll_inplace")
optdb.register('Coll_inplace',
        coll_inplace,
        100.0, 'fast_run', 'inplace', 'coll_inplace')

@local_optimizer([Wait()], inplace=True)
def local_wait_inplace(node):
    op = node.op
    if isinstance(op, Wait) and op.inplace is False:
        return [Wait(inplace=True,priority=op.priority,seqn=op.seqn,color=op.color)(*node.inputs)]
wait_inplace = in2out(local_wait_inplace,name="wait_inplace")
optdb.register('Wait_inplace',
        wait_inplace,
        101.0, 'fast_run', 'inplace', 'wait_inplace')

###########################################################################
########## User interface - to be exposed for Theano-based scripts ########
###########################################################################
"""
partition()
Returns a partition, no other side effects

Inputs:
    - instance of {int,slice,tuple,list,numpy.ndarray,theano.variable} type
    - optional layout (default 'x'). Other than 'x' or equivalent parameters
      don't work with int.

Outputs:
    - object of the same type as input.0. Returns a partition of the input
      object which belong to particular rank in distributed environment.
      The partitioned dimension is set by input.1 - layout.
"""
def partition(x,layout='x'):
    """
    Partition numpy objects and scalars (that could be seen as shapes)
    """
    dist = Distribution()
    rank = dist.rank
    size = dist.size
    if isinstance(x, np.ndarray) and len(x.shape) > 0:
        part = x.shape[0]/size;
        if len(x.shape) == 1:
            return x[rank*part: (rank+1)*part]
        else:
            return x[rank*part: (rank+1)*part,:]
    elif isinstance(x, Variable) and x.ndim > 0:
        part = T.shape(x)[0]/size;
        if x.ndim == 1:
            return x[rank*part: (rank+1)*part]
        else:
            return x[rank*part: (rank+1)*part,:]
    elif isinstance(x, list) and len(x) > 0:
        y = x
        y[0] = y[0]/size
        return y
    elif isinstance(x, tuple) and len(x) > 0:
        y = list(x)
        y[0] = y[0]/size
        return tuple(y)
    elif isinstance(x, slice):
        start, stop, step = x.start, x.stop, x.step
        if start is None: start = 0
        if step is None: step = 1
        n = (stop-start)/step
        part = n/size
        return slice(start+part*step*rank, start+part*step*(rank+1), step)
    else:
        return x/size

"""
distributed()
Set 'layout' attribute to input tensor.

Inputs:
    - Theano variable (tensor) for which we need to specify the distribution properties
      in order to make appropriate communication
    - optional layout (default 'x'). Defines current tensor distribution properties.
"""
def distributed(x,layout='x'):
    x.layout = layout

"""
distribute()
Returns transformed object due to distribution layout change. To be used anywhere in
forward prop computation.

Inputs:
    - Theano variable (tensor), which has 'layout' attribute defined (specifically with
      distributed() call on this object prior to this function call)
    - optional layout (default 'x'). Specify the final tensor distribution properties.
Outputs:
    - object of the same type as input.0. Result of re-distribution wrt to input and
      output shape.
"""
def distribute(x,layout='x'):
    if not hasattr(x,'layout'):
        distributed(x)
    if x.layout == 'x' and layout == '1':
        kind = 'Allgather'
        io_layout = None
    elif x.layout == '1' and layout == 'x':
        kind = 'Reduce_scatter'
        io_layout = None
    elif (x.layout == 'x' or x.layout == 'x,1') and layout == '1,x':
        kind = 'Alltoall'
        io_layout = ('1,x','x,1')
    elif x.layout == '1,x' and (layout == 'x' or layout == 'x,1'):
        kind = 'Alltoall'
        io_layout = ('x,1','1,x')
    else:
        return x
    # Not sure if we want it non-blocking for here should be a critical path
    # I've got implression non-blocking hangs sometimes
    #y = coll(x,kind=kind,blocking=False,inplace=False,root=0,layout=io_layout,priority=None)
    y = coll(x,kind=kind,blocking=True,inplace=False,root=0,layout=io_layout,priority=None)
    y.layout = layout
    return y

"""
collect()
Collects parts of gradients wrt distributed/replicated parameter. To be used upon
gradient computation.

Inputs:
    - Theano variable (tensor), which should be collected
    - Theano variable (tensor) representing parameter, wrt which gradient is computed.
      Its distribution properties should be set with distributed(), otherwise
      replication ('1') is assumed.
Outputs:
    - object of the same type as input.0. Result of collecting gradients taking into
      account the parameter distribution properties.
"""
def collect(x,y):
    dist = Distribution()
    #if dist.size >1 and (not hasattr(y,'layout') or y.layout == '1'):
    if not hasattr(y,'layout') or y.layout == '1':
        return (coll(x,kind='Allreduce',blocking=False,priority='low') / dist.size)
    else:
        return x
