import numpy
import pyopencl as cl
import pyopencl.array

import theano
from theano import Op, Apply
from theano import tensor

import conf
from type import CLArrayType

def as_cl_variable(x):
    if hasattr(x, '_as_cl_variable'):
        return x._as_cl_variable()
    tensor_x = tensor.as_tensor_variable(x)
    return cl_from_host(tensor_x)

def as_cl_array(obj):
    if isinstance(obj, numpy.ndarray):
        return cl.array.to_device(conf.default_queue, obj)
    elif isinstance(obj, cl.array.Array):
        return obj
    else:
        raise TypeError("Can't convert %s to cl Array"%(type(obj),))

class HostFromCL(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'HostFromCL'

    def make_node(self, x):
        if not isinstance(x.type, CLArrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype, broadcastable=x.broadcastable)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = inp.get()

    def grad(self, inputs, grads):
        gz, = grads
        return [cl_from_host(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isintance(ev, tensor.TensorType):
            return cl_from_host(ev)
        else:
            return [ev]

    def infer_shape(self, node, xshp):
        return xshp

host_from_cl = HostFromCL()

class CLFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'CLFromHost'

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CLArrayType(dtype=x.dtype, broadcastable=x.broadcastable)()])
    
    def perfor(self, x):
        x, = inp
        z, = out
        z[0] = cl.array.to_device(x)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, CLArrayType):
            return [host_from_gpu(ev)]
        else:
            return [ev]

    def infer_shape(self, node, xshp):
        return xshp

cl_from_host = CLFromHost()
