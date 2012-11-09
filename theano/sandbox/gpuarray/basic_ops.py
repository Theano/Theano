import numpy

import theano
from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar, config
from theano.scalar import Scalar

from theano.gof.python25 import all, any

import pygpu
from pygpu import gpuarray, elemwise

from type import GpuArrayType

def as_gpuarray_variable(x):
    if hasattr(x, '_as_GpuArrayVariable'):
        return x._as_GpuArrayVariable()
    # TODO we need to have the cuda -> gpu path taken care of.
    tensor_x = tensor.as_tensor_variable(x)
    return gpu_from_host(tensor_x)


def as_gpuarray(x, kind, context):
    return gpuarray.array(x, kind=kind, context=context, copy=False)


class HostFromGpu(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        return Apply(self, [x],
                     [tensor.TensorType(dtype=x.dtype,
                                        broadcastable=x.broadcastable,)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = numpy.asarray(x)

    def grad(self, inputs, grads):
        gz, = grads
        return [gpu_from_host(gz)]
                                                   
    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, tensor.TensorType):
            return [gpu_from_host(ev)]
        else:
            return [ev]

    def infer_shape(self, node, xshp):
        return xshp


host_from_gpu = HostFromGpu()


class GpuFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        type = node.outputs[0].type
        z[0] = gpuarray.array(x, kind=type.kind, context=type.context)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isintance(ev, GpuArrayType):
            return [host_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp


gpu_from_host = GpuFromHost()


class GpuFromCuda(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        if not isinstance(x.type, CudaNdArrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)]())

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if globals.kind == 'cuda':
            base = x
            while hasattr(base, 'base') and base.base is not None:
                base = base.base
            raise NotImplementedError("How are we going to get a gpudata pointer from here")
            x[0] = gpuarray.from_gpudata(b, 0, x.dtype, x.shape,
                                         base=base, kind=globals.kind,
                                         context=globals.context,
                                         strides=x.strides)
        else:
            z[0] = gpuarray.array(numpy.asarray(x), kind=globals.kind,
                                  context=globals.context)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isintance(ev, GpuArrayType):
            return [host_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp
