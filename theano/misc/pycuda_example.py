"""
This file show how we can use Pycuda compiled fct in a Theano Op. Do no use them in production code. See the TODO.

You can use them as a guide to use your pycuda code into a Theano op.

The PycudaElemwiseSourceModule op use pycuda code generated with pycuda.compiler.SourceModule

The PycudaElemwiseKernel op use pycuda code generated with pycuda.elementwise.ElementwiseKernel

Their is a test in test_pycuda.py. 

This don't work with broadcast and non-contiguous memory as pycuda don't support that, but we make sure we don't introduce problem.

"""

import numpy

import theano
import theano.tensor as T
from theano.gof import Op, Apply, local_optimizer, EquilibriumDB
from theano.sandbox.cuda import GpuElemwise, CudaNdarrayType, CudaNdarray
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous, host_from_gpu
from theano.sandbox.cuda.opt import gpu_seqopt

from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from pycuda.gpuarray import splay

import pycuda.autoinit

class PycudaElemwiseSourceModule(Op):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, inplace_pattern = {}, name = None):
        self.name = name
        self.scalar_op = scalar_op
        self.inplace_pattern=None

    def __str__(self):
        if self.name is None:
            if self.inplace_pattern:
                items = self.inplace_pattern.items()
                items.sort()
                return "PycudaElemwiseSourceModule{%s}%s" % (self.scalar_op, str(items))
            else:
                return "PycudaElemwiseSourceModule{%s}" % (self.scalar_op)
        else:
            return self.name

    def make_node(self, *inputs):
        _inputs = [gpu_contiguous(as_cuda_ndarray_variable(i)) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('different ranks among inputs')

        assert not any([any(i.type.broadcastable) for i in inputs])
        assert len(inputs)==2#TODO remove

        otype = CudaNdarrayType(broadcastable=[False]*_inputs[0].type.ndim)
        assert self.nout == 1

        #TODO change the scalar op with the good c_code!
        fct_name = "pycuda_elemwise_%s"%str(self.scalar_op)
        out_node = Apply(self, _inputs, [otype() for o in xrange(self.nout)])
        in_name = ["i"+str(id) for id in range(len(inputs))]
        out_name = ["o"+str(id) for id in range(self.nout)]
        c_code = self.scalar_op.c_code(out_node, "some_name", tuple([n+"[i]"for n in in_name]), tuple(n+"[i]"for n in out_name), {})
        c_code_param = ", ".join([var.type.dtype_specs()[1]+" *"+name for var,name in zip(inputs,in_name) + zip(out_node.outputs,out_name)])
        mod = SourceModule("""
#include<Python.h>
#include <numpy/arrayobject.h>
  __global__ void %s(%s)
  {
    int i = threadIdx.x + threadIdx.y*blockDim.x;
    %s
  }
  """%(fct_name,c_code_param,c_code))
        self.pycuda_fct = mod.get_function(fct_name)
        return out_node

    def perform(self, node, inputs, (z,)):
        #TODO support broadcast!
        #TODO assert all input have the same shape
        if z[0] is None or z[0].shape!=inputs[0].shape:
            z[0] = theano.sandbox.cuda.CudaNdarray.zeros(inputs[0].shape)
        self.pycuda_fct(inputs[0],inputs[1],z[0], block=(inputs[0].shape[0],inputs[0].shape[1],1))


class PycudaElemwiseKernel(Op):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, inplace_pattern = {}, name = None):
        self.name = name
        self.scalar_op = scalar_op
        self.inplace_pattern=None

    def __str__(self):
        if self.name is None:
            if self.inplace_pattern:
                items = self.inplace_pattern.items()
                items.sort()
                return "PycudaElemwiseKernel{%s}%s" % (self.scalar_op, str(items))
            else:
                return "PycudaElemwiseKernel{%s}" % (self.scalar_op)
        else:
            return self.name

    def make_node(self, *inputs):
        _inputs = [gpu_contiguous(as_cuda_ndarray_variable(i)) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('different ranks among inputs')

        assert not any([any(i.type.broadcastable) for i in inputs])
        assert len(inputs)==2#TODO remove

# output is broadcastable only along dimensions where all inputs are broadcastable
        broadcastable = []
        for d in xrange(_inputs[0].type.ndim):
            bcast_d = True
            for i in _inputs:
                if not i.type.broadcastable[d]:
                    bcast_d = False
                    break
            broadcastable.append(bcast_d)
        assert len(broadcastable) == _inputs[0].type.ndim

        otype = CudaNdarrayType(broadcastable=broadcastable)
        assert self.nout == 1

        out_node = Apply(self, _inputs, [otype() for o in xrange(self.nout)])
        in_name = ["i"+str(id) for id in range(len(inputs))]
        out_name = ["o"+str(id) for id in range(self.nout)]
        c_code = self.scalar_op.c_code(out_node, "some_name", tuple([n+"[i]"for n in in_name]), tuple(n+"[i]"for n in out_name), {})
        
        self.pycuda_fct = ElementwiseKernel(
            ", ".join([var.type.dtype_specs()[1]+" *"+name for var,name in zip(inputs,in_name) + zip(out_node.outputs,out_name)]),
            c_code,
            "pycuda_elemwise_kernel_%s"%str(self.scalar_op),
            preamble="""#include<Python.h>
#include <numpy/arrayobject.h>""")
        return out_node

    def perform(self, node, inputs, (z,)):
        #TODO assert all input have the same shape
        if z[0] is None or z[0].shape!=inputs[0].shape:
            z[0] = theano.sandbox.cuda.CudaNdarray.zeros(inputs[0].shape)
        i = inputs + z
        sp = splay(i[0].mem_size)
        self.pycuda_fct(*i)#, grid=sp[0], block=sp[1])

pycuda_optimizer = EquilibriumDB()
gpu_seqopt.register("pycuda_optimizer", pycuda_optimizer, 1.5, "fast_run")

@local_optimizer([])
def local_pycuda_gpu_elemwise(node):
    """
       GpuElemwise -> PycudaElemwiseSourceModule
    """
    if isinstance(node.op, GpuElemwise):
        if not any([ any(i.type.broadcastable) for i in node.inputs]) and all([i.ndim<=2 for i in node.inputs]):
            new_op = PycudaElemwiseSourceModule(node.op.scalar_op, node.op.inplace_pattern)(*node.inputs)
            return [new_op]

pycuda_optimizer.register("local_pycuda_gpu_elemwise", local_pycuda_gpu_elemwise)

@local_optimizer([])
def local_pycuda_gpu_elemwise_kernel(node):
    """
       GpuElemwise -> PycudaElemwiseKernel
    """
    if isinstance(node.op, GpuElemwise):
        if not any([ any(i.type.broadcastable) for i in node.inputs]):
            new_op = PycudaElemwiseKernel(node.op.scalar_op, node.op.inplace_pattern)(*node.inputs)
            return [new_op]

pycuda_optimizer.register("local_pycuda_gpu_elemwise_kernel", local_pycuda_gpu_elemwise_kernel, 1.5)
