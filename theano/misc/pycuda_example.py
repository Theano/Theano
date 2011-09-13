"""
This file show how we can use Pycuda compiled fct in a Theano Op. Do no use those op in production code. See the TODO.

You can use them as a guide to use your pycuda code into a Theano op.

The PycudaElemwiseSourceModuleOp is a Theano op use pycuda code generated with pycuda.compiler.SourceModule

The PycudaElemwiseKernelOp op use pycuda code generated with pycuda.elementwise.ElementwiseKernel. It must be wrapper by TheanoElementwiseKernel.

Their is a test in test_pycuda.py.

This don't work with broadcast and non-contiguous memory as pycuda don't support that, but we make sure we don't introduce problem.
  If the memory is non-contiguous, we create a new copy that is contiguous.
  If their is broadcasted dimensions, we raise an error.
"""

import numpy

import theano
from theano.gof import Op, Apply, local_optimizer, EquilibriumDB
from theano.sandbox.cuda import GpuElemwise, CudaNdarrayType
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous
from theano.sandbox.cuda.opt import gpu_seqopt

import pycuda_init
if not pycuda_init.pycuda_available:
    raise Exception("No pycuda available. You can't load pycuda_example.py")

import pycuda
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from pycuda.tools import VectorArg

def theano_parse_c_arg(c_arg):
    c_arg = c_arg.replace('npy_float32','float')
    c_arg = c_arg.replace('npy_float64','double')
    c_arg = c_arg.replace('npy_int32','int')
    c_arg = c_arg.replace('npy_int8','char')
    c_arg = c_arg.replace('npy_ucs4','unsigned int')
    c_arg = c_arg.replace('npy_uint32','unsigned int')
    c_arg = c_arg.replace('npy_uint16','unsigned short')
    c_arg = c_arg.replace('npy_uint8','unsigned char')
    return pycuda.tools.parse_c_arg(c_arg)

class TheanoElementwiseKernel(pycuda.elementwise.ElementwiseKernel):
    def __init__(self, arguments, operation,
                 name="kernel", keep=False, options=[], **kwargs):
        if isinstance(arguments, basestring):
            arguments = [theano_parse_c_arg(arg) for arg in arguments.split(",")]
            pycuda.elementwise.ElementwiseKernel.__init__(self, arguments, operation, name, keep, options, **kwargs)

    def __call__(self, *args):
        vectors = []

        invocation_args = []
        for arg, arg_descr in zip(args, self.arguments):
            if isinstance(arg_descr, VectorArg):
                vectors.append(arg)
                invocation_args.append(arg.gpudata)
            else:
                invocation_args.append(arg)

        repr_vec = vectors[0]
        invocation_args.append(repr_vec.mem_size)
        if hasattr(repr_vec,"_block") and hasattr(repr_vec,"_grid"):
            self.func.set_block_shape(*repr_vec._block)
            self.func.prepared_call(repr_vec._grid, *invocation_args)
        else:
            _grid, _block = pycuda.gpuarray.splay(repr_vec.mem_size)
            self.func.set_block_shape(*_block)
            self.func.prepared_call(_grid, *invocation_args)


class PycudaElemwiseSourceModuleOp(Op):
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
                return self.__class__.__name__+"{%s}%s" % (self.scalar_op, str(items))
            else:
                return self.__class__.__name__+"{%s}" % (self.scalar_op)
        else:
            return self.name

    def make_node(self, *inputs):
        _inputs = [gpu_contiguous(as_cuda_ndarray_variable(i)) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('different ranks among inputs')

        if any([any(i.type.broadcastable) for i in inputs]):
            raise Exception("pycuda don't support broadcasted dimensions")
        assert len(inputs)==2#TODO remove

        otype = CudaNdarrayType(broadcastable=[False]*_inputs[0].type.ndim)
        assert self.nout == 1

        fct_name = "pycuda_elemwise_%s"%str(self.scalar_op)
        out_node = Apply(self, _inputs, [otype() for o in xrange(self.nout)])
        in_name = ["i"+str(id) for id in range(len(inputs))]
        out_name = ["o"+str(id) for id in range(self.nout)]
        c_code = self.scalar_op.c_code(out_node, "some_name", tuple([n+"[i]"for n in in_name]), tuple(n+"[i]"for n in out_name), {})
        c_code_param = ", ".join([var.type.dtype_specs()[1]+" *"+name for var,name in zip(inputs,in_name) + zip(out_node.outputs,out_name)]+["int size"])
        mod = SourceModule("""
#include<Python.h>
#include <numpy/arrayobject.h>
  __global__ void %s(%s)
  {
    int i = (blockIdx.x+blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y);
    i += threadIdx.x + threadIdx.y*blockDim.x;
    if(i<size){
        %s
    }
  }
  """%(fct_name,c_code_param,c_code))
        self.pycuda_fct = mod.get_function(fct_name)
        return out_node

    def perform(self, node, inputs, out):
        #TODO support broadcast!
        #TODO assert all input have the same shape
        z, = out
        if z[0] is None or z[0].shape!=inputs[0].shape:
            z[0] = theano.sandbox.cuda.CudaNdarray.zeros(inputs[0].shape)
        if inputs[0].shape != inputs[1].shape:
            raise TypeError("PycudaElemwiseSourceModuleOp: inputs don't have the same shape!")

        if inputs[0].size > 512:
            grid = (int(numpy.ceil(inputs[0].size / 512.)),1)
            block = (512,1,1)
        else:
            grid = (1,1)
            block = (inputs[0].shape[0],inputs[0].shape[1],1)
        self.pycuda_fct(inputs[0], inputs[1], z[0], numpy.intc(inputs[1].size), block=block, grid=grid)


class PycudaElemwiseKernelOp(Op):
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
                return self.__class__.__name__+"{%s}%s" % (self.scalar_op, str(items))
            else:
                return self.__class__.__name__+"{%s}" % (self.scalar_op)
        else:
            return self.name

    def make_node(self, *inputs):
        _inputs = [gpu_contiguous(as_cuda_ndarray_variable(i)) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('different ranks among inputs')

        if any([any(i.type.broadcastable) for i in inputs]):
            raise Exception("pycuda don't support broadcasted dimensions")
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

        self.pycuda_fct = TheanoElementwiseKernel(
            ", ".join([var.type.dtype_specs()[1]+" *"+name for var,name in zip(inputs,in_name) + zip(out_node.outputs,out_name)]),
            c_code,
            "pycuda_elemwise_kernel_%s"%str(self.scalar_op),
            preamble="""#include<Python.h>
#include <numpy/arrayobject.h>""")
        return out_node

    def perform(self, node, inputs, out):
        #TODO assert all input have the same shape
        z, = out
        if z[0] is None or z[0].shape!=inputs[0].shape:
            z[0] = theano.sandbox.cuda.CudaNdarray.zeros(inputs[0].shape)
        i = inputs + z
        self.pycuda_fct(*i)

pycuda_optimizer = EquilibriumDB()
gpu_seqopt.register("pycuda_optimizer", pycuda_optimizer, 1.5, "fast_run")

@local_optimizer([])
def local_pycuda_gpu_elemwise(node):
    """
       GpuElemwise -> PycudaElemwiseSourceModuleOp
    """
    if isinstance(node.op, GpuElemwise):
        if not any([ any(i.type.broadcastable) for i in node.inputs]) and all([i.ndim<=2 for i in node.inputs]):
            new_op = PycudaElemwiseSourceModuleOp(node.op.scalar_op, node.op.inplace_pattern)(*node.inputs)
            return [new_op]

pycuda_optimizer.register("local_pycuda_gpu_elemwise", local_pycuda_gpu_elemwise)

@local_optimizer([])
def local_pycuda_gpu_elemwise_kernel(node):
    """
       GpuElemwise -> PycudaElemwiseKernelOp
    """
    if isinstance(node.op, GpuElemwise):
        if not any([ any(i.type.broadcastable) for i in node.inputs]):
            new_op = PycudaElemwiseKernelOp(node.op.scalar_op, node.op.inplace_pattern)(*node.inputs)
            return [new_op]

pycuda_optimizer.register("local_pycuda_gpu_elemwise_kernel", local_pycuda_gpu_elemwise_kernel, 1.5)
