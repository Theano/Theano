from theano import tensor, gof
from theano import tensor, scalar

from .basic_ops import *

@gof.local_optimizer([GpuFromHost(), None])
def local_gpu_host_gpu(node):
    if not tensor.opt.opt.check_chain(node, GpuFromHost(), HostFromGpu()):
        return False
    return [node.inputs[0].owner.inputs[0]]
tensor.opt.register_specialize(local_gpu_host_gpu, 'gpu')
@gof.local_optimizer([HostFromGpu(), None])
def local_host_gpu_host(node):
    if not tensor.opt.opt.check_chain(node, HostFromGpu(), GpuFromHost()):
        return False
    return [node.inputs[0].owner.inputs[0]]
tensor.opt.register_specialize(local_host_gpu_host, 'gpu')


@gof.local_optimizer([])
def local_gpu_elemwise(node):
    if isinstance(node.op, tensor.Elemwise):
        if any(hasattr(i.owner, 'op') and isinstance(i.owner.op, HostFromGpu) for i in node.inputs):
            # move the add to a GpuAdd
            new_op = GpuElemwise(node.op.scalar_op, node.op.inplace_pattern)
            return [host_from_gpu(new_op(*(gpu_from_host(i) for i in node.inputs)))]
    return False
tensor.opt.register_specialize(local_gpu_elemwise, 'gpu')

@gof.local_optimizer([])
def local_gpu_dimshuffle(node):
    if isinstance(node.op, tensor.DimShuffle):
        input, = node.inputs
        if input.owner and isinstance(input.owner.op, HostFromGpu):
            # move the add to a GpuAdd
            new_op = GpuDimShuffle(node.op.input_broadcastable, 
                    node.op.new_order)
            if node.op.inplace:
                return [host_from_gpu(new_op(gpu_from_host(input)))]
            else:
                return [host_from_gpu(new_op(gpu_from_host(tensor.tensor_copy(input))))]
    return False
tensor.opt.register_specialize(local_gpu_dimshuffle, 'gpu')
