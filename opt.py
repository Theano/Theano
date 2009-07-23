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
def local_gpu_elemwise_0(node):
    if isinstance(node.op, tensor.Elemwise):
        if any(hasattr(i.owner, 'op') and isinstance(i.owner.op, HostFromGpu) for i in node.inputs):
            # move the add to a GpuAdd
            new_op = GpuElemwise(node.op.scalar_op, node.op.inplace_pattern)
            return [host_from_gpu(new_op(*(gpu_from_host(i) for i in node.inputs)))]
    return False
tensor.opt.register_specialize(local_gpu_elemwise_0, 'gpu')
@gof.local_optimizer([])
def local_gpu_elemwise_1(node):
    """
    gpu_from_host(Elemwise)) -> GpuElemwise(gpu_from_host(...))
    """
    if node.op == gpu_from_host:
        host_i, = node.inputs
        if host_i.owner and isinstance(host_i.owner.op, tensor.Elemwise) and len(host_i.clients)==1:
            elemwise_node = host_i.owner
            new_op = GpuElemwise(elemwise_node.op.scalar_op, elemwise_node.op.inplace_pattern)
            return [new_op(*(gpu_from_host(i) for i in elemwise_node.inputs))]
    return False
tensor.opt.register_specialize(local_gpu_elemwise_1, 'gpu')

@gof.local_optimizer([])
def local_gpu_dimshuffle_0(node):
    """
    dimshuffle(host_from_gpu()) -> host_from_gpu(gpu_dimshuffle)
    """
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
tensor.opt.register_specialize(local_gpu_dimshuffle_0, 'gpu')

@gof.local_optimizer([])
def local_gpu_dimshuffle_1(node):
    """
    gpu_from_host(dimshuffle) -> gpu_dimshuffle(gpu_from_host)
    """
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, tensor.DimShuffle):
            dimshuffle_node = host_input.owner
            new_op = GpuDimShuffle(dimshuffle_node.op.input_broadcastable, 
                    dimshuffle_node.op.new_order)
            return [new_op(gpu_from_host(dimshuffle_node.inputs[0]))]
    return False
tensor.opt.register_specialize(local_gpu_dimshuffle_1, 'gpu')

