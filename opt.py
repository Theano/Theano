from theano import tensor, scalar, compile
from theano.gof import local_optimizer, EquilibriumDB

from .basic_ops import *
from .blas import gpu_dot22, gpu_gemm

from theano.compile import optdb
#optdb.print_summary()  # this shows what is currently registered (in a so-far crude way...)

gpu_optimizer = EquilibriumDB()
optdb.register('gpu', 
        gpu_optimizer, 
        optdb.__priority__.get('inplace_opt', 75) + 5, 
        'fast_run',
        'inplace')

def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'inplace', *tags)
        return local_opt
    return f

@register_opt()
@local_optimizer([GpuFromHost(), None])
def local_gpu_host_gpu(node):
    if not tensor.opt.opt.check_chain(node, GpuFromHost(), HostFromGpu()):
        return False
    return [node.inputs[0].owner.inputs[0]]

@register_opt()
@local_optimizer([HostFromGpu(), None])
def local_host_gpu_host(node):
    if not tensor.opt.opt.check_chain(node, HostFromGpu(), GpuFromHost()):
        return False
    return [node.inputs[0].owner.inputs[0]]

@register_opt()
@local_optimizer([])
def local_gpu_elemwise_0(node):
    if isinstance(node.op, tensor.Elemwise):
        if any(hasattr(i.owner, 'op') and isinstance(i.owner.op, HostFromGpu) for i in node.inputs):
            # move the add to a GpuAdd
            new_op = GpuElemwise(node.op.scalar_op, node.op.inplace_pattern)
            return [host_from_gpu(new_op(*(gpu_from_host(i) for i in node.inputs)))]
    return False

@register_opt()
@local_optimizer([])
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

@register_opt()
@local_optimizer([])
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

@register_opt()
@local_optimizer([])
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

@register_opt()
@local_optimizer([])
def local_gpu_dot(node):
    """
    gpu_from_host(dot) -> gpudot(gpu_from_host)

    dot(host_from_gpu) -> host_from_gpu(gpudot)
    """
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op == tensor.blas._dot22:
            x, y = host_input.owner.inputs
            return [gpu_dot22(gpu_from_host(x), gpu_from_host(y))]
    if node.op == tensor.blas._dot22:
        if any((i.owner and i.owner.op == host_from_gpu) for i in node.inputs):
            x, y = node.inputs
            return [host_from_gpu(gpu_dot22(gpu_from_host(x), gpu_from_host(y)))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_gemm(node):
    """
    gpu_from_host(gemm) -> gpu_gemm(gpu_from_host)

    gemm(host_from_gpu) -> host_from_gpu(gpu_gemm)
    """
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op == tensor.blas.gemm:
            z, a, x, y, b = host_input.owner.inputs
            return [gpu_gemm(gpu_from_host(z), a, gpu_from_host(x), gpu_from_host(y), b)]
    if node.op == tensor.blas.gemm:
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and x.owner.op == host_from_gpu)
        y_on_gpu = (y.owner and y.owner.op == host_from_gpu)
        z_on_gpu = (z.owner and z.owner.op == host_from_gpu)
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(gpu_gemm(gpu_from_host(z), a, gpu_from_host(x), gpu_from_host(y), b))]
    return False
