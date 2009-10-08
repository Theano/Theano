import sys
from theano import tensor, scalar, compile
from theano.gof import local_optimizer, EquilibriumDB, SequenceDB

from theano_cuda_ndarray.basic_ops import *
from theano_cuda_ndarray.blas import gpu_dot22, gpu_gemm, GpuConv
from theano_cuda_ndarray.blas import GpuDownsampleFactorMax, GpuDownsampleFactorMaxGrad
from theano_cuda_ndarray.nnet import (
        GpuCrossentropySoftmaxArgmax1HotWithBias,
        GpuCrossentropySoftmax1HotWithBiasDx)
from theano.compile import optdb
#optdb.print_summary()  # this shows what is currently registered (in a so-far crude way...)

gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()
gpu_seqopt = SequenceDB()
gpu_seqopt.register('gpu_local_optimizations', gpu_optimizer, 1, 'fast_run', 'inplace')
gpu_seqopt.register('gpu_cut_transfers', gpu_cut_copies, 2, 'fast_run', 'inplace')
optdb.register('gpu', 
        gpu_seqopt, 
        optdb.__priority__.get('inplace_opt', 75) + 5, 
        'fast_run',
        'inplace')

def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'inplace', *tags)
        return local_opt
    return f

@local_optimizer([])
def local_cut_gpu_host_gpu(node):
    if tensor.opt.opt.check_chain(node, gpu_from_host, host_from_gpu):
        return [node.inputs[0].owner.inputs[0]]
    if tensor.opt.opt.check_chain(node, host_from_gpu, gpu_from_host):
        return [node.inputs[0].owner.inputs[0]]
    return False
gpu_cut_copies.register('cut_gpu_host_transfers', local_cut_gpu_host_gpu, 
        'fast_run', 'inplace', 'gpu')
gpu_cut_copies.register('cut_gpu_constant_transfers', tensor.opt.constant_folding, 
        'fast_run', 'gpu')

@register_opt()
@local_optimizer([])
def local_gpu_elemwise_0(node):
    if isinstance(node.op, tensor.Elemwise):
        if any(hasattr(i.owner, 'op') and isinstance(i.owner.op, HostFromGpu) for i in node.inputs):
            if any(o.type.dtype == 'float64' for o in node.outputs):
                print 'WARNING: THERE ARE STILL float64s in your graph local_gpu_elemwise_0', node
            else:
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
    gpu_from_host(dimshuffle) -> gpu_dimshuffle(gpu_from_host)
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

@register_opt()
@local_optimizer([])
def local_gpu_sum(node):
    if isinstance(node.op, tensor.elemwise.CAReduce):
        if node.op.scalar_op == scalar.add:
            x, = node.inputs
            if x.owner and x.owner.op == host_from_gpu:
                if node.op.axis is None:
                    reduce_mask = [1] * x.type.ndim
                else:
                    reduce_mask = [0] * x.type.ndim
                    for a in node.op.axis:
                        assert reduce_mask[a] == 0
                        reduce_mask[a] = 1
                return [host_from_gpu(GpuSum(reduce_mask)(gpu_from_host(x)))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_reshape(node):
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, tensor.Reshape):
            rshp = host_input.owner.op
            x, shp = host_input.owner.inputs
            return [GpuReshape(rshp.ndim)(gpu_from_host(x), shp)]
    if isinstance(node.op, tensor.Reshape):
        x, shp = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuReshape(node.op.ndim)(gpu_x, shp))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_flatten(node):
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, tensor.Flatten):
            outdim = host_input.owner.op.outdim
            return [GpuFlatten(outdim)(gpu_from_host(host_input.owner.inputs[0]))]
    if isinstance(node.op, tensor.Flatten):
        x, = node.inputs
        outdim = node.op.outdim
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuFlatten(outdim)(gpu_x))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_subtensor(node):
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, tensor.Subtensor):
            subt = host_input.owner.op
            x = host_input.owner.inputs[0]
            coords = host_input.owner.inputs[1:]
            return [GpuSubtensor(subt.idx_list)(gpu_from_host(x), *coords)]
    if isinstance(node.op, tensor.Subtensor):
        x  = node.inputs[0]
        coords = node.inputs[1:]
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuSubtensor(node.op.idx_list)(gpu_x, *coords))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_incsubtensor(node):
    if node.op == gpu_from_host:
        host_output = node.inputs[0]
        if host_output.owner and type(host_output.owner.op) == tensor.IncSubtensor:
            incsubt = host_output.owner.op
            x, y = host_output.owner.inputs[0:2]
            coords = host_output.owner.inputs[2:]
            return [GpuIncSubtensor(incsubt.idx_list, inplace=incsubt.inplace)(
                gpu_from_host(x),
                gpu_from_host(y),
                *coords)]
    if type(node.op) == tensor.IncSubtensor:
        x, y = node.inputs[0:2]
        assert isinstance(x.type, tensor.TensorType)
        assert isinstance(y.type, tensor.TensorType)
        coords = node.inputs[2:]
        go_gpu = False
        if x.owner and x.owner.op == host_from_gpu:
            go_gpu = True
            gpu_x, = x.owner.inputs
        else:
            gpu_x = gpu_from_host(x)
        if y.owner and y.owner.op == host_from_gpu:
            go_gpu = True
            gpu_y, = y.owner.inputs
        else:
            gpu_y = gpu_from_host(y)
        if go_gpu:
            return [host_from_gpu(GpuIncSubtensor(
                node.op.idx_list, inplace=node.op.inplace)(
                    gpu_x, gpu_y, *coords))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_shape(node):
    if isinstance(node.op, tensor.Shape):
        x, = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            return [gpu_shape(gpu_x)]
    return False


def cast(x, dtype):
    stype = theano.scalar.Scalar(dtype)
    cast_op = theano.tensor.Elemwise(scalar.Identity(scalar.specific_out(stype)))
    return cast_op(x)

import theano.tensor.nnet
@register_opt()
@local_optimizer([])
def local_gpu_crossentorpy_softmax_argmax_1hot_with_bias(node):
    if isinstance(node.op, tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias):
        x,b,y = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            gpu_nll, gpu_sm, gpu_am = GpuCrossentropySoftmaxArgmax1HotWithBias()(
                gpu_x,
                gpu_from_host(b),
                gpu_from_host(cast(y, 'float32')))
            am_dtype = node.outputs[2].type.dtype
            return [host_from_gpu(gpu_nll), 
                    host_from_gpu(gpu_sm), 
                    cast(host_from_gpu(gpu_am), am_dtype)]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_crossentorpy_softmax_1hot_with_bias_dx(node):
    if isinstance(node.op, tensor.nnet.CrossentropySoftmax1HotWithBiasDx):
        dnll,sm,yidx = node.inputs
        if sm.owner and sm.owner.op == host_from_gpu:
            gpu_sm, = sm.owner.inputs
            gpu_dx = GpuCrossentropySoftmax1HotWithBiasDx()(
                gpu_from_host(dnll),
                gpu_sm,
                gpu_from_host(cast(yidx, 'float32')))
            return [host_from_gpu(gpu_dx)]
    return False


#### Convolution, maxpooling
import theano.sandbox.conv
@register_opt()
@local_optimizer([])
def local_gpu_conv(node):
    """
    gpu_from_host(conv) -> gpu_conv(gpu_from_host)

    conv(host_from_gpu) -> host_from_gpu(conv)
    """
    def GpuConvOp_from_ConvOp(op):
        ret = GpuConv(border_mode=op.out_mode,
                    subsample=(op.dx, op.dy),
                    logical_img_hw=op.imshp_logical[1:3],
                    logical_kern_hw=op.kshp_logical,
                    logical_kern_align_top=op.kshp_logical_top_aligned
                    )
        #HACK to print the number of MFlops in the profiler output.
        if hasattr(op,'flops'):
            ret.flops=op.flops
        return ret

    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, theano.sandbox.conv.ConvOp):
            gpu_conv = GpuConvOp_from_ConvOp(host_input.owner.op)
            img, kern = host_input.owner.inputs
            return [gpu_conv(gpu_from_host(img), gpu_from_host(kern))]

    if isinstance(node.op, theano.sandbox.conv.ConvOp):
        img, kern = node.inputs
        img_on_gpu = (img.owner and img.owner.op == host_from_gpu)
        kern_on_gpu = (kern.owner and kern.owner.op == host_from_gpu)
        if img_on_gpu or kern_on_gpu:
            gpu_conv = GpuConvOp_from_ConvOp(node.op)
            return [host_from_gpu(gpu_conv(gpu_from_host(img), gpu_from_host(kern)))]

import theano.sandbox.downsample
@register_opt()
@local_optimizer([])
def local_gpu_downsample_factor_max(node):
    if isinstance(node.op, theano.sandbox.downsample.DownsampleFactorMax):
        x, = node.inputs
        if (x.owner and x.owner.op == host_from_gpu):
            gpu_ds = GpuDownsampleFactorMax(node.op.ds, node.op.ignore_border)
            return [host_from_gpu(gpu_ds(x.owner.inputs[0]))]

@register_opt()
@local_optimizer([])
def local_gpu_downsample_factor_max_grad(node):
    if isinstance(node.op, theano.sandbox.downsample.DownsampleFactorMaxGrad):
        x,z,gz = node.inputs
        if (x.owner and x.owner.op == host_from_gpu):
            gpu_ds_grad = GpuDownsampleFactorMaxGrad(node.op.ds, node.op.ignore_border)
            return [host_from_gpu(gpu_ds_grad(x.owner.inputs[0], gpu_from_host(z), gpu_from_host(gz)))]

