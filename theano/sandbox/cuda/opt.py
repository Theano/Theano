import logging
_logger = logging.getLogger('theano.sandbox.cuda.opt')

import copy
import sys
import warnings

import numpy

import theano
from theano import scalar as scal
from theano import config, tensor, gof
import theano.ifelse

from theano.compile import optdb
from theano.gof import (local_optimizer, EquilibriumDB, SequenceDB, ProxyDB,
                        Optimizer, toolbox)
from theano.gof.python25 import all, any
from theano.sandbox.cuda.basic_ops import (
    device_properties, gpu_eye,
    gpu_from_host, host_from_gpu, GpuFromHost, HostFromGpu,
    GpuElemwise, GpuDimShuffle, GpuReshape, GpuCAReduce, GpuFlatten,
    GpuSubtensor, GpuAdvancedSubtensor1,
    GpuAdvancedIncSubtensor1, GpuAdvancedIncSubtensor1_dev20,
    GpuIncSubtensor, gpu_alloc, GpuAlloc, gpu_shape)
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.blas import (gpu_dot22, gpu_dot22scalar,
        gpu_gemm_inplace, gpu_gemm_no_inplace, GpuConv)
from theano.sandbox.cuda.blas import gpu_gemv_inplace
from theano.sandbox.cuda.blas import gpu_gemv_no_inplace
from theano.sandbox.cuda.blas import gpu_ger_inplace
from theano.sandbox.cuda.blas import gpu_ger_no_inplace
from theano.sandbox.cuda.blas import (GpuDownsampleFactorMax,
        GpuDownsampleFactorMaxGrad)
from theano.sandbox.cuda.nnet import (
        GpuCrossentropySoftmaxArgmax1HotWithBias,
        GpuCrossentropySoftmax1HotWithBiasDx,
        GpuSoftmax, GpuSoftmaxWithBias)
from theano.sandbox.cuda.elemwise import SupportCodeError
from theano.scalar.basic_scipy import Erfinv
from theano.sandbox.cuda.elemwise import erfinv_gpu
from theano.sandbox.cuda.var import CudaNdarrayConstant
from theano.sandbox.cuda.fftconv import conv2d_fft
from theano.scan_module import scan_utils, scan_op, scan_opt
from theano.tensor.blas import _is_real_vector, _is_real_matrix
linalg = None

#optdb.print_summary()  # shows what is currently registered

#ignore_newtrees is to speed the optimization as this is the pattern
#we use for optimization. Otherwise, we can iterate 100s of time on
#the graph and apply only a few optimizations each time.
gpu_optimizer = EquilibriumDB(ignore_newtrees=False)
gpu_cut_copies = EquilibriumDB()
gpu_seqopt = SequenceDB()
gpu_seqopt.register('gpu_local_optimizations', gpu_optimizer, 1,
        'fast_run', 'inplace', 'gpu')
gpu_seqopt.register('gpu_cut_transfers', gpu_cut_copies, 2,
        'fast_run', 'gpu')
# DO NOT PUT fast_run in gpu_opt! This will ALWAYS enable the GPU!
optdb.register('gpu_opt',
               gpu_seqopt,
               optdb.__position__.get('add_destroy_handler', 49.5) - 1,
               'gpu')
# DO NOT PUT fast_run in gpu_after_fusion! This will ALWAYS enable the GPU!
# This second pass is needed as the fusion can put all the non float32 code
# inside the elemwise. When there is no float64 op, this is working.
optdb.register('gpu_after_fusion',
               ProxyDB(gpu_seqopt),
               optdb.__position__.get('elemwise_fusion', 49) + .1,
               'gpu')

## Register merge_optimizer as a global opt
gpu_optimizer.register('gpu_merge', theano.gof.opt.merge_optimizer, 'fast_run')


def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'gpu', *tags)
        return local_opt
    return f

#register local_track_shape_i at this level too
#to make multi-level lift of shape work.
register_opt()(theano.tensor.opt.local_track_shape_i)
register_opt(name='gpu_constant_folding')(
    tensor.opt.constant_folding)


class InputToGpuOptimizer(Optimizer):
    """
    Transfer the input of a graph to the gpu if it is necessary.
    It should make this part of the optimizer faster we will will need only 1
    pass on the fgraph.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        for input in fgraph.inputs:
            if isinstance(input.type, CudaNdarrayType):
                return

            # This happen frequently as we do 2 pass of the gpu optimizations
            if (len(input.clients) == 1 and
                (input.clients[0][0] == 'output' or
                 input.clients[0][0].op == gpu_from_host)):
                return

            try:
                new_input = host_from_gpu(gpu_from_host(input))

                if new_input.type == input.type:
                    fgraph.replace_validate(input, new_input,
                                         "InputToGpuOptimizer")
            except TypeError:
                #as we currently only support float32, this can fail.
                #Using try except make that we won't need
                pass

# we register it before all other gpu optimizer to be sure that the input
# are on the gpu.
gpu_seqopt.register('InputToGpuOptimizer', InputToGpuOptimizer(),
                    0,
                    'fast_run',
                    'fast_compile',
                    'merge')  # TODO: how to make it mandatory for gpu_seqopt?


@local_optimizer([gpu_from_host, host_from_gpu])
def local_cut_gpu_host_gpu(node):
    if tensor.opt.opt.check_chain(node, gpu_from_host, host_from_gpu):
        return [node.inputs[0].owner.inputs[0]]
    if tensor.opt.opt.check_chain(node, host_from_gpu, gpu_from_host):
        return [node.inputs[0].owner.inputs[0]]
    return False
gpu_cut_copies.register('cut_gpu_host_transfers', local_cut_gpu_host_gpu,
        'fast_run', 'gpu')
gpu_cut_copies.register('cut_gpu_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_run', 'gpu')
#register it into canonicalize to allow other optimization to work without
#botering with this useless pattern.
optdb['canonicalize'].register('local_cut_gpu_host_gpu',
                               local_cut_gpu_host_gpu, 'fast_run', 'gpu')

# 'float64', 'complex128' and 'complex64' are not supported in elemwise
# on the gpu.
elemwise_cuda_dtype_supported = ['float32', 'uint8', 'int8', 'uint16', 'int16',
                                 'uint32', 'int32', 'uint64', 'int64']


def dtype_in_elemwise_supported(op):
    """
    Return True of the Elemwise op is supported on the gpu.
    Return False otherwise.

    :note: We need to check inside the Composite op.
    """
    def get_all_basic_scalar(composite_op):
        l = []
        for i in composite_op.fgraph.toposort():
            if isinstance(i, theano.scalar.Composite):
                l += get_all_basic_scalar(i)
            else:
                l.append(i)
        return l
    if isinstance(op, GpuElemwise) or isinstance(op, tensor.Elemwise):
        if isinstance(op.scalar_op, theano.scalar.Composite):
            scals = get_all_basic_scalar(op.scalar_op)
            for s in scals:
                if any([i.type.dtype not in elemwise_cuda_dtype_supported
                        for i in s.inputs + s.outputs]):
                    return False
    return True


@register_opt()
@local_optimizer([tensor.Elemwise])
def local_gpu_elemwise_0(node):
    """elemwise(..., host_from_gpu, ...)
       -> host_from_gpu(elemwise(gpu_from_host, ..., gpu_from_host)
    """
    if (isinstance(node.op, tensor.Elemwise) and
        dtype_in_elemwise_supported(node.op)):
        if any([i.owner and
                isinstance(i.owner.op, HostFromGpu)
                for i in node.inputs]):
            if all([o.type.dtype == 'float32' for o in node.outputs]):
                # Don't set any inplace pattern.
                # gpu_inplace_elemwise_optimizer will do it later

                if isinstance(node.op.scalar_op, Erfinv):
                    new_op = GpuElemwise(erfinv_gpu)
                else:
                    try:
                        new_op = GpuElemwise(node.op.scalar_op)
                    except SupportCodeError:
                        # This happens when scalar_op requires support code
                        return False

                #   first establish that float32 can store all inputs
                upcastable = set(['float32', 'int8', 'int16', 'uint8',
                                  'uint16'])
                # case 1 - all inputs are already float32
                if all([i.type.dtype == 'float32' for i in node.inputs]):
                    #TODO: change this when fusion makes Elemwise with multiple
                    # outputs
                    gpu_elemwise = new_op(*(gpu_from_host(i)
                                            for i in node.inputs))
                # case 2 - it is still ok if some inputs were upcast to float32
                elif all([i.type.dtype in upcastable
                          for i in node.inputs]):
                    # second - establish that a new node with upcasted inputs
                    # has the same outputs types as the original node
                    upcasted = node.op.make_node(*[tensor.cast(i, 'float32')
                                                   for i in node.inputs])
                    if [o.type for o in upcasted.outputs] ==\
                       [o.type for o in node.outputs]:

                        new_inputs = [gpu_from_host(tensor.cast(i, 'float32'))
                                      for i in node.inputs]
                        gpu_elemwise = new_op(*new_inputs)
                    else:
                        return False
                else:
                    return False

                gpu_elemwise = split_huge_add_or_mul(gpu_elemwise.owner)
                if not gpu_elemwise:
                    return False
                if max_inputs_to_GpuElemwise(node) < len(gpu_elemwise.inputs):
                    return False
                return [host_from_gpu(gpu_elemwise.outputs[0])]


@register_opt()
@local_optimizer([gpu_from_host])
def local_gpu_elemwise_1(node):
    """
    gpu_from_host(Elemwise)) -> GpuElemwise(gpu_from_host(...))
    """
    if isinstance(node.op, GpuFromHost):
        host_i, = node.inputs
        if (host_i.owner and
            isinstance(host_i.owner.op, tensor.Elemwise) and
            len(host_i.clients) == 1 and
            dtype_in_elemwise_supported(node.op)):

            elemwise_node = host_i.owner
            # Don't set any inplace pattern.
            # gpu_inplace_elemwise_optimizer will do it later

            if isinstance(elemwise_node.op.scalar_op, Erfinv):
                new_op = GpuElemwise(erfinv_gpu)
            else:
                try:
                    new_op = GpuElemwise(elemwise_node.op.scalar_op)
                except SupportCodeError:
                    # This happens when scalar_op requires support code
                    return False

            if all([i.dtype == 'float32' for i in elemwise_node.inputs]):
                gpu_elemwise = new_op(*[gpu_from_host(i)
                                        for i in elemwise_node.inputs])
                gpu_elemwise = split_huge_add_or_mul(gpu_elemwise.owner)
                if not gpu_elemwise:
                    return False
                return [gpu_elemwise.outputs[0]]
    return False


@register_opt()
@local_optimizer([tensor.DimShuffle, gpu_from_host])
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
            return [host_from_gpu(new_op(gpu_from_host(input)))]
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.DimShuffle):
            dimshuffle_node = host_input.owner
            new_op = GpuDimShuffle(dimshuffle_node.op.input_broadcastable,
                                   dimshuffle_node.op.new_order)
            return [new_op(gpu_from_host(dimshuffle_node.inputs[0]))]
    return False


@register_opt()
@local_optimizer([tensor.SpecifyShape, gpu_from_host])
def local_gpu_specifyShape_0(node):
    """
    specify_shape(host_from_gpu()) -> host_from_gpu(specify_shape)
    gpu_from_host(specify_shape) -> specify_shape(gpu_from_host)
    """
    if isinstance(node.op, tensor.SpecifyShape):
        input = node.inputs[0]
        if input.owner and isinstance(input.owner.op, HostFromGpu):
            return [host_from_gpu(tensor.specify_shape(gpu_from_host(input),
                                                      *node.inputs[1:]))]
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.SpecifyShape):
            specifyshape_node = host_input.owner
            return [tensor.specify_shape(
                gpu_from_host(specifyshape_node.inputs[0]),
                *specifyshape_node.inputs[1:])]
    return False


@register_opt()
@local_optimizer([gpu_from_host]) # XXX: broken: tensor.basic.dot is not an op
def local_gpu_dot_to_dot22(node):
    """
    gpu_from_host(dot) -> gpudot(gpu_from_host)
    dot(host_from_gpu) -> host_from_gpu(gpudot)

    This optimization solves the vector-matrix multiplication issue by
    transforming the vector into a matrix, apply gpudot22 and reshaping
    the output.

    A more suitable solution would be to use the right cublas call
    """

    # In case the got do input upcast, we much check that we can
    # make it run on the gpu.
    if isinstance(node.op, GpuFromHost):
        if node.outputs[0].type.dtype != 'float32':
            return False
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op == tensor.basic.dot:
            x, y = host_input.owner.inputs
            # case one: vector X matrix
            if _is_real_vector(x) and _is_real_matrix(y):
                new_op = GpuDimShuffle((False,), ['x', 0])
                shape_out = y.shape[1].dimshuffle(['x'])
                gpu_x = new_op(gpu_from_host(x))
                gpu_y = gpu_from_host(y)
            # case two: matrix X vector
            elif _is_real_matrix(x) and _is_real_vector(y):
                new_op = GpuDimShuffle((False,), [0, 'x'])
                shape_out = x.shape[0].dimshuffle(['x'])
                gpu_x = gpu_from_host(x)
                gpu_y = new_op(gpu_from_host(y))
            else:
                return False

            return [GpuReshape(1)(gpu_dot22(gpu_x, gpu_y), shape_out)]
    if node.op == tensor.basic.dot:
        if node.outputs[0].type.dtype != 'float32':
            return False
        if any([i.owner and isinstance(i.owner.op, HostFromGpu)
                for i in node.inputs]):
            x, y = node.inputs
            if _is_real_vector(x) and _is_real_matrix(y):
                new_op = GpuDimShuffle((False,), ['x', 0])
                shape_out = y.shape[1].dimshuffle(['x'])
                gpu_x = new_op(gpu_from_host(x))
                gpu_y = gpu_from_host(y)

            elif _is_real_matrix(x) and _is_real_vector(y):
                new_op = GpuDimShuffle((False,), [0, 'x'])
                shape_out = x.shape[0].dimshuffle(['x'])
                gpu_x = gpu_from_host(x)
                gpu_y = new_op(gpu_from_host(y))
            else:
                return False

            return [host_from_gpu(GpuReshape(1)(gpu_dot22(gpu_x, gpu_y),
                                                shape_out))]
    return False


@register_opt()
@local_optimizer([theano.ifelse.IfElse, gpu_from_host])
def local_gpu_lazy_ifelse(node):
    """
    gpu_from_host(ifelse) -> gpu_ifelse(gpu_from_host)

    ifelse(host_from_gpu) -> host_from_gpu(ifelse)
    """
    if isinstance(node.op, theano.ifelse.IfElse) and not node.op.gpu:
        gpu_ifelse = theano.ifelse.IfElse(node.op.n_outs, gpu=True)
        outs_clients = reduce(list.__add__,
                              [out.clients for out in node.outputs])
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]) or any(
                    [c != 'output' and c.op == gpu_from_host for c, idx
                     in outs_clients]):

            c = node.inputs[0]
            outs = node.inputs[1:]
            # Should not happen, but just in case
            if isinstance(c.type, CudaNdarrayType):
                c = host_from_gpu(c)

            for i in range(len(outs)):
                if not isinstance(outs[i], CudaNdarrayType):
                    outs[i] = gpu_from_host(outs[i])
            return [host_from_gpu(out) for out in
                    gpu_ifelse.make_node(c, *outs).outputs]

    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op, theano.ifelse.IfElse) and
            not host_input.owner.op.gpu and
            # If there is more then 1 outputs, we can't replace it
            # here with a local optimizer as we replace the
            # GpuFromHost node and the other output of the if won't be
            # replaced.
            host_input.owner.op.n_outs == 1):
            gpu_ifelse = theano.ifelse.IfElse(host_input.owner.op.n_outs,
                                                  gpu=True)

            c = host_input.owner.inputs[0]
            outs = host_input.owner.inputs[1:]
            # Should not happen, but just in case
            if isinstance(c.type, CudaNdarrayType):
                c = host_from_gpu(c)

            for i in range(len(outs)):
                if not isinstance(outs[i], CudaNdarrayType):
                    outs[i] = gpu_from_host(outs[i])

            outs = gpu_ifelse.make_node(c, *outs).outputs
            return outs

    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.blas.Dot22])
def local_gpu_dot22(node):
    """
    gpu_from_host(dot22) -> gpudot(gpu_from_host)

    dot(host_from_gpu) -> host_from_gpu(gpudot22)
    """
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.blas.Dot22):
            x, y = host_input.owner.inputs
            return [gpu_dot22(gpu_from_host(x), gpu_from_host(y))]
    if isinstance(node.op, tensor.blas.Dot22):
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]):
            x, y = node.inputs
            return [host_from_gpu(gpu_dot22(gpu_from_host(x),
                                            gpu_from_host(y)))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.blas.Dot22Scalar])
def local_gpu_dot22scalar(node):
    """
    gpu_from_host(dot22scalar) -> gpudot(gpu_from_host)

    dot(host_from_gpu) -> host_from_gpu(gpudot22scalar)
    """
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op,
                       tensor.blas.Dot22Scalar)):
            x, y, scalar = host_input.owner.inputs
            return [gpu_dot22scalar(gpu_from_host(x), gpu_from_host(y),
                                    tensor.blas._as_scalar(scalar))]
    if isinstance(node.op, tensor.blas.Dot22Scalar):
        if any([i.owner and isinstance(i.owner.op, HostFromGpu)
                for i in node.inputs]):
            x, y, scalar = node.inputs
            return [host_from_gpu(
                gpu_dot22scalar(gpu_from_host(x),
                                gpu_from_host(y),
                                tensor.blas._as_scalar(scalar)))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.blas_c.CGemv, tensor.blas.Gemv])
def local_gpu_gemv(node):
    """
    gpu_from_host(gemv) -> gpu_gemv(gpu_from_host)
    gemv(host_from_gpu) -> host_from_gpu(gpu_gemv)

    """
    gemvs = (tensor.blas.Gemv,
             tensor.blas_c.CGemv,
            )
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, gemvs):
            z, a, x, y, b = host_input.owner.inputs
            return [gpu_gemv_no_inplace(
                    gpu_from_host(z),
                    a,
                    gpu_from_host(x),
                    gpu_from_host(y),
                    b)]
    if isinstance(node.op, gemvs):
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and isinstance(x.owner.op, HostFromGpu))
        y_on_gpu = (y.owner and isinstance(y.owner.op, HostFromGpu))
        z_on_gpu = (z.owner and isinstance(z.owner.op, HostFromGpu))
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(
                gpu_gemv_no_inplace(
                    gpu_from_host(z),
                    a,
                    gpu_from_host(x),
                    gpu_from_host(y),
                    b))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.blas_c.CGer, tensor.blas.Ger,
                  tensor.blas_scipy.ScipyGer])
def local_gpu_ger(node):
    """
    gpu_from_host(ger) -> gpu_ger(gpu_from_host)
    ger(host_from_gpu) -> host_from_gpu(gpu_ger)

    """
    gers = (tensor.blas_c.CGer,
            tensor.blas.Ger,
            tensor.blas_scipy.ScipyGer,
        )

    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, gers):
            z, a, x, y = host_input.owner.inputs
            return [gpu_ger_no_inplace(
                    gpu_from_host(z),
                    a,
                    gpu_from_host(x),
                    gpu_from_host(y)
                    )]
    if isinstance(node.op, gers):
        z, a, x, y = node.inputs
        x_on_gpu = (x.owner and isinstance(x.owner.op, HostFromGpu))
        y_on_gpu = (y.owner and isinstance(y.owner.op, HostFromGpu))
        z_on_gpu = (z.owner and isinstance(z.owner.op, HostFromGpu))
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(
                gpu_ger_no_inplace(
                    gpu_from_host(z),
                    a,
                    gpu_from_host(x),
                    gpu_from_host(y)
                    ))]
    return False


@register_opt()
@local_optimizer([tensor.blas.Gemm, gpu_from_host])
def local_gpu_gemm(node):
    """
    gpu_from_host(gemm) -> gpu_gemm(gpu_from_host)

    gemm(host_from_gpu) -> host_from_gpu(gpu_gemm)
    """
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.blas.Gemm):
            z, a, x, y, b = host_input.owner.inputs
            return [gpu_gemm_no_inplace(gpu_from_host(z),
                                        a,
                                        gpu_from_host(x),
                                        gpu_from_host(y),
                                        b)]
    if isinstance(node.op, tensor.blas.Gemm):
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and isinstance(x.owner.op, HostFromGpu))
        y_on_gpu = (y.owner and isinstance(y.owner.op, HostFromGpu))
        z_on_gpu = (z.owner and isinstance(z.owner.op, HostFromGpu))
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(gpu_gemm_no_inplace(gpu_from_host(z),
                                                 a,
                                                 gpu_from_host(x),
                                                 gpu_from_host(y),
                                                 b))]
    return False


@register_opt()
@local_optimizer([tensor.elemwise.CAReduce,
                  tensor.elemwise.All,
                  tensor.elemwise.Any,
                  tensor.elemwise.CAReduceDtype,
                  tensor.elemwise.Sum,
                  tensor.elemwise.Prod,
                  tensor.elemwise.ProdWithoutZeros])
def local_gpu_careduce(node):
    if isinstance(node.op, tensor.elemwise.CAReduce):
        scalar_op = node.op.scalar_op
        # currently, only these two ops are supported at all,
        # and max does not support all combinations of axes
        if isinstance(node.op.scalar_op, (scal.Add, scal.Mul,
                                          scal.Maximum, scal.Minimum)):
            x, = node.inputs
            if x.owner and isinstance(x.owner.op, HostFromGpu):
                if node.op.axis is None:
                    reduce_mask = [1] * x.type.ndim
                else:
                    reduce_mask = [0] * x.type.ndim
                    for a in node.op.axis:
                        assert reduce_mask[a] == 0
                        reduce_mask[a] = 1
                greduce = GpuCAReduce(reduce_mask, scalar_op)
                if greduce.supports_c_code([gpu_from_host(x)]):
                    rval = host_from_gpu(greduce(gpu_from_host(x)))
                    if rval.type == node.outputs[0].type:
                        return [rval]
                    else:
                        print >> sys.stderr, (
                            "WARNING: local_gpu_careduce got type wrong",
                            rval.type, node.outputs[0].type,
                            node.inputs[0].type,
                            node)
                        return None
                else:

                    # Try to make a simpler pattern based on reshaping
                    # The principle is that if two adjacent dimensions have
                    # the same value in the reduce_mask, then we can reshape
                    # to make them a single dimension, do the reduction, and
                    # then reshape to get them back.

                    shape_of = node.fgraph.shape_feature.shape_of

                    x_shape = shape_of[x]

                    new_in_shp = [x_shape[0]]
                    new_mask = [reduce_mask[0]]
                    for i in xrange(1, x.type.ndim):
                        if reduce_mask[i] == reduce_mask[i - 1]:
                            new_in_shp[-1] *= x_shape[i]
                        else:
                            new_mask.append(reduce_mask[i])
                            new_in_shp.append(x_shape[i])

                    new_greduce = GpuCAReduce(new_mask, scalar_op)
                    reshaped_x = x.reshape(tensor.stack(*new_in_shp))
                    gpu_reshaped_x = gpu_from_host(reshaped_x)
                    reshaped_gpu_inputs = [gpu_reshaped_x]
                    if new_greduce.supports_c_code(reshaped_gpu_inputs):
                        reduce_reshaped_x = host_from_gpu(
                            new_greduce(gpu_reshaped_x))

                        if reduce_reshaped_x.ndim != node.outputs[0].ndim:
                            unreshaped_reduce = reduce_reshaped_x.reshape(
                                tensor.stack(*shape_of[node.outputs[0]]))
                        else:
                            unreshaped_reduce = reduce_reshaped_x
                        if unreshaped_reduce.type == node.outputs[0].type:
                            return [unreshaped_reduce]
                        else:
                            print >> sys.stderr, (
                                "WARNING: local_gpu_careduce got type wrong",
                                unreshaped_reduce.type, node.outputs[0].type,
                                node.inputs[0].type,
                                node)
                            return None

    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.Reshape])
def local_gpu_reshape(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           isinstance(host_input.owner.op, tensor.Reshape):
            rshp = host_input.owner.op
            x, shp = host_input.owner.inputs
            gpu_reshape = GpuReshape(rshp.ndim)(gpu_from_host(x), shp)
            if gpu_reshape.broadcastable != node.outputs[0].broadcastable:
                # this can happen as we always return False for all broadcast
                # dim in GpuReshape but not for Reshape
                # Event if we did the same think, with the constant
                # optimization that could happen.
                gpu_reshape = theano.tensor.patternbroadcast(
                    gpu_reshape, node.outputs[0].broadcastable)
            return [gpu_reshape]
    if isinstance(node.op, tensor.Reshape):
        x, shp = node.inputs
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            gpu_reshape = GpuReshape(node.op.ndim)(gpu_x, shp)
            if gpu_reshape.broadcastable != node.outputs[0].broadcastable:
                # this can happen as we always return False for all broadcast
                # dim in GpuReshape but not for Reshape
                # Event if we did the same think, with the constant
                # optimization that could happen.
                gpu_reshape = theano.tensor.patternbroadcast(
                    gpu_reshape, node.outputs[0].broadcastable)
            return [host_from_gpu(gpu_reshape)]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.Flatten])
def local_gpu_flatten(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           isinstance(host_input.owner.op, tensor.Flatten):
            outdim = host_input.owner.op.outdim
            return [GpuFlatten(outdim)(
                gpu_from_host(host_input.owner.inputs[0]))]
    if isinstance(node.op, tensor.Flatten):
        x, = node.inputs
        outdim = node.op.outdim
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuFlatten(outdim)(gpu_x))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.Subtensor])
def local_gpu_subtensor(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           isinstance(host_input.owner.op, tensor.Subtensor):
            subt = host_input.owner.op
            x = host_input.owner.inputs[0]
            coords = host_input.owner.inputs[1:]
            return [GpuSubtensor(subt.idx_list)(gpu_from_host(x), *coords)]
    if isinstance(node.op, tensor.Subtensor):
        x = node.inputs[0]
        if (x.owner and
            isinstance(x.owner.op, HostFromGpu) and
            x.dtype == "float32"):
            gpu_x, = x.owner.inputs
            coords = node.inputs[1:]
            return [host_from_gpu(GpuSubtensor(
                node.op.idx_list)(gpu_x, *coords))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.AdvancedSubtensor1])
def local_gpu_advanced_subtensor1(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           host_input.owner.op.__class__ is tensor.AdvancedSubtensor1:
            x = host_input.owner.inputs[0]
            coords = host_input.owner.inputs[1:]
            return [GpuAdvancedSubtensor1()(gpu_from_host(x), *coords)]
    if node.op.__class__ is tensor.AdvancedSubtensor1:
        x = node.inputs[0]
        coords = node.inputs[1:]
        if x.owner and isinstance(x.owner.op, HostFromGpu) and x.dtype == "float32":
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuAdvancedSubtensor1()(gpu_x, *coords))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.AdvancedIncSubtensor1])
def local_gpu_advanced_incsubtensor1(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        # Should not execute for GpuAdvancedIncSubtensor1
        if host_input.owner and \
           host_input.owner.op.__class__ is tensor.AdvancedIncSubtensor1:
            x, y = host_input.owner.inputs[0:2]
            coords = host_input.owner.inputs[2:]
            set_instead_of_inc = host_input.owner.op.set_instead_of_inc
            if set_instead_of_inc and config.warn.gpu_set_subtensor1:
                warnings.warn(
                    'Although your current code is fine, please note that '
                    'Theano versions prior to 0.6 (more specifically, '
                    'prior to commitd 2240bddd on March 29, 2012) may have '
                    'yielded an incorrect result. To remove this warning, '
                    'either set the `warn.gpu_set_subtensor1` config '
                    'option to False, or `warn.ignore_bug_before` to at '
                    'least \'0.6\'.', stacklevel=1)
            active_device_no = theano.sandbox.cuda.active_device_number()
            compute_capability = device_properties(active_device_no)['major']
            if (compute_capability < 2 or
                x.ndim != 2 or
                y.ndim != 2):
                gpu_op = GpuAdvancedIncSubtensor1(
                    set_instead_of_inc=set_instead_of_inc)
            else:
                gpu_op = GpuAdvancedIncSubtensor1_dev20(
                    set_instead_of_inc=set_instead_of_inc)
            return [gpu_op(gpu_from_host(x), gpu_from_host(y), *coords)]

    # Should not execute for GpuAdvancedIncSubtensor1
    if node.op.__class__ is tensor.AdvancedIncSubtensor1 and \
       node.inputs[0].dtype == "float32":
        x, y = node.inputs[0:2]
        coords = node.inputs[2:]
        go_gpu = False
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            go_gpu = True
            gpu_x, = x.owner.inputs
        else:
            gpu_x = gpu_from_host(x)
        if y.owner and isinstance(y.owner.op, HostFromGpu):
            go_gpu = True
            gpu_y, = y.owner.inputs
        else:
            gpu_y = gpu_from_host(y)
        if go_gpu:
            set_instead_of_inc = node.op.set_instead_of_inc
            if set_instead_of_inc and config.warn.gpu_set_subtensor1:
                warnings.warn(
                    'Although your current code is fine, please note that '
                    'Theano versions prior to 0.6 (more specifically, '
                    'prior to commit d2240bddd on March 29, 2012) may have '
                    'yielded an incorrect result. To remove this warning, '
                    'either set the `warn.gpu_set_subtensor1` config '
                    'option to False, or `warn.ignore_bug_before` to at '
                    'least \'0.6\'.', stacklevel=1)

            active_device_no = theano.sandbox.cuda.active_device_number()
            compute_capability = device_properties(active_device_no)['major']
            if (compute_capability < 2 or
                x.ndim != 2 or
                y.ndim != 2):
                gpu_op = GpuAdvancedIncSubtensor1(
                    set_instead_of_inc=set_instead_of_inc)
            else:
                gpu_op = GpuAdvancedIncSubtensor1_dev20(
                    set_instead_of_inc=set_instead_of_inc)
            return [host_from_gpu(gpu_op(gpu_x, gpu_y, *coords))]
    return False


@register_opt()
@local_optimizer([gpu_from_host, tensor.IncSubtensor])
def local_gpu_incsubtensor(node):
    if isinstance(node.op, GpuFromHost):
        host_output = node.inputs[0]
        if host_output.owner and \
           type(host_output.owner.op) == tensor.IncSubtensor:
            incsubt = host_output.owner.op
            x, y = host_output.owner.inputs[0:2]
            coords = host_output.owner.inputs[2:]
            return [GpuIncSubtensor(
                incsubt.idx_list,
                inplace=incsubt.inplace,
                set_instead_of_inc=incsubt.set_instead_of_inc)(
                    gpu_from_host(x),
                    gpu_from_host(y),
                    *coords)]
    # Incrementing a float32 x results in a float32
    # output even if y is float64, so we can downcast
    # y to put it on GPU
    if type(node.op) == tensor.IncSubtensor and \
       node.inputs[0].dtype == "float32":
        x, y = node.inputs[0:2]
        assert isinstance(x.type, tensor.TensorType)
        assert isinstance(y.type, tensor.TensorType)
        coords = node.inputs[2:]
        go_gpu = False
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            go_gpu = True
            gpu_x, = x.owner.inputs
        else:
            gpu_x = gpu_from_host(x)
        if y.owner and isinstance(y.owner.op, HostFromGpu):
            go_gpu = True
            gpu_y, = y.owner.inputs
        else:
            if y.dtype != 'float32':
                y = tensor.cast(y, 'float32')
            gpu_y = gpu_from_host(y)
        if go_gpu:
            return [host_from_gpu(GpuIncSubtensor(
                node.op.idx_list, inplace=node.op.inplace,
                set_instead_of_inc=node.op.set_instead_of_inc)(
                    gpu_x, gpu_y, *coords))]
    return False


@register_opt()
@local_optimizer([tensor.Shape])
def local_gpu_shape(node):
    if isinstance(node.op, tensor.Shape):
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            return [gpu_shape(gpu_x)]
    return False


@register_opt()
@local_optimizer([tensor.Rebroadcast])
def local_gpu_rebroadcast(node):
    '''rebroadcast(host_from_gpu(x)) -> host_from_gpu(rebroadcast(x))'''
    if isinstance(node.op, tensor.Rebroadcast):
        x, = node.inputs
        if (x.owner and isinstance(x.owner.op, HostFromGpu)):
            gpu_x = x.owner.inputs[0]
            return [host_from_gpu(node.op(gpu_x))]


def gpu_print_wrapper(op, cnda):
    op.old_op.global_fn(op.old_op, numpy.asarray(cnda))


@register_opt()
@local_optimizer([tensor.printing.Print])
def local_gpu_print_op(node):
    if isinstance(node.op, tensor.printing.Print):
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            new_op = node.op.__class__(global_fn=gpu_print_wrapper)
            new_op.old_op = node.op
            return [host_from_gpu(new_op(gpu_x))]
    return False


def cast(x, dtype):
    stype = scal.Scalar(dtype)
    cast_op = theano.tensor.Elemwise(scal.Identity(scal.specific_out(stype)))
    return cast_op(x)

import theano.tensor.nnet


@register_opt()
@local_optimizer([tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias])
def local_gpu_crossentorpy_softmax_argmax_1hot_with_bias(node):
    if isinstance(node.op, tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias):
        x, b, y = node.inputs
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            # if y is a cast to integers, we can go to the underlying
            # thing if we want, since this gpu op will cast to integers
            # internally anyway
            int_cast_ops = (
                    tensor.basic._convert_to_int32,
                    tensor.basic._convert_to_int8,
                    tensor.basic._convert_to_int16,
                    tensor.basic._convert_to_int64,
                    )
            while y.owner and y.owner.op in int_cast_ops:
                y = y.owner.inputs[0]
            gpu_nll, gpu_sm, gpu_am = \
                    GpuCrossentropySoftmaxArgmax1HotWithBias()(
                        gpu_x,
                        gpu_from_host(b),
                        gpu_from_host(cast(y, 'float32')))
            am_dtype = node.outputs[2].type.dtype
            return [host_from_gpu(gpu_nll),
                    host_from_gpu(gpu_sm),
                    cast(host_from_gpu(gpu_am), am_dtype)]
    return False


@register_opt()
@local_optimizer([tensor.nnet.CrossentropySoftmax1HotWithBiasDx])
def local_gpu_crossentorpy_softmax_1hot_with_bias_dx(node):
    if isinstance(node.op, tensor.nnet.CrossentropySoftmax1HotWithBiasDx):
        dnll, sm, yidx = node.inputs
        if sm.owner and isinstance(sm.owner.op, HostFromGpu):
            gpu_sm, = sm.owner.inputs
            gpu_dx = GpuCrossentropySoftmax1HotWithBiasDx()(
                gpu_from_host(dnll),
                gpu_sm,
                gpu_from_host(cast(yidx, 'float32')))
            return [host_from_gpu(gpu_dx)]
    return False


@register_opt()
@local_optimizer([tensor.nnet.Softmax])
def local_gpu_softmax(node):
    if isinstance(node.op, tensor.nnet.Softmax):
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            gpu_sm = GpuSoftmax()(gpu_x)
            return [host_from_gpu(gpu_sm)]
    return False


@register_opt()
@local_optimizer([tensor.nnet.SoftmaxWithBias])
def local_gpu_softmax_with_bias(node):
    if isinstance(node.op, tensor.nnet.SoftmaxWithBias):
        x, b = node.inputs
        x_on_gpu = x.owner and isinstance(x.owner.op, HostFromGpu)
        b_on_gpu = b.owner and isinstance(b.owner.op, HostFromGpu)
        if x_on_gpu or b_on_gpu:
            gpu_sm = GpuSoftmaxWithBias()(gpu_from_host(x), gpu_from_host(b))
            return [host_from_gpu(gpu_sm)]
    return False

#### Convolution, maxpooling
from theano.tensor.nnet import conv


@register_opt()
@local_optimizer([gpu_from_host, conv.ConvOp])
def local_gpu_conv(node):
    """
    gpu_from_host(conv) -> gpu_conv(gpu_from_host)

    conv(host_from_gpu) -> host_from_gpu(gpu_conv)
    """
    def GpuConvOp_from_ConvOp(op):
        logical_img_hw = None

        if op.kshp_logical is not None and op.kshp_logical != op.kshp:
            return None
        #print op.kshp, op.imshp[1:3]
        #print op.kshp_logical, logical_img_hw
        ret = GpuConv(border_mode=op.out_mode,
                    subsample=(op.dx, op.dy),
                    logical_img_hw=logical_img_hw,
                    logical_kern_hw=op.kshp_logical,
                    logical_kern_align_top=op.kshp_logical_top_aligned,
                    kshp=op.kshp,
                    version=op.version,
                    verbose=op.verbose,
                    imshp=op.imshp,
                    )
        if op.imshp_logical is not None:
            logical_img_hw = op.imshp_logical[1:3]
            if logical_img_hw != op.imshp[1:3]:
                # this case is not implemented
                #return None
                rstride = int(numpy.ceil(op.imshp_logical[1] /
                                         float(op.imshp[1])))
                cstride = int(numpy.ceil(op.imshp_logical[2] /
                                         float(op.imshp[2])))

                def make_graph(img, kern):
                    buf = tensor.alloc(numpy.asarray(0, dtype=img.dtype),
                                       img.shape[0], *op.imshp_logical)
                    img = tensor.set_subtensor(buf[:, :, ::rstride, ::cstride],
                                               img)
                    img = gpu_from_host(img)
                    return ret(img, kern)

                return make_graph
        return ret

    def values_eq_approx(a, b):
        """This fct is needed to don't have DebugMode raise useless
        error due to ronding error.

        This happen as We reduce on the two last dimensions, so this
        can raise the absolute error if the number of element we
        reduce on is significant.

        """
        assert a.ndim == 4
        atol = None
        if a.shape[-1] * a.shape[-2] > 100:
            #For float32 the default atol is 1e-5
            atol = 3e-5
        return CudaNdarrayType.values_eq_approx(a, b, atol=atol)

    if isinstance(node.op, GpuFromHost):
        #gpu_from_host(conv) -> gpu_conv(gpu_from_host)
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, conv.ConvOp):
            gpu_conv = GpuConvOp_from_ConvOp(host_input.owner.op)
            if gpu_conv is None:
                return
            img, kern = host_input.owner.inputs
            out = gpu_conv(gpu_from_host(img),
                           gpu_from_host(kern))
            out = tensor.patternbroadcast(out,
                                          node.outputs[0].broadcastable)
            out.values_eq_approx = values_eq_approx
            # in some case the ConvOp broadcast the last 2 dimensions
            # differently then the gpu ConvOp
            return [out]

    if isinstance(node.op, conv.ConvOp):
        #conv(host_from_gpu) -> host_from_gpu(gpu_conv)
        img, kern = node.inputs
        img_on_gpu = (img.owner and isinstance(img.owner.op, HostFromGpu))
        kern_on_gpu = (kern.owner and isinstance(kern.owner.op, HostFromGpu))
        if img_on_gpu or kern_on_gpu:
            gpu_conv = GpuConvOp_from_ConvOp(node.op)
            if gpu_conv is None:
                return
            out = gpu_conv(gpu_from_host(img),
                           gpu_from_host(kern))
            out = tensor.patternbroadcast(
                host_from_gpu(out),
                node.outputs[0].broadcastable)
            out.values_eq_approx = values_eq_approx
            # in some case the ConvOp broadcast the last 2 dimensions
            # differently then the gpu ConvOp
            return [out]


@register_opt()
@local_optimizer([GpuConv])
def local_conv_fft(node):
    if (theano.config.enable_conv2d_fft and
        isinstance(node.op, GpuConv) and
        node.op.border_mode == 'valid' and
        node.op.subsample == (1, 1)):
        return [conv2d_fft(node.inputs[0], node.inputs[1])]

import theano.tensor.signal.downsample as downsample


@register_opt()
@local_optimizer([downsample.DownsampleFactorMax])
def local_gpu_downsample_factor_max(node):
    if isinstance(node.op, downsample.DownsampleFactorMax):
        x, = node.inputs
        if (x.owner and isinstance(x.owner.op, HostFromGpu)):
            gpu_ds = GpuDownsampleFactorMax(node.op.ds, node.op.ignore_border)
            return [host_from_gpu(gpu_ds(x.owner.inputs[0]))]


@register_opt()
@local_optimizer([downsample.DownsampleFactorMaxGrad])
def local_gpu_downsample_factor_max_grad(node):
    if isinstance(node.op, downsample.DownsampleFactorMaxGrad):
        x, z, gz = node.inputs
        if (x.owner and isinstance(x.owner.op, HostFromGpu)):
            gpu_ds_grad = GpuDownsampleFactorMaxGrad(node.op.ds,
                                                     node.op.ignore_border)
            return [host_from_gpu(gpu_ds_grad(x.owner.inputs[0],
                                              gpu_from_host(z),
                                              gpu_from_host(gz)))]


from theano.sandbox.cuda.basic_ops import gpu_join, GpuJoin


@register_opt()
@local_optimizer([tensor.Join])
def local_gpu_join(node):
    """
    Inspired by the opt for convop.

    Very loose notation follows.

    Subgraphs concerned first look like
        [array of HostTensor] -> HostToGpu -> GpuToHost
        -> Join -> HostToGpu -> GpuToHost

    First we apply this Opt:

    join(host_from_gpu) -> host_from_gpu(gpu_join)

    then, as an intermediate result, there should be
    host_from_gpu(gpu_join) -> HostToGpu -> GpuToHost
    this unnecessary GpuToHost -> HostToGpu should be removed
    by other opts, leaving us with
    host_from_gpu(gpu_join)


    For intermediate places in the graph not covered by the first opt, the
    following could be useful:

    gpu_from_host(join) -> gpu_join(gpu_from_host)

    not implemented yet.

    """
    if isinstance(node.op, tensor.Join):
        # optimizing this case:
        # join(host_from_gpu) -> host_from_gpu(gpu_join)

        # print "OPT: we've got a Join instance"

        axis_and_tensors = node.inputs

        #print "OPT: axis_and_tensors=", axis_and_tensors

        matches = [(not t.owner is None and isinstance(t.owner.op, HostFromGpu)) or
                   isinstance(t, gof.Constant) for t in axis_and_tensors[1:]]
        #print "OPT: matches =", matches

        # if all input tensors are host_from_gpu'ified
        if all(matches):
            # the extra gpu_from_host introduced here will
            # be removed by further optimizations
            new_tensors = [gpu_from_host(t) for t in axis_and_tensors[1:]]
            new_a_and_t = [axis_and_tensors[0]] + new_tensors

            replacement_node = host_from_gpu(gpu_join(*new_a_and_t))

            # print "OPT: replacement_node", replacement_node

            return [replacement_node]

# This is a copy of the same opt in tensor to make the tests happy,
# but I'm not convinced it is actually needed.
@register_opt()
@local_optimizer([GpuJoin])
def local_gpujoin_1(node):
    tensors = node.inputs[1:]
    if len(tensors) == 1:
        return [tensors[0]]

# Commented out because it can result in
#   shared =  dimshuffle(gemm_inplace(dimshuffle(shared)))
# which causes memory leaks (long term fix is to make the above not leak
# memory)
@local_optimizer([gpu_gemm_no_inplace], inplace=True)
def local_inplace_gemm(node):
    if node.op == gpu_gemm_no_inplace:
        return [gpu_gemm_inplace(*node.inputs)]


@local_optimizer([gpu_gemv_no_inplace], inplace=True)
def local_inplace_gemv(node):
    if node.op == gpu_gemv_no_inplace:
        return [gpu_gemv_inplace(*node.inputs)]


@local_optimizer([gpu_ger_no_inplace], inplace=True)
def local_inplace_ger(node):
    if node.op == gpu_ger_no_inplace:
        return [gpu_ger_inplace(*node.inputs)]

# After destroyhandler is in but before we try to make elemwise things inplace
# Try to make gpu gemm inplace
# Also, need to make the gemm optimisation(step 70) happen before the fusion of
# elemwise(step 71)
optdb.register('InplaceGpuBlasOpt',
               tensor.opt.in2out(local_inplace_gemm,
                                 local_inplace_gemv,
                                 local_inplace_ger,
                                 name="InplaceGpuBlasOpt"),
               70.0, 'fast_run', 'inplace', 'gpu')


def get_device_type_sizes():
    """
    :return:(gpu ptr size, cpu ptr size, int sizes(gpu and cpu))
    :return type: tuple
    """
    if hasattr(get_device_type_sizes, 'rval'):
        return get_device_type_sizes.rval
    gpu_ptr_size = 8
    cpu_ptr_size = 8
    int_size = 8
    try:

        cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
        t = cuda_ndarray.ptr_int_size()
        gpu_ptr_size, cpu_ptr_size, int_size, gpu_int_size = t
        assert int_size == gpu_int_size
        del gpu_int_size
        del t
    except Exception, e:
        _logger.warning(("Optimization Warning: "
            "Got the following error, but we can ignore it. "
            "This could cause less GpuElemwise fused together.\n"
            "%s") % e)

    rval = get_device_type_sizes.rval = locals()
    return rval


def max_inputs_to_GpuElemwise(node):
    """
    return the maximum number of inputs this GpuElemwise Apply node can
    accept.

    This is needed as currently there is a limit of 256 bytes of
    parameter for the gpu function on devices with compute capability
    1.x. There is a 4 kbyte limit on devices with compute capability
    2.x (not used).

    This measures the number of parameters we put in our GPU function and
    computes the maximum number of inputs that respect the 256 byte
    limit.
    """
    type_sizes = get_device_type_sizes()
    int_size = type_sizes['int_size']
    gpu_ptr_size = type_sizes['gpu_ptr_size']

    # some bytes are used for block and thread coords etc.
    argument_limit = 232
    ndim = node.inputs[0].type.ndim
    size_param_mandatory = int_size  # for numels
    size_param_mandatory += int_size * ndim  # for the shape
    size_param_mandatory += sum((gpu_ptr_size + int_size * ndim)
                                 for i in node.outputs)

    nb_bytes_avail = argument_limit - size_param_mandatory
    nb_bytes_per_inputs = (ndim * int_size) + gpu_ptr_size
    max_nb_inputs = nb_bytes_avail // nb_bytes_per_inputs

    # There is a case this don't algorithm doesn't work. Is this related to
    # the order of the parameters to the gpu function?
    if node.inputs[0].type.ndim == 1 and max_nb_inputs > 14:
        return 14

    return max_nb_inputs


def split_huge_add_or_mul(node):
    """
    For add and mul, it can happen that we have too much input
    That will make nvcc fail compilation of our current code.
    We don't want node in the graph that can't execute
    as this break DebugMode.

    This should not happen for other GpuElemwise as their is only the fusion
    that can generate op with too much input and it check for that.
    """
    if node.op.scalar_op in (scal.add, scal.mul):
        max_nb_inputs = max_inputs_to_GpuElemwise(node)
        if max_nb_inputs <= 1 and len(node.inputs) > 1:
            return False
        while len(node.inputs) > max_nb_inputs:
            inner_op = []
            for i in xrange(0,
                            len(node.inputs),
                            max_nb_inputs):
                inner_op.append(node.op(*node.inputs[i: i + max_nb_inputs]))
            node = node.op(*inner_op).owner
    return node

#GpuElemwise fusion
gpu_local_elemwise_fusion = tensor.opt.local_elemwise_fusion_op(
        GpuElemwise,
        max_inputs_to_GpuElemwise)
if config.gpu.local_elemwise_fusion:
    _logger.debug("enabling optimization fusion of gpu elemwise in fast_run")
    #Must be after cpu fusion at 40, gpu at 48.5 and before AddDestroyHandler at 49.5
    optdb.register('gpu_elemwise_fusion',
                   tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion),
                   49, 'fast_run', 'fusion',
                   'local_elemwise_fusion', 'gpu')
else:
    _logger.debug(("not enabling optimization fusion of gpu elemwise in "
                   "fast_run"))
    optdb.register('gpu_elemwise_fusion',
                   tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion),
                   71.00, 'fusion', 'local_elemwise_fusion')

#GpuElemwise inplace
gpu_inplace_elemwise_optimizer = tensor.opt.inplace_elemwise_optimizer_op(
        GpuElemwise)
optdb.register('gpu_inplace_elemwise_opt', gpu_inplace_elemwise_optimizer, 75,
               'fast_run', 'inplace', 'gpu_inplace', 'gpu')


@register_opt()
@local_optimizer([tensor.alloc])
def local_gpualloc(node):
    replace = False
    if node.op == tensor.alloc:
        if node.inputs[0].owner and \
           isinstance(node.inputs[0].owner.op, HostFromGpu):
            replace = True
        elif all([c != 'output' and c.op == gpu_from_host
                  for c, idx in node.outputs[0].clients]):
            # if all clients are on gpu
            replace = True
        elif all([c != 'output' and
                  c.op == tensor.join and
                  all([i.owner and
                       i.owner.op in [host_from_gpu, tensor.alloc]
                       for i in c.inputs[1:]])
                  for c, idx in node.outputs[0].clients]):
            # if the client is a subtensor with input on gpu or alloc
            replace = True
        if replace and node.inputs[0].dtype != 'float32':
            replace = False
    if replace:
        val = node.inputs[0]
        shp = node.inputs[1:]
        old_out = node.outputs[0]
        val2 = tensor.shape_padleft(val, len(shp) - val.ndim)
        new_out = host_from_gpu(gpu_alloc(val2, *shp))
        # Sigh. it's an annoying thing about theano
        # that you can't add information to the graph.
        # If for some reason it has come to light that
        # one of the dimensions is broadcastable, we have to hide that
        # or the optimization won't go through.
        if new_out.type != old_out.type:
            assert new_out.type.ndim == old_out.type.ndim
            assert new_out.type.dtype == old_out.type.dtype
            # it seems to have happened that new_out has some broadcastable
            # dimensions that old_out did not have
            for b_old, b_new in zip(old_out.type.broadcastable,
                                    new_out.type.broadcastable):
                assert b_new or (not b_old)
            new_out = tensor.patternbroadcast(new_out, old_out.broadcastable)
        #if old_out.type != new_out.type:
            #import pdb; pdb.set_trace()
        return [new_out]


@register_opt()
@local_optimizer([GpuAlloc])
def local_gpualloc_memset_0(node):
    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
        inp = node.inputs[0]
        if (isinstance(inp, CudaNdarrayConstant) and
            inp.data.size == 1 and
            (numpy.asarray(inp.data) == 0).all()):
            new_out = GpuAlloc(memset_0=True)(*node.inputs)
            return [new_out]


@register_opt()
@local_optimizer([gpu_from_host, tensor.Eye])
def local_gpu_eye(node):
    """
    gpu_from_host(eye) -> gpueye(gpu_from_host)

    eye(host_from_gpu) -> host_from_gpu(gpueye)
    """
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op, tensor.Eye) and
            host_input.owner.op.dtype == "float32"):
            return [gpu_eye(*host_input.owner.inputs)]
    if isinstance(node.op, tensor.Eye) and node.op.dtype == "float32":
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]):
            return [host_from_gpu(gpu_eye(*node.inputs))]
    return False


def safe_to_gpu(x):
    if (isinstance(x.type, tensor.TensorType) and
        x.type.dtype == 'float32'):
        return gpu_from_host(x)
    else:
        return x


def safe_to_cpu(x):
    if isinstance(x.type, CudaNdarrayType):
        return host_from_gpu(x)
    else:
        return x


def gpu_safe_new(x, tag=''):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name ( old name + tag). This function is used
    by gradient, or the R-op to construct new variables for the inputs of
    the inner graph such that there is no interference between the original
    graph and the newly constructed graph.
    """
    if hasattr(x, 'name') and x.name is not None:
        nw_name = x.name + tag
    else:
        nw_name = None
    if isinstance(x, theano.Constant):
        return x.clone()

    nw_x = x.type()
    nw_x.name = nw_name
    return nw_x


def gpu_reconstruct_graph(inputs, outputs, tag=None):
    """
    Different interface to clone, that allows you to pass inputs.
    Compared to clone, this method always replaces the inputs with
    new variables of the same type, and returns those ( in the same
    order as the original inputs).
    """
    if tag is None:
        tag = ''
    nw_inputs = [gpu_safe_new(x, tag) for x in inputs]
    givens = {}
    for nw_x, x in zip(nw_inputs, inputs):
        givens[x] = nw_x
    nw_outputs = scan_utils.clone(outputs, replace=givens)
    return (nw_inputs, nw_outputs)


def tensor_to_cuda(x):
    if (isinstance(x.type, tensor.TensorType) and
        x.type.dtype == 'float32'):
        y = CudaNdarrayType(broadcastable=x.type.broadcastable)()
        if x.name:
            y.name = x.name + '[cuda]'
        return y
    else:
        return x


@register_opt()
@local_optimizer(None) # XXX: linalg is in sandbox, so don't import it globally
def local_gpu_extract_diagonal(node):
    """
    extract_diagonal(host_from_gpu()) -> host_from_gpu(extract_diagonal)
    gpu_from_host(extract_diagonal) -> extract_diagonal(gpu_from_host)
    """
    global linalg
    if linalg is None:
        from theano.sandbox import linalg
        linalg = theano.sandbox.linalg

    if (isinstance(node.op, linalg.ops.ExtractDiag) and
        isinstance(node.inputs[0].type,
                   theano.tensor.TensorType)):
        inp = node.inputs[0]
        if inp.owner and isinstance(inp.owner.op, HostFromGpu):
            return [host_from_gpu(linalg.extract_diag(gpu_from_host(inp)))]
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op, linalg.ops.ExtractDiag) and
            isinstance(host_input.owner.inputs[0].type,
                       theano.tensor.TensorType)):
            diag_node = host_input.owner
            return [linalg.extract_diag(
                gpu_from_host(diag_node.inputs[0]))]
    return False

def typeConstructor(broadcastable, dtype):
    if dtype == 'float32':
        return CudaNdarrayType(broadcastable=broadcastable)
    else:
        return tensor.TensorType(broadcastable=broadcastable, dtype=dtype)

@register_opt('scan')
@local_optimizer([gpu_from_host, scan_op.Scan])
def gpuScanOptimization(node):
    """
    scan(host_from_gpu) -> host_from_gpu(GPUscan)
    gpu_from_host(scan) -> GPUscan(gpu_from_host)
    """

    #gpu_from_host(scan) -> GPUscan(gpu_from_host)
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op, scan_op.Scan) and
            not host_input.owner.op.info['gpu'] and
            len(host_input.owner.outputs) == 1):
            # Note that we are not doing the right thing here !!
            # This is because the local optimizer expects only one
            # output that corresponds to the input of ``node``
            # If we do this for each output seperately we will have
            # multiple scan ops in the graph ( as many as outputs )
            # and I'm not sure they will get merged into one again
            # So for now I will just cover a limited case when there
            # is only one output and the local optimizer can be used
            # TODO (fix) : either make sure the different scans get
            # merged or implement this optimization as a global
            # optimization
            thescan = host_input.owner.op
            info = copy.deepcopy(thescan.info)
            info['gpu'] = True
            inputs = host_input.owner.inputs
            nw_ins = [inputs[0]]
            e = (1 +
                 thescan.n_seqs +
                 thescan.n_mit_mot +
                 thescan.n_mit_sot +
                 thescan.n_sit_sot +
                 thescan.n_shared_outs)
            nw_ins += [safe_to_gpu(x) for x in inputs[1:e]]
            b = e
            e = e + thescan.n_nit_sot
            nw_ins += inputs[b:e]
            nw_ins += [safe_to_gpu(x) for x in inputs[e:]]
            scan_ins = [tensor_to_cuda(x) for x in thescan.inputs]
            scan_outs = [safe_to_gpu(x) for x in thescan.outputs]
            scan_outs = scan_utils.clone(
                scan_outs,
                replace=zip(thescan.inputs,
                            [safe_to_cpu(x) for x in scan_ins]))
            # We need to construct the hash here, because scan
            # __init__ does not know about cuda ndarray and can not
            # handle graphs with inputs being Cuda Ndarrays
            tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins,
                                                    scan_outs)
            local_fgraph = gof.FunctionGraph(tmp_in, tmp_out)
            _cmodule_key = gof.CLinker().cmodule_key_(local_fgraph, [])
            info['gpu_hash'] = hash(_cmodule_key)

            nw_op = scan_op.Scan(scan_ins,
                                 scan_outs,
                                 info,
                                 typeConstructor=typeConstructor).make_node(
                                     *nw_ins)
            _outputs = nw_op.outputs
            return _outputs

    #scan(host_from_gpu) -> host_from_gpu(GPUscan)
    if (type(node.op) == scan_op.Scan
        and not node.op.info['gpu']):
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]):

            thescan = node.op
            info = copy.deepcopy(thescan.info)
            info['gpu'] = True
            inputs = node.inputs
            nw_ins = [inputs[0]]
            e = (1 +
                 thescan.n_seqs +
                 thescan.n_mit_mot +
                 thescan.n_mit_sot +
                 thescan.n_sit_sot +
                 thescan.n_shared_outs)
            nw_ins += [safe_to_gpu(x) for x in inputs[1:e]]
            b = e
            e = e + thescan.n_nit_sot
            nw_ins += inputs[b:e]
            nw_ins += [safe_to_gpu(x) for x in inputs[e:]]

            scan_ins = [tensor_to_cuda(x) for x in thescan.inputs]
            scan_outs = [safe_to_gpu(x) for x in thescan.outputs]
            scan_outs = scan_utils.clone(
                scan_outs,
                replace=zip(thescan.inputs,
                            [safe_to_cpu(x) for x in scan_ins]))

            # We need to construct the hash here, because scan
            # __init__ does not know about cuda ndarray and can not
            # handle graphs with inputs being Cuda Ndarrays
            tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins,
                                                    scan_outs)
            local_fgraph = gof.FunctionGraph(tmp_in, tmp_out)
            _cmodule_key = gof.CLinker().cmodule_key_(local_fgraph, [])
            info['gpu_hash'] = hash(_cmodule_key)

            _outputs = scan_op.Scan(
                scan_ins,
                scan_outs,
                info,
                typeConstructor=typeConstructor).make_node(*nw_ins).outputs
            outputs = []
            for x, y in zip(_outputs, node.outputs):
                if isinstance(y.type, CudaNdarrayType):
                    outputs += [x]
                else:
                    outputs += [safe_to_cpu(x)]
            return outputs
    return False


optdb.register('gpu_scanOp_make_inplace',
               scan_opt.ScanInplaceOptimizer(typeConstructor=typeConstructor,
                                             gpu_flag=True),
               75,
               'gpu',
               'fast_run',
               'inplace',
               'scan')
