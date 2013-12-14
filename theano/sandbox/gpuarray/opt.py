import copy
import theano
import numpy
from theano import tensor, scalar
from theano.compile import optdb
from theano.gof import (local_optimizer, EquilibriumDB,
                        SequenceDB, ProxyDB,
                        Optimizer, toolbox, DestroyHandler,
                        InconsistencyError, EquilibriumOptimizer)

from theano.gof.python25 import all, any
from theano.sandbox.gpuarray.type import GpuArrayType

from theano.sandbox.gpuarray.basic_ops import (host_from_gpu,
                                               gpu_from_host,
                                               gpu_alloc,
                                               gpu_shape,
                                               GpuAlloc,
                                               GpuShape,
                                               GpuReshape,
                                               GpuEye)
from theano.sandbox.gpuarray.blas import gpu_dot22, GpuGemv, GpuGemm
from theano.sandbox.gpuarray.nnet import (GpuCrossentropySoftmaxArgmax1HotWithBias,
                                          GpuCrossentropySoftmax1HotWithBiasDx)
from theano.sandbox.gpuarray.elemwise import (GpuElemwise, _is_scalar,
                                              GpuDimShuffle, GpuCAReduce)
from theano.sandbox.gpuarray.subtensor import GpuIncSubtensor, GpuSubtensor
from theano.sandbox.gpuarray.type import GpuArrayConstant

gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()

gpu_seqopt = SequenceDB()

gpu_seqopt.register('gpuarray_local_optimiziations', gpu_optimizer, 1,
                    'fast_run', 'inplace', 'gpuarray')
gpu_seqopt.register('gpuarray_cut_transfers', gpu_cut_copies, 2,
                    'fast_run', 'gpuarray')

# do not add 'fast_run' to these two as this would always enable gpuarray mode
optdb.register('gpuarray_opt', gpu_seqopt,
               optdb.__position__.get('add_destroy_handler', 49.5) - 1,
               'gpuarray')


def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'gpuarray', *tags)
        return local_opt
    return f

register_opt()(theano.tensor.opt.local_track_shape_i)


def op_lifter(OP):
    """
    OP(..., host_from_gpu(), ...) -> host_from_gpu(GpuOP(...))
    gpu_from_host(OP(inp0, ...)) -> GpuOP(inp0, ...)
    """
    def f(maker):
        def local_opt(node):
            if type(node.op) in OP:
                # This does not support nodes that have more than one output.
                # either one of our inputs is on the gpu or
                # all of our client are on the gpu
                if (any([i.owner and i.owner.op == host_from_gpu
                         for i in node.inputs]) or
                    all([c != 'output' and c.op == gpu_from_host
                         for c, idx in node.outputs[0].clients])):
                    new_op = maker(node)
                    # This is needed as sometimes new_op inherit from OP.
                    if new_op and new_op != node.op:
                        if isinstance(new_op, theano.Op):
                            return [host_from_gpu(o) for o in new_op(*node.inputs, return_list=True)]
                        elif isinstance(new_op, (tuple, list)):
                            return [host_from_gpu(o) for o in new_op]
                        else:  # suppose it is a variable on the GPU
                            return [host_from_gpu(new_op)]
            return False
        local_opt.__name__ = maker.__name__
        return local_optimizer(OP)(local_opt)
    return f


class InputToGpuOptimizer(Optimizer):
    "Transfer the input to the gpu to start the rolling wave."

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())
        fgraph.attach_feature(DestroyHandler())

    def apply(self, fgraph):
        for input in fgraph.inputs:
            if isinstance(input.type, GpuArrayType):
                continue

            if (len(input.clients) == 1 and
                (input.clients[0][0] == 'output' or
                 input.clients[0][0].op == gpu_from_host)):
                continue

            try:
                new_input = host_from_gpu(gpu_from_host(input))
                fgraph.replace_validate(input, new_input,
                                        "InputToGpuOptimizer")
            except TypeError, e:
                # This could fail if the inputs are not TensorTypes
                pass

gpu_seqopt.register('InputToGpuArrayOptimizer', InputToGpuOptimizer(),
                    0, 'fast_run', 'fast_compile', 'merge')


@local_optimizer([gpu_from_host, host_from_gpu])
def local_cut_gpu_host_gpu(node):
    if tensor.opt.opt.check_chain(node, gpu_from_host, host_from_gpu):
        return [node.inputs[0].owner.inputs[0]]
    if tensor.opt.opt.check_chain(node, host_from_gpu, gpu_from_host):
        return [node.inputs[0].owner.inputs[0]]
    return False
gpu_cut_copies.register('cut_gpua_host_transfers', local_cut_gpu_host_gpu,
                        'fast_run', 'inplace', 'gpuarray')
gpu_cut_copies.register('cut_gpua_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_run', 'gpuarray')
optdb['canonicalize'].register('local_cut_gpua_host_gpua',
                               local_cut_gpu_host_gpu, 'fast_run', 'gpuarray')


@register_opt()
@op_lifter([tensor.Alloc])
def local_gpualloc(node):
    return gpu_alloc


@register_opt()
@local_optimizer([GpuAlloc])
def local_gpualloc_memset_0(node):
    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
        inp = node.inputs[0]
        if (isinstance(inp, GpuArrayConstant) and
            inp.data.size == 1 and
            (numpy.asarray(inp.data) == 0).all()):
            new_out = GpuAlloc(memset_0=True)(*node.inputs)
            return [new_out]


@register_opt()
@op_lifter([tensor.Reshape])
def local_gpureshape(node):
    op = node.op
    name = op.name
    if name:
        name = 'Gpu' + name
    res = GpuReshape(op.ndim, op.name)
    return res


@register_opt()
@op_lifter([tensor.Flatten])
def local_gpuflatten(node):
    op = node.op
    shp =[]
    if op.outdim != 1:
        shp = [node.inputs[0].shape[i] for i in range(op.outdim - 1)]
    shp += [-1]
    res = GpuReshape(op.outdim, None)
    o = res(node.inputs[0], theano.tensor.as_tensor_variable(shp))
    return o


@register_opt()
@op_lifter([tensor.Elemwise])
def local_gpu_elemwise(node):
    op = node.op
    name = op.name
    if node.outputs[0].ndim == 0:
        return
    if name:
        name = 'Gpu'+name
    res = GpuElemwise(op.scalar_op, name=name,
                      inplace_pattern=copy.copy(op.inplace_pattern),
                      nfunc_spec=op.nfunc_spec)
    return res


def max_inputs_to_GpuElemwise(node):
    ptr_size = 8
    int_size = 4

    # we take the limit from CUDA for now
    argument_limit = 232
    ndim = node.inputs[0].type.ndim
    # number of elements and shape
    size_param_mandatory = (int_size * (ndim + 1)) + \
        (ptr_size + int_size * ndim) * len(node.outputs)

    nb_bytes_avail = argument_limit - size_param_mandatory
    nb_bytes_per_input = ptr_size + ndim * int_size
    max_nb_inputs = nb_bytes_avail // nb_bytes_per_input

    return max_nb_inputs

gpu_local_elemwise_fusion = tensor.opt.local_elemwise_fusion_op(
    GpuElemwise,
    max_inputs_to_GpuElemwise)
optdb.register('gpua_elemwise_fusion',
               tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion), 71.00,
               'fast_run', 'fusion', 'local_elemwise_fusion', 'gpuarray')

inplace_gpu_elemwise_opt = tensor.opt.inplace_elemwise_optimizer_op(
    GpuElemwise)
optdb.register('gpua_inplace_opt', inplace_gpu_elemwise_opt, 75,
               'inplace_elemwise_optimizer', 'fast_run', 'inplace', 'gpuarray')


@register_opt()
@op_lifter([tensor.DimShuffle])
def local_gpua_dimshuffle(node):
    return GpuDimShuffle(node.op.input_broadcastable,
                         node.op.new_order)


@register_opt()
@op_lifter([tensor.SpecifyShape])
def local_gpua_specifyShape(node):
    return tensor.specify_shape


@register_opt()
@op_lifter([tensor.Subtensor])
def local_gpua_subtensor(node):
    return GpuSubtensor(node.op.idx_list)


@register_opt()
@op_lifter([tensor.IncSubtensor])
def local_gpua_incsubtensor(node):
    return GpuIncSubtensor(node.op.idx_list, node.op.inplace,
                           node.op.set_instead_of_inc,
                           node.op.destroyhandler_tolerate_aliased)


@register_opt()
@op_lifter([tensor.CAReduce, tensor.Sum])
def local_gpua_careduce(node):
    if (isinstance(node.op.scalar_op, scalar.basic.Add) or
        isinstance(node.op.scalar_op, scalar.basic.Mul)):
        return GpuCAReduce(node.op.scalar_op, axis=node.op.axis,
                           dtype=getattr(node.op, 'dtype', None),
                           acc_dtype=getattr(node.op, 'acc_dtype', None))


@register_opt()
@op_lifter([tensor.blas.Gemv])
def local_gpua_gemv(node):
    return GpuGemv(inplace=node.op.inplace)


@register_opt()
@op_lifter([tensor.blas_c.CGemv])
def local_gpua_gemv2(node):
    return GpuGemv(inplace=node.op.inplace)


@register_opt()
@op_lifter([tensor.blas.Gemm])
def local_gpua_gemm(node):
    return GpuGemm(inplace=node.op.inplace)


@register_opt()
@op_lifter([tensor.blas.Dot22])
def local_gpua_dot22(node):
    return gpu_dot22


@register_opt()
@op_lifter([tensor.basic.Eye])
def local_gpua_eye(node):
    return GpuEye(dtype=node.op.dtype)


@register_opt()
@op_lifter([tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias])
def local_gpua_crossentropysoftmaxargmax1hotwithbias(node):
    return GpuCrossentropySoftmaxArgmax1HotWithBias()


@register_opt()
@op_lifter([tensor.nnet.CrossentropySoftmax1HotWithBiasDx])
def local_gpua_crossentropysoftmax1hotwithbiasdx(node):
    return GpuCrossentropySoftmax1HotWithBiasDx()


@register_opt()
@local_optimizer([tensor.Shape])
def local_gpua_shape(node):
    """
    Can't use op_lifter as the output is on the GPU.
    """
    if isinstance(node.op, tensor.Shape):
        x, = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            return [gpu_shape(gpu_x)]
    return False


@register_opt()
@local_optimizer([])
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
            atol = 3e-5
        return tensor.TensorType.values_eq_approx(a, b, atol=atol)

    if node.op == gpu_from_host:
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
        img_on_gpu = (img.owner and img.owner.op == host_from_gpu)
        kern_on_gpu = (kern.owner and kern.owner.op == host_from_gpu)
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
