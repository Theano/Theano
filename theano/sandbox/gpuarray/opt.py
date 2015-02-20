import copy
import theano
import numpy

try:
    import pygpu
except ImportError:
    pass

from theano import tensor, scalar, gof
from theano.compile import optdb, Rebroadcast
from theano.gof import (local_optimizer, EquilibriumDB,
                        SequenceDB, ProxyDB,
                        Optimizer, toolbox,
                        InconsistencyError, EquilibriumOptimizer)

from theano.scan_module import scan_utils, scan_op, scan_opt

from theano.gof.python25 import all, any
from theano.tensor.nnet.conv import ConvOp

from .type import GpuArrayType, GpuArrayConstant, get_context
from .basic_ops import (
    host_from_gpu, HostFromGpu, GpuFromHost, GpuFromGpu, GpuSplit,
    GpuAlloc, GpuReshape, GpuEye, GpuJoin, as_gpuarray_variable
)
from .blas import gpu_dot22, GpuGemv, GpuGemm, GpuGer
from .conv import GpuConv
from .nnet import (
    GpuCrossentropySoftmaxArgmax1HotWithBias,
    GpuCrossentropySoftmax1HotWithBiasDx,
    GpuSoftmaxWithBias, GpuSoftmax
)
from .elemwise import (GpuElemwise, _is_scalar,
                       GpuDimShuffle, GpuCAReduceCuda,
                       GpuCAReduceCPY)
from .subtensor import (GpuIncSubtensor, GpuSubtensor,
                        GpuAdvancedIncSubtensor1,
                        GpuAdvancedIncSubtensor1_dev20)

gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()

gpu_seqopt = SequenceDB()

gpu_seqopt.register('gpuarray_local_optimiziations', gpu_optimizer, 1,
                    'fast_compile', 'fast_run', 'inplace', 'gpuarray')
gpu_seqopt.register('gpuarray_cut_transfers', gpu_cut_copies, 2,
                    'fast_compile', 'fast_run', 'gpuarray')

# do not add 'fast_run' to this as it would always enable gpuarray mode
optdb.register('gpuarray_opt', gpu_seqopt,
               optdb.__position__.get('add_destroy_handler', 49.5) - 1,
               'gpuarray')


def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'gpuarray', *tags)
        return local_opt
    return f

register_opt('fast_compile')(theano.tensor.opt.local_track_shape_i)


def safe_to_gpu(x, ctx):
    if isinstance(x.type, tensor.TensorType):
        return GpuFromHost(ctx)(x)
    else:
        return x


def safe_to_cpu(x):
    if isinstance(x.type, GpuArrayType):
        return host_from_gpu(x)
    else:
        return x


def op_lifter(OP, cuda_only=False):
    """
    OP(..., host_from_gpu(), ...) -> host_from_gpu(GpuOP(...))
    gpu_from_host(OP(inp0, ...)) -> GpuOP(inp0, ...)
    """
    def f(maker):
        def local_opt(node):
            if type(node.op) in OP:
                # Either one of our inputs is on the gpu or
                # all of our client are on the gpu
                replace = False
                context = None
                # We replace if any input is a host_from_gpu
                for i in node.inputs:
                    if i.owner and i.owner.op == host_from_gpu:
                        # Inherit the context from the inputs
                        context = i.owner.inputs[0].type.context
                        replace = True
                        break
                if not replace:
                    # We replace if *all* clients are on the GPU
                    clients = [c for o in node.outputs for c in o.clients]
                    # Only replace a node if it has at least one client
                    replace = len(clients) != 0
                    for c, idx in clients:
                        if c == 'output' or not isinstance(c.op, GpuFromHost):
                            replace = False
                    # TODO: should we check that all clients want the same context?
                    if replace:
                        # We are sure that we have at least one client
                        # and it is a GpuFromHost
                        context = clients[0][0].op.context
                if (not replace or
                    (cuda_only and get_context(context).kind != 'cuda')):
                    return False
                new_op = maker(node, context)
                # This is needed as sometimes new_op inherit from OP.
                if new_op and new_op != node.op:
                    if isinstance(new_op, theano.Op):
                        # tag the inputs with the context in case the
                        # context was derived from the outputs.
                        def tag(i, ctx):
                            i.tag.context = ctx
                            return i
                        inputs = [tag(i, context) for i in node.inputs]
                        return [safe_to_cpu(o) for o in
                                new_op(*inputs, return_list=True)]
                    elif isinstance(new_op, (tuple, list)):
                        return [safe_to_cpu(o) for o in new_op]
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

    def apply(self, fgraph):
        for input in fgraph.inputs:
            if isinstance(input.type, GpuArrayType):
                continue

            if (len(input.clients) == 1 and
                (input.clients[0][0] == 'output' or
                 isinstance(input.clients[0][0].op, GpuFromHost))):
                continue

            try:
                # This will fail if there is no default context
                ctx = getattr(input.tag, 'context', None)
                new_input = host_from_gpu(GpuFromHost(ctx)(input))
                fgraph.replace_validate(input, new_input,
                                        "InputToGpuOptimizer")
            except TypeError, e:
                # This could fail if the inputs are not TensorTypes
                pass
            except ValueError, e:
                # If there is no context tag and no default context
                # then we give up
                assert ctx is None
                pass

gpu_seqopt.register('InputToGpuArrayOptimizer', InputToGpuOptimizer(),
                    0, 'fast_run', 'fast_compile', 'merge')


# This cuts down on CPU <-> GPU and GPU <-> GPU transfers
@local_optimizer([GpuFromHost, GpuFromGpu, host_from_gpu])
def local_cut_gpu_host_gpu(node):
    # gpu[ab] -> host -> gpub
    if (isinstance(node.op, GpuFromHost) and
        node.inputs[0].owner is not None and
        node.inputs[0].owner.op == host_from_gpu):

        other = node.inputs[0].owner.inputs[0]
        if node.op.context == other.type.context:
            return [other]
        else:
            return [GpuFromGpu(node.op.context)(other)]

    # ? -> gpua -> host
    elif (node.op == host_from_gpu and
          node.inputs[0].owner is not None):

        # host ->
        if isinstance(node.inputs[0].owner.op, GpuFromHost):
            return [node.inputs[0].owner.inputs[0]]

        # gpub ->
        if isinstance(node.inputs[0].owner.op, GpuFromGpu):
            return [host_from_gpu(node.inputs[0].owner.inputs[0])]

    # ? -> gpua -> gpub
    elif isinstance(node.op, GpuFromGpu):
        # Transfer within same context
        if node.inputs[0].type.context == node.op.context:
            return [node.inputs[0]]
        if node.inputs[0].owner is not None:

            # host ->
            if isinstance(node.inputs[0].owner.op, GpuFromHost):
                return [GpuFromHost(node.op.context)(
            node.inputs[0].owner.inputs[0])]

            # gpuc ->
            if isinstance(node.inputs[0].owner.op, GpuFromGpu):
                other = node.inputs[0].owner.inputs[0]
                if node.op.context == other.type.context:
                    return [other]
                else:
                    return [node.op(other)]

    return False

gpu_cut_copies.register('cut_gpua_host_transfers', local_cut_gpu_host_gpu,
                        'fast_compile', 'fast_run', 'inplace', 'gpuarray')
gpu_cut_copies.register('cut_gpua_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_compile', 'fast_run', 'gpuarray')
optdb['canonicalize'].register('local_cut_gpua_host_gpua',
                               local_cut_gpu_host_gpu,
                               'fast_compile', 'fast_run', 'gpuarray')


# TODO might need more work to figure out the proper context
@register_opt('fast_compile')
@local_optimizer([tensor.Alloc])
def local_gpuaalloc2(node):
    """
    Join(axis, {Alloc or HostFromGPU}, ...) -> Join(axis, GpuAlloc, Alloc, ...)

    Moves an alloc that is an input to join to the gpu.
    """
    if isinstance(node.op, tensor.Alloc):
        replace = True
        # We try to match an existing context if there is one
        context = None
        for c, idx in node.outputs[0].clients:
            if c == 'output':
                replace = False
            elif c.op == tensor.join:
                for i in c.inputs[1:]:
                    if not i.owner:
                        replace = False
                    if i.owner.op == host_from_gpu:
                        context = i.owner.inputs[0].type.context
                    elif i.owner.op != tensor.alloc:
                        replace = False
                if context is None:
                    context = getattr(c.outputs[0].tag, 'context', None)
        if replace:
            if context is None:
                context = getattr(node.outputs[0].tag, 'context', None)
            try:
                return [host_from_gpu(GpuAlloc(context)(*node.inputs))]
            except ValueError:
                if context is None:
                    # If there is no default context, ignore failures
                    return False
                raise


@register_opt('fast_compile')
@op_lifter([tensor.Alloc])
def local_gpuaalloc(node, context):
    new_out = GpuAlloc(context)(*node.inputs)
    # We need to hide new broadcastable dimensions because
    # ReplaceValidate doesn't like when they change.
    if new_out.broadcastable != node.outputs[0].broadcastable:
        # but if a dim is suddenly not broadcastable anymore then that's a bug
        for b_old, b_new in zip(node.outputs[0].broadcastable,
                                new_out.broadcastable):
            assert b_new or (not b_old)
        new_out = tensor.patternbroadcast(new_out,
                                          node.outputs[0].broadcastable)
    return (new_out,)


@register_opt()
@local_optimizer([GpuAlloc])
def local_gpualloc_memset_0(node):
    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
        inp = node.inputs[0]
        if (isinstance(inp, GpuArrayConstant) and
            inp.data.size == 1 and
            (numpy.asarray(inp.data) == 0).all()):
            new_out = GpuAlloc(node.op.context,
                               memset_0=True)(*node.inputs)
            return [new_out]


@register_opt('fast_compile')
@op_lifter([tensor.Reshape])
def local_gpureshape(node, context):
    op = node.op
    name = op.name
    if name:
        name = 'Gpu' + name
    res = GpuReshape(op.ndim, op.name)
    return res


@register_opt('fast_compile')
@op_lifter([Rebroadcast])
def local_gpu_rebroadcast(node, context):
    if node.inputs[0].owner.op == host_from_gpu:
        return node.op(node.inputs[0].owner.inputs[0])


@register_opt('fast_compile')
@op_lifter([tensor.Flatten])
def local_gpuflatten(node, context):
    op = node.op
    shp = []
    if op.outdim != 1:
        shp = [node.inputs[0].shape[i] for i in range(op.outdim - 1)]
    shp += [-1]
    res = GpuReshape(op.outdim, None)
    o = res(node.inputs[0], theano.tensor.as_tensor_variable(shp))
    return o


@register_opt('fast_compile')
@op_lifter([tensor.Elemwise])
def local_gpu_elemwise(node, context):
    op = node.op
    name = op.name
    if name:
        name = 'Gpu'+name
    res = GpuElemwise(op.scalar_op, name=name,
                      inplace_pattern=copy.copy(op.inplace_pattern),
                      nfunc_spec=op.nfunc_spec, context=context)
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
    max_inputs_to_GpuElemwise,
    maker=lambda node, C: GpuElemwise(C, context=node.op.context))
optdb.register('gpua_elemwise_fusion',
               tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion), 71.00,
               'fast_run', 'fusion', 'local_elemwise_fusion', 'gpuarray')

inplace_gpu_elemwise_opt = tensor.opt.inplace_elemwise_optimizer_op(
    GpuElemwise)
optdb.register('gpua_inplace_opt', inplace_gpu_elemwise_opt, 75,
               'inplace_elemwise_optimizer', 'fast_run', 'inplace', 'gpuarray')


@register_opt('fast_compile')
@op_lifter([tensor.DimShuffle])
def local_gpua_dimshuffle(node, context):
    return GpuDimShuffle(node.op.input_broadcastable,
                         node.op.new_order)


@register_opt('fast_compile')
@op_lifter([tensor.SpecifyShape])
def local_gpua_specifyShape(node, context):
    if isinstance(node.inputs[0].type, GpuArrayType):
        return
    inp = [GpuFromHost(context)(node.inputs[0])] + node.inputs[1:]
    return tensor.specify_shape(*inp)


@register_opt('fast_compile')
@op_lifter([theano.compile.ops.Shape])
def local_gpua_shape(node, context):
    # op_lifter will call this opt too frequently as the output is
    # always on the CPU.
    if isinstance(node.inputs[0].type, GpuArrayType):
        return
    return [GpuFromHost(context)(node.inputs[0]).shape]


def gpu_print_wrapper(op, cnda):
    op.old_op.global_fn(op.old_op, numpy.asarray(cnda))


@register_opt('fast_compile')
@op_lifter([tensor.printing.Print])
def local_gpu_print_op(node, context):
    x, = node.inputs
    gpu_x, = x.owner.inputs
    new_op = node.op.__class__(global_fn=gpu_print_wrapper)
    new_op.old_op = node.op
    return new_op(gpu_x)


@register_opt('fast_compile')
@op_lifter([tensor.Join])
def local_gpua_join(node, context):
    return GpuJoin(context)


@register_opt('fast_compile')
@local_optimizer([GpuJoin])
def local_gpuajoin_1(node):
    # join of a single element
    if (isinstance(node.op, GpuJoin) and
        len(node.inputs) == 2):
        return [node.inputs[1]]


@register_opt('fast_compile')
@op_lifter([tensor.Split])
def local_gpua_split(node, context):
    return GpuSplit(node.op.len_splits)


@register_opt('fast_compile')
@op_lifter([tensor.Subtensor])
def local_gpua_subtensor(node, context):
    x = node.inputs[0]
    if (x.owner and isinstance(x.owner.op, HostFromGpu)):
        gpu_x = x.owner.inputs[0]
        if (gpu_x.owner and
            isinstance(gpu_x.owner.op, GpuFromHost) and
            # And it is a shared var or an input of the graph.
            not gpu_x.owner.inputs[0].owner):
            if len(x.clients) == 1:
                if any([n == 'output' or any([isinstance(v.type, GpuArrayType)
                                              for v in n.inputs + n.outputs])
                        for n,_  in node.outputs[0].clients]):
                    return
                else:
                    return [host_from_gpu(GpuFromHost(context)(node.outputs[0]))]
    return GpuSubtensor(node.op.idx_list)


@register_opt('fast_compile')
@op_lifter([tensor.IncSubtensor])
def local_gpua_incsubtensor(node, context):
    return GpuIncSubtensor(node.op.idx_list, node.op.inplace,
                           node.op.set_instead_of_inc,
                           node.op.destroyhandler_tolerate_aliased)


@register_opt('fast_compile')
@op_lifter([tensor.AdvancedIncSubtensor1])
def local_gpua_advanced_incsubtensor(node, context):

    x, y = node.inputs[0:2]
    coords = node.inputs[2:]

    ctx = get_context(context)

    # This optimization is disabled if cuda is not active
    if ctx.kind != "cuda":
        return None

    set_instead_of_inc = node.op.set_instead_of_inc

    # HACK ALERT: currently the bin_id for cuda contexts is sm_<major><minor>
    # this is not set in stone and may change later
    cc = ctx.bin_id
    assert cc.startswith('sm_')
    compute_capability = int(cc[3])

    if (compute_capability < 2 or x.ndim != 2 or y.ndim != 2):
        return GpuAdvancedIncSubtensor1(
            inplace=node.op.inplace,
            set_instead_of_inc=set_instead_of_inc)
    else:
        return GpuAdvancedIncSubtensor1_dev20(
            inplace=node.op.inplace,
            set_instead_of_inc=set_instead_of_inc,
            context=context)


@register_opt('fast_compile')
@op_lifter([tensor.CAReduce, tensor.Sum, tensor.elemwise.Prod])
def local_gpua_careduce(node, context):
    if isinstance(node.op.scalar_op, (scalar.Add, scalar.Mul,
                                      scalar.Maximum, scalar.Minimum)):
        x, = node.inputs
        ctx = get_context(context)
        if ctx.kind == 'opencl':
            if node.op.scalar_op not in [scalar.add, scalar.mul]:
                # We don't support yet all reduction with cpy code.
                return
            greduce = GpuCAReduceCPY(
                node.op.scalar_op, axis=node.op.axis,
                dtype=getattr(node.op, 'dtype', None),
                acc_dtype=getattr(node.op, 'acc_dtype', None))
        else:
            greduce = GpuCAReduceCuda(
                node.op.scalar_op, axis=node.op.axis,
                dtype=getattr(node.op, 'dtype', None),
                acc_dtype=getattr(node.op, 'acc_dtype', None),
                context=context)
        gvar = greduce(x)
        # We need to have the make node called, otherwise the mask can
        # be None
        if (isinstance(greduce, GpuCAReduceCPY) or
            gvar.owner.op.supports_c_code([GpuFromHost(context)(x)])):
            return greduce
        else:
            # Try to make a simpler pattern based on reshaping
            # The principle is that if two adjacent dimensions have
            # the same value in the reduce_mask, then we can reshape
            # to make them a single dimension, do the reduction, and
            # then reshape to get them back.

            if node.op.axis is None:
                reduce_mask = [1] * x.type.ndim
            else:
                reduce_mask = [0] * x.type.ndim
                for a in node.op.axis:
                    assert reduce_mask[a] == 0
                    reduce_mask[a] = 1

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
            new_axis = []
            for idx, m in enumerate(new_mask):
                if m == 1:
                    new_axis.append(idx)
            greduce = GpuCAReduceCuda(
                node.op.scalar_op,
                axis=new_axis, reduce_mask=new_mask,
                dtype=getattr(node.op, 'dtype', None),
                acc_dtype=getattr(node.op, 'acc_dtype', None),
                context=context)

            reshaped_x = x.reshape(tensor.stack(*new_in_shp))
            gpu_reshaped_x = GpuFromHost(context)(reshaped_x)
            gvar = greduce(gpu_reshaped_x)
            # We need to have the make node called, otherwise the mask can
            # be None
            reshaped_gpu_inputs = [gpu_reshaped_x]
            if greduce.supports_c_code(reshaped_gpu_inputs):
                reduce_reshaped_x = host_from_gpu(
                    greduce(gpu_reshaped_x))

                if reduce_reshaped_x.ndim != node.outputs[0].ndim:
                    unreshaped_reduce = reduce_reshaped_x.reshape(
                        tensor.stack(*shape_of[node.outputs[0]]))
                else:
                    unreshaped_reduce = reduce_reshaped_x
                return [unreshaped_reduce]


@register_opt('fast_compile')
@op_lifter([tensor.blas.Gemv, tensor.blas_c.CGemv])
def local_gpua_gemv(node, context):
    return GpuGemv(inplace=node.op.inplace)


@register_opt('fast_compile')
@op_lifter([tensor.blas.Gemm])
def local_gpua_gemm(node, context):
    return GpuGemm(inplace=node.op.inplace)


@register_opt('fast_compile')
@op_lifter([tensor.blas.Ger, tensor.blas_c.CGer, tensor.blas_scipy.ScipyGer])
def local_gpua_ger(node, context):
    return GpuGer(destructive=node.op.destructive)


@register_opt('fast_compile')
@op_lifter([tensor.blas.Dot22])
def local_gpua_dot22(node, context):
    return gpu_dot22


@register_opt('fast_compile')
@op_lifter([tensor.basic.Eye])
def local_gpua_eye(node, context):
    return GpuEye(dtype=node.op.dtype,
                  context=context)


@register_opt('fast_compile')
@op_lifter([tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias], cuda_only=True)
def local_gpua_crossentropysoftmaxargmax1hotwithbias(node, context):
    return GpuCrossentropySoftmaxArgmax1HotWithBias(context)


@register_opt('fast_compile')
@op_lifter([tensor.nnet.CrossentropySoftmax1HotWithBiasDx], cuda_only=True)
def local_gpua_crossentropysoftmax1hotwithbiasdx(node, context):
    return GpuCrossentropySoftmax1HotWithBiasDx(context)


@register_opt('fast_compile')
@op_lifter([tensor.nnet.Softmax], cuda_only=True)
def local_gpua_softmax(node, context):
    return GpuSoftmax(context)


@register_opt('fast_compile')
@op_lifter([tensor.nnet.SoftmaxWithBias], cuda_only=True)
def local_gpua_softmaxwithbias(node, context):
    return GpuSoftmaxWithBias(context)


@register_opt('fast_compile')
@op_lifter([theano.tensor.opt.Assert])
def local_assert(node, context):
    return [host_from_gpu(node.op(node.inputs[0].owner.inputs[0],
                                  *node.inputs[1:]))]


@register_opt('fast_compile')
@op_lifter([ConvOp])
def local_gpu_conv(node, context):
    def GpuConvOp_from_ConvOp(op):
        logical_img_hw = None

        if op.kshp_logical is not None and op.kshp_logical != op.kshp:
            return None
        # print op.kshp, op.imshp[1:3]
        # print op.kshp_logical, logical_img_hw
        ret = GpuConv(border_mode=op.out_mode,
                      subsample=(op.dx, op.dy),
                      logical_img_hw=logical_img_hw,
                      logical_kern_hw=op.kshp_logical,
                      logical_kern_align_top=op.kshp_logical_top_aligned,
                      kshp=op.kshp,
                      version=op.version,
                      verbose=op.verbose,
                      imshp=op.imshp,
                      context=context
        )
        if op.imshp_logical is not None:
            logical_img_hw = op.imshp_logical[1:3]
            if logical_img_hw != op.imshp[1:3]:
                # this case is not implemented
                # return None
                rstride = int(numpy.ceil(op.imshp_logical[1] /
                                         float(op.imshp[1])))
                cstride = int(numpy.ceil(op.imshp_logical[2] /
                                         float(op.imshp[2])))

                def make_graph(img, kern):
                    buf = tensor.alloc(numpy.asarray(0, dtype=img.dtype),
                                       img.shape[0], *op.imshp_logical)
                    img = tensor.set_subtensor(buf[:, :, ::rstride, ::cstride],
                                               img)
                    img = GpuFromHost(context)(img)
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
            # For float32 the default atol is 1e-5
            atol = 3e-5
        return GpuArrayType.values_eq_approx(a, b, atol=atol)

    img, kern = node.inputs
    gpu_conv = GpuConvOp_from_ConvOp(node.op)
    if gpu_conv is None:
        return
    out = gpu_conv(img, kern)
    # in some case the ConvOp broadcast the last 2 dimensions
    # differently then the gpu ConvOp
    out = tensor.patternbroadcast(
        host_from_gpu(out),
        node.outputs[0].broadcastable)
    # op_lifter wants the output on the GPU.
    out = GpuFromHost(context)(out)
    out.values_eq_approx = values_eq_approx
    return [out]


@register_opt("low_memory")
@local_optimizer([GpuCAReduceCuda])
def local_gpu_elemwise_careduce(node):
    """ Merge some GpuCAReduceCuda and GPUElemwise"""
    if (isinstance(node.op, GpuCAReduceCuda) and
        node.op.pre_scalar_op is None and
        node.inputs[0].owner and
        isinstance(node.inputs[0].owner.op, GpuElemwise) and
        # The Op support all scalar with 1 inputs.  We don't
        # automatically add more case, as some like trigonometic
        # operation with some reduction pattern will probably result
        # to slow down.
        isinstance(node.inputs[0].owner.op.scalar_op, scalar.basic.Sqr)
        ):
        op = node.op
        inp = node.inputs[0].owner.inputs[0]
        return [GpuCAReduceCuda(scalar_op=op.scalar_op,
                                reduce_mask=op.reduce_mask,
                                pre_scalar_op=scalar.basic.sqr,
                                context=node.op.context)(inp)]


def tensor_to_gpu(x, context):
    if isinstance(x.type, tensor.TensorType):
        y = GpuArrayType(broadcastable=x.type.broadcastable,
                         dtype=x.type.dtype,
                         context=context)()
        if x.name:
            y.name = x.name + '[Gpua]'
        return y
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


@register_opt('scan', 'fast_compile')
@op_lifter([scan_op.Scan])
def local_scan_to_gpua(node, ctx):
    info = copy.deepcopy(node.op.info)
    if info.get('gpua', False):
        return
    info['gpua'] = True
    nw_ins = [node.inputs[0]]
    e = (1 +
         node.op.n_seqs +
         node.op.n_mit_mot +
         node.op.n_mit_sot +
         node.op.n_sit_sot +
         node.op.n_shared_outs)
    nw_ins += [safe_to_gpu(x, ctx) for x in node.inputs[1:e]]
    b = e
    e = e + node.op.n_nit_sot
    nw_ins += node.inputs[b:e]
    nw_ins += [safe_to_gpu(x, ctx) for x in node.inputs[e:]]
    scan_ins = [tensor_to_gpu(x, ctx) for x in node.op.inputs]
    scan_outs = [safe_to_gpu(x, ctx) for x in node.op.outputs]
    scan_outs = scan_utils.clone(
        scan_outs,
        replace=zip(node.op.inputs,
                    [safe_to_cpu(x) for x in scan_ins]))

    # We need to construct the hash here, because scan
    # __init__ does not know about the gpu and can not
    # handle graphs with inputs being on the gpu
    tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins, scan_outs)
    local_fgraph = gof.FunctionGraph(tmp_in, tmp_out, clone=False)
    _cmodule_key = gof.CLinker().cmodule_key_(local_fgraph, [])
    info['gpu_hash'] = hash(_cmodule_key)

    nw_op = scan_op.Scan(scan_ins, scan_outs, info).make_node(*nw_ins)
    return nw_op.outputs

optdb.register('gpua_scanOp_make_inplace',
               scan_opt.ScanInplaceOptimizer(gpua_flag=True),
               75,
               'gpua',
               'fast_run',
               'inplace',
               'scan')
