from __future__ import absolute_import, print_function, division
import copy
import numpy
import logging
from six.moves import xrange

import theano
from theano import tensor, scalar, gof
from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.gof import (local_optimizer, EquilibriumDB,
                        SequenceDB, Optimizer, toolbox)
from theano.gof.optdb import LocalGroupDB
from theano.ifelse import IfElse

from theano.scalar.basic import Scalar, Pow, Cast
from theano.scan_module import scan_utils, scan_op, scan_opt

from theano.tensor.nnet.conv import ConvOp
from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs)

from theano.tests.breakpoint import PdbBreakpoint

from .type import (GpuArrayType, GpuArrayConstant, get_context,
                   ContextNotDefined)
from .basic_ops import (as_gpuarray_variable, infer_context_name,
                        host_from_gpu, GpuToGpu,
                        HostFromGpu, GpuFromHost,
                        GpuSplit, GpuContiguous,
                        GpuAlloc, GpuAllocEmpty, GpuReshape,
                        GpuEye, gpu_join, GpuJoin)
from .blas import (gpu_dot22, GpuGemv, GpuGemm, GpuGer,
                   gpugemm_no_inplace)
from .nnet import (GpuCrossentropySoftmaxArgmax1HotWithBias,
                   GpuCrossentropySoftmax1HotWithBiasDx,
                   GpuSoftmaxWithBias, GpuSoftmax)
from .elemwise import (GpuElemwise, GpuDimShuffle, GpuCAReduceCuda,
                       GpuCAReduceCPY)
from .subtensor import (GpuIncSubtensor, GpuSubtensor,
                        GpuAdvancedSubtensor1,
                        GpuAdvancedIncSubtensor1,
                        GpuAdvancedIncSubtensor1_dev20)
from .opt_util import alpha_merge, output_merge

_logger = logging.getLogger("theano.sandbox.gpuarray.opt")

gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()

gpu_seqopt = SequenceDB()

# Don't register this right now
conv_groupopt = LocalGroupDB()
conv_groupopt.__name__ = "gpua_conv_opts"

gpu_seqopt.register('gpuarray_local_optimiziations', gpu_optimizer, 1,
                    'fast_compile', 'fast_run', 'gpuarray')
gpu_seqopt.register('gpuarray_cut_transfers', gpu_cut_copies, 2,
                    'fast_compile', 'fast_run', 'gpuarray')

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

register_opt('fast_compile')(theano.tensor.opt.local_track_shape_i)
register_opt(final_opt=True, name='gpua_constant_folding')(
    tensor.opt.constant_folding)
gpu_optimizer.register('local_remove_all_assert',
                       theano.tensor.opt.local_remove_all_assert,
                       'unsafe')


def safe_to_gpu(x, ctx_name):
    if isinstance(x.type, tensor.TensorType):
        return GpuFromHost(ctx_name)(x)
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
                # all of our clients are on the gpu
                replace = False
                # TODO: Maybe set context_name with infer_context_name()?
                context_name = None
                # We replace if any input is a host_from_gpu
                for i in node.inputs:
                    if i.owner and i.owner.op == host_from_gpu:
                        context_name = i.owner.inputs[0].type.context_name
                        replace = True
                        break
                if not replace:
                    # We replace if *all* clients are on the GPU
                    clients = [c for o in node.outputs for c in o.clients]
                    replace = len(clients) != 0
                    for c, idx in clients:
                        if (c == 'output' or
                                not isinstance(c.op, GpuFromHost)):
                            replace = False
                    # TODO: check that the clients want the same context?
                    if replace:
                        # All clients are GpuFromHost and we have at least one
                        context_name = clients[0][0].op.context_name

                # Check if we should replace
                if (not replace or
                    (cuda_only and
                     get_context(context_name).kind != 'cuda')):
                    return False

                # tag the inputs with the context in case
                # the context was derived from the outputs
                for i in node.inputs:
                    i.tag.context_name = context_name
                new_op = maker(node, context_name)
                # This is needed as sometimes new_op inherits from OP.
                if new_op and new_op != node.op:
                    if isinstance(new_op, theano.Op):
                        return [safe_to_cpu(o) for o in
                                new_op(*node.inputs, return_list=True)]
                    elif isinstance(new_op, (tuple, list)):
                        return [safe_to_cpu(o) for o in new_op]
                    else:  # suppose it is a variable on the GPU
                        return [host_from_gpu(new_op)]
            return False
        local_opt.__name__ = maker.__name__
        return local_optimizer(OP)(local_opt)
    return f


class InputToGpuOptimizer(Optimizer):
    """
    Transfer the input to the gpu to start the rolling wave.

    """
    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        for input in fgraph.inputs:
            if isinstance(input.type, GpuArrayType):
                continue

            # If all clients are outputs or transfers don't do anything.
            if (all(cl[0] == 'output' or isinstance(cl[0].op, GpuFromHost)
                    for cl in input.clients)):
                continue

            target = getattr(input.tag, 'target', None)
            if target == 'cpu':
                continue

            try:
                new_input = host_from_gpu(GpuFromHost(target)(input))
                fgraph.replace_validate(input, new_input,
                                        "InputToGpuOptimizer")
            except TypeError:
                # This could fail if the inputs are not TensorTypes
                pass
            except ContextNotDefined:
                if hasattr(input.tag, 'target'):
                    raise
                # If there is no context tag and no default context
                # then it stays on the CPU
                pass


gpu_seqopt.register('InputToGpuArrayOptimizer', InputToGpuOptimizer(),
                    0, 'fast_run', 'fast_compile', 'merge')


@local_optimizer([GpuFromHost, GpuToGpu, host_from_gpu])
def local_cut_gpu_transfers(node):
    # gpu[ab] -> host -> gpub
    if (isinstance(node.op, GpuFromHost) and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, HostFromGpu)):
        other = node.inputs[0].owner.inputs[0]
        if node.op.context_name == other.type.context_name:
            return [other]
        else:
            return [GpuToGpu(node.op.context_name)(other)]

    # ? -> gpua -> host
    elif (isinstance(node.op, HostFromGpu) and
          node.inputs[0].owner):
        n2 = node.inputs[0].owner

        # host ->
        if isinstance(n2.op, GpuFromHost):
            return [n2.inputs[0]]

        # gpub ->
        if isinstance(n2.op, GpuToGpu):
            return [host_from_gpu(n2.inputs[0])]

    # ? -> gpua -> gpub
    elif isinstance(node.op, GpuToGpu):
        # Transfer within same context
        if node.inputs[0].type.context_name == node.op.context_name:
            return [node.inputs[0]]

        if node.inputs[0].owner:
            n2 = node.inputs[0].owner

            # host ->
            if isinstance(n2.op, GpuFromHost):
                return [GpuFromHost(node.op.context_name)(n2.inputs[0])]

            # gpuc ->
            if isinstance(n2.op, GpuToGpu):
                if node.op.context_name == n2.inputs[0].type.context_name:
                    return [n2.inputs[0]]
                else:
                    return [node.op(n2.inputs[0])]

gpu_cut_copies.register('cut_gpua_host_transfers', local_cut_gpu_transfers,
                        'fast_compile', 'fast_run', 'inplace', 'gpuarray')
gpu_cut_copies.register('cut_gpua_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_compile', 'fast_run', 'gpuarray')
optdb['canonicalize'].register('local_cut_gpua_host_gpua',
                               local_cut_gpu_transfers,
                               'fast_compile', 'fast_run', 'gpuarray')


@register_opt('fast_compile')
@local_optimizer([tensor.Alloc])
def local_gpuaalloc2(node):
    """
    Join(axis, {Alloc or HostFromGPU}, ...) -> Join(axis, GpuAlloc, Alloc, ...)

    Moves an alloc that is an input to join to the gpu.

    """
    try:
        get_context(None)
    except ContextNotDefined:
        # If there is no default context then we do not perform the move here.
        return
    if (isinstance(node.op, tensor.Alloc) and
        all(c != 'output' and
            c.op == tensor.join and
            all(i.owner and
                i.owner.op in [host_from_gpu, tensor.alloc]
                for i in c.inputs[1:])
            for c, idx in node.outputs[0].clients)):
        return [host_from_gpu(GpuAlloc(None)(*node.inputs))]


@register_opt('fast_compile')
@op_lifter([tensor.Alloc])
def local_gpuaalloc(node, context_name):
    return GpuAlloc(context_name)(*node.inputs)


@register_opt('fast_compile')
@op_lifter([tensor.AllocEmpty])
def local_gpuaallocempty(node, context_name):
    # We use _props_dict() to make sure that the GPU op know all the
    # CPU op props.
    return GpuAllocEmpty(context_name=context_name,
                         **node.op._props_dict())(*node.inputs)


@register_opt()
@local_optimizer([GpuAlloc])
def local_gpualloc_memset_0(node):
    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
        inp = node.inputs[0]
        if (isinstance(inp, GpuArrayConstant) and
                inp.data.size == 1 and
                (numpy.asarray(inp.data) == 0).all()):
            new_op = GpuAlloc(node.op.context_name, memset_0=True)
            return [new_op(*node.inputs)]


# Don't register by default.
@gof.local_optimizer([GpuAllocEmpty])
def local_gpua_alloc_empty_to_zeros(node):
    if isinstance(node.op, GpuAllocEmpty):
        context_name = infer_context_name(*node.inputs)
        z = numpy.asarray(0, dtype=node.outputs[0].dtype)
        return [GpuAlloc()(as_gpuarray_variable(z, context_name),
                           *node.inputs)]
optdb.register('local_gpua_alloc_empty_to_zeros',
               theano.tensor.opt.in2out(local_gpua_alloc_empty_to_zeros),
               # After move to gpu and merge2, before inplace.
               49.3,
               'alloc_empty_to_zeros',)


@register_opt()
@local_optimizer([GpuContiguous])
def local_gpu_contiguous_gpu_contiguous(node):
    """
    gpu_contiguous(gpu_contiguous(x)) -> gpu_contiguous(x)

    """
    if isinstance(node.op, GpuContiguous):
        inp = node.inputs[0]
        if inp.owner and isinstance(inp.owner.op, GpuContiguous):
            return [inp]


@register_opt('fast_compile')
@op_lifter([tensor.Reshape])
def local_gpureshape(node, context_name):
    op = node.op
    name = op.name
    if name:
        name = 'Gpu' + name
    res = GpuReshape(op.ndim, op.name)
    return res


@register_opt('fast_compile')
@op_lifter([tensor.Rebroadcast])
def local_gpu_rebroadcast(node, context_name):
    return node.op(as_gpuarray_variable(node.inputs[0], context_name))


@register_opt('fast_compile')
@op_lifter([tensor.Flatten])
def local_gpuflatten(node, context_name):
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
def local_gpu_elemwise(node, context_name):
    op = node.op
    scal_op = op.scalar_op
    name = op.name
    if name:
        name = 'Gpu' + name
    if len(node.outputs) > 1:
        return
    res = GpuElemwise(scal_op, name=name,
                      inplace_pattern=copy.copy(op.inplace_pattern),
                      nfunc_spec=op.nfunc_spec)

    # If the elemwise operation is a pow, casts might be required on the
    # inputs and or outputs because only the (float, float)->float and
    # (double, double)->double cases are implemented at the moment.
    if isinstance(op.scalar_op, Pow):

        # Only transfer the computation on the gpu if the output dtype is
        # floating point. Else, give up on the transfer to the gpu.
        out_dtype = node.outputs[0].dtype
        if out_dtype not in ['float16', 'float32', 'float64']:
            return

        # Transfer the inputs on the GPU and cast them to the right dtype.
        new_inputs = []
        for inp in node.inputs:
            if inp.dtype != out_dtype:
                gpu_cast_op = GpuElemwise(Cast(Scalar(out_dtype)))
                new_inputs.append(gpu_cast_op(as_gpuarray_variable(inp, context_name)))
            else:
                new_inputs.append(as_gpuarray_variable(inp, context_name))

        # Perform the exponent on the gpu and transfer the output back to the
        # cpu.
        gpu_output = res(*new_inputs)
        cpu_output = host_from_gpu(gpu_output)
        return [cpu_output]
    else:
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


@register_opt('fast_compile')
@op_lifter([tensor.DimShuffle])
def local_gpua_dimshuffle(node, context_name):
    return GpuDimShuffle(node.op.input_broadcastable,
                         node.op.new_order)


@register_opt('fast_compile')
@op_lifter([tensor.SpecifyShape])
def local_gpua_specifyShape(node, context_name):
    if isinstance(node.inputs[0].type, GpuArrayType):
        return
    inp = [GpuFromHost(context_name)(node.inputs[0])] + node.inputs[1:]
    return tensor.specify_shape(*inp)


@register_opt('fast_compile')
@op_lifter([theano.compile.ops.Shape])
def local_gpua_shape(node, context_name):
    # op_lifter will call this opt too frequently as the output is
    # always on the CPU.
    if isinstance(node.inputs[0].type, GpuArrayType):
        return
    return [GpuFromHost(context_name)(node.inputs[0]).shape]


def gpu_print_wrapper(op, cnda):
    op.old_op.global_fn(op.old_op, numpy.asarray(cnda))


@register_opt('fast_compile')
@op_lifter([tensor.printing.Print])
def local_gpu_print_op(node, context_name):
    x, = node.inputs
    gpu_x = as_gpuarray_variable(x, context_name=context_name)
    new_op = node.op.__class__(global_fn=gpu_print_wrapper)
    new_op.old_op = node.op
    return new_op(gpu_x)


@register_opt('fast_compile')
@local_optimizer([PdbBreakpoint])
def local_gpu_pdbbreakpoint_op(node):
    if isinstance(node.op, PdbBreakpoint):

        old_inputs = node.inputs
        old_outputs = node.outputs

        new_inputs = node.inputs[:1]
        input_transfered = []

        # Go through the monitored variables, only transfering on GPU those
        # for which the input comes from the GPU or the output will be
        # transfered on the GPU.
        nb_monitored_vars = len(node.outputs)
        for i in range(nb_monitored_vars):

            inp = old_inputs[i + 1]
            out = old_outputs[i]

            input_is_from_gpu = (inp.owner and
                                 isinstance(inp.owner.op, HostFromGpu))
            output_goes_to_gpu = False
            for c in out.clients:
                if c == 'output':
                    continue
                if isinstance(c[0].op, GpuFromHost):
                    output_goes_to_gpu = True
                    context_name = c[0].op.context_name
                    break

            if input_is_from_gpu:
                # The op should be applied on the GPU version of the input
                new_inputs.append(inp.owner.inputs[0])
                input_transfered.append(True)

            elif output_goes_to_gpu:
                # The input should be transfered to the gpu
                new_inputs.append(GpuFromHost(context_name)(inp))
                input_transfered.append(True)

            else:
                # No transfer is required.
                new_inputs.append(inp)
                input_transfered.append(False)

        # Only continue the optimization if at least one input has been
        # transfered to the gpu
        if not any(input_transfered):
            return False

        # Apply the op on the new inputs
        new_op_outputs = node.op(*new_inputs, return_list=True)

        # Propagate the transfer to the gpu through the outputs that require
        # it
        new_outputs = []
        for i in range(len(new_op_outputs)):
            if input_transfered[i]:
                new_outputs.append(host_from_gpu(new_op_outputs[i]))
            else:
                new_outputs.append(new_op_outputs[i])

        return new_outputs

    return False


@register_opt('fast_compile')
@op_lifter([IfElse])
def local_gpua_lazy_ifelse(node, context_name):
    if node.op.gpu:
        return
    c = node.inputs[0]
    inps = []
    for v in node.inputs[1:]:
        if isinstance(v.type, (tensor.TensorType, GpuArrayType)):
            inps.append(as_gpuarray_variable(v, context_name))
        else:
            inps.append(v)
    return IfElse(node.op.n_outs, gpu=True)(c, *inps, return_list=True)


@register_opt('fast_compile')
@op_lifter([tensor.Join])
def local_gpua_join(node, context_name):
    return gpu_join


@register_opt('fast_compile')
@local_optimizer([GpuJoin])
def local_gpuajoin_1(node):
    # join of a single element
    if (isinstance(node.op, GpuJoin) and
            len(node.inputs) == 2):
        return [node.inputs[1]]


@register_opt('fast_compile')
@op_lifter([tensor.Split])
def local_gpua_split(node, context_name):
    return GpuSplit(node.op.len_splits)


@register_opt('fast_compile')
@op_lifter([tensor.Subtensor])
def local_gpua_subtensor(node, context_name):
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
                        for n, _ in node.outputs[0].clients]):
                    return
                else:
                    return [host_from_gpu(gpu_x.owner.op(node.outputs[0]))]

    return GpuSubtensor(node.op.idx_list)


@register_opt('fast_compile')
@op_lifter([tensor.IncSubtensor])
def local_gpua_incsubtensor(node, context_name):
    op = GpuIncSubtensor(node.op.idx_list, node.op.inplace,
                         node.op.set_instead_of_inc,
                         node.op.destroyhandler_tolerate_aliased)
    ret = op(*node.inputs)
    val = getattr(node.outputs[0].tag, 'nan_guard_mode_check', True)
    ret.tag.nan_guard_mode_check = val
    return ret


@register_opt('fast_compile')
@op_lifter([tensor.AdvancedSubtensor1])
def local_gpua_advanced_subtensor(node, context_name):
    return GpuAdvancedSubtensor1()


@register_opt('fast_compile')
@op_lifter([tensor.AdvancedIncSubtensor1])
def local_gpua_advanced_incsubtensor(node, context_name):

    # This is disabled on non-cuda contexts
    if get_context(context_name).kind != 'cuda':
        return None

    x, y, ilist = node.inputs

    # Gpu Ops needs both inputs to have the same dtype
    if (x.type.dtype != y.type.dtype):
        dtype = scalar.upcast(x.type.dtype, y.type.dtype)
        if x.type.dtype != dtype:
            x = tensor.cast(x, dtype)
        if y.type.dtype != dtype:
            y = tensor.cast(y, dtype)

    set_instead_of_inc = node.op.set_instead_of_inc
    active_device_no = theano.sandbox.cuda.active_device_number()
    device_properties = theano.sandbox.cuda.device_properties

    compute_capability = device_properties(active_device_no)['major']

    if (compute_capability < 2 or x.ndim != 2 or y.ndim != 2):
        return GpuAdvancedIncSubtensor1(
            set_instead_of_inc=set_instead_of_inc)
    else:
        return GpuAdvancedIncSubtensor1_dev20(
            set_instead_of_inc=set_instead_of_inc)


@register_opt('fast_compile')
@op_lifter([tensor.CAReduce, tensor.Sum, tensor.elemwise.Prod])
def local_gpua_careduce(node, context_name):
    if isinstance(node.op.scalar_op, (scalar.Add, scalar.Mul,
                                      scalar.Maximum, scalar.Minimum)):
        ctx = get_context(context_name)
        if ctx.kind == 'opencl':
            op = GpuCAReduceCPY
            if node.op.scalar_op not in [scalar.add, scalar.mul]:
                # We don't support yet all reduction with cpy code.
                return
        elif ctx.kind == 'cuda':
            op = GpuCAReduceCuda
        else:
            return False
        x, = node.inputs

        greduce = op(
            node.op.scalar_op, axis=node.op.axis,
            dtype=getattr(node.op, 'dtype', None),
            acc_dtype=getattr(node.op, 'acc_dtype', None))
        gvar = greduce(x)
        # We need to have the make node called, otherwise the mask can
        # be None
        if (op is GpuCAReduceCPY or
                gvar.owner.op.supports_c_code([GpuFromHost(context_name)(x)])):
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
            greduce = op(
                node.op.scalar_op,
                axis=new_axis, reduce_mask=new_mask,
                dtype=getattr(node.op, 'dtype', None),
                acc_dtype=getattr(node.op, 'acc_dtype', None))

            reshaped_x = x.reshape(tensor.stack(new_in_shp))
            gpu_reshaped_x = GpuFromHost(context_name)(reshaped_x)
            gvar = greduce(gpu_reshaped_x)
            # We need to have the make node called, otherwise the mask can
            # be None
            reshaped_gpu_inputs = [gpu_reshaped_x]
            if greduce.supports_c_code(reshaped_gpu_inputs):
                reduce_reshaped_x = host_from_gpu(
                    greduce(gpu_reshaped_x))

                if reduce_reshaped_x.ndim != node.outputs[0].ndim:
                    unreshaped_reduce = reduce_reshaped_x.reshape(
                        tensor.stack(shape_of[node.outputs[0]]))
                else:
                    unreshaped_reduce = reduce_reshaped_x
                return [unreshaped_reduce]


@register_opt('fast_compile')
@op_lifter([tensor.blas.Gemv, tensor.blas_c.CGemv])
def local_gpua_gemv(node, context_name):
    return GpuGemv(inplace=node.op.inplace)


@register_opt('fast_compile')
@op_lifter([tensor.blas.Gemm])
def local_gpua_gemm(node, context_name):
    return GpuGemm(inplace=node.op.inplace)


@register_opt('fast_compile')
@op_lifter([tensor.basic.Dot])
def local_gpua_hgemm(node, context_name):
    from theano.sandbox.cuda import nvcc_compiler
    if nvcc_compiler.nvcc_version < '7.5':
        _logger.warning("Not performing dot of float16 on the GPU since "
                        "cuda 7.5 is not available. Updating could speed up "
                        "your code.")
        return
    A = node.inputs[0]
    B = node.inputs[1]
    if (A.ndim == 2 and B.ndim == 2 and
            A.dtype == 'float16' and B.dtype == 'float16'):
        fgraph = node.inputs[0].fgraph
        C = GpuAllocEmpty(dtype='float16', context_name=context_name)(
            shape_i(A, 0, fgraph),
            shape_i(B, 1, fgraph))
        return gpugemm_no_inplace(C, 1.0, A, B, 0.0)


@register_opt()
@alpha_merge(GpuGemm, alpha_in=1, beta_in=4)
def local_gpuagemm_alpha_merge(node, *inputs):
    return [gpugemm_no_inplace(*inputs)]


@register_opt()
@output_merge(GpuGemm, alpha_in=1, beta_in=4, out_in=0)
def local_gpuagemm_output_merge(node, *inputs):
    return [gpugemm_no_inplace(*inputs)]


@register_opt('fast_compile')
@op_lifter([tensor.blas.Ger, tensor.blas_c.CGer, tensor.blas_scipy.ScipyGer])
def local_gpua_ger(node, context_name):
    return GpuGer(inplace=node.op.destructive)


@register_opt('fast_compile')
@op_lifter([tensor.blas.Dot22])
def local_gpua_dot22(node, context_name):
    return gpu_dot22


@register_opt('fast_compile')
@op_lifter([tensor.blas.Dot22Scalar])
def local_gpua_dot22scalar(node, context_name):
    x, y, a = node.inputs
    x = as_gpuarray_variable(x, context_name)
    y = as_gpuarray_variable(y, context_name)
    z = GpuAllocEmpty(x.dtype, context_name)(x.shape[0], y.shape[1])
    return [GpuGemm(inplace=False)(z, a, x, y, 0)]


@register_opt('fast_compile')
@op_lifter([tensor.basic.Eye])
def local_gpua_eye(node, context_name):
    return GpuEye(dtype=node.op.dtype, context_name=context_name)


@register_opt('fast_compile')
@op_lifter([tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias], cuda_only=True)
def local_gpua_crossentropysoftmaxargmax1hotwithbias(node, context_name):
    return GpuCrossentropySoftmaxArgmax1HotWithBias()


@register_opt('fast_compile')
@op_lifter([tensor.nnet.CrossentropySoftmax1HotWithBiasDx], cuda_only=True)
def local_gpua_crossentropysoftmax1hotwithbiasdx(node, context_name):
    return GpuCrossentropySoftmax1HotWithBiasDx()


@register_opt('fast_compile')
@op_lifter([tensor.nnet.Softmax], cuda_only=True)
def local_gpua_softmax(node, context_name):
    return GpuSoftmax()


@register_opt('fast_compile')
@op_lifter([tensor.nnet.SoftmaxWithBias], cuda_only=True)
def local_gpua_softmaxwithbias(node, context_name):
    return GpuSoftmaxWithBias()


@register_opt('fast_compile')
@op_lifter([theano.tensor.opt.Assert])
def local_assert(node, context_name):
    return [host_from_gpu(node.op(as_gpuarray_variable(node.inputs[0],
                                                       context_name),
                                  *node.inputs[1:]))]


@register_opt('fast_compile')
@op_lifter([ConvOp])
def local_error_convop(node, context_name):
    assert False, """
ConvOp does not work with the gpuarray backend.

Use the new convolution interface to have GPU convolution working:
theano.tensor.nnet.conv2d()
"""


# This deals with any abstract convs that have a transfer somewhere
@register_opt('fast_compile')
@op_lifter([AbstractConv2d,
            AbstractConv2d_gradWeights,
            AbstractConv2d_gradInputs])
def local_lift_abstractconv2d(node, context_name):
    if isinstance(node.outputs[0].type, GpuArrayType):
        # Don't handle this node here, it's already on the GPU.
        return
    inps = list(node.inputs)
    inps[0] = as_gpuarray_variable(node.inputs[0],
                                   context_name=context_name)
    inps[1] = as_gpuarray_variable(node.inputs[1],
                                   context_name=context_name)
    return [node.op(*inps)]

# Register this here so that it goes after the abstract lifting
register_opt()(conv_groupopt)


@register_opt("low_memory")
@local_optimizer([GpuCAReduceCuda])
def local_gpu_elemwise_careduce(node):
    """
    Merge some GpuCAReduceCuda and GPUElemwise.

    """
    if (isinstance(node.op, GpuCAReduceCuda) and
            node.op.pre_scalar_op is None and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, GpuElemwise) and
            # The Op support all scalar with 1 inputs.  We don't
            # automatically add more case, as some like trigonometic
            # operation with some reduction pattern will probably results
            # in slow down.
            isinstance(node.inputs[0].owner.op.scalar_op, scalar.basic.Sqr)):
        op = node.op
        inp = node.inputs[0].owner.inputs[0]
        return [GpuCAReduceCuda(scalar_op=op.scalar_op,
                                axis=op.axis,
                                reduce_mask=op.reduce_mask,
                                pre_scalar_op=scalar.basic.sqr)(inp)]


def tensor_to_gpu(x, context_name):
    if isinstance(x.type, tensor.TensorType):
        y = GpuArrayType(broadcastable=x.type.broadcastable,
                         context_name=context_name,
                         dtype=x.type.dtype)()
        if x.name:
            y.name = x.name + '[Gpua]'
        return y
    else:
        return x


def gpu_safe_new(x, tag=''):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name (old name + tag). This function is used
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
    new variables of the same type, and returns those (in the same
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
def local_scan_to_gpua(node, context_name):
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
    nw_ins += [safe_to_gpu(x, context_name) for x in node.inputs[1:e]]
    b = e
    e = e + node.op.n_nit_sot
    nw_ins += node.inputs[b:e]
    nw_ins += [safe_to_gpu(x, context_name) for x in node.inputs[e:]]
    scan_ins = [tensor_to_gpu(x, context_name) for x in node.op.inputs]

    # The inner output corresponding to the looping condition should not be
    # moved to the gpu
    if node.op.info['as_while']:
        scan_outs = [safe_to_gpu(x, context_name) for x in node.op.outputs[:-1]]
        scan_outs += [node.op.outputs[-1]]
    else:
        scan_outs = [safe_to_gpu(x, context_name) for x in node.op.outputs]
    scan_outs = scan_utils.clone(
        scan_outs,
        replace=list(zip(node.op.inputs,
                         (safe_to_cpu(x) for x in scan_ins))))

    # We need to construct the hash here, because scan
    # __init__ does not know about the gpu and can not
    # handle graphs with inputs being on the gpu
    tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins, scan_outs)
    local_fgraph = gof.FunctionGraph(tmp_in, tmp_out, clone=True)
    _cmodule_key = gof.CLinker().cmodule_key_(local_fgraph, [])
    info['gpu_hash'] = hash(_cmodule_key)

    def typebuild(dtype, broadcastable, context_name=context_name):
        return GpuArrayType(dtype=dtype, broadcastable=broadcastable,
                            context_name=context_name)

    nw_op = scan_op.Scan(scan_ins, scan_outs, info,
                         typeConstructor=typebuild).make_node(*nw_ins)
    return nw_op.outputs


def _scan_type_infer(node):
    context_name = infer_context_name(*node.inputs)

    def typebuild(dtype, broadcastable, context_name=context_name):
        return GpuArrayType(dtype=dtype, broadcastable=broadcastable,
                            context_name=context_name)
    return typebuild

# Do not register in fast_run or fast_compile.
# It will be added to fast_run if the GPU is enabled.
optdb.register('gpua_scanOp_make_inplace',
               scan_opt.ScanInplaceOptimizer(typeInfer=_scan_type_infer,
                                             gpua_flag=True),
               75,
               'gpuarray',
               'inplace',
               'scan')
