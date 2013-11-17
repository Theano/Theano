import copy
import theano
import numpy
from theano import tensor, scalar
from theano.compile import optdb
from theano.gof import (local_optimizer, EquilibriumDB, SequenceDB, ProxyDB,
                        Optimizer, toolbox, DestroyHandler,
                        InconsistencyError, EquilibriumOptimizer)

from theano.gof.python25 import all, any
from theano.sandbox.gpuarray.type import GpuArrayType

from theano.sandbox.gpuarray.basic_ops import (host_from_gpu,
                                               gpu_from_host,
                                               gpu_alloc, GpuReshape,
                                               GpuEye)
from theano.sandbox.gpuarray.elemwise import (GpuElemwise, _is_scalar,
                                              GpuDimShuffle, GpuCAReduce)
from theano.sandbox.gpuarray.subtensor import GpuSubtensor
from theano.sandbox.gpuarray.blas import GpuGemv, GpuGemm

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
            if type(node.op) is OP:
                # This does not support nodes that have more than one output.
                assert len(node.outputs) == 1
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
                            return [host_from_gpu(new_op(*node.inputs))]
                        else:  # suppose it is a variable on the GPU
                            return [host_from_gpu(new_op)]
            return False
        local_opt.__name__ = maker.__name__
        return local_optimizer([OP])(local_opt)
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


@local_optimizer([])
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
@op_lifter(tensor.Alloc)
def local_gpualloc(node):
    return gpu_alloc


@register_opt()
@op_lifter(tensor.Reshape)
def local_gpureshape(node):
    op = node.op
    name = op.name
    if name:
        name = 'Gpu' + name
    res = GpuReshape(op.ndim, op.name)
    return res


@register_opt()
@op_lifter(tensor.Flatten)
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
@op_lifter(tensor.Elemwise)
def local_gpu_elemwise(node):
    op = node.op
    name = op.name
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
@op_lifter(tensor.DimShuffle)
def local_gpua_dimshuffle(node):
    return GpuDimShuffle(node.op.input_broadcastable,
                         node.op.new_order)


@register_opt()
@op_lifter(tensor.SpecifyShape)
def local_gpua_specifyShape(node):
    return tensor.specify_shape


@register_opt()
@op_lifter(tensor.Subtensor)
def local_gpua_subtensor(node):
    return GpuSubtensor(node.op.idx_list)


@register_opt()
@op_lifter(tensor.CAReduce)
def local_gpua_careduce(node):
    if (isinstance(node.op.scalar_op, scalar.basic.Add) or
        isinstance(node.op.scalar_op, scalar.basic.Mul)):
        return GpuCAReduce(node.op.scalar_op, axis=node.op.axis,
                           dtype=getattr(node.op, 'dtype', None),
                           acc_dtype=getattr(node.op, 'acc_dtype', None))

@register_opt()
@op_lifter(tensor.blas.Gemv)
def local_gpua_gemv(node):
    return GpuGemv(inplace=node.op.inplace)

@register_opt()
@op_lifter(tensor.blas_c.CGemv)
def local_gpua_gemv2(node):
    return GpuGemv(inplace=node.op.inplace)

@register_opt()
@op_lifter(tensor.blas.Gemm)
def local_gpua_gemm(node):
    return GpuGemm(inplace=node.op.inplace)


@register_opt()
@op_lifter(tensor.basic.Eye)
def local_gpua_eye(node):
    return GpuEye(dtype=node.op.dtype)
