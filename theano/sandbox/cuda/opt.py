from __future__ import absolute_import, print_function, division

import copy
import logging
import pdb
import sys
import time
import warnings

import numpy
from six.moves import reduce, xrange

from . import dnn
import theano
from theano import scalar as scal
from theano import config, tensor, gof
from theano.compile.ops import shape_i
import theano.ifelse
import theano.tensor.signal.pool
import theano.tensor.nnet
import theano.tensor.nnet.neighbours
# Convolution
from theano.tensor.nnet import conv
from theano.tensor.nnet.ConvGrad3D import ConvGrad3D
from theano.tensor.nnet.ConvTransp3D import ConvTransp3D
# Pooling
import theano.tensor.signal.pool as pool
from theano.compile import optdb
from theano.gof import (local_optimizer, EquilibriumDB, ProxyDB,
                        Optimizer, TopoOptimizer, toolbox)
from theano.gof.opt import LocalMetaOptimizer
from theano.sandbox.cuda.basic_ops import gpu_join, GpuJoin
from theano.sandbox.cuda import as_cuda_ndarray_variable
from theano.sandbox.cuda.basic_ops import (
    gpu_eye, gpu_contiguous,
    gpu_from_host, host_from_gpu, GpuFromHost, HostFromGpu,
    GpuContiguous,
    GpuElemwise, GpuDimShuffle, GpuReshape, GpuCAReduce,
    gpu_flatten,
    GpuSubtensor, GpuAdvancedSubtensor1,
    GpuAdvancedIncSubtensor1, GpuAdvancedIncSubtensor1_dev20,
    GpuIncSubtensor, gpu_alloc, GpuAlloc, gpu_shape, GpuSplit, GpuAllocEmpty)
from theano.sandbox.cuda.opt_util import pad_dims, unpad_dims

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.blas import (
    gpu_dot22, gpu_dot22scalar, gpu_gemm_inplace, gpu_gemm_no_inplace, GpuConv,
    GpuBatchedDot, GpuCorrMM, GpuCorrMM_gradInputs, GpuCorrMM_gradWeights,
    GpuCorr3dMM, GpuCorr3dMM_gradInputs, GpuCorr3dMM_gradWeights)

from theano.sandbox.cuda.blas import gpu_gemv_inplace

from theano.sandbox.cuda.blas import gpu_gemv_no_inplace
from theano.sandbox.cuda.blas import gpu_ger_inplace
from theano.sandbox.cuda.blas import gpu_ger_no_inplace
from theano.sandbox.cuda.blas import (
    GpuDownsampleFactorMax, GpuDownsampleFactorMaxGrad,
    GpuDownsampleFactorMaxGradGrad)

from theano.tensor.nnet.blocksparse import SparseBlockGemv, SparseBlockOuter
from theano.sandbox.cuda.blocksparse import (
    GpuSparseBlockGemv,
    GpuSparseBlockOuter,
    gpu_sparse_block_gemv_inplace,
    gpu_sparse_block_outer_inplace)


from theano.sandbox.cuda.nnet import (
    GpuCrossentropySoftmaxArgmax1HotWithBias,
    GpuCrossentropySoftmax1HotWithBiasDx,
    GpuSoftmax, GpuSoftmaxWithBias)

from theano.sandbox.cuda.elemwise import SupportCodeError
from theano.scalar.basic_scipy import Erfinv
from theano.scalar.basic_scipy import Erfcx
from theano.sandbox.cuda.elemwise import erfinv_gpu
from theano.sandbox.cuda.elemwise import erfcx_gpu
from theano.sandbox.cuda.var import CudaNdarrayConstant
from theano.sandbox.cuda import gpu_optimizer, register_opt, gpu_seqopt, GpuOp
import theano.sandbox.cuda.extra_ops
from theano.scan_module import scan_utils, scan_op, scan_opt
from theano.tensor.blas import _is_real_vector, _is_real_matrix

from theano.tensor import nlinalg

from theano.tensor.nnet.Conv3D import Conv3D
from theano.tests.breakpoint import PdbBreakpoint

from theano.tensor.nnet.abstract_conv import (BaseAbstractConv,
                                              AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs,
                                              AbstractConv3d,
                                              AbstractConv3d_gradWeights,
                                              AbstractConv3d_gradInputs)
from theano.tensor.opt import register_specialize_device


try:
    # We need to be able to import this file even if cuda isn't avail.
    from theano.sandbox.cuda import device_properties
except ImportError:
    pass


_logger = logging.getLogger('theano.sandbox.cuda.opt')

# optdb.print_summary()  # shows what is currently registered

gpu_cut_copies = EquilibriumDB()
gpu_seqopt.register('gpu_local_optimizations', gpu_optimizer, 1,
                    'fast_run', 'fast_compile', 'gpu')
gpu_seqopt.register('gpu_cut_transfers', gpu_cut_copies, 2,
                    'fast_run', 'fast_compile', 'gpu')
# DO NOT PUT fast_run or fast_compile in gpu_opt! This will ALWAYS
# enable the GPU!
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

# Register merge_optimizer as a global opt
gpu_optimizer.register('gpu_merge', theano.gof.opt.MergeOptimizer(),
                       'fast_run', 'fast_compile', final_opt=True)


# register local_track_shape_i at this level too
# to make multi-level lift of shape work.
register_opt()(theano.tensor.opt.local_track_shape_i)
register_opt(final_opt=True, name='gpu_constant_folding')(
    tensor.opt.constant_folding)
register_opt()(theano.tensor.opt.local_subtensor_make_vector)

# Register local_remove_all_assert as a global opt
gpu_optimizer.register('local_remove_all_assert',
                       theano.tensor.opt.local_remove_all_assert,
                       'unsafe')

# Register local_reshape_chain
register_opt(name='local_gpu_reshape_chain')(
    theano.tensor.opt.local_reshape_chain(GpuReshape))

# This is a partial list of CPU ops that can be in some circonstance
# moved to the GPU. This list is used by an optimization.
# Hopefully, we can keep this list up to date.
cpu_ops_moved_to_gpu = [
    tensor.blas.Dot22, tensor.blas.Dot22Scalar, tensor.blas.Gemm,
    tensor.blas.Gemv, tensor.blas.Ger, tensor.nnet.conv.ConvOp,
    tensor.signal.pool.Pool,
    tensor.signal.pool.MaxPoolGrad,
    tensor.signal.pool.AveragePoolGrad,
    theano.tensor.nnet.neighbours.Images2Neibs,
    tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias,
    tensor.nnet.CrossentropySoftmax1HotWithBiasDx,
    tensor.nnet.Softmax, tensor.nnet.SoftmaxWithBias,
    tensor.Elemwise, tensor.DimShuffle, tensor.CAReduce,
    tensor.elemwise.All, tensor.elemwise.Any,
    tensor.elemwise.CAReduceDtype, tensor.elemwise.Sum,
    tensor.elemwise.Prod, tensor.elemwise.ProdWithoutZeros,
    tensor.Reshape, tensor.flatten, tensor.Subtensor,
    tensor.AdvancedSubtensor1, tensor.AdvancedIncSubtensor1,
    tensor.IncSubtensor, tensor.Shape, tensor.Join,
    tensor.Alloc, tensor.Eye, tensor.blas.BatchedDot]


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
                continue

            # This happen frequently as we do 2 pass of the gpu optimizations
            if (len(input.clients) == 1 and
                (input.clients[0][0] == 'output' or
                 isinstance(input.clients[0][0].op, GpuFromHost))):
                continue

            try:
                new_input = host_from_gpu(gpu_from_host(input))

                if new_input.type == input.type:
                    fgraph.replace_validate(input, new_input,
                                            "InputToGpuOptimizer")
            except TypeError:
                # as we currently only support float32, this can fail.
                # Using try except make that we won't need
                pass

# we register it before all other gpu optimizer to be sure that the input
# are on the gpu.
gpu_seqopt.register('InputToGpuOptimizer', InputToGpuOptimizer(),
                    0,
                    'fast_run',
                    'fast_compile',
                    'merge')  # TODO: how to make it mandatory for gpu_seqopt?


@local_optimizer([GpuFromHost, HostFromGpu])
def local_cut_gpu_host_gpu(node):
    if tensor.opt.opt.check_chain(node, gpu_from_host, host_from_gpu):
        return [node.inputs[0].owner.inputs[0]]
    if tensor.opt.opt.check_chain(node, host_from_gpu, gpu_from_host):
        return [node.inputs[0].owner.inputs[0]]
    return False
gpu_cut_copies.register('cut_gpu_host_transfers', local_cut_gpu_host_gpu,
                        'fast_run', 'fast_compile', 'gpu')
gpu_cut_copies.register('cut_gpu_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_run', 'fast_compile', 'gpu')
# register it into canonicalize to allow other optimization to work without
# botering with this useless pattern.
optdb['canonicalize'].register('local_cut_gpu_host_gpu',
                               local_cut_gpu_host_gpu,
                               'fast_run', 'fast_compile', 'gpu')

# 'float64', 'complex128' and 'complex64' are not supported in elemwise
# on the gpu.
elemwise_cuda_dtype_supported = ['float32', 'bool',
                                 'uint8', 'int8',
                                 'uint16', 'int16',
                                 'uint32', 'int32',
                                 'uint64', 'int64']


def dtype_in_elemwise_supported(op):
    """
    Return True of the Elemwise op is supported on the gpu.
    Return False otherwise.

    Notes
    -----
    We need to check inside the Composite op.

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
    """
    Elemwise(..., host_from_gpu, ...)
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
                elif isinstance(node.op.scalar_op, Erfcx):
                    new_op = GpuElemwise(erfcx_gpu)
                else:
                    try:
                        new_op = GpuElemwise(node.op.scalar_op)
                    except SupportCodeError:
                        # This happens when scalar_op requires support code
                        return False

                #   first establish that float32 can store all inputs
                upcastable = set(['float32', 'bool', 'int8', 'int16',
                                  'uint8', 'uint16'])
                # case 1 - all inputs are already float32
                if all([i.type.dtype == 'float32' for i in node.inputs]):
                    # TODO: change this when fusion makes Elemwise with
                    # multiple outputs
                    gpu_elemwise = new_op(*(as_cuda_ndarray_variable(i)
                                            for i in node.inputs),
                                          return_list=True)
                # case 2 - it is still ok if some inputs were upcast to float32
                elif all([i.type.dtype in upcastable
                          for i in node.inputs]):
                    # second - establish that a new node with upcasted inputs
                    # has the same outputs types as the original node
                    upcasted = node.op.make_node(*[tensor.cast(i, 'float32')
                                                   for i in node.inputs])
                    if [o.type for o in upcasted.outputs] ==\
                       [o.type for o in node.outputs]:

                        new_inputs = [as_cuda_ndarray_variable(tensor.cast(i, 'float32'))
                                      for i in node.inputs]
                        gpu_elemwise = new_op(*new_inputs, return_list=True)
                    else:
                        return False
                else:
                    return False

                gpu_elemwise = split_huge_add_or_mul(gpu_elemwise[0].owner)
                if not gpu_elemwise:
                    return False
                if (max_inputs_to_GpuElemwise(node) <
                        len(gpu_elemwise.inputs)):
                    return False
                return [host_from_gpu(out) for out in gpu_elemwise.outputs]


@register_opt()
@local_optimizer([GpuFromHost])
def local_gpu_elemwise_1(node):
    """
    gpu_from_host(Elemwise)) -> GpuElemwise(gpu_from_host(...))

    """
    if isinstance(node.op, GpuFromHost):
        host_i, = node.inputs
        if (host_i.owner and
                isinstance(host_i.owner.op, tensor.Elemwise) and
                len(host_i.owner.outputs) == 1 and
                len(host_i.clients) == 1 and
                dtype_in_elemwise_supported(node.op)):

            elemwise_node = host_i.owner
            # Don't set any inplace pattern.
            # gpu_inplace_elemwise_optimizer will do it later

            if isinstance(elemwise_node.op.scalar_op, Erfinv):
                new_op = GpuElemwise(erfinv_gpu)
            elif isinstance(elemwise_node.op.scalar_op, Erfcx):
                new_op = GpuElemwise(erfcx_gpu)
            else:
                try:
                    new_op = GpuElemwise(elemwise_node.op.scalar_op)
                except SupportCodeError:
                    # This happens when scalar_op requires support code
                    return False

            if all([i.dtype == 'float32' for i in elemwise_node.inputs]):
                gpu_elemwise = new_op(*[as_cuda_ndarray_variable(i)
                                        for i in elemwise_node.inputs])
                gpu_elemwise = split_huge_add_or_mul(gpu_elemwise.owner)
                if not gpu_elemwise:
                    return False
                return [gpu_elemwise.outputs[0]]
    return False


@register_opt()
@local_optimizer([tensor.Split])
def local_gpu_split(node):
    if isinstance(node.op, tensor.Split):
        input = node.inputs[0]
        outs_clients = reduce(list.__add__,
                              [out.clients for out in node.outputs])
        if (input.owner and isinstance(input.owner.op, HostFromGpu) or
            any(c != 'output' and isinstance(c.op, GpuFromHost) for c, idx
                in outs_clients)):
            new_op = GpuSplit(**node.op._props_dict())
            split_res = new_op(as_cuda_ndarray_variable(input),
                               *node.inputs[1:], return_list=True)
            return [host_from_gpu(o) for o in split_res]
    return False


@register_opt()
@local_optimizer([tensor.DimShuffle, GpuFromHost])
def local_gpu_dimshuffle_0(node):
    """
    dimshuffle(host_from_gpu()) -> host_from_gpu(gpu_dimshuffle)
    gpu_from_host(dimshuffle) -> gpu_dimshuffle(gpu_from_host)

    """
    if isinstance(node.op, tensor.DimShuffle):
        input, = node.inputs
        if input.owner and isinstance(input.owner.op, HostFromGpu):
            # move the add to a GpuAdd
            p_dict = node.op._props_dict()
            p_dict.pop('inplace', None)
            new_op = GpuDimShuffle(**p_dict)
            return [host_from_gpu(new_op(as_cuda_ndarray_variable(input)))]
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.DimShuffle):
            dimshuffle_node = host_input.owner
            p_dict = dimshuffle_node.op._props_dict()
            p_dict.pop('inplace', None)
            new_op = GpuDimShuffle(**p_dict)
            return [new_op(
                as_cuda_ndarray_variable(dimshuffle_node.inputs[0]))]
    return False


@register_opt()
@local_optimizer([tensor.SpecifyShape, GpuFromHost])
def local_gpu_specifyShape_0(node):
    """
    specify_shape(host_from_gpu()) -> host_from_gpu(specify_shape)
    gpu_from_host(specify_shape) -> specify_shape(gpu_from_host)

    """
    if isinstance(node.op, tensor.SpecifyShape):
        input = node.inputs[0]
        if input.owner and isinstance(input.owner.op, HostFromGpu):
            return [host_from_gpu(tensor.specify_shape(
                as_cuda_ndarray_variable(input), *node.inputs[1:]))]
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.SpecifyShape):
            specifyshape_node = host_input.owner
            return [tensor.specify_shape(
                as_cuda_ndarray_variable(specifyshape_node.inputs[0]),
                *specifyshape_node.inputs[1:])]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.basic.Dot])
def local_gpu_dot_to_dot22(node):
    """
    gpu_from_host(dot) -> gpudot(gpu_from_host)
    dot(host_from_gpu) -> host_from_gpu(gpudot)

    This optimization solves the vector-matrix multiplication issue by
    transforming the vector into a matrix, apply gpudot22 and reshaping
    the output.

    A more suitable solution would be to use the right cublas call.

    This is needed in fast_compile.

    """
    # In case the got do input upcast, we much check that we can
    # make it run on the gpu.
    if isinstance(node.op, GpuFromHost):
        if node.outputs[0].type.dtype != 'float32':
            return False
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.basic.Dot):
            x, y = host_input.owner.inputs
            # case one: vector X matrix
            if _is_real_vector(x) and _is_real_matrix(y):
                new_op = GpuDimShuffle((False,), ('x', 0))
                shape_out = y.shape[1].dimshuffle(['x'])
                gpu_x = new_op(as_cuda_ndarray_variable(x))
                gpu_y = as_cuda_ndarray_variable(y)
            # case two: matrix X vector
            elif _is_real_matrix(x) and _is_real_vector(y):
                new_op = GpuDimShuffle((False,), (0, 'x'))
                shape_out = x.shape[0].dimshuffle(['x'])
                gpu_x = as_cuda_ndarray_variable(x)
                gpu_y = new_op(as_cuda_ndarray_variable(y))
            else:
                return False

            return [GpuReshape(1)(gpu_dot22(gpu_x, gpu_y), shape_out)]
    if isinstance(node.op, tensor.basic.Dot):
        if node.outputs[0].type.dtype != 'float32':
            return False
        if any([i.owner and isinstance(i.owner.op, HostFromGpu)
                for i in node.inputs]):
            x, y = node.inputs
            if _is_real_vector(x) and _is_real_matrix(y):
                new_op = GpuDimShuffle((False,), ('x', 0))
                shape_out = y.shape[1].dimshuffle(['x'])
                gpu_x = new_op(as_cuda_ndarray_variable(x))
                gpu_y = as_cuda_ndarray_variable(y)

            elif _is_real_matrix(x) and _is_real_vector(y):
                new_op = GpuDimShuffle((False,), (0, 'x'))
                shape_out = x.shape[0].dimshuffle(['x'])
                gpu_x = as_cuda_ndarray_variable(x)
                gpu_y = new_op(as_cuda_ndarray_variable(y))
            else:
                return False

            return [host_from_gpu(GpuReshape(1)(gpu_dot22(gpu_x, gpu_y),
                                                shape_out))]
    return False


@local_optimizer(None)
def local_assert_no_cpu_op(node):
    if (not isinstance(node.op, GpuOp) and
        all([var.owner and isinstance(var.owner.op, HostFromGpu)
             for var in node.inputs]) and
        any([[c for c in var.clients if isinstance(c[0].op, GpuFromHost)]
             for var in node.outputs])):

            if config.assert_no_cpu_op == "warn":
                _logger.warning(("CPU op %s is detected in the computational"
                                 " graph") % node)
            elif config.assert_no_cpu_op == "raise":
                raise AssertionError("The op %s is on CPU." % node)
            elif config.assert_no_cpu_op == "pdb":
                pdb.set_trace()

    return None

# Register the local_assert_no_cpu_op:
assert_no_cpu_op = theano.tensor.opt.in2out(local_assert_no_cpu_op,
                                            name='assert_no_cpu_op')
# 49.2 is after device specialization & fusion optimizations for last transfers
optdb.register('gpu_assert_no_cpu_op', assert_no_cpu_op, 49.2,
               'assert_no_cpu_op')


@register_opt()
@local_optimizer([theano.ifelse.IfElse, GpuFromHost])
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
            if all([isinstance(o.type, CudaNdarrayType) or
                    getattr(o, 'dtype', None) != 'float32'
                    for o in outs]):
                return

            for i in range(len(outs)):
                if (not isinstance(outs[i].type, CudaNdarrayType) and
                        getattr(outs[i], 'dtype', None) == 'float32'):
                    outs[i] = as_cuda_ndarray_variable(outs[i])
            outs = gpu_ifelse(c, *outs, return_list=True)
            for i in range(len(outs)):
                if isinstance(outs[i].type, CudaNdarrayType):
                    outs[i] = host_from_gpu(outs[i])
            return outs

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
            if all([isinstance(o.type, CudaNdarrayType) or o.dtype != 'float32'
                    for o in outs]):
                return

            for i in range(len(outs)):
                if (not isinstance(outs[i].type, CudaNdarrayType) and
                        outs[i].dtype == 'float32'):
                    outs[i] = as_cuda_ndarray_variable(outs[i])
            outs = gpu_ifelse.make_node(c, *outs).outputs
            return outs

    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.blas.Dot22])
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
            return [gpu_dot22(as_cuda_ndarray_variable(x),
                              as_cuda_ndarray_variable(y))]
    if isinstance(node.op, tensor.blas.Dot22):
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]):
            x, y = node.inputs
            return [host_from_gpu(gpu_dot22(as_cuda_ndarray_variable(x),
                                            as_cuda_ndarray_variable(y)))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.blas.BatchedDot])
def local_gpu_batched_dot(node):
    """
    gpu_from_host(batched_dot) -> gpu_batched_dot(gpu_from_host)

    batched_dot(host_from_gpu) -> host_from_gpu(gpu_batched_dot)

    """
    def gpu_batched_dot(x, y):
        # pad x and y shapes to be third-order tensors
        x_, y_ = x, y
        if x.ndim == 2:
            x_ = x_.dimshuffle(0, "x", 1)
        if y.ndim == 2:
            y_ = y_.dimshuffle(0, 1, "x")
        z = GpuBatchedDot()(as_cuda_ndarray_variable(x_),
                            as_cuda_ndarray_variable(y_))
        # unpad z shape
        if x.ndim == 2:
            z = z.dimshuffle(0, *range(2, z.ndim))
        if y.ndim == 2:
            z = z.dimshuffle(*range(z.ndim - 1))
        return as_cuda_ndarray_variable(z)

    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           tensor.blas.BatchedDot):
            x, y = host_input.owner.inputs
            return [gpu_batched_dot(x, y)]
    if isinstance(node.op, tensor.blas.BatchedDot):
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]):
            x, y = node.inputs
            return [host_from_gpu(gpu_batched_dot(x, y))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.blas.Dot22Scalar])
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
            return [gpu_dot22scalar(as_cuda_ndarray_variable(x),
                                    as_cuda_ndarray_variable(y),
                                    tensor.blas._as_scalar(scalar))]
    if isinstance(node.op, tensor.blas.Dot22Scalar):
        if any([i.owner and isinstance(i.owner.op, HostFromGpu)
                for i in node.inputs]):
            x, y, scalar = node.inputs
            return [host_from_gpu(
                gpu_dot22scalar(as_cuda_ndarray_variable(x),
                                as_cuda_ndarray_variable(y),
                                tensor.blas._as_scalar(scalar)))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.blas_c.CGemv, tensor.blas.Gemv])
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
                    as_cuda_ndarray_variable(z),
                    a,
                    as_cuda_ndarray_variable(x),
                    as_cuda_ndarray_variable(y),
                    b)]
    if isinstance(node.op, gemvs):
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and isinstance(x.owner.op, HostFromGpu))
        y_on_gpu = (y.owner and isinstance(y.owner.op, HostFromGpu))
        z_on_gpu = (z.owner and isinstance(z.owner.op, HostFromGpu))
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(
                gpu_gemv_no_inplace(
                    as_cuda_ndarray_variable(z),
                    a,
                    as_cuda_ndarray_variable(x),
                    as_cuda_ndarray_variable(y),
                    b))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.blas_c.CGer, tensor.blas.Ger,
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
                    as_cuda_ndarray_variable(z),
                    a,
                    as_cuda_ndarray_variable(x),
                    as_cuda_ndarray_variable(y)
                    )]
    if isinstance(node.op, gers):
        z, a, x, y = node.inputs
        x_on_gpu = (x.owner and isinstance(x.owner.op, HostFromGpu))
        y_on_gpu = (y.owner and isinstance(y.owner.op, HostFromGpu))
        z_on_gpu = (z.owner and isinstance(z.owner.op, HostFromGpu))
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(
                gpu_ger_no_inplace(
                    as_cuda_ndarray_variable(z),
                    a,
                    as_cuda_ndarray_variable(x),
                    as_cuda_ndarray_variable(y)))]
    return False


@register_opt()
@local_optimizer([tensor.blas.Gemm, GpuFromHost])
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
            return [gpu_gemm_no_inplace(as_cuda_ndarray_variable(z),
                                        a,
                                        as_cuda_ndarray_variable(x),
                                        as_cuda_ndarray_variable(y),
                                        b)]
    if isinstance(node.op, tensor.blas.Gemm):
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and isinstance(x.owner.op, HostFromGpu))
        y_on_gpu = (y.owner and isinstance(y.owner.op, HostFromGpu))
        z_on_gpu = (z.owner and isinstance(z.owner.op, HostFromGpu))
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(gpu_gemm_no_inplace(
                as_cuda_ndarray_variable(z),
                a,
                as_cuda_ndarray_variable(x),
                as_cuda_ndarray_variable(y),
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
            # Otherwise, is some corner case, we will try to move it
            # to the GPU later and this cause not wanted user warning.
            if x.dtype != 'float32' or node.outputs[0].dtype != "float32":
                return
            replace = False
            if x.owner and isinstance(x.owner.op, HostFromGpu):
                replace = True
            # If this is a useless reduce, remove it as
            # local_cut_useless_reduce.  This is needed as the code
            # below do not support when x.ndim == 0.
            if x.type == node.outputs[0].type:
                return [x]
            elif (all([c != "output" and isinstance(c.op, GpuFromHost)
                      for c, i in node.outputs[0].clients]) and
                  x.owner and x.owner.op.__class__ in
                  cpu_ops_moved_to_gpu):
                # It is not always good to transfer the reduction to
                # the GPU when the clients are on the GPU but not the
                # reduction input. It mean we will transfer the
                # (bigger) input to the GPU instead of the
                # output(smaller) if we stop optimization there. Most
                # of the time, we will also move to the GPU what
                # created the input of the reduction. In that case, we
                # don't introduce a bigger transfer. It is hard to
                # know if after all optimization we will do the bigger
                # transfer or not. I'm guessing an heuristic to find
                # that. I suppose that if the input of the reduction is
                # generated by an op that we can in some cases move to
                # the GPU, that we will move it. If some CPU ops are
                # supported only in some cases on the GPU, this will
                # move to the GPU the reduction when it wasn't a good
                # idea.

                replace = True

            if replace:
                if node.op.axis is None:
                    reduce_mask = [1] * x.type.ndim
                else:
                    reduce_mask = [0] * x.type.ndim
                    for a in node.op.axis:
                        assert reduce_mask[a] == 0
                        reduce_mask[a] = 1
                greduce = GpuCAReduce(reduce_mask, scalar_op)
                out = node.outputs[0]
                if greduce.supports_c_code([as_cuda_ndarray_variable(x)]):
                    rval = host_from_gpu(greduce(as_cuda_ndarray_variable(x)))
                else:
                    # Try to make a simpler pattern based on reshaping
                    # The principle is that if two adjacent dimensions have
                    # the same value in the reduce_mask, then we can reshape
                    # to make them a single dimension, do the reduction, and
                    # then reshape to get them back.

                    new_in_shp = [shape_i(x, 0)]
                    new_mask = [reduce_mask[0]]
                    for i in xrange(1, x.type.ndim):
                        if reduce_mask[i] == reduce_mask[i - 1]:
                            new_in_shp[-1] *= shape_i(x, i)
                        else:
                            new_mask.append(reduce_mask[i])
                            new_in_shp.append(shape_i(x, i))

                    new_greduce = GpuCAReduce(new_mask, scalar_op)
                    new_x = x.reshape(tensor.stack(new_in_shp))
                    gpu_new_x = as_cuda_ndarray_variable(new_x)
                    if not new_greduce.supports_c_code([gpu_new_x]):
                        if not new_mask == [1, 0, 1]:
                            return
                        # The reduced mask [1, 0, 1] is not supported but
                        # [1, 0, 1, 1] is. Therefore, we add a broadcastable
                        # dimension to new_x and change the mask to
                        # [1, 0, 1, 1].
                        new_x = new_x.dimshuffle(0, 1, 2, 'x')
                        gpu_new_x = as_cuda_ndarray_variable(new_x)

                        new_greduce = GpuCAReduce([1, 0, 1, 1], scalar_op)
                        if not new_greduce.supports_c_code([gpu_new_x]):
                            raise Exception('Reduction mask [1, 0, 1, 1] is'
                                            'supposed to be supported.')

                    rval = host_from_gpu(
                        new_greduce(gpu_new_x))

                    # Restore the expected shape of the output
                    if rval.ndim != out.ndim:
                        out_shp = []
                        for i in range(x.ndim):
                            if i not in node.op.axis:
                                out_shp.append(shape_i(x, i))
                        rval = rval.reshape(tensor.stack(out_shp))

                if rval.type == out.type:
                    return [rval]
                else:
                    for b1, b2 in zip(rval.broadcastable,
                                      out.type.broadcastable):
                        if b1 is True:
                            # It can happen that during
                            # optimization we discover that the
                            # input can be broadcasted, but didn't
                            # know that at graph build time.
                            continue
                        if b1 is False and b2 is True:
                            # We should not loose the information
                            # that one dimensions was
                            # broadcastable.
                            print((
                                "WARNING: local_gpu_careduce got type"
                                " wrong",
                                rval.type, out.type,
                                node.inputs[0].type, x.type,
                                node), file=sys.stderr)
                            return None
                    rval = tensor.patternbroadcast(rval,
                                                   out.broadcastable)
                    return [rval]

    return False


@register_opt("low_memory")
@local_optimizer([GpuCAReduce])
def local_gpu_elemwise_careduce(node):
    if (isinstance(node.op, GpuCAReduce) and
        node.op.pre_scalar_op is None and
        node.inputs[0].owner and
        isinstance(node.inputs[0].owner.op, GpuElemwise) and
        # The Op support all scalar with 1 inputs.  We don't
        # automatically add more case, as some like trigonometic
        # operation with some reduction pattern will probably result
        # to slow down.
       isinstance(node.inputs[0].owner.op.scalar_op, scal.basic.Sqr)):

        op = node.op
        inp = node.inputs[0].owner.inputs[0]
        return [GpuCAReduce(op.reduce_mask, op.scalar_op, scal.basic.sqr)(inp)]


@register_opt()
@local_optimizer([GpuFromHost, tensor.Reshape])
def local_gpu_reshape(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           isinstance(host_input.owner.op, tensor.Reshape):
            x, shp = host_input.owner.inputs
            gpu_reshape = GpuReshape(**host_input.owner.op._props_dict())(as_cuda_ndarray_variable(x), shp)
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
            gpu_reshape = GpuReshape(**node.op._props_dict())(gpu_x, shp)
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
@local_optimizer([GpuFromHost, tensor.Flatten])
def local_gpu_flatten(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           isinstance(host_input.owner.op, tensor.Flatten):
            outdim = host_input.owner.op.outdim
            return [gpu_flatten(host_input.owner.inputs[0], outdim)(
                as_cuda_ndarray_variable(host_input.owner.inputs[0]))]
    if isinstance(node.op, tensor.Flatten):
        x, shp = node.inputs
        outdim = node.op.outdim
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            outdim = node.op.outdim
            gpu_x, = x.owner.inputs
            return [host_from_gpu(gpu_flatten(gpu_x, outdim))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.Subtensor])
def local_gpu_subtensor(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           isinstance(host_input.owner.op, tensor.Subtensor):
            subt = host_input.owner.op
            x = host_input.owner.inputs[0]
            if len(x.clients) == 1:
                # It mean, the input of the subtensor is used only by
                # the subtensor. We do not want to move the subtensor
                # to the GPU in that case.
                return
            coords = host_input.owner.inputs[1:]
            return [GpuSubtensor(subt.idx_list)(as_cuda_ndarray_variable(x),
                                                *coords)]
    if isinstance(node.op, tensor.Subtensor):
        x = node.inputs[0]
        if (x.owner and x.dtype == "float32" and
                isinstance(x.owner.op, HostFromGpu)):

            gpu_x = x.owner.inputs[0]
            if (gpu_x.owner and  # And it is a shared var or an input of the graph.
                    not(gpu_x.owner.inputs[0].owner) and
                    isinstance(gpu_x.owner.op, GpuFromHost)):

                if len(x.clients) == 1:
                    if any([n == 'output' or isinstance(n.op, GpuOp)
                            for n, _ in node.outputs[0].clients]):
                        return
                    else:
                        return [host_from_gpu(as_cuda_ndarray_variable(
                            node.outputs[0]))]
                    return

            gpu_x, = x.owner.inputs
            coords = node.inputs[1:]
            return [host_from_gpu(GpuSubtensor(
                **node.op._props_dict())(gpu_x, *coords))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.AdvancedSubtensor1])
def local_gpu_advanced_subtensor1(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           host_input.owner.op.__class__ is tensor.AdvancedSubtensor1:
            x = host_input.owner.inputs[0]
            coords = host_input.owner.inputs[1:]
            return [GpuAdvancedSubtensor1()(as_cuda_ndarray_variable(x),
                                            *coords)]
    if node.op.__class__ is tensor.AdvancedSubtensor1:
        x = node.inputs[0]
        coords = node.inputs[1:]
        if (x.owner and isinstance(x.owner.op, HostFromGpu) and
                x.dtype == "float32"):
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuAdvancedSubtensor1()(gpu_x, *coords))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.AdvancedIncSubtensor1])
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
            props_dict = host_input.owner.op._props_dict()
            if (compute_capability < 2 or y.ndim != 2 or x.ndim != 2):

                gpu_op = GpuAdvancedIncSubtensor1(**props_dict)
            else:
                gpu_op = GpuAdvancedIncSubtensor1_dev20(**props_dict)
            return [gpu_op(as_cuda_ndarray_variable(x),
                           as_cuda_ndarray_variable(y), *coords)]

    # Should not execute for GpuAdvancedIncSubtensor1
    if (node.op.__class__ is tensor.AdvancedIncSubtensor1 and
            node.inputs[0].dtype == "float32" and
            node.inputs[1].dtype == "float32"):
        x, y = node.inputs[0:2]
        coords = node.inputs[2:]
        go_gpu = False
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            go_gpu = True
            gpu_x, = x.owner.inputs
        else:
            gpu_x = as_cuda_ndarray_variable(x)
        if y.owner and isinstance(y.owner.op, HostFromGpu):
            go_gpu = True
            gpu_y, = y.owner.inputs
        else:
            gpu_y = as_cuda_ndarray_variable(y)
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
            if (compute_capability < 2 or y.ndim != 2 or x.ndim != 2):
                gpu_op = GpuAdvancedIncSubtensor1(**node.op._props_dict())
            else:
                gpu_op = GpuAdvancedIncSubtensor1_dev20(**node.op._props_dict())
            return [host_from_gpu(gpu_op(gpu_x, gpu_y, *coords))]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.IncSubtensor])
def local_gpu_incsubtensor(node):
    if isinstance(node.op, GpuFromHost):
        host_output = node.inputs[0]
        if host_output.owner and \
           type(host_output.owner.op) == tensor.IncSubtensor:
            incsubt = host_output.owner.op
            x, y = host_output.owner.inputs[0:2]
            coords = host_output.owner.inputs[2:]
            if x.dtype != "float32":
                return
            if y.dtype != "float32":
                # The IncSubtensor upcast to float32 y, so we do it
                # explicitly to move it to the GPU.
                y = y.astype('float32')
            ret = GpuIncSubtensor(**incsubt._props_dict())(as_cuda_ndarray_variable(x),
                                                           as_cuda_ndarray_variable(y),
                                                           *coords)
            ret.tag.nan_guard_mode_check = getattr(
                host_output.tag, 'nan_guard_mode_check', True)
            return [ret]
    # Incrementing a float32 x results in a float32
    # output even if y is float64, so we can downcast
    # y to put it on GPU
    elif (type(node.op) == tensor.IncSubtensor and
          node.inputs[0].dtype == "float32"):
        x, y = node.inputs[0:2]
        assert isinstance(x.type, tensor.TensorType)
        assert isinstance(y.type, tensor.TensorType)
        coords = node.inputs[2:]
        go_gpu = False
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            go_gpu = True
            gpu_x, = x.owner.inputs
        else:
            gpu_x = as_cuda_ndarray_variable(x)
        if y.owner and isinstance(y.owner.op, HostFromGpu):
            go_gpu = True
            gpu_y, = y.owner.inputs
        else:
            if y.dtype != 'float32':
                y = tensor.cast(y, 'float32')
            gpu_y = as_cuda_ndarray_variable(y)
        if go_gpu:
            ret = GpuIncSubtensor(**node.op._props_dict())(gpu_x, gpu_y, *coords)

            val = getattr(node.outputs[0].tag, 'nan_guard_mode_check', True)
            ret.tag.nan_guard_mode_check = val
            ret = host_from_gpu(ret)
            ret.tag.nan_guard_mode_check = val
            return [ret]
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
    """
    rebroadcast(host_from_gpu(x)) -> host_from_gpu(rebroadcast(x))

    """
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


@register_opt()
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
            output_goes_to_gpu = any(c[0] != "output" and
                                     isinstance(c[0].op, GpuFromHost)
                                     for c in out.clients)

            if input_is_from_gpu:
                # The op should be applied on the GPU version of the input
                new_inputs.append(inp.owner.inputs[0])
                input_transfered.append(True)

            elif output_goes_to_gpu:
                # The input should be transfered to the gpu
                new_inputs.append(as_cuda_ndarray_variable(inp))
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


def cast(x, dtype):
    stype = scal.Scalar(dtype)
    cast_op = theano.tensor.Elemwise(scal.Identity(scal.specific_out(stype)))
    return cast_op(x)


@register_opt()
@local_optimizer([tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias],
                 'local_gpu_crossentorpy_softmax_argmax_1hot_with_bias')
def local_gpu_crossentropy_softmax_argmax_1hot_with_bias(node):
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
                tensor.basic._convert_to_int64)
            while y.owner and y.owner.op in int_cast_ops:
                y = y.owner.inputs[0]
            gpu_nll, gpu_sm, gpu_am = \
                GpuCrossentropySoftmaxArgmax1HotWithBias()(
                    gpu_x,
                    as_cuda_ndarray_variable(b),
                    as_cuda_ndarray_variable(cast(y, 'float32')))
            am_dtype = node.outputs[2].type.dtype
            return [host_from_gpu(gpu_nll),
                    host_from_gpu(gpu_sm),
                    cast(host_from_gpu(gpu_am), am_dtype)]
    return False


@register_opt()
@local_optimizer([tensor.nnet.CrossentropySoftmax1HotWithBiasDx],
                 'local_gpu_crossentorpy_softmax_1hot_with_bias_dx')
def local_gpu_crossentropy_softmax_1hot_with_bias_dx(node):
    if isinstance(node.op, tensor.nnet.CrossentropySoftmax1HotWithBiasDx):
        dnll, sm, yidx = node.inputs
        if sm.owner and isinstance(sm.owner.op, HostFromGpu):
            gpu_sm, = sm.owner.inputs
            gpu_dx = GpuCrossentropySoftmax1HotWithBiasDx()(
                as_cuda_ndarray_variable(dnll),
                gpu_sm,
                as_cuda_ndarray_variable(cast(yidx, 'float32')))
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
            gpu_sm = GpuSoftmaxWithBias()(as_cuda_ndarray_variable(x),
                                          as_cuda_ndarray_variable(b))
            return [host_from_gpu(gpu_sm)]
    return False


def _gpu_conv_to_fftconv(node):
    # shared helper function for local_conv_fft_valid and local_conv_fft_full.
    # we import conv2d_fft locally to avoid pycuda warnings
    from theano.sandbox.cuda.fftconv import conv2d_fft
    kwargs = {'border_mode': node.op.border_mode}
    if (node.op.imshp is not None and
            node.op.imshp[-1] is not None and
            node.op.imshp[-1] % 2 == 1):

        kwargs['pad_last_dim'] = True
    # If the user supplied the full nonsymbolic image_shape and
    # filter_shape in conv2d(), we can pass it on to conv2d_fft().
    if ((node.op.imshp is not None) and
            (len(node.op.imshp) == 3) and
            (None not in node.op.imshp) and
            (node.op.bsize is not None)):
        kwargs['image_shape'] = (node.op.bsize,) + node.op.imshp
    if ((node.op.kshp is not None) and
            (None not in node.op.kshp) and
            (node.op.nkern is not None) and
            (len(node.op.imshp) == 3) and
            (node.op.imshp[0] is not None)):
        kwargs['filter_shape'] = (node.op.nkern, node.op.imshp[0]) + \
            node.op.kshp
    rval = conv2d_fft(node.inputs[0], node.inputs[1], **kwargs)
    if node.outputs[0].broadcastable != rval.broadcastable:
        # With given shape information, conv2d_fft may return a different
        # broadcast pattern than GpuConv. This is forbidden, so we fix it.
        rval = tensor.patternbroadcast(
            rval, node.outputs[0].type.broadcastable)
    return rval


@local_optimizer([GpuConv])
def local_conv_fft_valid(node):
    if isinstance(node.op, GpuConv):
        if (node.op.border_mode == 'valid' and
                node.op.subsample == (1, 1) and
                node.op.fft_opt):

            return [_gpu_conv_to_fftconv(node)]
        return False


@local_optimizer([GpuConv])
def local_conv_fft_full(node):
    if isinstance(node.op, GpuConv):
        if (node.op.border_mode == 'full' and
                node.op.subsample == (1, 1) and
                node.op.fft_opt):

            return [_gpu_conv_to_fftconv(node)]
        return


def values_eq_approx_high_tol(a, b):
    """
    This fct is needed to don't have DebugMode raise useless
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
    return CudaNdarrayType.values_eq_approx(a, b, atol=atol)


@local_optimizer([GpuFromHost, conv.ConvOp])
def local_gpu_conv(node):
    """
    gpu_from_host(conv) -> gpu_conv(gpu_from_host)

    conv(host_from_gpu) -> host_from_gpu(gpu_conv)

    """
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
                      direction_hint=op.direction_hint,
                      verbose=op.verbose,
                      imshp=op.imshp,
                      nkern=op.nkern,
                      bsize=op.bsize,
                      fft_opt=op.fft_opt
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
                    img = as_cuda_ndarray_variable(img)
                    return ret(img, kern)

                return make_graph
        return ret

    if isinstance(node.op, GpuFromHost):
        # gpu_from_host(conv) -> gpu_conv(gpu_from_host)
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, conv.ConvOp):
            gpu_conv = GpuConvOp_from_ConvOp(host_input.owner.op)
            if gpu_conv is None:
                return
            img, kern = host_input.owner.inputs
            out = gpu_conv(as_cuda_ndarray_variable(img),
                           as_cuda_ndarray_variable(kern))
            out = tensor.patternbroadcast(out,
                                          node.outputs[0].broadcastable)
            out.tag.values_eq_approx = values_eq_approx_high_tol
            # in some case the ConvOp broadcast the last 2 dimensions
            # differently then the gpu ConvOp
            return [out]

    if isinstance(node.op, conv.ConvOp):
        # conv(host_from_gpu) -> host_from_gpu(gpu_conv)
        img, kern = node.inputs
        img_on_gpu = (img.owner and isinstance(img.owner.op, HostFromGpu))
        kern_on_gpu = (kern.owner and isinstance(kern.owner.op, HostFromGpu))
        if img_on_gpu or kern_on_gpu:
            gpu_conv = GpuConvOp_from_ConvOp(node.op)
            if gpu_conv is None:
                return
            out = gpu_conv(as_cuda_ndarray_variable(img),
                           as_cuda_ndarray_variable(kern))
            out = tensor.patternbroadcast(
                host_from_gpu(out),
                node.outputs[0].broadcastable)
            out.tag.values_eq_approx = values_eq_approx_high_tol
            # in some case the ConvOp broadcast the last 2 dimensions
            # differently then the gpu ConvOp
            return [out]


@local_optimizer([GpuConv])
def local_conv_gemm(node):
    if (isinstance(node.op, GpuConv) and
            node.op.border_mode in ['full', 'valid']):

        img, kern = node.inputs
        border_mode = node.op.border_mode
        subsample = node.op.subsample
        if (border_mode == 'valid') or (subsample != (1, 1)):
            # need to flip the kernel for valid convolution
            kern = kern[:, :, ::-1, ::-1]
            # By default use GpuCorrMM
            rval = GpuCorrMM(border_mode, subsample)(
                gpu_contiguous(img), gpu_contiguous(kern))

            # call GpuCorrMM_gradWeights if good
            # (the latter is faster if batchsize * kernelHeight * kernelWidth
            # is larger than inputChannels * outputHeight * outputWidth.
            # GpuConv does not always store information on the batchsize and
            # channels, though, so we only use what information we have.)
            if ((subsample == (1, 1)) and
                    (node.op.imshp is not None) and
                    (None not in node.op.imshp[-2:]) and
                    (node.op.kshp is not None) and
                    (None not in node.op.kshp)):
                # we know the kernel and output size
                prod1 = node.op.kshp[0] * node.op.kshp[1]
                prod2 = ((node.op.imshp[-2] - node.op.kshp[0] + 1) *
                         (node.op.imshp[-1] - node.op.kshp[1] + 1))
                if ((node.op.bsize is not None) and
                        (len(node.op.imshp) == 3) and
                        (node.op.imshp[0] is not None)):
                    # we also know batchsize and input channels
                    prod1 *= node.op.bsize
                    prod2 *= node.op.imshp[0]
                # compare to decide
                if prod1 > prod2:
                    # (we need to wrap the result in as_cuda_ndarray_variable,
                    # because we are not allowed to replace a CudaNdarray with
                    # a DimShuffle instance in a graph optimization)
                    rval = theano.sandbox.cuda.as_cuda_ndarray_variable(
                        GpuCorrMM_gradWeights(border_mode,
                                              subsample)(
                            gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
                            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3))
                        ).dimshuffle(1, 0, 2, 3))
        elif (border_mode == 'full'):
            # need to dimshuffle the kernel for full convolution
            kern = kern.dimshuffle(1, 0, 2, 3)
            # call GpuCorrMM_gradInputs
            rval = GpuCorrMM_gradInputs('valid', subsample)(
                gpu_contiguous(kern), gpu_contiguous(img))
        if node.outputs[0].broadcastable != rval.broadcastable:
            # With given shape information, conv2d_fft may return a different
            # broadcast pattern than GpuConv. This is forbidden, so we fix it.
            rval = tensor.patternbroadcast(
                rval, node.outputs[0].type.broadcastable)
        return [rval]

# First we register the optimizer that moves convolutions to the GPU.
register_opt()(local_gpu_conv)

# Then we create a group of optimizers that replace the legacy GpuConv
# with other implementations. They are tried in a specific order so we
# can control which ones take precedence over others.
conv_groupopt = theano.gof.optdb.LocalGroupDB()
conv_groupopt.__name__ = "gpu_conv_opts"
register_opt()(conv_groupopt)

# FFT gets the highest priority (lowest number), but is disabled by default.
# It can be enabled by including 'conv_fft'.
conv_groupopt.register('conv_fft_valid', local_conv_fft_valid,
                       'conv_fft', position=10)
conv_groupopt.register('conv_fft_full', local_conv_fft_full,
                       'conv_fft', position=10)
# cuDNN is the second, but only registered if cuDNN is available.
# It can be disabled by excluding 'conv_dnn' or 'cudnn'.
# We can't check at import if dnn is available, so we must always
# register it. This do not cause problem as if it is not avail, the
# opt will do nothing.
conv_groupopt.register('local_conv_dnn', dnn.local_conv_dnn,
                       'conv_dnn',
                       'fast_compile', 'fast_run', 'cudnn', position=20)
# The GEMM-based convolution comes last to catch all remaining cases.
# It can be disabled by excluding 'conv_gemm'.
conv_groupopt.register('local_conv_gemm', local_conv_gemm,
                       'conv_gemm',
                       'fast_compile', 'fast_run', position=30)


class LocalCudaMetaOptimizer(LocalMetaOptimizer):
    """
    Base class for CUDA-based LocalMetaOptimizers.

    """

    def time_call(self, fn):
        # Override time_call() to do device synchronization
        theano.sandbox.cuda.synchronize()
        start = time.time()
        fn()
        theano.sandbox.cuda.synchronize()
        return time.time() - start


# Convolution Meta-optimizer

class ConvMetaOptimizer(LocalCudaMetaOptimizer):
    def __init__(self, optimizers):
        super(ConvMetaOptimizer, self).__init__([GpuConv], optimizers)

    def provide_inputs(self, node, inputs):
        # We need to provide dummy data for the given inputs.
        # We can make use of the fact that GpuConv often knows its shapes.
        result = {}
        img, kern = node.inputs
        # provide dummy image and filters if needed
        vars = (img, kern)
        if node.op.imshp is not None and len(node.op.imshp) == 3:
            nchannels = node.op.imshp[0]
        else:
            nchannels = None
        shapes = ((node.op.bsize,) + node.op.imshp,
                  (node.op.nkern, nchannels) + node.op.kshp)
        for (var, shape) in zip(vars, shapes):
            if ((var in inputs) and (shape is not None) and
                    not any(s is None for s in shape)):

                result[var] = theano.shared(
                    # TODO: Use var.type.filter when cuda_ndarray.filter
                    # supports non-strict casts
                    # var.type.filter(numpy.random.randn(*shape),
                    # allow_downcast=True),
                    numpy.require(numpy.random.randn(*shape),
                                  dtype=var.dtype),
                    var.name,
                    broadcastable=var.broadcastable,
                    borrow=True)
        # return mapping
        return result

# We just register all optimizers from conv_groupopt with the metaoptimizer
conv_metaopt = ConvMetaOptimizer(
    conv_groupopt.query(*['+' + name for name in conv_groupopt._names]).opts)
# Then we add some optimizers that try less obvious options
conv_metaopt.register(dnn.local_conv_dnn_alternative)
# Finally, we register the metaoptimizer as the first optimizer in
# conv_groupopt
conv_groupopt.register('conv_meta', conv_metaopt, position=0)


@local_optimizer([Conv3D])
def local_conv3d_fft(node):
    if not isinstance(node.op, Conv3D):
        return
    try:
        stride_x = tensor.get_scalar_constant_value(node.inputs[3][0])
        stride_y = tensor.get_scalar_constant_value(node.inputs[3][1])
        stride_z = tensor.get_scalar_constant_value(node.inputs[3][2])
    except tensor.NotScalarConstantError:
        return False
    if (stride_x, stride_y, stride_z) == (1, 1, 1):
        # we import conv3d_fft locally to avoid pycuda warnings
        from theano.sandbox.cuda.fftconv import conv3d_fft
        # Shuffle inputs signal from (b, 0, 1, t, c) to (b, c, 0, 1, t)
        x = node.inputs[0]
        x = gpu_from_host(x.dimshuffle(0, 4, 1, 2, 3))
        # Shuffle filters from (oc, 0, 1, t, ic) to (oc, ic, 0, 1, t)
        f = node.inputs[1]
        f = gpu_from_host(f.dimshuffle(0, 4, 1, 2, 3))
        # filter flip
        f = f[:, :, ::-1, ::-1, ::-1]
        rval = conv3d_fft(x, f, border_mode='valid', pad_last_dim=True)
        # Shuffle from (oc, c, 0, 1, t) to (oc, 0, 1, t, c)
        return [rval.dimshuffle(0, 2, 3, 4, 1) + node.inputs[2]]


gpu_optimizer.register("conv3d_fft", local_conv3d_fft)


@local_optimizer([ConvGrad3D])
def local_convgrad3d_fft(node):
    try:
        stride_x = tensor.get_scalar_constant_value(node.inputs[1][0])
        stride_y = tensor.get_scalar_constant_value(node.inputs[1][1])
        stride_z = tensor.get_scalar_constant_value(node.inputs[1][2])
    except tensor.NotScalarConstantError:
        return False
    if (isinstance(node.op, ConvGrad3D) and
            (stride_x, stride_y, stride_z) == (1, 1, 1)):

        # we import conv3d_fft locally to avoid pycuda warnings
        from theano.sandbox.cuda.fftconv import conv3d_fft
        # Shuffle inputs signal from (b, 0, 1, t, ic) to (ic, b, 0, 1, t)
        x = node.inputs[0]
        x = x.dimshuffle(4, 0, 1, 2, 3)
        # Shuffle dCdH from (b, 0, 1, t, oc) to (oc, b, 0, 1, t)
        f = node.inputs[3]
        f = f.dimshuffle(4, 0, 1, 2, 3)
        # filter flip
        f = f[:, :, ::-1, ::-1, ::-1]
        rval = conv3d_fft(x, f, border_mode='valid', pad_last_dim=True)
        # Shuffle from (ic, oc, 0, 1, t) to (oc, 0, 1, t, ic)
        return [rval.dimshuffle(1, 2, 3, 4, 0)]


gpu_optimizer.register("convgrad3d_fft", local_convgrad3d_fft)


@local_optimizer([ConvTransp3D])
def local_convtransp3d_fft(node):
    try:
        stride_x = tensor.get_scalar_constant_value(node.inputs[2][0])
        stride_y = tensor.get_scalar_constant_value(node.inputs[2][1])
        stride_z = tensor.get_scalar_constant_value(node.inputs[2][2])
    except tensor.NotScalarConstantError:
        return False
    if (isinstance(node.op, ConvTransp3D) and
            (stride_x, stride_y, stride_z) == (1, 1, 1)):
        # we import conv3d_fft locally to avoid pycuda warnings
        from theano.sandbox.cuda.fftconv import conv3d_fft
        # Shuffle filters from (oc, 0, 1, t, ic) to (ic, oc, 0, 1, t)
        x = node.inputs[0]
        x = x.dimshuffle(4, 0, 1, 2, 3)
        # Shuffle dCdH from (b, 0, 1, t, oc) to (b, oc, 0, 1, t)
        f = node.inputs[3]
        f = f.dimshuffle(0, 4, 1, 2, 3)
        rval = conv3d_fft(f, x, border_mode='full', pad_last_dim=True)
        # Shuffle from (ic, b, 0, 1, t) to (b, 0, 1, t, ic)
        return [rval.dimshuffle(0, 2, 3, 4, 1) + node.inputs[1]]

gpu_optimizer.register("convtransp3d_fft", local_convtransp3d_fft)


@local_optimizer([Conv3D])
def local_conv3d_gemm(node):
    if not isinstance(node.op, Conv3D):
        return
    try:
        sx = tensor.get_scalar_constant_value(node.inputs[3][0])
        sy = tensor.get_scalar_constant_value(node.inputs[3][1])
        sz = tensor.get_scalar_constant_value(node.inputs[3][2])
    except tensor.NotScalarConstantError:
        return False
    if isinstance(node.op, Conv3D):
        # Shuffle inputs signal from (b, 0, 1, t, c) to (b, c, 0, 1, t)
        x = node.inputs[0]
        x = x.dimshuffle(0, 4, 1, 2, 3)
        # Shuffle filters from (oc, 0, 1, t, ic) to (oc, ic, 0, 1, t)
        f = node.inputs[1]
        f = f.dimshuffle(0, 4, 1, 2, 3)
        rval = GpuCorr3dMM(border_mode='valid', subsample=(sx, sy, sz))(x, f)
        # Shuffle from (oc, c, 0, 1, t) to (oc, 0, 1, t, c)
        return [rval.dimshuffle(0, 2, 3, 4, 1) + node.inputs[2]]

gpu_optimizer.register("conv3d_gemm", local_conv3d_gemm)


@local_optimizer([ConvGrad3D])
def local_convgrad3d_gemm(node):
    try:
        sx = tensor.get_scalar_constant_value(node.inputs[1][0])
        sy = tensor.get_scalar_constant_value(node.inputs[1][1])
        sz = tensor.get_scalar_constant_value(node.inputs[1][2])
    except tensor.NotScalarConstantError:
        return False
    if isinstance(node.op, ConvGrad3D):
        # Shuffle inputs signal from (b, 0, 1, t, c) to (b, c, 0, 1, t)
        x = node.inputs[0]
        x = gpu_contiguous(x.dimshuffle(0, 4, 1, 2, 3))

        # Shuffle dCdH from (b, 0, 1, t, oc) to (oc, b, 0, 1, t)
        f = node.inputs[3]
        f = gpu_contiguous(f.dimshuffle(0, 4, 1, 2, 3))

        rval = GpuCorr3dMM_gradWeights(subsample=(sx, sy, sz))(
            x, f, shape=node.inputs[2][1:4])
        # Shuffle from (ic, oc, 0, 1, t) to (oc, 0, 1, t, ic)
        return [rval.dimshuffle(0, 2, 3, 4, 1)]

gpu_optimizer.register("convgrad3d_gemm", local_convgrad3d_gemm)


@local_optimizer([ConvTransp3D])
def local_convtransp3d_gemm(node):
    try:
        sx = tensor.get_scalar_constant_value(node.inputs[2][0])
        sy = tensor.get_scalar_constant_value(node.inputs[2][1])
        sz = tensor.get_scalar_constant_value(node.inputs[2][2])
    except tensor.NotScalarConstantError:
        return False
    if isinstance(node.op, ConvTransp3D) and (sx, sy, sz) == (1, 1, 1):
        # Shuffle filters from (oc, 0, 1, t, ic) to (ic, oc, 0, 1, t)
        x = node.inputs[0]
        x = gpu_contiguous(x.dimshuffle(0, 4, 1, 2, 3))
        # Shuffle dCdH from (b, 0, 1, t, oc) to (b, oc, 0, 1, t)
        f = node.inputs[3]
        f = gpu_contiguous(f.dimshuffle(0, 4, 1, 2, 3))
        rval = GpuCorr3dMM_gradInputs(subsample=(sx, sy, sz))(kern=x,
                                                              topgrad=f)
        # Shuffle from (ic, b, 0, 1, t) to (b, 0, 1, t, ic)
        return [rval.dimshuffle(0, 2, 3, 4, 1) + node.inputs[1]]

gpu_optimizer.register("convtransp3d_gemm", local_convtransp3d_gemm)


def _check_constant_args_pool(ndim, ws, stride, pad, node):
    """Check if the args of pool are constants. Warns if not."""
    try:
        ws = tuple(tensor.get_scalar_constant_value(ws[i]) for i in range(ndim))
        stride = tuple(tensor.get_scalar_constant_value(stride[i]) for i in range(ndim))
        pad = tuple(tensor.get_scalar_constant_value(pad[i]) for i in range(ndim))
    except tensor.NotScalarConstantError:
        msg = ("Pool with tensor variable for the window size, stride or "
               "padding is only supported in the new GPU backend, so this op "
               "will run on CPU. (op %s)" % node)
        if config.assert_no_cpu_op == "warn":
            _logger.warning(msg)
        elif config.assert_no_cpu_op == "raise":
            raise AssertionError(msg)
        return None
    return ws, stride, pad


@register_opt()
@local_optimizer([pool.Pool])
def local_gpu_downsample_factor_max(node):
    if (isinstance(node.op, pool.Pool)):
        assert node.op.__props__ == ('ignore_border', 'mode', 'ndim')
        x, ws, stride, pad = node.inputs
        nd = node.op.ndim if node.op.ndim else (x.ndim - 2)
        ret = _check_constant_args_pool(nd, ws, stride, pad, node)
        if ret is None:
            return
        ws, stride, pad = ret
        if (nd != 2 or
                max(pad) != 0 or
                node.op.mode != 'max' or
                stride != ws):
            return
        if (x.owner and isinstance(x.owner.op, HostFromGpu)):
            gpu_ws = GpuDownsampleFactorMax(ws, node.op.ignore_border)
            if node.inputs[0].ndim == 4:
                return [host_from_gpu(gpu_ws(x.owner.inputs[0]))]
            else:
                input_4D = pad_dims(x.owner.inputs[0], 2, 2)
                output_4D = gpu_ws(input_4D)
                output = unpad_dims(output_4D, x.owner.inputs[0], 2, 2)
                return [host_from_gpu(output)]


@register_opt()
@local_optimizer([pool.MaxPoolGrad])
def local_gpu_downsample_factor_max_grad(node):
    if (isinstance(node.op, pool.MaxPoolGrad)):
        assert node.op.__props__ == ('ignore_border', 'mode', 'ndim')
        x, z, gz, ws, stride, pad = node.inputs
        nd = node.op.ndim if node.op.ndim else (x.ndim - 2)
        ret = _check_constant_args_pool(nd, ws, stride, pad, node)
        if ret is None:
            return
        ws, stride, pad = ret
        if (nd != 2 or
                max(pad) != 0 or
                node.op.mode != 'max' or
                stride != ws):
            return
        if (x.owner and isinstance(x.owner.op, HostFromGpu)):
            gpu_ws_grad = GpuDownsampleFactorMaxGrad(ws, node.op.ignore_border)
            if node.inputs[0].ndim == 4:
                return [host_from_gpu(gpu_ws_grad(x.owner.inputs[0],
                                                  as_cuda_ndarray_variable(z),
                                                  as_cuda_ndarray_variable(gz)))]
            else:
                x_4D = pad_dims(x.owner.inputs[0], 2, 2)
                z_4D = pad_dims(as_cuda_ndarray_variable(z), 2, 2)
                gz_4D = pad_dims(as_cuda_ndarray_variable(gz), 2, 2)
                output_4D = gpu_ws_grad(x_4D, z_4D, gz_4D)
                output = unpad_dims(output_4D, x.owner.inputs[0], 2, 2)
                return [host_from_gpu(output)]


@register_opt()
@local_optimizer([pool.DownsampleFactorMaxGradGrad])
def local_gpu_downsample_factor_max_grad_grad(node):
    if isinstance(node.op, pool.DownsampleFactorMaxGradGrad):
        assert node.op.__props__ == ('ignore_border', 'mode', 'ndim')
        x, z, gx, ws, stride, pad = node.inputs
        nd = node.op.ndim if node.op.ndim else (x.ndim - 2)
        ret = _check_constant_args_pool(nd, ws, stride, pad, node)
        if ret is None:
            return
        ws, stride, pad = ret
        if (nd != 2 or
                max(pad) != 0 or
                node.op.mode != 'max' or
                stride != ws):
            return
        if (x.owner and isinstance(x.owner.op, HostFromGpu)):
            op = GpuDownsampleFactorMaxGradGrad(ws, node.op.ignore_border)
            if node.inputs[0].ndim == 4:
                return [host_from_gpu(op(x.owner.inputs[0],
                                         as_cuda_ndarray_variable(z),
                                         as_cuda_ndarray_variable(gx)))]
            else:
                x_4D = pad_dims(x.owner.inputs[0], 2, 2)
                z_4D = pad_dims(as_cuda_ndarray_variable(z), 2, 2)
                gx_4D = pad_dims(as_cuda_ndarray_variable(gx), 2, 2)
                output_4D = op(x_4D, z_4D, gx_4D)
                output = unpad_dims(output_4D, x.owner.inputs[0], 2, 2)
                return [host_from_gpu(output)]


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

        axis_and_tensors = node.inputs

        matches = [t.dtype == 'float32' and
                   ((t.owner is not None and
                     isinstance(t.owner.op, HostFromGpu)) or
                    isinstance(t, gof.Constant)) for t in axis_and_tensors[1:]]

        if all(matches):
            new_tensors = [as_cuda_ndarray_variable(t)
                           for t in axis_and_tensors[1:]]
            new_a_and_t = [axis_and_tensors[0]] + new_tensors

            replacement_node = host_from_gpu(gpu_join(*new_a_and_t))

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
    Returns
    -------
    tuple
        (gpu ptr size, cpu ptr size, int sizes(gpu and cpu)).

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
        assert int_size == gpu_int_size, (int_size, gpu_int_size)
        del gpu_int_size
        del t
    except Exception as e:
        _logger.warning((
            "Optimization Warning: "
            "Got the following error, but you can ignore it. "
            "This could cause less GpuElemwise fused together.\n"
            "%s") % e)

    rval = get_device_type_sizes.rval = dict(gpu_ptr_size=gpu_ptr_size,
                                             cpu_ptr_size=cpu_ptr_size,
                                             int_size=int_size)
    return rval


def max_inputs_to_GpuElemwise(node):
    """
    Return the maximum number of inputs this GpuElemwise Apply node can
    accept.

    This is needed as currently there is a limit of 256 bytes of
    parameter for the gpu function on devices with compute capability
    1.x. There is a 4 kbyte limit on devices with compute capability
    2.x (not used).

    This measures the number of parameters we put in our GPU function and
    computes the maximum number of inputs that respect the 256 byte limit.

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

# GpuElemwise fusion
gpu_local_elemwise_fusion = tensor.opt.local_elemwise_fusion_op(
    GpuElemwise, max_inputs_to_GpuElemwise)
if config.gpu.local_elemwise_fusion:
    _logger.debug("enabling optimization fusion of gpu elemwise in fast_run")
    # Must be after cpu fusion at 40, gpu at 48.5 and before
    # AddDestroyHandler at 49.5
    optdb.register('gpu_elemwise_fusion',
                   tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion),
                   49, 'fast_run', 'fusion',
                   'local_elemwise_fusion', 'gpu')
else:
    _logger.debug(("not enabling optimization fusion of gpu elemwise in "
                   "fast_run"))
    optdb.register('gpu_elemwise_fusion',
                   tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion),
                   49, 'fusion', 'local_elemwise_fusion')

# GpuElemwise inplace
gpu_inplace_elemwise_optimizer = tensor.opt.InplaceElemwiseOptimizer(
    GpuElemwise)
# DO NOT PLACE add a 'gpu' tag here! This would enable it in fast_compile.
# It still will be run in fast_run with device=gpu with the current tag.
optdb.register('gpu_inplace_elemwise_opt', gpu_inplace_elemwise_optimizer, 75,
               'fast_run', 'inplace', 'gpu_inplace')


register_opt()(tensor.opt.local_remove_useless_assert)

register_opt()(tensor.opt.local_shape_to_shape_i)
gpu_elemwise_alloc = gof.local_optimizer([GpuElemwise])(
    tensor.opt.local_elemwise_alloc_op(GpuElemwise, GpuAlloc, GpuDimShuffle)
)
register_opt()(gpu_elemwise_alloc)
# needed by gpu_elemwise_alloc
register_opt()(tensor.opt.local_useless_elemwise)
tensor.opt.register_specialize_device(gpu_elemwise_alloc)


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
                  isinstance(c.op, tensor.Join) and
                  all(i.owner and
                      i.owner.op in [host_from_gpu, tensor.alloc]
                      for i in c.inputs[1:])
                  for c, idx in node.outputs[0].clients]):
            # if the client is on gpu or alloc
            replace = True
        if replace and node.inputs[0].dtype != 'float32':
            replace = False
    if replace:
        val = node.inputs[0]
        shp = node.inputs[1:]
        old_out = node.outputs[0]
        new_out = host_from_gpu(gpu_alloc(val, *shp))

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

        return [new_out]


@register_opt()
@local_optimizer([theano.tensor.opt.Assert, GpuFromHost])
def local_assert(node):
    if (isinstance(node.op, theano.tensor.opt.Assert) and
        node.inputs[0].owner and
        isinstance(node.inputs[0].owner.op,
                   HostFromGpu)):
        return [host_from_gpu(node.op(node.inputs[0].owner.inputs[0],
                                      *node.inputs[1:]))]
    elif (isinstance(node.op, GpuFromHost) and
          node.inputs[0].owner and
          isinstance(node.inputs[0].owner.op,
                     theano.tensor.opt.Assert)):
        a = node.inputs[0].owner
        new = a.op(gpu_from_host(a.inputs[0]), *a.inputs[1:])
        return [new]


@register_opt()
@local_optimizer([GpuAlloc])
def local_gpualloc_memset_0(node):
    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
        inp = node.inputs[0]
        if (isinstance(inp, CudaNdarrayConstant) and
                inp.data.size == 1 and
                (numpy.asarray(inp.data) == 0).all()):

            new_out = GpuAlloc(memset_0=True)(*node.inputs)
            old_bcast = node.outputs[0].type.broadcastable
            if new_out.type.broadcastable != old_bcast:
                # check that we did not try discarding a broadcastable
                # dimension
                assert not any(b_old and not b_new for b_old, b_new in
                               zip(old_bcast, new_out.type.broadcastable))
                # force old broadcasting pattern; we must not change it here
                new_out = tensor.patternbroadcast(new_out, old_bcast)
            return [new_out]


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
@local_optimizer([GpuFromHost, tensor.extra_ops.CpuContiguous])
def local_gpu_contiguous(node):
    if isinstance(node.op, tensor.extra_ops.CpuContiguous):
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            gpu_x, = x.owner.inputs
            return [tensor.as_tensor_variable(gpu_contiguous(gpu_x))]
    if isinstance(node.op, GpuFromHost):
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, tensor.extra_ops.CpuContiguous):
            gpu_x, = x.owner.inputs
            return [gpu_contiguous(gpu_x)]
    return False


@register_opt()
@local_optimizer([GpuFromHost, tensor.Eye])
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

            if tensor.extract_constant(host_input.owner.inputs[2]) != 0:
                return
            return [gpu_eye(*host_input.owner.inputs)]
    if isinstance(node.op, tensor.Eye) and node.op.dtype == "float32":
        if any([(i.owner and isinstance(i.owner.op, HostFromGpu))
                for i in node.inputs]):
            if tensor.extract_constant(node.inputs[2]) != 0:
                return
            return [host_from_gpu(gpu_eye(*node.inputs))]
    return False


def safe_to_gpu(x):
    if (isinstance(x.type, tensor.TensorType) and
            x.type.dtype == 'float32'):

        return as_cuda_ndarray_variable(x)
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
@local_optimizer([nlinalg.ExtractDiag])
def local_gpu_extract_diagonal(node):
    """
    extract_diagonal(host_from_gpu()) -> host_from_gpu(extract_diagonal)

    gpu_from_host(extract_diagonal) -> extract_diagonal(gpu_from_host)

    """
    if (isinstance(node.op, nlinalg.ExtractDiag) and
        isinstance(node.inputs[0].type,
                   theano.tensor.TensorType)):
        inp = node.inputs[0]
        if inp.owner and isinstance(inp.owner.op, HostFromGpu):
            return [host_from_gpu(nlinalg.extract_diag(
                as_cuda_ndarray_variable(inp)))]
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op, nlinalg.ExtractDiag) and
            isinstance(host_input.owner.inputs[0].type,
                       theano.tensor.TensorType)):
            diag_node = host_input.owner
            return [nlinalg.extract_diag(
                as_cuda_ndarray_variable(diag_node.inputs[0]))]
    return False


def typeConstructor(broadcastable, dtype):
    if dtype == 'float32':
        return CudaNdarrayType(broadcastable=broadcastable)
    else:
        return tensor.TensorType(broadcastable=broadcastable, dtype=dtype)


@register_opt('scan')
@local_optimizer([GpuFromHost, scan_op.Scan])
def gpuScanOptimization(node):
    """
    scan(host_from_gpu) -> host_from_gpu(GPUscan)

    gpu_from_host(scan) -> GPUscan(gpu_from_host)

    """
    # gpu_from_host(scan) -> GPUscan(gpu_from_host)
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
                replace=list(zip(thescan.inputs,
                                 (safe_to_cpu(x) for x in scan_ins))))
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

    # scan(host_from_gpu) -> host_from_gpu(GPUscan)
    if (type(node.op) == scan_op.Scan and
            not node.op.info['gpu']):

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
                replace=list(zip(thescan.inputs,
                                 (safe_to_cpu(x) for x in scan_ins))))

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


@register_opt()
@local_optimizer([tensor.AllocEmpty, GpuFromHost])
def local_gpu_allocempty(node):
    if (isinstance(node.op, tensor.AllocEmpty) and
            node.op.dtype == "float32"):
        ret = host_from_gpu(GpuAllocEmpty()(*node.inputs))
        # Keep the check that we don't care about the value.
        ret.tag.values_eq_approx = node.outputs[0].tag.values_eq_approx
        return [ret]
    return False


# Don't register by default.
@gof.local_optimizer([GpuAllocEmpty])
def local_gpu_alloc_empty_to_zeros(node):
    # We need the exact match as GpuAlloc inherit from GpuAllocEmpty.
    if type(node.op) is GpuAllocEmpty:
        return [gpu_alloc(theano.tensor.constant(0, dtype='float32'),
                          *node.inputs)]
optdb.register('local_gpu_alloc_empty_to_zeros',
               theano.tensor.opt.in2out(local_gpu_alloc_empty_to_zeros),
               # After move to gpu and merge2, before inplace.
               49.3,
               'alloc_empty_to_zeros',)


def typeInfer(node):
    return typeConstructor

# Do not register in fast_run or fast_compile.
# It will be added to fast_run if the GPU is enabled.
optdb.register('gpu_scanOp_make_inplace',
               scan_opt.ScanInplaceOptimizer(typeInfer=typeInfer,
                                             gpu_flag=True),
               75,
               'gpu',
               'inplace',
               'scan')


# XXX: these optimisations were badly broken and now require a working
# beta param (could only be a 0/1 thing for outer_merge, but
# alpha_merge needs the full range).

#    @register_opt()
#    @alpha_merge(GpuSparseBlockOuter, alpha_in=5, beta_in=?, nd=4)
#    def local_merge_blocksparse_alpha(node, *inputs):
#        """
#            GpuElemwise{mul}(lr, GpuSparseBlockOuter) ->
#                GpuSparseBlockOuter(..., alpha=lr)
#        """
#        return [gpu_sparse_block_outer(*inputs)]

#    @register_opt()
#    @output_merge(GpuSparseBlockOuter, alpha_in=5, beta_in=? out_in=0, nd=4)
#    def local_merge_blocksparse_output(node, *inputs):
#        return [gpu_sparse_block_outer(*inputs)]


def _owner_isinstance(inp, test_class):
    """
        Tests whether input has an owner and if its owner is
        of type `test_class`
    """
    return bool(inp.owner) and isinstance(inp.owner.op, test_class)


def _clear_host_from_gpu(inputs):
    """
        Replace any HostFromGpu by its input
    """
    clean_inputs = []
    for inp in inputs:
        if _owner_isinstance(inp, HostFromGpu):
            clean_inputs.append(inp.owner.inputs[0])
        else:
            clean_inputs.append(inp)
    return clean_inputs


@register_opt()
@local_optimizer([SparseBlockGemv, GpuFromHost])
def gpu_sparse_block_gemv_opt(node):
    """
        SparseBlockGemv(HostFromGpu(input)) ->
        HostFromGpu(GpuSparseBlockGemv(input))

        or

        GpuFromHost(SparseBlockGemv) -> GpuSparseBlockGemv
    """
    if isinstance(node.op, SparseBlockGemv) and \
            any(_owner_isinstance(inp, HostFromGpu) for inp in node.inputs):

        inputs = _clear_host_from_gpu(node.inputs)

        return [host_from_gpu(GpuSparseBlockGemv(node.op.inplace)(*inputs))]

    elif isinstance(node.op, GpuFromHost) and \
            _owner_isinstance(node.inputs[0], SparseBlockGemv):

        meta_node = node.inputs[0].owner
        inputs = _clear_host_from_gpu(meta_node.inputs)

        return [GpuSparseBlockGemv(meta_node.op.inplace)(*inputs)]


@register_opt()
@local_optimizer([SparseBlockOuter, GpuFromHost])
def gpu_sparse_block_outer_opt(node):
    """
        SparseBlockOuter(HostFromGpu(input)) ->
        HostFromGpu(GpuSparseBlockOuter(input))

        or

        GpuFromHost(SparseBlockOuter) -> GpuSparseBlockOuter
    """

    if isinstance(node.op, SparseBlockOuter) and \
            any(_owner_isinstance(inp, HostFromGpu) for inp in node.inputs):

        inputs = _clear_host_from_gpu(node.inputs)

        return [host_from_gpu(GpuSparseBlockOuter()(*inputs))]

    elif isinstance(node.op, GpuFromHost) and \
            _owner_isinstance(node.inputs[0], SparseBlockOuter):

        meta_node = node.inputs[0].owner
        inputs = _clear_host_from_gpu(meta_node.inputs)

        return [GpuSparseBlockOuter()(*inputs)]


@local_optimizer([GpuSparseBlockGemv], inplace=True)
def local_inplace_gpu_sparse_block_gemv(node):
    """
        GpuSparseBlockGemv(inplace=False) -> GpuSparseBlockGemv(inplace=True)
    """
    if isinstance(node.op, GpuSparseBlockGemv) and not node.op.inplace:
        new_node = gpu_sparse_block_gemv_inplace(*node.inputs)
        return [new_node]
    return False
optdb.register('local_inplace_gpu_sparse_block_gemv',
               TopoOptimizer(
                   local_inplace_gpu_sparse_block_gemv,
                   failure_callback=TopoOptimizer.warn_inplace),
               60, 'fast_run', 'inplace', 'gpu')  # DEBUG


@local_optimizer([GpuSparseBlockOuter], inplace=True)
def local_inplace_gpu_sparse_block_outer(node):
    """
        GpuSparseBlockOuter(inplace=False) -> GpuSparseBlockOuter(inplace=True)
    """
    if isinstance(node.op, GpuSparseBlockOuter) and not node.op.inplace:
        new_node = gpu_sparse_block_outer_inplace(*node.inputs)
        return [new_node]
    return False
optdb.register('local_inplace_gpu_sparse_block_outer',
               TopoOptimizer(
                   local_inplace_gpu_sparse_block_outer,
                   failure_callback=TopoOptimizer.warn_inplace),
               60, 'fast_run', 'inplace', 'gpu')  # DEBUG


# Move to Gpu optimization
@local_optimizer([GpuFromHost,
                  AbstractConv2d,
                  AbstractConv2d_gradWeights,
                  AbstractConv2d_gradInputs,
                  AbstractConv3d,
                  AbstractConv3d_gradWeights,
                  AbstractConv3d_gradInputs])
def local_conv_gpu_conv(node):
    """
    gpu_from_host(AbstractConv) -> AbstractConv(gpu_from_host)

    AbstractConv(host_from_gpu) -> host_from_gpu(AbstractConv)
    """
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           BaseAbstractConv):

            conv = host_input.owner.op
            inps = list(host_input.owner.inputs)
            inps[0] = as_cuda_ndarray_variable(inps[0])
            inps[1] = as_cuda_ndarray_variable(inps[1])
            out = conv(*inps)
            # out is on the GPU because both inputs are.
            out = theano.tensor.patternbroadcast(out,
                                                 node.outputs[0].broadcastable)
            out.tag.values_eq_approx = values_eq_approx_high_tol
            return [out]

    if isinstance(node.op, BaseAbstractConv):
        # conv(host_from_gpu) -> host_from_gpu(gpu_conv)
        inp1 = node.inputs[0]
        inp2 = node.inputs[1]
        if ((isinstance(inp1.type, CudaNdarrayType) and
             isinstance(inp2.type, CudaNdarrayType))):
            # Both inputs are already directly on the GPU, nothing to do
            return

        inp1_on_gpu = (isinstance(inp1.type, CudaNdarrayType) or
                       (inp1.owner and isinstance(inp1.owner.op, HostFromGpu)))
        inp2_on_gpu = (isinstance(inp2.type, CudaNdarrayType) or
                       (inp2.owner and isinstance(inp2.owner.op, HostFromGpu)))

        if inp1_on_gpu or inp2_on_gpu:
            conv = node.op
            inps = list(node.inputs)
            inps[0] = as_cuda_ndarray_variable(inps[0])
            inps[1] = as_cuda_ndarray_variable(inps[1])
            out = conv(*inps)
            # out is on the GPU because both inputs are.
            out = theano.tensor.patternbroadcast(
                out,
                node.outputs[0].broadcastable)
            out.tag.values_eq_approx = values_eq_approx_high_tol
            # If the original output was on CPU, we have to transfer it
            if isinstance(node.outputs[0].type, tensor.TensorType):
                return [tensor.as_tensor_variable(out)]
            else:
                return [out]
register_opt()(local_conv_gpu_conv)


# Corrmm opt
@local_optimizer([AbstractConv2d])
def local_abstractconv_gemm(node):
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if (not isinstance(img.type, CudaNdarrayType) or
            not isinstance(kern.type, CudaNdarrayType)):
        return None

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    if ((border_mode == 'full') and (subsample == (1, 1))):
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1]
        # need to dimshuffle the kernel for full convolution
        kern = kern.dimshuffle(1, 0, 2, 3)
        # call GpuCorrMM_gradInputs
        rval = GpuCorrMM_gradInputs('valid',
                                    subsample,
                                    filter_dilation)(
            gpu_contiguous(kern), gpu_contiguous(img))
    else:
        # need to flip the kernel if necessary
        if node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1]
        # By default use GpuCorrMM
        rval = GpuCorrMM(border_mode,
                         subsample,
                         filter_dilation)(gpu_contiguous(img),
                                          gpu_contiguous(kern))

        # call GpuCorrMM_gradWeights if good
        # (the latter is faster if batchsize * kernelHeight * kernelWidth
        # is larger than inputChannels * outputHeight * outputWidth.
        # GpuConv does not always store information on the batchsize and
        # channels, though, so we only use what information we have.)
        if ((subsample == (1, 1)) and (filter_dilation == (1, 1)) and
                (node.op.imshp is not None) and
                (None not in node.op.imshp[-2:]) and
                (node.op.kshp is not None) and
                (None not in node.op.kshp) and
                border_mode != "half"):
            # we know the kernel and output size
            prod1 = node.op.kshp[0] * node.op.kshp[1]
            prod2 = ((node.op.imshp[-2] - node.op.kshp[0] + 1) *
                     (node.op.imshp[-1] - node.op.kshp[1] + 1))
            if (None not in node.op.imshp[:1]):
                # we also know batchsize and input channels
                prod1 *= node.op.imshp[0]
                prod2 *= node.op.imshp[1]
            # compare to decide
            if prod1 > prod2:
                # (we need to wrap the result in as_cuda_ndarray_variable,
                # because we are not allowed to replace a CudaNdarray with
                # a DimShuffle instance in a graph optimization)
                rval = theano.sandbox.cuda.as_cuda_ndarray_variable(
                    GpuCorrMM_gradWeights(border_mode,
                                          subsample,
                                          filter_dilation)(
                        gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
                        gpu_contiguous(kern.dimshuffle(1, 0, 2, 3))
                    ).dimshuffle(1, 0, 2, 3))
    return [rval]


# Corrmm opt
@local_optimizer([AbstractConv3d])
def local_abstractconv3d_gemm(node):
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if (not isinstance(img.type, CudaNdarrayType) or
            not isinstance(kern.type, CudaNdarrayType)):
        return None

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    if ((border_mode == 'full') and (subsample == (1, 1, 1))):
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        # need to dimshuffle the kernel for full convolution
        kern = kern.dimshuffle(1, 0, 2, 3, 4)
        # call GpuCorr3dMM_gradInputs
        rval = GpuCorr3dMM_gradInputs('valid',
                                      subsample,
                                      filter_dilation)(
            gpu_contiguous(kern), gpu_contiguous(img))
    else:
        # need to flip the kernel if necessary
        if node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        # By default use GpuCorr3dMM
        rval = GpuCorr3dMM(border_mode,
                           subsample,
                           filter_dilation)(gpu_contiguous(img),
                                            gpu_contiguous(kern))

        # call GpuCorr3dMM_gradWeights if good
        # (the latter is faster if
        #   batchsize * kernelHeight * kernelWidth * kernelDepth
        # is larger than
        #   inputChannels * outputHeight * outputWidth * outputDepth.
        # GpuConv does not always store information on the batchsize and
        # channels, though, so we only use what information we have.)
        if ((subsample == (1, 1, 1)) and (filter_dilation == (1, 1, 1)) and
                (node.op.imshp is not None) and
                (None not in node.op.imshp[-3:]) and
                (node.op.kshp is not None) and
                (None not in node.op.kshp) and
                border_mode != "half"):
            # we know the kernel and output size
            prod1 = node.op.kshp[0] * node.op.kshp[1] * node.op.kshp[2]
            prod2 = ((node.op.imshp[-3] - node.op.kshp[0] + 1) *
                     (node.op.imshp[-2] - node.op.kshp[1] + 1) *
                     (node.op.imshp[-1] - node.op.kshp[2] + 1))
            if (None not in node.op.imshp[:1]):
                # we also know batchsize and input channels
                prod1 *= node.op.imshp[0]
                prod2 *= node.op.imshp[1]
            # compare to decide
            if prod1 > prod2:
                # (we need to wrap the result in as_cuda_ndarray_variable,
                # because we are not allowed to replace a CudaNdarray with
                # a DimShuffle instance in a graph optimization)
                rval = theano.sandbox.cuda.as_cuda_ndarray_variable(
                    GpuCorr3dMM_gradWeights(border_mode,
                                            subsample,
                                            filter_dilation)(
                        gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
                        gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4))
                    ).dimshuffle(1, 0, 2, 3, 4))
    return [rval]


@local_optimizer([AbstractConv2d_gradWeights])
def local_abstractconv_gradweight_gemm(node):
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, CudaNdarrayType) or \
            not isinstance(topgrad.type, CudaNdarrayType):
        return None

    rval = GpuCorrMM_gradWeights(border_mode=node.op.border_mode,
                                 subsample=node.op.subsample,
                                 filter_dilation=node.op.filter_dilation)(
        gpu_contiguous(img), gpu_contiguous(topgrad), shape)
    if node.op.filter_flip:
        rval = rval[:, :, ::-1, ::-1]
    rval = tensor.patternbroadcast(rval, node.outputs[0].broadcastable)
    rval = as_cuda_ndarray_variable(rval)
    return [rval]


@local_optimizer([AbstractConv3d_gradWeights])
def local_abstractconv3d_gradweight_gemm(node):
    if not isinstance(node.op, AbstractConv3d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, CudaNdarrayType) or \
            not isinstance(topgrad.type, CudaNdarrayType):
        return None

    rval = GpuCorr3dMM_gradWeights(border_mode=node.op.border_mode,
                                   subsample=node.op.subsample,
                                   filter_dilation=node.op.filter_dilation)(
        gpu_contiguous(img), gpu_contiguous(topgrad), shape)
    if node.op.filter_flip:
        rval = rval[:, :, ::-1, ::-1, ::-1]
    rval = tensor.patternbroadcast(rval, node.outputs[0].broadcastable)
    rval = as_cuda_ndarray_variable(rval)
    return [rval]


@local_optimizer([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_gemm(node):
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, CudaNdarrayType) or \
            not isinstance(topgrad.type, CudaNdarrayType):
        return None

    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1]

    rval = GpuCorrMM_gradInputs(border_mode=node.op.border_mode,
                                subsample=node.op.subsample,
                                filter_dilation=node.op.filter_dilation)(
        gpu_contiguous(kern), gpu_contiguous(topgrad), shape)
    return [rval]


@local_optimizer([AbstractConv3d_gradInputs])
def local_abstractconv3d_gradinputs_gemm(node):
    if not isinstance(node.op, AbstractConv3d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, CudaNdarrayType) or \
            not isinstance(topgrad.type, CudaNdarrayType):
        return None

    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]

    rval = GpuCorr3dMM_gradInputs(border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_dilation=node.op.filter_dilation)(
        gpu_contiguous(kern), gpu_contiguous(topgrad), shape)
    return [rval]


# Register GPU convolution implementation
# They are tried in a specific order so we can control
# which ones take precedence over others.
abstractconv_groupopt = theano.gof.optdb.LocalGroupDB()
abstractconv_groupopt.__name__ = "gpu_abstractconv_opts"
register_specialize_device(abstractconv_groupopt, 'gpu', 'fast_compile')

# cuDNN is first, but only registered if cuDNN is available.
conv_groupopt.register('local_abstractconv_dnn',
                       dnn.local_abstractconv_cudnn, 20,
                       'conv_dnn',
                       'gpu', 'fast_compile', 'fast_run', 'cudnn')
conv_groupopt.register('local_abstractconv3d_dnn',
                       dnn.local_abstractconv3d_cudnn, 20,
                       'conv_dnn',
                       'gpu', 'fast_compile', 'fast_run', 'cudnn')
# The GEMM-based convolution comes last to catch all remaining cases.
# It can be disabled by excluding 'conv_gemm'.
conv_groupopt.register('local_abstractconv_gemm', local_abstractconv_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')

conv_groupopt.register('local_abstractconv3d_gemm', local_abstractconv3d_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')

conv_groupopt.register('local_abstractconv_gradweight_gemm',
                       local_abstractconv_gradweight_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')

conv_groupopt.register('local_abstractconv3d_gradweight_gemm',
                       local_abstractconv3d_gradweight_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')

conv_groupopt.register('local_abstractconv_gradinputs_gemm',
                       local_abstractconv_gradinputs_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')

conv_groupopt.register('local_abstractconv3d_gradinputs_gemm',
                       local_abstractconv3d_gradinputs_gemm, 30,
                       'conv_gemm',
                       'gpu', 'fast_compile', 'fast_run')


# Register cuDNN batch normalization implementation
abstract_batch_norm_groupopt = theano.gof.optdb.LocalGroupDB()
abstract_batch_norm_groupopt.__name__ = "gpu_batchnorm_opts"
register_opt('fast_compile')(abstract_batch_norm_groupopt)

# cuDNN optimizations are only registered if cuDNN is available.
# (we import these opts here instead of at the top of this file
# to avoid a circular dependency problem with dnn)
from .dnn import (local_abstract_batch_norm_train_cudnn,
                  local_abstract_batch_norm_train_grad_cudnn,
                  local_abstract_batch_norm_inference_cudnn)     # noqa: 402
abstract_batch_norm_groupopt.register('local_abstract_batch_norm_train_dnn',
                                      local_abstract_batch_norm_train_cudnn, 20,
                                      'batchnorm_dnn',
                                      'gpu', 'fast_compile', 'fast_run', 'cudnn')
abstract_batch_norm_groupopt.register('local_abstract_batch_norm_train_grad_dnn',
                                      local_abstract_batch_norm_train_grad_cudnn, 20,
                                      'batchnorm_dnn',
                                      'gpu', 'fast_compile', 'fast_run', 'cudnn')
abstract_batch_norm_groupopt.register('local_abstract_batch_norm_inference_dnn',
                                      local_abstract_batch_norm_inference_cudnn, 20,
                                      'batchnorm_dnn',
                                      'gpu', 'fast_compile', 'fast_run', 'cudnn')
