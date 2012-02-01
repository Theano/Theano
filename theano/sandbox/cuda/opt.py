import logging
_logger = logging.getLogger('theano.sandbox.cuda.opt')

import sys
import theano
import numpy
from theano.scan_module import scan_utils, scan_op
from theano import scalar as scal
from theano import tensor, compile, gof

from theano.gof import (local_optimizer, EquilibriumDB, SequenceDB, ProxyDB,
                        Optimizer, toolbox, DestroyHandler,
                        EquilibriumOptimizer)

from theano.sandbox.cuda.basic_ops import *
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.blas import (gpu_dot22, gpu_dot22scalar,
        gpu_gemm_inplace, gpu_gemm_no_inplace, gpu_outer, GpuConv)
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
from theano.compile import optdb
from theano.tensor.blas import _is_real_vector, _is_real_matrix

#optdb.print_summary()  # shows what is currently registered

gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()
gpu_seqopt = SequenceDB()
gpu_seqopt.register('gpu_local_optimizations', gpu_optimizer, 1,
        'fast_run', 'inplace')
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
               optdb.__position__.get('elemwise_fusion', 71) + .1,
               'gpu')

def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'inplace', *tags)
        return local_opt
    return f

#register local_track_shape_i at this level too
#to make multi-level lift of shape work.
register_opt()(theano.tensor.opt.local_track_shape_i)

class InputToGpuOptimizer(Optimizer):
    """Transfert the input of a graph to the gpu if needed
    It should make this part of the optimizer faster we will will need only 1
    pass on the env.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())
        env.extend(DestroyHandler())

    def apply(self, env):
        for input in env.inputs:
            if not isinstance(input.type, CudaNdarrayType):
                try:
                    new_input = host_from_gpu(gpu_from_host(input))

                    if new_input.type==input.type:
                        env.replace_validate(input, new_input, "InputToGpuOptimizer")
                except TypeError, e:
                    #as we currently only support float32, this can fail.
                    #Using try except make that we won't need
                    pass

#we register it before all other gpu optimizer to be sure that the input are on the gpu.
gpu_seqopt.register('InputToGpuOptimizer', InputToGpuOptimizer(),
                    0, 'fast_run', 'fast_compile', 'merge')#TODO: how to make it mandatory for gpu_seqopt?

@local_optimizer([])
def local_cut_gpu_host_gpu(node):
    if tensor.opt.opt.check_chain(node, gpu_from_host, host_from_gpu):
        return [node.inputs[0].owner.inputs[0]]
    if tensor.opt.opt.check_chain(node, host_from_gpu, gpu_from_host):
        return [node.inputs[0].owner.inputs[0]]
    return False
gpu_cut_copies.register('cut_gpu_host_transfers', local_cut_gpu_host_gpu,
        'fast_run', 'inplace', 'gpu')
gpu_cut_copies.register('cut_gpu_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_run', 'gpu')
#register it into canonicalize to allow other optimization to work without
#botering with this useless pattern.
optdb['canonicalize'].register('local_cut_gpu_host_gpu',
                               local_cut_gpu_host_gpu, 'fast_run', 'gpu')

#'float64', 'complex128' and 'complex64' are not supported in elemwise on the gpu.
elemwise_cuda_dtype_supported=['float32','uint8','int8','uint16','int16',
                               'uint32','int32','uint64','int64']

def dtype_in_elemwise_supported(op):
    """
    Return True of the Elemwise op is supported on the gpu.
    Return False otherwise.

    :note: We need to check inside the Composite op.
    """
    def get_all_basic_scalar(composite_op):
        l=[]
        for i in composite_op.env.toposort():
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
                        for i in s.inputs+s.outputs]):
                    return False
    return True




@register_opt()
@local_optimizer([])
def local_gpu_elemwise_0(node):
    """elemwise(..., host_from_gpu, ...)
       -> host_from_gpu(elemwise(gpu_from_host, ..., gpu_from_host)
    """
    if isinstance(node.op, tensor.Elemwise) and dtype_in_elemwise_supported(node.op):
        if numpy.any([i.owner and isinstance(i.owner.op, HostFromGpu) for i in node.inputs]):
            if numpy.all([o.type.dtype == 'float32' for o in node.outputs]):
                # Don't set any inplace pattern.
                # gpu_inplace_elemwise_optimizer will do it later
                try:
                    new_op = GpuElemwise(node.op.scalar_op)
                except SupportCodeError:
                    # This happens when scalar_op requires support code
                    return False

                #   first establish that float32 can store all inputs
                upcastable = set(['float32', 'int8', 'int16', 'uint8', 'uint16'])
                # case 1 - all inputs are already float32
                if numpy.all([i.type.dtype == 'float32' for i in node.inputs]):
                    #TODO: change this when fusion makes Elemwise with multiple outputs
                    gpu_elemwise = new_op(*(gpu_from_host(i) for i in node.inputs))
                # case 2 - it is still ok if some inputs were upcast to float32
                elif numpy.all([i.type.dtype in upcastable for i in node.inputs]):
                    # second - establish that a new node with upcasted inputs has the same outputs
                    # types as the original node
                    upcasted = node.op.make_node(*[tensor.cast(i, 'float32') for i in node.inputs])
                    if [o.type for o in upcasted.outputs] == [o.type for o in node.outputs]:

                        new_inputs = [gpu_from_host(tensor.cast(i, 'float32')) for i in node.inputs]
                        gpu_elemwise = new_op(*new_inputs)
                    else:
                        return False
                else:
                    return False

                gpu_elemwise = split_huge_add_or_mul(gpu_elemwise.owner)
                if not gpu_elemwise:
                    return False
                if max_inputs_to_GpuElemwise(node)<len(gpu_elemwise.inputs):
                    return False
                return [host_from_gpu(gpu_elemwise.outputs[0])]
@register_opt()
@local_optimizer([])
def local_gpu_elemwise_1(node):
    """
    gpu_from_host(Elemwise)) -> GpuElemwise(gpu_from_host(...))
    """
    if node.op == gpu_from_host:
        host_i, = node.inputs
        if (host_i.owner and
            isinstance(host_i.owner.op, tensor.Elemwise) and
            len(host_i.clients)==1 and
            dtype_in_elemwise_supported(node.op)):

            elemwise_node = host_i.owner
            # Don't set any inplace pattern.
            # gpu_inplace_elemwise_optimizer will do it later
            try:
                new_op = GpuElemwise(elemwise_node.op.scalar_op)
            except SupportCodeError:
                # This happens when scalar_op requires support code
                return False
            if all([i.dtype=='float32' for i in elemwise_node.inputs]):
                gpu_elemwise = new_op(*[gpu_from_host(i) for i in elemwise_node.inputs])
                gpu_elemwise = split_huge_add_or_mul(gpu_elemwise.owner)
                if not gpu_elemwise:
                    return False
                return [gpu_elemwise.outputs[0]]
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
            return [host_from_gpu(new_op(gpu_from_host(input)))]
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
    if node.op == gpu_from_host:
        if node.outputs[0].type.dtype != 'float32':
            return False
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op == tensor.basic.dot:
            x, y = host_input.owner.inputs
            # case one: vector X matrix
            if _is_real_vector(x) and _is_real_matrix(y):
                new_op = GpuDimShuffle((False,), ['x',0])
                shape_out = y.shape[1].dimshuffle(['x'])
                gpu_x = new_op(gpu_from_host(x))
                gpu_y = gpu_from_host(y)
            # case two: matrix X vector
            elif _is_real_matrix(x) and _is_real_vector(y):
                new_op = GpuDimShuffle((False,), [0,'x'])
                shape_out = x.shape[0].dimshuffle(['x'])
                gpu_x = gpu_from_host(x)
                gpu_y = new_op(gpu_from_host(y))
            else:
                return False

            return [GpuReshape(1)(gpu_dot22(gpu_x, gpu_y), shape_out)]
    if node.op == tensor.basic.dot:
        if node.outputs[0].type.dtype != 'float32':
            return False
        if numpy.any([(i.owner and i.owner.op == host_from_gpu) for i in node.inputs]):
            x, y = node.inputs
            if _is_real_vector(x) and _is_real_matrix(y):
                new_op = GpuDimShuffle((False,), ['x',0])
                shape_out = y.shape[1].dimshuffle(['x'])
                gpu_x = new_op(gpu_from_host(x))
                gpu_y = gpu_from_host(y)

            elif _is_real_matrix(x) and _is_real_vector(y):
                new_op = GpuDimShuffle((False,), [0,'x'])
                shape_out = x.shape[0].dimshuffle(['x'])
                gpu_x = gpu_from_host(x)
                gpu_y = new_op(gpu_from_host(y))
            else:
                return False

            return [host_from_gpu(GpuReshape(1)(gpu_dot22(gpu_x, gpu_y),
                                                shape_out))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_lazy_ifelse(node):
    """
    gpu_from_host(ifelse) -> gpu_ifelse(gpu_from_host)

    ifelse(host_from_gpu) -> host_from_gpu(ifelse)
    """
    if hasattr(theano, "lazycond"):
        gpu_ifelse = theano.lazycond.IfElse(gpu = True)

        if node.op == gpu_from_host:
            host_input = node.inputs[0]
            if (host_input.owner
                    and host_input.owner.op == theano.lazycond.ifelse):
                c, t, f = host_input.owner.inputs
                if not isinstance(f.type,CudaNdarrayType):
                    f = gpu_from_host(f)
                if not isinstance(t.type,CudaNdarrayType):
                    t = gpu_from_host(t)
                if isinstance(c.type,CudaNdarrayType):
                    c = host_from_gpu(c)

                return [gpu_ifelse(c, t, f)]

        if node.op == theano.lazycond.ifelse:
            if numpy.any([(i.owner and i.owner.op == host_from_gpu) for i in node.inputs]):
                c, t, f = node.inputs

                if not isinstance(f.type,CudaNdarrayType):
                    f = gpu_from_host(f)
                if not isinstance(t.type,CudaNdarrayType):
                    t = gpu_from_host(t)
                if isinstance(c.type,CudaNdarrayType):
                    c = host_from_gpu(c)

                return [host_from_gpu(gpu_ifelse(c, t, f))]

    return False


@register_opt()
@local_optimizer([])
def local_gpu_dot22(node):
    """
    gpu_from_host(dot22) -> gpudot(gpu_from_host)

    dot(host_from_gpu) -> host_from_gpu(gpudot22)
    """
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op == tensor.blas._dot22:
            x, y = host_input.owner.inputs
            return [gpu_dot22(gpu_from_host(x), gpu_from_host(y))]
    if node.op == tensor.blas._dot22:
        if numpy.any([(i.owner and i.owner.op == host_from_gpu) for i in node.inputs]):
            x, y = node.inputs
            return [host_from_gpu(gpu_dot22(gpu_from_host(x), gpu_from_host(y)))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_dot22scalar(node):
    """
    gpu_from_host(dot22scalar) -> gpudot(gpu_from_host)

    dot(host_from_gpu) -> host_from_gpu(gpudot22scalar)
    """
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op == tensor.blas._dot22scalar:
            x, y, scalar = host_input.owner.inputs
            return [gpu_dot22scalar(gpu_from_host(x), gpu_from_host(y), tensor.blas._as_scalar(scalar))]
    if node.op == tensor.blas._dot22scalar:
        if numpy.any([(i.owner and i.owner.op == host_from_gpu) for i in node.inputs]):
            x, y, scalar = node.inputs
            return [host_from_gpu(gpu_dot22scalar(gpu_from_host(x), gpu_from_host(y),tensor.blas._as_scalar(scalar)))]
    return False


@register_opt()
@local_optimizer([])
def local_gpu_gemv(node):
    """
    gpu_from_host(gemv) -> gpu_gemv(gpu_from_host)
    gemv(host_from_gpu) -> host_from_gpu(gpu_gemv)

    """
    gemvs = {
            tensor.blas.gemv_inplace: gpu_gemv_no_inplace,
            tensor.blas.gemv_no_inplace: gpu_gemv_no_inplace,
            tensor.blas_c.CGemv(inplace=True): gpu_gemv_no_inplace,
            tensor.blas_c.CGemv(inplace=False): gpu_gemv_no_inplace,
            }
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op in gemvs:
            op = host_input.owner.op
            z, a, x, y, b = host_input.owner.inputs
            return [gemvs[op](
                    gpu_from_host(z)
                    , a
                    , gpu_from_host(x)
                    , gpu_from_host(y)
                    , b)]
    if node.op in gemvs:
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and x.owner.op == host_from_gpu)
        y_on_gpu = (y.owner and y.owner.op == host_from_gpu)
        z_on_gpu = (z.owner and z.owner.op == host_from_gpu)
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(
                gemvs[node.op](
                    gpu_from_host(z)
                    , a
                    , gpu_from_host(x)
                    , gpu_from_host(y)
                    , b))]
    return False


@register_opt()
@local_optimizer([])
def local_gpu_ger(node):
    """
    gpu_from_host(ger) -> gpu_ger(gpu_from_host)
    ger(host_from_gpu) -> host_from_gpu(gpu_ger)

    """
    gers = {
            tensor.blas_c.CGer(destructive=True): gpu_ger_no_inplace,
            tensor.blas_c.CGer(destructive=False): gpu_ger_no_inplace,
            tensor.blas.Ger(destructive=True): gpu_ger_no_inplace,
            tensor.blas.Ger(destructive=False): gpu_ger_no_inplace,
            }
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op in gers:
            op = host_input.owner.op
            z, a, x, y = host_input.owner.inputs
            return [gers[op](
                    gpu_from_host(z)
                    , a
                    , gpu_from_host(x)
                    , gpu_from_host(y)
                    )]
    if node.op in gers:
        z, a, x, y = node.inputs
        x_on_gpu = (x.owner and x.owner.op == host_from_gpu)
        y_on_gpu = (y.owner and y.owner.op == host_from_gpu)
        z_on_gpu = (z.owner and z.owner.op == host_from_gpu)
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(
                gers[node.op](
                    gpu_from_host(z)
                    , a
                    , gpu_from_host(x)
                    , gpu_from_host(y)
                    ))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_gemm(node):
    """
    gpu_from_host(gemm) -> gpu_gemm(gpu_from_host)

    gemm(host_from_gpu) -> host_from_gpu(gpu_gemm)
    """
    gemms = {
            #tensor.blas.gemm_inplace: gpu_gemm_inplace,
            tensor.blas.gemm_no_inplace: gpu_gemm_no_inplace}
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op in gemms:
            op = host_input.owner.op
            z, a, x, y, b = host_input.owner.inputs
            return [gemms[op](gpu_from_host(z), a, gpu_from_host(x), gpu_from_host(y), b)]
    if node.op in gemms:
        z, a, x, y, b = node.inputs
        x_on_gpu = (x.owner and x.owner.op == host_from_gpu)
        y_on_gpu = (y.owner and y.owner.op == host_from_gpu)
        z_on_gpu = (z.owner and z.owner.op == host_from_gpu)
        if x_on_gpu or y_on_gpu or z_on_gpu:
            return [host_from_gpu(gemms[node.op](gpu_from_host(z), a, gpu_from_host(x), gpu_from_host(y), b))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_outer(node):
    """
    gpu_dot22(col, row) -> gpu_outer
    """
    if node.op == gpu_dot22:
        l, r = node.inputs
        if l.type.broadcastable[1] and r.type.broadcastable[0]:
            # TODO: we would like to remove the double-dimshuffle when l or r is
            # already the output of a GpuDimshuffle. To do this, refactor the
            # logic in tensor/opt.py that collapses dimshuffle chains so that we
            # can call it from here.
            lvec = GpuDimShuffle(l.broadcastable, [0])(l)
            rvec = GpuDimShuffle(r.broadcastable, [1])(r)
            return [gpu_outer(lvec, rvec)]

    return False

@register_opt()
@local_optimizer([])
def local_gpu_sum(node):
    if isinstance(node.op, tensor.elemwise.CAReduce):
        if node.op.scalar_op == scal.add:
            x, = node.inputs
            if x.owner and x.owner.op == host_from_gpu:
                if node.op.axis is None:
                    reduce_mask = [1] * x.type.ndim
                else:
                    reduce_mask = [0] * x.type.ndim
                    for a in node.op.axis:
                        assert reduce_mask[a] == 0
                        reduce_mask[a] = 1
                gsum=GpuSum(reduce_mask)
                pattern=(''.join(str(i) for i in reduce_mask))
                if hasattr(gsum, 'c_code_reduce_%s'%pattern):
                    rval = host_from_gpu(gsum(gpu_from_host(x)))
                    if rval.type == node.outputs[0].type:
                        return [rval]
                    else:
                        print >> sys.stderr, "WARNING: local_gpu_sum got type wrong"
                        return None
                else:

                    # Try to make a simpler pattern based on reshaping
                    # The principle is that if two adjacent dimensions have the same value in
                    # the reduce_mask, then we can reshape to make them a single dimension, do
                    # the sum, and then reshape to get them back.

                    shape_of = node.env.shape_feature.shape_of

                    x_shape = shape_of[x]

                    new_in_shp = [x_shape[0]]
                    new_mask = [reduce_mask[0]]
                    for i in xrange(1, x.type.ndim):
                        if reduce_mask[i] == reduce_mask[i-1]:
                            new_in_shp[-1] *= x_shape[i]
                        else:
                            new_mask.append(reduce_mask[i])
                            new_in_shp.append(x_shape[i])

                    pattern=(''.join(str(i) for i in new_mask))
                    new_gsum = GpuSum(new_mask)
                    if hasattr(new_gsum, 'c_code_reduce_%s'%pattern):
                        reshaped_x = x.reshape(tensor.stack(*new_in_shp))
                        sum_reshaped_x = host_from_gpu(new_gsum(gpu_from_host(reshaped_x)))

                        if sum_reshaped_x.ndim != node.outputs[0].ndim:
                            unreshaped_sum = sum_reshaped_x.reshape(tensor.stack(*shape_of[node.outputs[0]]))
                        else:
                            unreshaped_sum = sum_reshaped_x
                        if unreshaped_sum.type == node.outputs[0].type:
                            return [unreshaped_sum]
                        else:
                            print >> sys.stderr, "WARNING: local_gpu_sum got type wrong"
                            return None

                        raise Exception("GpuSum don't have implemented the pattern",pattern)
    return False

@register_opt()
@local_optimizer([])
def local_gpu_reshape(node):
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, tensor.Reshape):
            rshp = host_input.owner.op
            x, shp = host_input.owner.inputs
            gpu_reshape = GpuReshape(rshp.ndim)(gpu_from_host(x), shp)
            if gpu_reshape.broadcastable != node.outputs[0].broadcastable:
                #this can happen as we always return False for all broadcast dim in GpuReshape but not for Reshape
                #Event if we did the same think, with the constant optimization that could happen.
                gpu_reshape = theano.tensor.patternbroadcast(gpu_reshape,node.outputs[0].broadcastable)
            return [gpu_reshape]
    if isinstance(node.op, tensor.Reshape):
        x, shp = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            gpu_reshape = GpuReshape(node.op.ndim)(gpu_x, shp)
            if gpu_reshape.broadcastable != node.outputs[0].broadcastable:
                #this can happen as we always return False for all broadcast dim in GpuReshape but not for Reshape
                #Event if we did the same think, with the constant optimization that could happen.
                gpu_reshape = theano.tensor.patternbroadcast(gpu_reshape,node.outputs[0].broadcastable)
            return [host_from_gpu(gpu_reshape)]
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
        if x.owner and x.owner.op == host_from_gpu and x.dtype == "float32":
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuSubtensor(node.op.idx_list)(gpu_x, *coords))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_advanced_subtensor1(node):
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and host_input.owner.op.__class__ is tensor.AdvancedSubtensor1:
            x = host_input.owner.inputs[0]
            coords = host_input.owner.inputs[1:]
            return [GpuAdvancedSubtensor1()(gpu_from_host(x), *coords)]
    if node.op.__class__ is tensor.AdvancedSubtensor1:
        x  = node.inputs[0]
        coords = node.inputs[1:]
        if x.owner and x.owner.op == host_from_gpu and x.dtype == "float32":
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuAdvancedSubtensor1()(gpu_x, *coords))]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_advanced_incsubtensor1(node):
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        # Should not execute for GpuAdvancedIncSubtensor1
        if host_input.owner and host_input.owner.op.__class__ is tensor.AdvancedIncSubtensor1:
            x, y = host_input.owner.inputs[0:2]
            coords = host_input.owner.inputs[2:]
            return [GpuAdvancedIncSubtensor1()(gpu_from_host(x),
                                               gpu_from_host(y), *coords)]

    # Should not execute for GpuAdvancedIncSubtensor1
    if node.op.__class__ is tensor.AdvancedIncSubtensor1 and node.inputs[0].dtype=="float32":
        x, y  = node.inputs[0:2]
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
            return [host_from_gpu(GpuAdvancedIncSubtensor1()(gpu_x, gpu_y, *coords))]
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
            return [GpuIncSubtensor(incsubt.idx_list, inplace=incsubt.inplace,
                                    set_instead_of_inc=incsubt.set_instead_of_inc)(
                gpu_from_host(x),
                gpu_from_host(y),
                *coords)]
    if type(node.op) == tensor.IncSubtensor and node.inputs[0].dtype=="float32":
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
                node.op.idx_list, inplace=node.op.inplace,
                set_instead_of_inc=node.op.set_instead_of_inc)(
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

@register_opt()
@local_optimizer([])
def local_gpu_rebroadcast(node):
    '''rebroadcast(host_from_gpu(x)) -> host_from_gpu(rebroadcast(x))'''
    if isinstance(node.op, tensor.Rebroadcast):
        x, = node.inputs
        if (x.owner and x.owner.op == host_from_gpu):
            gpu_x = x.owner.inputs[0]
            return [host_from_gpu(node.op(gpu_x))]


def gpu_print_wrapper(op, cnda):
    op.old_op.global_fn(op.old_op, numpy.asarray(cnda))

@register_opt()
@local_optimizer([])
def local_gpu_print_op(node):
    if isinstance(node.op, tensor.printing.Print):
        x, = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            new_op = node.op.__class__(global_fn=gpu_print_wrapper)
            new_op.old_op = node.op
            return [host_from_gpu(new_op(gpu_x))]
    return False


@register_opt()
@local_optimizer([tensor.TensorDot])
def local_gpu_tensordot(node):
    '''
    T.tensordot(host_from_gpu) -> basic_ops.tensordot(host_from_gpu)

    There is no Cuda Op for tensordot, however we can build a chain of
    CPU Ops implementing tensordot. These Ops all have a GPU equivalent.

    Note: applying this optimization at that stage is not ideal, because
    all blas-related optimizations have already been applied.
    However, if we want to apply it before the blas optimizations, then
    we don't know which variables may end up on the GPU or not.
    '''
    if (isinstance(node.op, tensor.TensorDot) and
            node.outputs[0].dtype == 'float32'):
        x, y = node.inputs
        if ((x.owner and
                x.owner.op == host_from_gpu and
                y.dtype=='float32') or
            (y.owner and
                y.owner.op == host_from_gpu and
                x.dtype=='float32')):

            axes = node.op.axes
            out = tensordot(x, y, axes=axes)
            return [out]


def cast(x, dtype):
    stype = scal.Scalar(dtype)
    cast_op = theano.tensor.Elemwise(scal.Identity(scal.specific_out(stype)))
    return cast_op(x)

import theano.tensor.nnet
@register_opt()
@local_optimizer([])
def local_gpu_crossentorpy_softmax_argmax_1hot_with_bias(node):
    if isinstance(node.op, tensor.nnet.CrossentropySoftmaxArgmax1HotWithBias):
        x,b,y = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            # if y is a cast to integers, we can go to the underlying thing if we want,
            # since this gpu op will cast to integers internally anyway
            int_cast_ops = (
                    tensor.basic._convert_to_int32,
                    tensor.basic._convert_to_int8,
                    tensor.basic._convert_to_int16,
                    tensor.basic._convert_to_int64,
                    )
            while y.owner and y.owner.op in int_cast_ops:
                y = y.owner.inputs[0]
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

@register_opt()
@local_optimizer([])
def local_gpu_softmax(node):
    if isinstance(node.op, tensor.nnet.Softmax):
        x, = node.inputs
        if x.owner and x.owner.op == host_from_gpu:
            gpu_x, = x.owner.inputs
            gpu_sm = GpuSoftmax()(gpu_x)
            return [host_from_gpu(gpu_sm)]
    return False

@register_opt()
@local_optimizer([])
def local_gpu_softmax_with_bias(node):
    if isinstance(node.op, tensor.nnet.SoftmaxWithBias):
        x, b = node.inputs
        x_on_gpu = x.owner and x.owner.op == host_from_gpu
        b_on_gpu = b.owner and b.owner.op == host_from_gpu
        if x_on_gpu or b_on_gpu:
            gpu_sm = GpuSoftmaxWithBias()(gpu_from_host(x), gpu_from_host(b))
            return [host_from_gpu(gpu_sm)]
    return False

#### Convolution, maxpooling
from theano.tensor.nnet import conv
@register_opt()
@local_optimizer([])
def local_gpu_conv(node):
    """
    gpu_from_host(conv) -> gpu_conv(gpu_from_host)

    conv(host_from_gpu) -> host_from_gpu(gpu_conv)
    """
    def GpuConvOp_from_ConvOp(op):
        logical_img_hw=None
        if op.imshp_logical is not None:
            logical_img_hw=op.imshp_logical[1:3]
            if logical_img_hw != op.imshp[1:3]:
                # this case is not implemented
                return None
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
        #HACK to print the number of MFlops in the profiler output.
        if hasattr(op,'flops'):
            ret.flops=op.flops
        return ret


    if node.op == gpu_from_host:
        #gpu_from_host(conv) -> gpu_conv(gpu_from_host)
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, conv.ConvOp):
            gpu_conv = GpuConvOp_from_ConvOp(host_input.owner.op)
            if gpu_conv is None:
                return
            img, kern = host_input.owner.inputs
            #in some case the ConvOp broadcast the last 2 dimensions differently then the gpu ConvOp
            return [tensor.patternbroadcast(gpu_conv(gpu_from_host(img), gpu_from_host(kern)),
                                            node.outputs[0].broadcastable)]

    if isinstance(node.op, conv.ConvOp):
        #conv(host_from_gpu) -> host_from_gpu(gpu_conv)
        img, kern = node.inputs
        img_on_gpu = (img.owner and img.owner.op == host_from_gpu)
        kern_on_gpu = (kern.owner and kern.owner.op == host_from_gpu)
        if img_on_gpu or kern_on_gpu:
            gpu_conv = GpuConvOp_from_ConvOp(node.op)
            if gpu_conv is None:
                return
            #in some case the ConvOp broadcast the last 2 dimensions differently then the gpu ConvOp
            return [tensor.patternbroadcast(host_from_gpu(gpu_conv(gpu_from_host(img),
                                                                   gpu_from_host(kern))),
                                            node.outputs[0].broadcastable)]

import theano.tensor.signal.downsample as downsample
@register_opt()
@local_optimizer([])
def local_gpu_downsample_factor_max(node):
    if isinstance(node.op, downsample.DownsampleFactorMax):
        x, = node.inputs
        if (x.owner and x.owner.op == host_from_gpu):
            gpu_ds = GpuDownsampleFactorMax(node.op.ds, node.op.ignore_border)
            return [host_from_gpu(gpu_ds(x.owner.inputs[0]))]

@register_opt()
@local_optimizer([])
def local_gpu_downsample_factor_max_grad(node):
    if isinstance(node.op, downsample.DownsampleFactorMaxGrad):
        x,z,gz = node.inputs
        if (x.owner and x.owner.op == host_from_gpu):
            gpu_ds_grad = GpuDownsampleFactorMaxGrad(node.op.ds, node.op.ignore_border)
            return [host_from_gpu(gpu_ds_grad(x.owner.inputs[0], gpu_from_host(z), gpu_from_host(gz)))]



from theano.sandbox.cuda.basic_ops import gpu_join

@register_opt()
@local_optimizer([])
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


    For intermediate places in the graph not covered by the first opt, the following could be useful:

    gpu_from_host(join) -> gpu_join(gpu_from_host)

    not implemented yet.

    """
    if isinstance(node.op, tensor.Join):
        # optimizing this case:
        # join(host_from_gpu) -> host_from_gpu(gpu_join)

        # print "OPT: we've got a Join instance"

        axis_and_tensors = node.inputs

        #print "OPT: axis_and_tensors=", axis_and_tensors

        matches = [(not t.owner is None and t.owner.op == host_from_gpu) or
                   isinstance(t, gof.Constant) for t in axis_and_tensors[1:]]
        #print "OPT: matches =", matches

        # if all input tensors are host_from_gpu'ified
        if numpy.all(matches):
            # the extra gpu_from_host introduced here will
            # be removed by further optimizations
            new_tensors = [gpu_from_host(t) for t in axis_and_tensors[1:]]
            new_a_and_t = [axis_and_tensors[0]]+new_tensors

            replacement_node = host_from_gpu(gpu_join(*new_a_and_t))

            # print "OPT: replacement_node", replacement_node

            return [replacement_node]


#Commented out because it can result in shared = dimshuffle(gemm_inplace(dimshuffle(shared)))
#which causes memory leaks (long term fix is to make the above not leak memory)
@local_optimizer([gpu_gemm_no_inplace])
def local_inplace_gemm(node):
    if node.op == gpu_gemm_no_inplace:
        return [gpu_gemm_inplace(*node.inputs)]


@local_optimizer([gpu_gemv_no_inplace])
def local_inplace_gemv(node):
    if node.op == gpu_gemv_no_inplace:
        return [gpu_gemv_inplace(*node.inputs)]


@local_optimizer([gpu_gemm_no_inplace])
def local_inplace_ger(node):
    if node.op == gpu_ger_no_inplace:
        return [gpu_ger_inplace(*node.inputs)]

# After destroyhandler is in but before we try to make elemwise things inplace
# Try to make gpu gemm inplace
# Also, need to make the gemm optimisation(step 70) happen before the fusion of
# elemwise(step 71)
optdb.register('InplaceGpuBlasOpt',
        EquilibriumOptimizer([local_inplace_gemm,
                              local_inplace_gemv,
                              local_inplace_ger,
                              ],
                            failure_callback=EquilibriumOptimizer.warn_inplace,
            max_use_ratio=5),
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

        t = cuda_ndarray.cuda_ndarray.ptr_int_size()
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

    argument_limit = 232  # some bytes are used for block and thread coords etc.
    ndim = node.inputs[0].type.ndim
    size_param_mandatory = int_size #for numels
    size_param_mandatory += int_size *  ndim # for the shape
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
        if max_nb_inputs<=1 and len(node.inputs)>1:
            return False
        while len(node.inputs)>max_nb_inputs:
            inner_op = []
            for i in xrange(0,len(node.inputs),max_nb_inputs):
                inner_op.append(node.op(*node.inputs[i:i+max_nb_inputs]))
            node = node.op(*inner_op).owner
    return node

#GpuElemwise fusion
gpu_local_elemwise_fusion = tensor.opt.local_elemwise_fusion_op(
        GpuElemwise,
        max_inputs_to_GpuElemwise)
if config.gpu.local_elemwise_fusion:
    _logger.debug("enabling optimization fusion of gpu elemwise in fast_run")
    optdb.register('gpu_elemwise_fusion',
                   tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion),
                   71.00, 'fast_run', 'fusion',
                   'local_elemwise_fusion','gpu')
else:
    _logger.debug("not enabling optimization fusion of gpu elemwise in fast_run")
    optdb.register('gpu_elemwise_fusion',
                   tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion),
                   71.00, 'fusion', 'local_elemwise_fusion')

#GpuElemwise inplace
gpu_inplace_elemwise_optimizer = tensor.opt.inplace_elemwise_optimizer_op(
        GpuElemwise)
optdb.register('gpu_inplace_elemwise_opt', gpu_inplace_elemwise_optimizer, 75,
               'fast_run', 'inplace','gpu_inplace', 'gpu')

@register_opt()
@local_optimizer([tensor.Alloc])
def local_gpualloc(node):
    replace=False
    if node.op == tensor.alloc:
        if node.inputs[0].owner and node.inputs[0].owner.op==host_from_gpu:#if the input was on the gpu
            replace = True
        if all([c!='output' and c.op == gpu_from_host for c,idx in node.outputs[0].clients]):#if all clients are on gpu
            replace=True
        if all([c!='output' and c.op == tensor.join and all([i.owner and i.owner.op in [host_from_gpu,tensor.alloc] for i in c.inputs[1:]]) for c,idx in node.outputs[0].clients]):#if the client is a subtensor with input on gpu or alloc
            replace=True
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
            for b_old,b_new in zip(old_out.type.broadcastable, new_out.type.broadcastable):
                assert b_new or (not b_old)
            new_out = tensor.patternbroadcast(new_out, old_out.broadcastable)
        #if old_out.type != new_out.type:
            #import pdb; pdb.set_trace()
        return [new_out]




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



def gpu_safe_new(x, tag = ''):
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

def gpu_reconstruct_graph(inputs, outputs, tag = None):
    """
    Different interface to clone, that allows you to pass inputs.
    Compared to clone, this method always replaces the inputs with
    new variables of the same type, and returns those ( in the same
    order as the original inputs).
    """
    if tag is None:
        tag = ''
    nw_inputs = [gpu_safe_new(x,tag) for x in inputs]
    givens = {}
    for nw_x, x in zip(nw_inputs, inputs):
        givens[x] = nw_x
    nw_outputs = scan_utils.clone( outputs, replace=givens)
    return (nw_inputs, nw_outputs)


def tensor_to_cuda(x):
    if (isinstance(x.type, tensor.TensorType) and
        x.type.dtype == 'float32'):
        y = CudaNdarrayType( broadcastable = x.type.broadcastable)()
        if x.name :
            y.name = x.name +'[cuda]'
        return y
    else:
        return x


@register_opt('scan')
@local_optimizer([])
def gpuScanOptimization(node):
    """
    scan(host_from_gpu) -> host_from_gpu(GPUscan)
    gpu_from_host(scan) -> GPUscan(gpu_from_host)
    """

    #gpu_from_host(scan) -> GPUscan(gpu_from_host)
    if node.op == gpu_from_host:
        host_input = node.inputs[0]
        if (host_input.owner and
            isinstance(host_input.owner.op, scan_op.Scan) and
            not host_input.owner.op.info['gpu'] and
            len(host_input.owner.outputs) == 1 ):
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
            info = thescan.info.copy()
            info['gpu'] = True
            inputs = host_input.owner.inputs
            nw_ins = [ inputs[0]]
            e = ( 1+ thescan.n_seqs
                 + thescan.n_mit_mot
                 + thescan.n_mit_sot
                 + thescan.n_sit_sot
                 + thescan.n_shared_outs)
            nw_ins += [safe_to_gpu(x) for x in inputs[1:e] ]
            b = e
            e = e + thescan.n_nit_sot
            nw_ins += inputs[b:e]
            nw_ins += [safe_to_gpu(x) for x in inputs[e:] ]
            scan_ins = [ tensor_to_cuda(x) for x in thescan.inputs]
            scan_outs = [ safe_to_gpu(x) for x in thescan.outputs ]
            scan_outs = scan_utils.clone(
                scan_outs
                , replace = zip(thescan.inputs,
                                [safe_to_cpu(x) for x in  scan_ins]))
            # We need to construct the hash here, because scan
            # __init__ does not know about cuda ndarray and can not
            # handle graphs with inputs being Cuda Ndarrays
            tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins,
                                                       scan_outs)
            local_env = gof.Env(tmp_in, tmp_out)
            _cmodule_key = gof.CLinker.cmodule_key_(local_env,[])
            info['gpu_hash'] = hash(_cmodule_key)

            typeConstructor = lambda broadcastable, dtype: CudaNdarrayType(
                    broadcastable = broadcastable)
            nw_op = scan_op.Scan( scan_ins
                                 , scan_outs
                                 , info
                                 , typeConstructor = typeConstructor
                                ).make_node(*nw_ins)
            _outputs = nw_op.outputs
            return _outputs

    #scan(host_from_gpu) -> host_from_gpu(GPUscan)
    if (type(node.op) == scan_op.Scan
        and not node.op.info['gpu']):
        if numpy.any([(i.owner and i.owner.op == host_from_gpu)
                      for i in node.inputs]):

            thescan = node.op
            info = thescan.info.copy()
            info['gpu'] = True
            inputs = node.inputs
            nw_ins = [ inputs[0]]
            e = ( 1+ thescan.n_seqs
                 + thescan.n_mit_mot
                 + thescan.n_mit_sot
                 + thescan.n_sit_sot
                 + thescan.n_shared_outs)
            nw_ins += [safe_to_gpu(x) for x in inputs[1:e] ]
            b = e
            e = e + thescan.n_nit_sot
            nw_ins += inputs[b:e]
            nw_ins += [safe_to_gpu(x) for x in inputs[e:] ]

            scan_ins = [ tensor_to_cuda(x) for x in thescan.inputs]
            scan_outs = [ safe_to_gpu(x) for x in thescan.outputs ]
            scan_outs = scan_utils.clone(
                scan_outs
                , replace = zip(thescan.inputs
                                ,[safe_to_cpu(x) for x in  scan_ins]))

            # We need to construct the hash here, because scan
            # __init__ does not know about cuda ndarray and can not
            # handle graphs with inputs being Cuda Ndarrays
            tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins,
                                                       scan_outs)
            local_env = gof.Env(tmp_in, tmp_out)
            _cmodule_key = gof.CLinker.cmodule_key_(local_env,[])
            info['gpu_hash'] = hash(_cmodule_key)
            typeConstructor = lambda broadcastable, dtype: CudaNdarrayType(
                    broadcastable = broadcastable)
            _outputs = scan_op.Scan(
                    scan_ins
                    , scan_outs
                    , info
                    , typeConstructor = typeConstructor
                    ).make_node(*nw_ins).outputs
            outputs = []
            for x,y in zip(_outputs, node.outputs):
                if isinstance(y.type, CudaNdarrayType):
                    outputs += [x]
                else:
                    outputs += [safe_to_cpu(x)]
            return outputs
    return False

@gof.local_optimizer([None])
def gpu_scan_make_inplace(node):
    op = node.op
    if ( isinstance(op, scan_op.Scan) and
        (not op.info['inplace']) and
        (op.info['gpu'])):
        info = op.info.copy()
        info['inplace'] = True
        # inputs corresponding to sequences and n_steps
        ls_begin = node.inputs[:1+op.n_seqs]
        ls  = op.outer_mitmot(node)
        ls += op.outer_mitsot(node)
        ls += op.outer_sitsot(node)
        ls_end  = op.outer_shared(node)
        ls_end += op.outer_nitsot(node)
        ls_end += op.outer_non_seqs(node)
        n_outs = len(ls)
        for idx in xrange(n_outs):
            if ls[idx] in ls[:idx]:
                ls[idx] = compile.function_module.deep_copy_op(ls[idx])

        inputs = ls_begin + ls + ls_end

        typeConstructor = lambda broadcastable, dtype: CudaNdarrayType(
                broadcastable = broadcastable)
        new_op = scan_op.Scan( op.inputs
                              , op.outputs
                              , info
                              , typeConstructor = typeConstructor
                             )
        return new_op.make_node(*inputs).outputs
    return False

optdb.register( 'gpu_scanOp_make_inplace'
               , theano.tensor.opt.in2out(gpu_scan_make_inplace,ignore_newtrees=True)
               , 75
               , 'gpu'
               , 'fast_run'
               , 'inplace'
               , 'scan')
