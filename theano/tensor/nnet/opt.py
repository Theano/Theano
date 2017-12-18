"""
Optimizations addressing the ops in nnet root directory
"""
from __future__ import absolute_import, print_function, division
import theano
from theano import compile, gof
from theano.compile import optdb
from theano.gof import local_optimizer
from theano.gof.opt import copy_stack_trace

from theano.tensor.nnet.corr import (
    CorrMM, CorrMM_gradInputs, CorrMM_gradWeights)
from theano.tensor.nnet.corr3d import (
    Corr3dMM, Corr3dMM_gradInputs, Corr3dMM_gradWeights)
from theano.tensor.nnet.blocksparse import (
    SparseBlockGemv,
    SparseBlockOuter,
    sparse_block_gemv_inplace,
    sparse_block_outer_inplace)
from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs)
from theano.tensor.nnet.abstract_conv import (AbstractConv3d,
                                              AbstractConv3d_gradWeights,
                                              AbstractConv3d_gradInputs)
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.opt import register_specialize_device
from theano.tensor import TensorType
from theano.tensor import opt

# Cpu implementation
from theano.tensor.nnet.conv import conv2d, ConvOp


@gof.local_optimizer([SparseBlockGemv], inplace=True)
def local_inplace_sparse_block_gemv(node):
    """
        SparseBlockGemv(inplace=False) -> SparseBlockGemv(inplace=True)
    """
    if isinstance(node.op, SparseBlockGemv) and not node.op.inplace:
        new_node = sparse_block_gemv_inplace(*node.inputs)
        copy_stack_trace(node.outputs[0], new_node)
        return [new_node]
    return False
compile.optdb.register('local_inplace_sparse_block_gemv',
                       gof.TopoOptimizer(
                           local_inplace_sparse_block_gemv,
                           failure_callback=gof.TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([SparseBlockOuter], inplace=True)
def local_inplace_sparse_block_outer(node):
    """
        SparseBlockOuter(inplace=False) -> SparseBlockOuter(inplace=True)
    """
    if isinstance(node.op, SparseBlockOuter) and not node.op.inplace:
        new_node = sparse_block_outer_inplace(*node.inputs)
        copy_stack_trace(node.outputs[0], new_node)
        return [new_node]
    return False
compile.optdb.register('local_inplace_sparse_block_outer',
                       gof.TopoOptimizer(
                           local_inplace_sparse_block_outer,
                           failure_callback=gof.TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


# Conv opts
@local_optimizer([AbstractConv2d])
def local_abstractconv_gemm(node):
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    if theano.config.cxx == "" or node.inputs[0].dtype == 'float16':
        return
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, TensorType) or \
       not isinstance(kern.type, TensorType):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        flip = (slice(None),) * (kern.ndim - 2) + \
            (slice(None, None, -1),) * 2
        kern = kern[flip]
    rval = CorrMM(border_mode=node.op.border_mode,
                  subsample=node.op.subsample,
                  filter_dilation=node.op.filter_dilation,
                  num_groups=node.op.num_groups,
                  unshared=node.op.unshared)(img, kern)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv3d])
def local_abstractconv3d_gemm(node):
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    if theano.config.cxx == "" or node.inputs[0].dtype == 'float16':
        return
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, TensorType) or \
       not isinstance(kern.type, TensorType):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]
    rval = Corr3dMM(border_mode=node.op.border_mode,
                    subsample=node.op.subsample,
                    filter_dilation=node.op.filter_dilation,
                    num_groups=node.op.num_groups)(img, kern)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv2d_gradWeights])
def local_abstractconv_gradweight_gemm(node):
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    if theano.config.cxx == "" or node.inputs[0].dtype == 'float16':
        return
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, TensorType) or \
       not isinstance(topgrad.type, TensorType):
        return None

    rval = CorrMM_gradWeights(border_mode=node.op.border_mode,
                              subsample=node.op.subsample,
                              filter_dilation=node.op.filter_dilation,
                              num_groups=node.op.num_groups,
                              unshared=node.op.unshared)(img, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        flip = (slice(None),) * (rval.ndim - 2) + \
            (slice(None, None, -1),) * 2
        rval = rval[flip]
    rval = theano.tensor.patternbroadcast(rval, node.outputs[0].broadcastable)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv3d_gradWeights])
def local_abstractconv3d_gradweight_gemm(node):
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    if theano.config.cxx == "" or node.inputs[0].dtype == 'float16':
        return
    if not isinstance(node.op, AbstractConv3d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, TensorType) or \
       not isinstance(topgrad.type, TensorType):
        return None

    rval = Corr3dMM_gradWeights(border_mode=node.op.border_mode,
                                subsample=node.op.subsample,
                                filter_dilation=node.op.filter_dilation,
                                num_groups=node.op.num_groups)(img, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        rval = rval[:, :, ::-1, ::-1, ::-1]
    rval = theano.tensor.patternbroadcast(rval, node.outputs[0].broadcastable)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_gemm(node):
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    if theano.config.cxx == "" or node.inputs[0].dtype == 'float16':
        return
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, TensorType) or \
       not isinstance(topgrad.type, TensorType):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        flip = (slice(None),) * (kern.ndim - 2) + \
            (slice(None, None, -1),) * 2
        kern = kern[flip]
    rval = CorrMM_gradInputs(border_mode=node.op.border_mode,
                             subsample=node.op.subsample,
                             filter_dilation=node.op.filter_dilation,
                             num_groups=node.op.num_groups,
                             unshared=node.op.unshared)(kern, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv3d_gradInputs])
def local_abstractconv3d_gradinputs_gemm(node):
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    if theano.config.cxx == "" or node.inputs[0].dtype == 'float16':
        return
    if not isinstance(node.op, AbstractConv3d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, TensorType) or \
       not isinstance(topgrad.type, TensorType):
        return None

    # need to flip the kernel if necessary
    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]
    rval = Corr3dMM_gradInputs(border_mode=node.op.border_mode,
                               subsample=node.op.subsample,
                               filter_dilation=node.op.filter_dilation,
                               num_groups=node.op.num_groups)(kern, topgrad,
                                                              shape)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv2d])
def local_conv2d_cpu(node):

    if (not isinstance(node.op, AbstractConv2d) or
            node.inputs[0].dtype == 'float16'):
        return None

    img, kern = node.inputs
    if ((not isinstance(img.type, TensorType) or
         not isinstance(kern.type, TensorType))):
        return None
    if node.op.border_mode not in ['full', 'valid']:
        return None
    if not node.op.filter_flip:
        # Not tested yet
        return None
    if node.op.num_groups > 1 or node.op.unshared:
        return None
    if node.op.filter_dilation != (1, 1):
        return None

    rval = conv2d(img, kern,
                  node.op.imshp, node.op.kshp,
                  border_mode=node.op.border_mode,
                  subsample=node.op.subsample)

    copy_stack_trace(node.outputs[0], rval)
    return [rval]


@local_optimizer([AbstractConv2d_gradWeights])
def local_conv2d_gradweight_cpu(node):
    if (not isinstance(node.op, AbstractConv2d_gradWeights) or
            node.inputs[0].dtype == 'float16'):
        return None

    img, topgrad, shape = node.inputs

    if ((not isinstance(img.type, TensorType) or
         not isinstance(topgrad.type, TensorType))):
        return None
    if node.op.border_mode not in ['full', 'valid']:
        return None
    if not node.op.filter_flip:
        # Not tested yet
        return
    if node.op.num_groups > 1 or node.op.unshared:
        return None

    if node.op.border_mode == 'valid' and \
            (node.op.subsample != (1, 1)):
        return None

    dx, dy = node.op.subsample
    if dx not in (1, 2) or dy not in (1, 2):
        # Not implemented in the gradient of ConvOp
        return None

    if node.op.imshp is None:
        op_imshp = (None, None, None, None)
    else:
        op_imshp = node.op.imshp

    if node.op.kshp is None:
        op_kshp = (None, None, None, None)
    else:
        op_kshp = node.op.kshp

    if None in op_imshp or None in op_kshp:
        if (dx, dy) != (1, 1):
            # We cannot infer the shapes
            return None

    # Determine gradient on kernels
    assert len(op_imshp) == 4 and len(op_kshp) == 4

    outshp = get_conv_output_shape(op_imshp, op_kshp,
                                   node.op.border_mode,
                                   node.op.subsample,
                                   node.op.filter_dilation)[2:]
    fulloutshp = get_conv_output_shape(op_imshp, op_kshp,
                                       node.op.border_mode, (1, 1))[2:]

    newimg = img.dimshuffle((1, 0, 2, 3))
    newtopgrad = topgrad.dimshuffle((1, 0, 2, 3))

    if node.op.border_mode == 'valid':
        (img, filters) = (newimg, newtopgrad)
        kshp_logical = fulloutshp
        kshp_logical_top_aligned = False
        imshp_logical = None
        (bsize, nkern) = (op_imshp[1], op_kshp[0])
        imshp = (op_imshp[0], op_imshp[2], op_imshp[3])
        kshp = outshp
    elif node.op.border_mode == 'full':
        (img, filters) = (newtopgrad, newimg)
        kshp_logical = None
        kshp_logical_top_aligned = True
        imshp_logical = (op_imshp[0],
                         fulloutshp[0],
                         fulloutshp[1])
        (bsize, nkern) = (op_kshp[0], op_imshp[1])
        imshp = (op_imshp[0], outshp[0], outshp[1])
        kshp = op_imshp[2:]
    else:
        raise NotImplementedError(
            'Only [full,valid] modes are currently supported.')

    # Flip the kernels
    filters = filters[:, :, ::-1, ::-1]

    dw = ConvOp(imshp, kshp, nkern, bsize, 1, 1, output_mode='valid',
                unroll_batch=None, unroll_kern=None, unroll_patch=None,
                imshp_logical=imshp_logical,
                kshp_logical=kshp_logical,
                kshp_logical_top_aligned=kshp_logical_top_aligned,
                direction_hint='bprop weights')
    res = dw(img, filters)
    copy_stack_trace(node.outputs[0], res)

    if node.op.border_mode == 'valid':
        res = res.dimshuffle((1, 0, 2, 3))
        res = res[:, :, ::-1, ::-1]

    res = theano.tensor.patternbroadcast(res, node.outputs[0].broadcastable)

    copy_stack_trace(node.outputs[0], res)
    return [res]


@local_optimizer([AbstractConv2d_gradInputs])
def local_conv2d_gradinputs_cpu(node):
    if (not isinstance(node.op, AbstractConv2d_gradInputs) or
            node.inputs[0].dtype == 'float16'):
        return None

    kern, topgrad, shape = node.inputs

    if ((not isinstance(kern.type, TensorType) or
         not isinstance(topgrad.type, TensorType))):
        return None
    if node.op.border_mode not in ['full', 'valid']:
        return None
    if not node.op.filter_flip:
        # Not tested yet
        return None
    if node.op.num_groups > 1 or node.op.unshared:
        return None

    # Conv 3d implementation, needed when subsample > 2
    if node.op.border_mode == 'valid' and node.op.subsample != (1, 1):
        # The op don't support that anymore.
        return False

    # Conv2d Implementation
    dx, dy = node.op.subsample
    if dx not in (1, 2) or dy not in (1, 2):
        # Not implemented in the gradient of ConvOp
        return None

    if node.op.imshp is None:
        op_imshp = (None, None, None, None)
    else:
        op_imshp = node.op.imshp

    if node.op.kshp is None:
        op_kshp = (None, None, None, None)
    else:
        op_kshp = node.op.kshp

    if None in op_imshp or None in op_kshp:
        if (dx, dy) != (1, 1):
            return None

    mode = 'valid'
    if not node.op.border_mode == 'full':
        mode = 'full'
    filters = kern.dimshuffle((1, 0, 2, 3))
    filters = filters[:, :, ::-1, ::-1]

    outshp = get_conv_output_shape(op_imshp, op_kshp,
                                   node.op.border_mode,
                                   node.op.subsample,
                                   node.op.filter_dilation)[2:]
    fulloutshp = get_conv_output_shape(op_imshp, op_kshp,
                                       node.op.border_mode, (1, 1))[2:]

    nkern = op_imshp[1]
    imshp = (op_kshp[0], outshp[0], outshp[1])
    imshp_logical = (op_kshp[0], fulloutshp[0], fulloutshp[1])
    din = ConvOp(imshp,
                 op_kshp[2:],
                 nkern,
                 op_imshp[0],
                 1, 1, output_mode=mode,
                 unroll_batch=None, unroll_kern=None,
                 unroll_patch=None,
                 imshp_logical=imshp_logical,
                 kshp_logical=None,
                 version=-1,
                 direction_hint='bprop inputs')
    din = din(topgrad, filters)
    copy_stack_trace(node.outputs[0], din)
    din = theano.tensor.patternbroadcast(din, node.outputs[0].broadcastable)
    copy_stack_trace(node.outputs[0], din)
    return [din]


# Register Cpu Optmization
conv_groupopt = theano.gof.optdb.LocalGroupDB()
conv_groupopt.__name__ = "conv_opts"
register_specialize_device(conv_groupopt, 'fast_compile', 'fast_run')

# GEMM-based convolution
# It can be disabled by excluding 'conv_gemm'.
conv_groupopt.register('local_abstractconv_gemm', local_abstractconv_gemm, 30,
                       'conv_gemm', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv_gradweight_gemm',
                       local_abstractconv_gradweight_gemm, 30,
                       'conv_gemm', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv_gradinputs_gemm',
                       local_abstractconv_gradinputs_gemm, 30,
                       'conv_gemm', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv3d_gemm', local_abstractconv3d_gemm, 30,
                       'conv_gemm', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv3d_gradweight_gemm',
                       local_abstractconv3d_gradweight_gemm, 30,
                       'conv_gemm', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv3d_gradinputs_gemm',
                       local_abstractconv3d_gradinputs_gemm, 30,
                       'conv_gemm', 'fast_compile', 'fast_run')

# Legacy convolution
conv_groupopt.register('local_conv2d_cpu', local_conv2d_cpu, 40,
                       'fast_compile', 'fast_run')
conv_groupopt.register('local_conv2d_gradweight_cpu',
                       local_conv2d_gradweight_cpu, 40,
                       'fast_compile', 'fast_run')
conv_groupopt.register('local_conv2d_gradinputs_cpu',
                       local_conv2d_gradinputs_cpu, 40,
                       'fast_compile', 'fast_run')


# Verify that no AbstractConv are present in the graph
@local_optimizer([AbstractConv2d,
                  AbstractConv2d_gradWeights,
                  AbstractConv2d_gradInputs,
                  AbstractConv3d,
                  AbstractConv3d_gradWeights,
                  AbstractConv3d_gradInputs])
def local_abstractconv_check(node):
    if isinstance(node.op, (AbstractConv2d,
                            AbstractConv2d_gradWeights,
                            AbstractConv2d_gradInputs,
                            AbstractConv3d,
                            AbstractConv3d_gradWeights,
                            AbstractConv3d_gradInputs)):
        raise gof.opt.LocalMetaOptimizerSkipAssertionError(
            '%s Theano optimization failed: there is no implementation '
            'available supporting the requested options. Did you exclude '
            'both "conv_dnn" and "conv_gemm" from the optimizer? If on GPU, '
            'is cuDNN available and does the GPU support it? If on CPU, '
            'do you have a BLAS library installed Theano can link against? '
            'On the CPU we do not support float16.' %
            node.op.__class__.__name__)

optdb.register('AbstractConvCheck',
               opt.in2out(local_abstractconv_check, name="AbstractConvCheck"),
               48.7, 'fast_compile', 'fast_run')
