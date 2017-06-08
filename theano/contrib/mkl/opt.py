from __future__ import absolute_import, print_function, division
import theano
from theano.compile import optdb
from theano.gof import local_optimizer
from theano.contrib.mkl import mkl_optimizer, register_opt, mkl_seqopt, mkl_available
from theano.contrib.mkl.basic_ops import U2IGrad, I2U, I2UGrad, U2IConv
from theano.contrib.mkl import mkl_conv

from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs)

import logging

_logger = logging.getLogger('theano.contrib.mkl.opt')

# global OPT
optdb.register('mkl_opt', mkl_seqopt, 0.09, 'mkl')

# local OPT
mkl_seqopt.register('mkl_local_optimizations', mkl_optimizer, 20,
                    'fast_run', 'fast_compile', 'mkl')


@local_optimizer([AbstractConv2d])
def local_Conv2D_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d):
        return

    if node.op.filter_dilation != (1, 1):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if None in node.op.kshp:
        return

    try:
        image, weight = node.inputs
        image_internal = U2IConv(imshp=node.op.imshp,
                                 kshp=node.op.kshp,
                                 subsample=node.op.subsample,
                                 border_mode=node.op.border_mode,
                                 filter_dilation=node.op.filter_dilation)(image)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation)(image_internal, weight)
        z_user = I2U()(convOut)
        reval = z_user
        return [reval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@local_optimizer([AbstractConv2d_gradInputs])
def local_ConvGradInputs_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if node.op.filter_dilation != (1, 1):
        return

    if None in node.op.kshp:
        return

    try:
        weight, gz, zshp = node.inputs
        image = node.inputs[2].owner.inputs[0].owner.inputs[0]
        image_internal = U2IConv(imshp=node.op.imshp,
                                 kshp=node.op.kshp,
                                 subsample=node.op.subsample,
                                 border_mode=node.op.border_mode,
                                 filter_dilation=node.op.filter_dilation)(image)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation)(image_internal, weight)
        gz_internal = I2UGrad()(convOut, gz)
        gradImage = mkl_conv.ConvGradInputs(border_mode=node.op.border_mode,
                                            subsample=node.op.subsample,
                                            imshp=node.op.imshp,
                                            kshp=node.op.kshp)(image_internal, weight, gz_internal)
        gradImage_user = U2IGrad()(image, gradImage)
        rval = gradImage_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@local_optimizer([AbstractConv2d_gradWeights])
def local_ConvGradWeights_mkl(node):

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if node.op.filter_dilation != (1, 1):
        return

    if None in node.op.kshp:
        return

    try:
        image, gz, zshp = node.inputs
        weight = node.inputs[2].owner.inputs[0].owner.inputs[0]
        image_internal = U2IConv(imshp=node.op.imshp,
                                 kshp=node.op.kshp,
                                 subsample=node.op.subsample,
                                 border_mode=node.op.border_mode,
                                 filter_dilation=node.op.filter_dilation)(image)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation)(image_internal, weight)
        gz_internal = I2UGrad()(convOut, gz)
        gradWeight = mkl_conv.ConvGradWeights(border_mode=node.op.border_mode,
                                              subsample=node.op.subsample,
                                              imshp=node.op.imshp,
                                              kshp=node.op.kshp)(image_internal, weight, gz_internal)
        rval = gradWeight
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


conv_groupopt = theano.gof.optdb.LocalGroupDB()
conv_groupopt.__name__ = "mkl_conv_opts"
register_opt()(conv_groupopt)

# MKL-based convolution, using the same group with theano.tensor.nnet.opt to avoid dumlicating GEMM functions
# It can be disabled by excluding 'conv_mkl'.
conv_groupopt.register('local_Conv2D_mkl', local_Conv2D_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_ConvGradInputs_mkl', local_ConvGradInputs_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_ConvGradWeights_mkl', local_ConvGradWeights_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
