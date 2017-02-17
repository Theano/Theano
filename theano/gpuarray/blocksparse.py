from __future__ import absolute_import, print_function, division
import logging
import os

import numpy as np
from theano import Apply, tensor
from theano.gof import COp
from theano.tensor import discrete_dtypes, as_tensor_variable

from theano.gradient import grad_undefined

from .type import gpu_context_type
from .basic_ops import as_gpuarray_variable, infer_context_name

_logger = logging.getLogger('theano.gpuarray.blocksparse')


class GpuSparseBlockGemv(COp):
    """
    GPU version of SparseBlockGemv. Check SparseBlockGemv's docstring for more
    information.

    This should not be directly called since the interface is subject
    to change without notice.  Use the sandbox.blocksparse.sparse_block_dot()
    function for a stable interface.
    """
    __props__ = ('inplace',)
    params_type = gpu_context_type

    def __init__(self, inplace=False):
        COp.__init__(self, "blockgemv.c", "APPLY_SPECIFIC(blockgemv)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def get_params(self, node):
        return node.inputs[0].type.context

    def get_op_params(self):
        if self.inplace:
            return [('INPLACE', '1')]
        else:
            return []

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_headers(self):
        return ['<gpuarray/buffer_blas.h>', '<gpuarray/buffer.h>',
                '<gpuarray_helper.h>']

    def make_node(self, o, W, h, inputIdx, outputIdx):
        ctx = infer_context_name(o, W, h)
        o = as_gpuarray_variable(o, ctx)
        W = as_gpuarray_variable(W, ctx)
        h = as_gpuarray_variable(h, ctx)
        inputIdx = as_tensor_variable(inputIdx)
        outputIdx = as_tensor_variable(outputIdx)
        assert o.ndim == 3
        assert W.ndim == 4
        assert h.ndim == 3
        assert inputIdx.ndim == 2
        assert outputIdx.ndim == 2

        assert inputIdx.type.dtype in discrete_dtypes
        assert outputIdx.type.dtype in discrete_dtypes

        return Apply(self, [o, W, h, inputIdx, outputIdx],
                     [o.type()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        Wgrad = gpu_sparse_block_outer(W.zeros_like(),
                                       h, go, inputIdx, outputIdx)
        hgrad = gpu_sparse_block_gemv(h.zeros_like(),
                                      W.dimshuffle((1, 0, 3, 2)),
                                      go,
                                      outputIdx, inputIdx)
        return [go, Wgrad, hgrad,
                grad_undefined(self, 3, inputIdx,
                               "grad of inputIdx makes no sense"),
                grad_undefined(self, 4, outputIdx,
                               "grad of outputIdx makes no sense")]


gpu_sparse_block_gemv = GpuSparseBlockGemv(False)
gpu_sparse_block_gemv_inplace = GpuSparseBlockGemv(True)


class GpuSparseBlockOuter(COp):
    """
    GPU version of SparseBlockOuter. See SparseBlockOuter's docstring for more
    information.

    This op should not be called directly since its interface is
    subject to change without notice.  It is involved in the gradient
    of GpuSparseBlockGemv. The gradient is not implemented.
    """
    __props__ = ('inplace',)
    params_type = gpu_context_type

    def __init__(self, inplace=False):
        COp.__init__(self, ["blockger.c"], "APPLY_SPECIFIC(blockger)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def get_params(self, node):
        return node.inputs[0].type.context

    def get_op_params(self):
        if self.inplace:
            return [('INPLACE', '1')]
        else:
            return []

    def make_node(self, o, x, y, xIdx, yIdx, alpha=None):
        ctx = infer_context_name(o, x, y)
        one = tensor.constant(np.asarray(1.0, dtype='float32'))
        o = as_gpuarray_variable(o, ctx)
        x = as_gpuarray_variable(x, ctx)
        y = as_gpuarray_variable(y, ctx)
        xIdx = as_tensor_variable(xIdx)
        yIdx = as_tensor_variable(yIdx)
        if alpha is None:
            alpha = one
        return Apply(self, [o, x, y, xIdx, yIdx, alpha],
                     [o.type()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_headers(self):
        return ['<gpuarray/buffer_blas.h>', '<gpuarray/buffer.h>',
                '<gpuarray_helper.h>']

gpu_sparse_block_outer = GpuSparseBlockOuter(False)
gpu_sparse_block_outer_inplace = GpuSparseBlockOuter(True)
