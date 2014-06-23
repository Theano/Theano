import numpy
import theano
from theano import Apply, tensor

from theano.gradient import grad_undefined, grad_not_implemented

from theano.sandbox.cuda import cuda_available, GpuOp

if cuda_available:
    from theano.sandbox.cuda import (basic_ops, CudaNdarrayType,
                                     CudaNdarray)

import theano.misc.pycuda_init
from theano.misc.pycuda_init import pycuda_available
if pycuda_available:
    import pycuda.gpuarray

try:
    import scikits.cuda
    from scikits.cuda import cublas
    import scikits.cuda.misc
    scikits.cuda.misc.init()
    scikits_cuda_available = True
except ImportError:
    scikits_cuda_available = False


def gemm_batched(Al, Bl, Cl, m, n, k, lda, ldb, ldc,
                 alpha=numpy.float32(1.0), beta=numpy.float32(1.0)):
    assert Al.shape[0] == Bl.shape[0]
    assert Al.shape[0] == Cl.shape[0]

    handle = scikits.cuda.misc._global_cublas_handle

    cublas.cublasSgemmBatched(handle, 'n', 'n', m, n, k, alpha,
                              Bl.gpudata, ldb, Al.gpudata, lda,
                              beta, Cl.gpuadata, ldc,
                              Cl.shape[0])


def gemv(alpha, A, x, beta, y):
    assert A.shape[1] == x.shape[0]
    assert A.shape[0] == y.shape[0]

    handle = scikits.cuda.misc._global_cublas_handle

    cublas.cublasSgemv(handle, 't', A.shape[1], A.shape[0], alpha,
                       A.gpudata, max(A.strides[0], A.strides[1]),
                       x.gpudata, x.strides[0],
                       beta, y.gpudata, y.strides[0])


def ger(alpha, x, y, A):
    assert A.shape[0] == x.shape[0]
    assert A.shape[1] == y.shape[0]

    handle = scikits.cuda.misc._global_cublas_handle

    cublas.cublasSger(handle, A.shape[1], A.shape[0], alpha,
                      y.gpudata, y.strides[0],
                      x.gpudata, x.strides[0],
                      A.gpudata, A.strides[0])


def bptr(a):
    assert (a.ndim == 3 and a.strides[2] == 1)
    return pycuda.gpuarray.arange(a.ptr,
                                  a.ptr + a.shape[0] * a.strides[0] * 4,
                                  a.strides[0] * 4,
                                  dtype=cublas.ctypes.c_void_p)


class SparseBlockGemvSS(GpuOp):
    def __init__(self, inplace):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, o, W, h, inputIdx, outputIdx):
        o = basic_ops.as_cuda_ndarray_variable(o)
        W = basic_ops.as_cuda_ndarray_variable(W)
        h = basic_ops.as_cuda_ndarray_variable(h)
        assert o.ndim == 2
        assert W.ndim == 4
        assert h.ndim == 2
        assert inputIdx.ndim == 1
        assert outputIdx.ndim == 1

        assert 'int' in inputIdx.type.dtype
        assert 'int' in outputIdx.type.dtype

        return Apply(self, [o, W, h, inputIdx, outputIdx],
                     [o.type()])

    def perform(self, node, inputs, outputs):
        o, W, h, inputIdx, outputIdx = inputs
        out = outputs[0]

        if not self.inplace:
            o = o.copy()

        for j in range(o.shape[0]):
            out_id = outputIdx[j]
            for i in range(h.shape[0]):
                inp_id = inputIdx[i]
                gemv(numpy.float32(1.0), W[inp_id, out_id],
                     h[i], numpy.float32(1.0), o[j])

        out[0] = o

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        # might revise that interface to not have a huge output
        Wgrad = sparse_block_outer_ss(W.zeros_like(),
                                      h, go, inputIdx, outputIdx)
        hgrad = sparse_block_gemv_ss(h.zeros_like(),
                                     W.dimshuffle((1, 0, 3, 2)),
                                     go,
                                     outputIdx, inputIdx)
        return [go, Wgrad, hgrad,
                grad_undefined(self, 3, inputIdx,
                               "grad of inputIdx makes no sense"),
                grad_undefined(self, 4, outputIdx,
                               "grad of outputIdx makes no sense")]


sparse_block_gemv_ss = SparseBlockGemvSS(False)


class SparseBlockOuterSS(GpuOp):
    def __init__(self):
        self.inplace = False

    def make_node(self, o, x, y, xIdx, yIdx):
        o = basic_ops.as_cuda_ndarray_variable(o)
        x = basic_ops.as_cuda_ndarray_variable(x)
        y = basic_ops.as_cuda_ndarray_variable(y)
        return Apply(self, [o, x, y, xIdx, yIdx],
                     [o.type()])

    def perform(self, node, inputs, outputs):
        o, x, y, xIdx, yIdx = inputs
        out = outputs[0]

        if not self.inplace:
            o = o.copy()

        for j in range(y.shape[0]):
            out_id = yIdx[j]
            for i in range(x.shape[0]):
                inp_id = xIdx[i]
                ger(numpy.float32(1.0), y[j],
                    x[i], o[inp_id, out_id])

        out[0] = o


sparse_block_outer_ss = SparseBlockOuterSS()

#############################################################
# All code above this line is unused (except for the imports)


def sparse_block_dot_SS(W, h, inputIdx, b, outputIdx):
    """
    var: shape, comment
    W: (iBlocks, oBlocks, iSize, oSize), weight matrix
    h: (iWin, iSize), input from lower layer (sparse)
    inputIdx: (iWin,), indexes of the input blocks
    b: (oBlocks, oSize), bias vector
    outputIdx: (oWin,), indexes of the output blocks

    returns (oBlocks, oSize), dot(W[i, j], h[i]) + b[j]
         but b[j] is only added once
    """
    o = b.take(outputIdx, axis=0)
    def outer_fn(out_id, W, h, b, iIdx):
        def inner_fn(inp_id, h_i, out_id, W):
            return tensor.dot(W[inp_id, out_id], h_i)
        return theano.scan(inner_fn, sequences=[iIdx, h],
                           outputs_info=None,
                           non_sequences=[out_id, W],
                           n_steps=iIdx.shape[0])[0].sum(axis=0) + b[out_id]
    return theano.scan(outer_fn, sequences=[outputIdx],
                       outputs_info=None,
                       non_sequences=[W, h, b, inputIdx],
                       n_steps=outputIdx.shape[0])[0]
