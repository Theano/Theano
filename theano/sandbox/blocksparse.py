import numpy
from collections import defaultdict

import theano
from theano import Op, Apply
import theano.tensor as T
from theano.tensor import discrete_dtypes
from theano.gradient import grad_undefined


class SparseBlockGemv(Op):

    register_opt = defaultdict(list)

    """
    This op computes the dot product of specified pieces of vectors
    and matrices, returning pieces of vectors.
    It computes something like this for each j:
      o[j] = sum_over_i(dot(W[i, j], h[i])) + o[j]
    The i and j are taken from the inputIdx and outputIdx lists
    respectively.
    """
    def __init__(self, inplace=False):
        self.inplace = False

    def make_node(self, o, W, h, inputIdx, outputIdx):
        """
        Compute the dot product (plus bias) of the specified pieces of vectors
        and matrices.
        Parameters
        ----------
        var: shape, comment
        W: (iBlocks, oBlocks, iSize, oSize), weight matrix
        h: (batch, iWin, iSize), input from lower layer (sparse)
        inputIdx: (batch, iWin), indexes of the input blocks
        b: (oBlocks, oSize), bias vector
        outputIdx: (batch, oWin), indexes of the output blocks
        returns (batch, oWin, oSize), dot(W[i, j], h[i]) + b[j]
             but b[j] is only added once
        Notation
        --------
        - `batch` is the number of examples in a minibatch (batch size).
        - `iBlocks` is the total number of blocks in the input (from lower layer).
        - `iSize` is the size of each of these input blocks.
        - `iWin` is the number of blocks that will be used as inputs. Which blocks
          will be used is specified in `inputIdx`.
        - `oBlocks` is the number or possible output blocks.
        - `oSize` is the size of each of these output blocks.
        - `oWin` is the number of output blocks that will actually be computed.
          Which blocks will be computed is specified in `outputIdx`.
        """
        o = theano.tensor.as_tensor_variable(o)
        W = theano.tensor.as_tensor_variable(W)
        h = theano.tensor.as_tensor_variable(h)
        inputIdx = theano.tensor.as_tensor_variable(inputIdx)
        outputIdx = theano.tensor.as_tensor_variable(outputIdx)

        if o.ndim != 3:
            raise TypeError('The output o must be a 3D tensor')
        if W.ndim != 4:
            raise TypeError('The weight matrix W must be a 4D tensor')
        if h.ndim != 3:
            raise TypeError('The input h must be a 3D tensor')
        if inputIdx.ndim != 2:
            raise TypeError('The input indices inputIdx must be a 2D tensor')
        if outputIdx.ndim != 2:
            raise TypeError('The output indices outputIdx must be a 2D tensor')

        assert inputIdx.type.dtype in discrete_dtypes
        assert outputIdx.type.dtype in discrete_dtypes

        return Apply(self, [o, W, h, inputIdx, outputIdx], [o.type()])

    def perform(self, node, inp, out_):
        raise NotImplementedError('Optimization of SparseBlockGemv failed.')

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        Wgrad = sparse_block_outer(W.zeros_like(),
                                      h, go, inputIdx, outputIdx)
        hgrad = sparse_block_gemv(h.zeros_like(),
                                     W.dimshuffle((1, 0, 3, 2)),
                                     go,
                                     outputIdx, inputIdx)
        return [go, Wgrad, hgrad,
                grad_undefined(self, 3, inputIdx,
                               "grad of inputIdx makes no sense"),
                grad_undefined(self, 4, outputIdx,
                               "grad of outputIdx makes no sense")]


sparse_block_gemv = SparseBlockGemv(False)
sparse_block_gemv_inplace = SparseBlockGemv(True)


class SparseBlockOuter(Op):
    """
    This computes the outer product of two sets of pieces of vectors
    updating a full matrix with the results.
    It computes something like this:
      o[i, j] = (alpha * outer(x[i], y[j])) + o[i, j]
    The i and j are taken from the xIdx and yIdx lists respectively.
    This op should not be called directly since its interface is
    subject to change without notice.  It is involved in the gradient
    of SparseBlockGemvSS.
    """
    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, o, x, y, xIdx, yIdx, alpha=None):
        """
            TODO: WRITEME
        """
        one = T.constant(numpy.asarray(1.0, dtype='float32'))
        o = basic_ops.as_cuda_ndarray_variable(o)
        x = basic_ops.as_cuda_ndarray_variable(x)
        y = basic_ops.as_cuda_ndarray_variable(y)
        if alpha is None:
            alpha = one
        return Apply(self, [o, x, y, xIdx, yIdx, alpha],
                     [o.type()])

    def perform(self, node, inp, out_):
        raise NotImplementedError('Optimization of SparseBlockOuter failed.')

    def grad(self, inputs, output_gradients):
        meta_grad_op = MetaGradSparseBlockGemv(output_gradients)
        return [meta_grad_op(inp) for inp in inputs]


sparse_block_outer = SparseBlockOuter(False)
sparse_block_outer_inplace = SparseBlockOuter(True)


def sparse_block_gemv_cpu(W, h, inputIdx, bias, outputIdx):
    """
    Creates a graph for the sparse block dot operation. Check SparseBlockGemv's
    docstring for information about the arguments.
    """

    def _loop_over_batch(b):

        def _loop_over_outputIdx(i):

            def _loop_over_inputIdx(j):
                return T.dot(h[b, j, :], W[inputIdx[b, j], outputIdx[b, i], :, :])

            res3 = theano.scan(fn=_loop_over_inputIdx,
                            sequences=T.arange(0, inputIdx.shape[1]),
                            name='_loop_over_inputIdx')[0]

            return res3.sum(axis=0)

        res2 = theano.scan(fn=_loop_over_outputIdx,
                        sequences=T.arange(0, outputIdx.shape[1]),
                        name='_loop_over_outputIdx')[0]

        return res2

    res1 = theano.scan(fn=_loop_over_batch,
                       sequences=T.arange(0, inputIdx.shape[0]),
                       name='_loop_over_batch')[0]

    return res1 + bias.take(outputIdx, axis=0)


def sparse_block_outer_cpu(x, y, xIdx, yIdx, alpha=None):
    if alpha is None:
        alpha = T.constant(numpy.asarray(1.0, dtype='float32'))

    def _loop_over_batch(b):

        def _loop_over_outputIdx(i):

            def _loop_over_inputIdx(j):
                return T.outer(x[b, i, :], y[b, j, :])

            res3 = theano.scan(fn=_loop_over_inputIdx,
                            sequences=T.arange(0, xIdx.shape[1]),
                            name='_loop_over_inputIdx')[0]

            return res3

        res2 = theano.scan(fn=_loop_over_outputIdx,
                        sequences=T.arange(0, yIdx.shape[1]),
                        name='_loop_over_outputIdx')[0]

        return res2

    res1 = theano.scan(fn=_loop_over_batch,
                       sequences=T.arange(0, xIdx.shape[0]),
                       name='_loop_over_batch')[0]

    return alpha * res1.sum(axis=0)


def sparse_block_dot(W, h, inputIdx, b, outputIdx, inplace=False):
    """
    Compute the dot product (plus bias) of the specified pieces of vectors
    and matrices.
    Parameters
    ----------
    var: shape, comment
    W: (iBlocks, oBlocks, iSize, oSize), weight matrix
    h: (batch, iWin, iSize), input from lower layer (sparse)
    inputIdx: (batch, iWin), indexes of the input blocks
    b: (oBlocks, oSize), bias vector
    outputIdx: (batch, oWin), indexes of the output blocks
    returns (batch, oWin, oSize), dot(W[i, j], h[i]) + b[j]
         but b[j] is only added once
    Notation
    --------
    - `batch` is the number of examples in a minibatch (batch size).
    - `iBlocks` is the total number of blocks in the input (from lower layer).
    - `iSize` is the size of each of these input blocks.
    - `iWin` is the number of blocks that will be used as inputs. Which blocks
      will be used is specified in `inputIdx`.
    - `oBlocks` is the number or possible output blocks.
    - `oSize` is the size of each of these output blocks.
    - `oWin` is the number of output blocks that will actually be computed.
      Which blocks will be computed is specified in `outputIdx`.
    """
    assert inputIdx.ndim == h.ndim - 1
    assert outputIdx.ndim == inputIdx.ndim
    if h.ndim == 2:
        h = h.dimshuffle('x', 0, 1)
        inputIdx = inputIdx.dimshuffle('x', 0)
        outputIdx = outputIdx.dimshuffle('x', 0)
    return SparseBlockGemv(inplace)(b.take(outputIdx, axis=0), W, h,
                                inputIdx, outputIdx)