from __future__ import absolute_import, print_function, division
import numpy as np

import theano
from theano import Op, Apply
from theano.tensor import discrete_dtypes
from theano.gradient import grad_undefined


class SparseBlockGemv(Op):
    """
    This op computes the dot product of specified pieces of vectors
    and matrices, returning pieces of vectors::

        for b in range(batch_size):
            for j in range(o.shape[1]):
                for i in range(h.shape[1]):
                    o[b, j, :] += numpy.dot(h[b, i], W[iIdx[b, i], oIdx[b, j]])

    where b, h, W, o iIdx, oIdx are defined in the docstring of make_node.

    .. image:: ../../../images/blocksparse.png
        :scale: 50 %

    """
    __props__ = ('inplace',)

    registered_opts = []

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, o, W, h, inputIdx, outputIdx):
        """
        Compute the dot product of the specified pieces of vectors
        and matrices.

        The parameter types are actually their expected shapes
        relative to each other.

        Parameters
        ----------
        o : batch, oWin, oSize
            output vector
        W : iBlocks, oBlocks, iSize, oSize
            weight matrix
        h : batch, iWin, iSize
            input from lower layer (sparse)
        inputIdx : batch, iWin
            indexes of the input blocks
        outputIdx : batch, oWin
            indexes of the output blocks

        Returns
        -------
        (batch, oWin, oSize)
            dot(W[i, j], h[i]) + o[j]

        Notes
        -----
        - `batch` is the number of examples in a minibatch (batch size).
        - `iBlocks` is the total number of blocks in the input (from lower
            layer).
        - `iSize` is the size of each of these input blocks.
        - `iWin` is the number of blocks that will be used as inputs. Which
           blocks will be used is specified in `inputIdx`.
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
            raise TypeError('The output o must be a 2D tensor')
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
        o, W, h, iIdx, oIdx = inp[:5]

        if not self.inplace:
            o = o.copy()

        for b in range(o.shape[0]):
            for j in range(o.shape[1]):
                outputIdx = oIdx[b, j]
                for i in range(h.shape[1]):
                    inputIdx = iIdx[b, i]
                    w = W[inputIdx, outputIdx]
                    o[b, j, :] += np.dot(h[b, i], w)
        out_[0][0] = o

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        outer_fun = SparseBlockOuter(self.inplace)
        gemv_fun = SparseBlockGemv(self.inplace)

        Wgrad = outer_fun(W.zeros_like(), h, go, inputIdx, outputIdx)
        hgrad = gemv_fun(h.zeros_like(), W.dimshuffle((1, 0, 3, 2)),
                         go, outputIdx, inputIdx)
        return [go, Wgrad, hgrad,
                grad_undefined(self, 3, inputIdx,
                               "grad of inputIdx makes no sense"),
                grad_undefined(self, 4, outputIdx,
                               "grad of outputIdx makes no sense")]


class SparseBlockOuter(Op):
    """
    This computes the outer product of two sets of pieces of vectors
    updating a full matrix with the results::

        for b in range(batch_size):
            o[xIdx[b, i], yIdx[b, j]] += (alpha * outer(x[b, i], y[b, j]))

    This op is involved in the gradient of SparseBlockGemv.

    """
    __props__ = ('inplace',)

    registered_opts = []

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, o, x, y, xIdx, yIdx, alpha=None):
        """
        Compute the dot product of the specified pieces of vectors
        and matrices.

        The parameter types are actually their expected shapes
        relative to each other.

        Parameters
        ----------
        o : xBlocks, yBlocks, xSize, ySize
        x : batch, xWin, xSize
        y : batch, yWin, ySize
        xIdx : batch, iWin
            indexes of the x blocks
        yIdx : batch, oWin
            indexes of the y blocks

        Returns
        -------
        (xBlocks, yBlocks, xSize, ySize)
            outer(x[i], y[j]) + o[i, j]

        Notes
        -----
        - `batch` is the number of examples in a minibatch (batch size).
        - `xBlocks` is the total number of blocks in x.
        - `xSize` is the size of each of these x blocks.
        - `xWin` is the number of blocks that will be used as x. Which blocks
          will be used is specified in `xIdx`.
        - `yBlocks` is the number or possible y blocks.
        - `ySize` is the size of each of these y blocks.
        - `yWin` is the number of y blocks that will actually be computed.
          Which blocks will be computed is specified in `yIdx`.

        """
        one = theano.tensor.constant(np.asarray(1.0, dtype='float32'))
        o = theano.tensor.as_tensor_variable(o)
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)

        if alpha is None:
            alpha = one

        return Apply(self, [o, x, y, xIdx, yIdx, alpha],
                     [o.type()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def perform(self, node, inp, out_):
        o, x, y, xIdx, yIdx, alpha = inp[:6]

        if not self.inplace:
            o = o.copy()

        for b in range(x.shape[0]):
            for i in range(xIdx.shape[1]):
                for j in range(yIdx.shape[1]):
                    o[xIdx[b, i], yIdx[b, j]] += np.outer(x[b, i],
                                                          y[b, j, :])
        out_[0][0] = o


sparse_block_gemv = SparseBlockGemv(False)
sparse_block_gemv_inplace = SparseBlockGemv(True)
sparse_block_outer = SparseBlockOuter(False)
sparse_block_outer_inplace = SparseBlockOuter(True)


def sparse_block_dot(W, h, inputIdx, b, outputIdx):
    """
    Compute the dot product (plus bias) of the specified pieces of vectors
    and matrices. See SparseBlockGemv to get more information.

    The parameter types are actually their expected shapes relative to
    each other.

    Parameters
    ----------
    W : iBlocks, oBlocks, iSize, oSize
        weight matrix
    h : batch, iWin, iSize
        input from lower layer (sparse)
    inputIdx : batch, iWin
        indexes of the input blocks
    b : oBlocks, oSize
        bias vector
    outputIdx : batch, oWin
        indexes of the output blocks

    Returns
    -------
    (batch, oWin, oSize)
        dot(W[i, j], h[i]) + b[j] but b[j] is only added once

    Notes
    -----
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
    return SparseBlockGemv()(b.take(outputIdx, axis=0), W, h,
                             inputIdx, outputIdx)
