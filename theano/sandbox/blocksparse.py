import numpy

import theano
from theano import Op, Apply
from theano import tensor
from theano.tensor import discrete_dtypes
from theano.gradient import grad_undefined


class SparseBlockGemv(Op):
    """
    This op computes the dot product of specified pieces of vectors
    and matrices, returning pieces of vectors:
        for b in range(batch_size):
            for j in range(o.shape[1]):
                for i in range(h.shape[1]):
                    o[b, j, :] += numpy.dot(h[b, i], W[iIdx[b, i], oIdx[b, j]])
    """

    registered_opts = []

    def __init__(self, inplace=False):
        self.inplace = False

    def make_node(self, o, W, h, inputIdx, outputIdx):
        """
        Compute the dot product of the specified pieces of vectors
        and matrices. 

        Parameters
        ----------
        var: shape, comment
        o: (batch, oWin, oSize) output vector
        W: (iBlocks, oBlocks, iSize, oSize), weight matrix
        h: (batch, iWin, iSize), input from lower layer (sparse)
        inputIdx: (batch, iWin), indexes of the input blocks
        outputIdx: (batch, oWin), indexes of the output blocks
        returns (batch, oWin, oSize), dot(W[i, j], h[i]) + o[j]

        Notation
        --------
        - `batch` is the number of examples in a minibatch (batch size).
        - `iBlocks` is the total number of blocks in the input (from lower
            layer).
        - `iSize` is the size of each of these input blocks.
        - `iWin` is the number of blocks that will be used as inputs. Which
            blocks
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

        output = o.type.__class__(dtype=o.type.dtype,
                                  broadcastable=(False,)*o.ndim)()

        return Apply(self, [o, W, h, inputIdx, outputIdx], [output])

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
    updating a full matrix with the results:
        for b in range(batch_size):
            o[xIdx[b, i], yIdx[b, j]] += (alpha * outer(x[xIdx[b, i]],
            y[yIdx[b, j]]))
    This op is involved in the gradient of SparseBlockGemv.
    """

    registered_opts = []

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, o, x, y, xIdx, yIdx, alpha=None):
        """

        Compute the dot product of the specified pieces of vectors
        and matrices. 

        Parameters
        ----------
        var: shape, comment
        o: (xBlocks, yBlocks, xSize, ySize)
        x: (batch, xWin, xSize)
        y: (batch, yWin, ySize)
        xIdx: (batch, iWin), indexes of the x blocks
        yIdx: (batch, oWin), indexes of the y blocks
        returns (xBlocks, yBlocks, xSize, ySize), outer(x[i], y[j]) + o[i, j]

        Notation
        --------
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
        one = tensor.constant(numpy.asarray(1.0, dtype='float32'))
        o = theano.tensor.as_tensor_variable(o)
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)

        if alpha is None:
            alpha = one

        output = o.type.__class__(dtype=o.type.dtype,
                                  broadcastable=(False,)*o.ndim)()

        return Apply(self, [o, x, y, xIdx, yIdx, alpha],
                     [output])

    def perform(self, node, inp, out_):
        raise NotImplementedError('Optimization of SparseBlockOuter failed.')

    def grad(self, inputs, output_gradients):
        raise NotImplementedError("SparseBlockOuter has no gradient "
                                  "implemented")


sparse_block_outer = SparseBlockOuter(False)
sparse_block_outer_inplace = SparseBlockOuter(True)


def cpu_sparse_block_gemv(o, W, h, inputIdx, outputIdx):
    """
    Creates a graph for the sparse block dot operation. Check SparseBlockGemv's
    docstring for information about the arguments.
    """
    def _loop_over_batch(b, W, h, inputIdx, outputIdx):

        def _loop_over_outputIdx(i, b, W, h, inputIdx, outputIdx):

            def _loop_over_inputIdx(j, b, i, W, h, inputIdx, outputIdx):
                return tensor.dot(h[b, j, :], W[inputIdx[b, j],
                                         outputIdx[b, i], :, :])

            res3 = theano.scan(fn=_loop_over_inputIdx,
                               sequences=tensor.arange(0, inputIdx.shape[1]),
                               non_sequences=[b, i, W, h, inputIdx, outputIdx],
                               name='_loop_over_inputIdx')[0]

            return res3.sum(axis=0)

        res2 = theano.scan(fn=_loop_over_outputIdx,
                           sequences=tensor.arange(0, outputIdx.shape[1]),
                           non_sequences=[b, W, h, inputIdx, outputIdx],
                           name='_loop_over_outputIdx')[0]

        return res2

    res1 = theano.scan(fn=_loop_over_batch,
                       sequences=tensor.arange(0, inputIdx.shape[0]),
                       non_sequences=[W, h, inputIdx, outputIdx],
                       name='_loop_over_batch')[0]

    return res1 + o


def cpu_sparse_block_outer(o, x, y, xIdx, yIdx, alpha=1.0):
    """
    Creates a graph for the sparse block outer operation. Check
    SparseBlockOuter's docstring for information about the arguments.
    """

    def _loop_over_batch(b, o1, x, y, xIdx, yIdx):

        def _loop_over_xIdx(i, o2, b, x, y, xIdx, yIdx):

            def _loop_over_yIdx(j, o3, i, b, x, y, xIdx, yIdx):
                return tensor.inc_subtensor(
                    o3[xIdx[b, i], yIdx[b, j]],
                    tensor.outer(x[b, xIdx[b, i], :], y[b, yIdx[b, j], :]))

            res3, updates3 = theano.scan(
                fn=_loop_over_yIdx,
                sequences=tensor.arange(0, yIdx.shape[1]),
                non_sequences=[i, b, x, y, xIdx, yIdx],
                outputs_info=[o2],
                name='_loop_over_yIdx')

            return res3[-1], updates3

        res2, updates2 = theano.scan(
            fn=_loop_over_xIdx,
            sequences=tensor.arange(0, xIdx.shape[1]),
            non_sequences=[b, x, y, xIdx, yIdx],
            outputs_info=[o1],
            name='_loop_over_xIdx')

        return res2[-1], updates2

    res1, updates1 = theano.scan(
        fn=_loop_over_batch,
        sequences=tensor.arange(0, xIdx.shape[0]),
        non_sequences=[x, y, xIdx, yIdx],
        outputs_info=[o],
        name='_loop_over_batch')

    return (alpha * res1[-1])


def sparse_block_dot(W, h, inputIdx, b, outputIdx, inplace=False):
    """
    Compute the dot product (plus bias) of the specified pieces of vectors
    and matrices. See SparseBlockGemv to get more information.

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
