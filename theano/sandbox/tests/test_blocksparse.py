import numpy
from numpy.random import randn

import theano
import theano.tensor as T
import theano.tests.unittest_tools as utt

from theano.sandbox.blocksparse import sparse_block_gemv_cpu, sparse_block_outer_cpu


def sparse_block_gemv_data():
    nInputBlock = 128
    nOutputBlock = 64
    inputSize = 40
    outputSize = 30
    inputWindowSize = 7
    outputWindowSize = 9
    batchSize = 2

    input = randn(batchSize, inputWindowSize, inputSize).astype('float32')
    permutation = numpy.random.permutation
    inputIndice = numpy.vstack(permutation(nInputBlock)[:inputWindowSize]
                               for _ in range(batchSize))
    outputIndice = numpy.vstack(permutation(nOutputBlock)[:outputWindowSize]
                                for _ in range(batchSize))
    weight = randn(nInputBlock, nOutputBlock,
                   inputSize, outputSize).astype('float32')
    bias = randn(nOutputBlock, outputSize).astype('float32')

    return weight, input, inputIndice, bias, outputIndice


def sparse_block_gemv_numpy(W, h, iIdx, b, oIdx):
    o = b.take(oIdx, axis=0)

    for b in range(o.shape[0]):
        for j in range(o.shape[1]):
            outputIdx = oIdx[b, j]

            for i in range(h.shape[1]):
                inputIdx = iIdx[b, i]
                w = W[inputIdx, outputIdx]
                # this below is a gemv I think
                o[b, j, :] += numpy.dot(h[b, i], w)
    return o


def test_sparse_block_gemv_cpu():
    b = T.fmatrix()
    W = T.ftensor4()
    h = T.ftensor3()
    iIdx = T.lmatrix()
    oIdx = T.lmatrix()

    o = sparse_block_gemv_cpu(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o)

    W_val, h_val, iIdx_val, b_val, oIdx_val = sparse_block_gemv_data()

    th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)
    ref_out = sparse_block_gemv_numpy(W_val, h_val, iIdx_val, b_val, oIdx_val)

    utt.assert_allclose(ref_out, th_out)


def sparse_block_outer_data():
    nInputBlock = 128
    nOutputBlock = 64
    xSize = 40
    ySize = 30
    xWindowSize = 7
    yWindowSize = 9
    batchSize = 2

    x = randn(batchSize, xWindowSize, xSize).astype('float32')
    y = randn(batchSize, yWindowSize, ySize).astype('float32')
    randint = numpy.random.randint
    xIdx = numpy.vstack(randint(0, xWindowSize, nInputBlock)
                        for _ in range(batchSize))
    yIdx = numpy.vstack(randint(0, yWindowSize, nOutputBlock)
                        for _ in range(batchSize))

    return x, y, xIdx, yIdx


def sparse_block_outer_numpy(x, y, xIdx, yIdx):

    o = numpy.zeros((yIdx.shape[1], xIdx.shape[1], y.shape[2], x.shape[2]),
                    dtype="float32")

    for b in range(x.shape[0]):
        for i in range(yIdx.shape[1]):
            for j in range(xIdx.shape[1]):
                o[i, j] += numpy.outer(y[b, yIdx[b, i], :], x[b, xIdx[b, j], :])
    return o


def test_sparse_block_outer_cpu():
    x = T.ftensor3()
    y = T.ftensor3()
    xIdx = T.lmatrix()
    yIdx = T.lmatrix()

    o = sparse_block_outer_cpu(x, y, xIdx, yIdx)

    f = theano.function([x, y, xIdx, yIdx], o)

    x_val, y_val, xIdx_val, yIdx_val = sparse_block_outer_data()

    th_out = f(x_val, y_val, xIdx_val, yIdx_val)
    ref_out = sparse_block_outer_numpy(x_val, y_val, xIdx_val, yIdx_val)

    utt.assert_allclose(ref_out, th_out)