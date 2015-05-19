import numpy
from numpy.random import randn

import theano
import theano.tensor as T
import theano.tests.unittest_tools as utt

from theano.sandbox.blocksparse import sparse_block_gemv_cpu


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