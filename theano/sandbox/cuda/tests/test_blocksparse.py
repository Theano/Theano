import theano
from theano import tensor
import theano.tests.unittest_tools as utt

import numpy
from numpy.random import randn

from theano.sandbox.cuda.blocksparse import sparse_block_dot_DS

def blocksparse_data():
    nInputBlock = 128
    nOutputBlock = 64
    inputSize = 40
    outputSize = 30
    inputWindowSize = 7
    outputWindowSize = 9

    input = randn(inputWindowSize, inputSize).astype('float32')
    inputIndice = numpy.random.permutation(nInputBlock)[:inputWindowSize]
    outputIndice = numpy.random.permutation(nOutputBlock)[:outputWindowSize]
    weight = randn(nInputBlock, nOutputBlock, outputSize, inputSize).astype('float32')
    bias = randn(nOutputBlock, outputSize).astype('float32')

    return weight, input, inputIndice, bias, outputIndice

def blocksparse(W, h, iIdx, b, oIdx):
    o = b.take(oIdx, axis=0).copy()

    for j in range(o.shape[0]):
        outputIdx = oIdx[j]

        for i in range(h.shape[0]):
            inputIdx = iIdx[i]
            w = W[inputIdx, outputIdx]
            # this below is a gemv I think
            o[j, :] += numpy.dot(w, h[i])

    return o

def test_blocksparse():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.fmatrix()
    iIdx = tensor.lvector()
    oIdx = tensor.lvector()

    o = sparse_block_dot_DS(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o)

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()

    th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)
    ref_out = blocksparse(W_val, h_val, iIdx_val, b_val, oIdx_val)

    utt.assert_allclose(ref_out, th_out)

