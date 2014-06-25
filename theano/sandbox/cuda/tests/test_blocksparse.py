import numpy
from numpy.random import randn

from unittest import TestCase

from nose.plugins.skip import SkipTest

import theano
from theano import tensor
import theano.tests.unittest_tools as utt

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

from theano.sandbox.cuda.blocksparse import (sparse_block_dot_SS,
                                             sparse_block_gemv_ss,
                                             sparse_block_gemv_ss_inplace,
                                             sparse_block_outer_ss,
                                             sparse_block_outer_ss_inplace)

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


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
    o = b.take(oIdx, axis=0)

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

    o = sparse_block_dot_SS(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o)

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()

    th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)
    ref_out = blocksparse(W_val, h_val, iIdx_val, b_val, oIdx_val)

    utt.assert_allclose(ref_out, th_out)


def test_blocksparse_grad():
    h_val = randn(2, 3).astype('float32')
    iIdx_val = numpy.random.permutation(3)[:2]
    oIdx_val = numpy.random.permutation(3)[:2]
    W_val = randn(3, 3, 4, 3).astype('float32')
    b_val = randn(3, 4).astype('float32')

    iIdx = theano.tensor.constant(iIdx_val)
    oIdx = theano.tensor.constant(oIdx_val)

    def f(b, h, W):
        return sparse_block_gemv_ss(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

    utt.verify_grad(f, [b_val, h_val, W_val])


def test_blocksparse_grad_shape():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.fmatrix()
    iIdx = tensor.lvector()
    oIdx = tensor.lvector()

    o = sparse_block_gemv_ss(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
    go = theano.grad(o.sum(), [b, W, h])

    f = theano.function([W, h, iIdx, b, oIdx], go)

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()

    # just make sure that it runs correcly and all the shapes are ok.
    b_g, W_g, h_g = f(W_val, h_val, iIdx_val, b_val, oIdx_val)

    assert b_g.shape == b_val.shape
    assert h_g.shape == h_val.shape
    assert W_g.shape == W_val.shape


class TestBlockSparseDot(TestCase, utt.TestOptimizationMixin):
    def test_opt_inplace(self):
        b = tensor.fmatrix()
        W = tensor.ftensor4()
        h = tensor.fmatrix()
        iIdx = tensor.lvector()
        oIdx = tensor.lvector()

        o = sparse_block_dot_SS(W, h, iIdx, b, oIdx)

        f = theano.function([W, h, iIdx, b, oIdx], o, mode=mode_with_gpu)

        self.assertFunctionContains0(f, sparse_block_gemv_ss)
        self.assertFunctionContains1(f, sparse_block_gemv_ss_inplace)

        gW = theano.grad(o.sum(), [W])

        f = theano.function([W, h, iIdx, b, oIdx], gW, mode=mode_with_gpu)

        self.assertFunctionContains0(f, sparse_block_outer_ss)
        self.assertFunctionContains1(f, sparse_block_outer_ss_inplace)
