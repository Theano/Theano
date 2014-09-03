import numpy
from numpy.random import randn

from unittest import TestCase

from nose.plugins.skip import SkipTest

import theano
from theano import tensor
import theano.tests.unittest_tools as utt

import theano.sandbox.cuda as cuda_ndarray
if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda disabled')

from theano.sandbox.cuda.basic_ops import (GpuDimShuffle,
                                           as_cuda_ndarray_variable)
from theano.sandbox.cuda.blocksparse import (sparse_block_dot_SS,
                                             sparse_block_gemv_ss,
                                             sparse_block_outer_ss,
                                             sparse_block_outer_ss_inplace)
from theano.sandbox.cuda.var import float32_shared_constructor


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


def setup():
    utt.seed_rng()


def blocksparse_data():
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


def blocksparse(W, h, iIdx, b, oIdx):
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


def test_blocksparse():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot_SS(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o, mode=mode_with_gpu)

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()

    th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)
    ref_out = blocksparse(W_val, h_val, iIdx_val, b_val, oIdx_val)

    utt.assert_allclose(ref_out, th_out)

test_blocksparse.setup = setup


# test the fortan order for W (which can happen in the grad for some graphs).
def test_blocksparseF():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot_SS(GpuDimShuffle((False, False, False, False),
                                          (0, 1, 3, 2))(
                                              as_cuda_ndarray_variable(W)),
                            h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o, mode=mode_with_gpu)

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()

    th_out = f(numpy.swapaxes(W_val, 2, 3), h_val, iIdx_val, b_val, oIdx_val)
    ref_out = blocksparse(W_val, h_val, iIdx_val, b_val, oIdx_val)

    utt.assert_allclose(ref_out, th_out)


def test_blocksparse_grad():
    h_val = randn(1, 2, 3).astype('float32')
    iIdx_val = numpy.random.permutation(3)[:2][None, :]
    oIdx_val = numpy.random.permutation(3)[:2][None, :]
    W_val = randn(3, 3, 3, 4).astype('float32')
    b_val = randn(3, 4).astype('float32')

    iIdx = theano.tensor.constant(iIdx_val)
    oIdx = theano.tensor.constant(oIdx_val)

    def f(b, h, W):
        return sparse_block_gemv_ss(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

    utt.verify_grad(f, [b_val, h_val, W_val], mode=mode_with_gpu)


def test_blocksparse_grad_1():
    # This tests that we correctly handle cases where dimensions are 1.
    h_val = randn(1, 1, 1).astype('float32')
    iIdx_val = numpy.random.permutation(1)[:1][None, :]
    oIdx_val = numpy.random.permutation(1)[:1][None, :]
    W_val = randn(1, 1, 1, 1).astype('float32')
    b_val = randn(1, 1).astype('float32')

    iIdx = theano.tensor.constant(iIdx_val)
    oIdx = theano.tensor.constant(oIdx_val)

    def f(b, h, W):
        return sparse_block_gemv_ss(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

    utt.verify_grad(f, [b_val, h_val, W_val], mode=mode_with_gpu)


def test_blocksparse_grad_shape():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_gemv_ss(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
    go = theano.grad(o.sum(), [b, W, h])

    f = theano.function([W, h, iIdx, b, oIdx], go, mode=mode_with_gpu)

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()

    # just make sure that it runs correcly and all the shapes are ok.
    b_g, W_g, h_g = f(W_val, h_val, iIdx_val, b_val, oIdx_val)

    assert b_g.shape == b_val.shape
    assert h_g.shape == h_val.shape
    assert W_g.shape == W_val.shape


def test_blocksparse_grad_merge():
    b = tensor.fmatrix()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    W_val, h_val, iIdx_val, b_val, oIdx_val = blocksparse_data()
    W = float32_shared_constructor(W_val)

    o = sparse_block_gemv_ss(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
    gW = theano.grad(o.sum(), W)

    lr = numpy.asarray(0.05, dtype='float32')

    upd = W - lr * gW

    f1 = theano.function([h, iIdx, b, oIdx], updates=[(W, upd)],
                         mode=mode_with_gpu)
    # not running with mode=gpu ensures that the elemwise is not merged in
    mode = None
    if theano.config.mode == 'FAST_COMPILE':
        mode = theano.compile.mode.get_mode('FAST_RUN')

    f2 = theano.function([h, iIdx, b, oIdx], updates=[(W, upd)], mode=mode)

    f2(h_val, iIdx_val, b_val, oIdx_val)
    W_ref = W.get_value()

    # reset the var
    W.set_value(W_val)
    f1(h_val, iIdx_val, b_val, oIdx_val)
    W_opt = W.get_value()

    utt.assert_allclose(W_ref, W_opt)
