"""
    Tests for block sparse dot
"""
import unittest

import numpy
from numpy.random import randn

import theano
from theano import tensor
import theano.tests.unittest_tools as utt

from theano.sandbox.blocksparse import sparse_block_dot, cpu_sparse_block_gemv, \
    cpu_sparse_block_outer, sparse_block_outer


class BlockSparse_Gemv_and_Outer(unittest.TestCase):

    def runTest(self):
        pass

    def setUp(self):
        utt.seed_rng()
        self.mode = theano.compile.get_default_mode().excluding(
            'constant_folding'
        )
        self.gemv_op = cpu_sparse_block_gemv
        self.outer_op = cpu_sparse_block_outer

    @staticmethod
    def gemv_data():

        nInputBlock = 8
        nOutputBlock = 7
        inputSize = 6
        outputSize = 5
        inputWindowSize = 4
        outputWindowSize = 3
        batchSize = 2

        input = randn(batchSize, inputWindowSize, inputSize).astype('float32')
        permutation = numpy.random.permutation
        inputIndice = numpy.vstack(permutation(nInputBlock)[:inputWindowSize]
                                   for _ in range(batchSize)).astype('int32')
        outputIndice = numpy.vstack(
            permutation(nOutputBlock)[:outputWindowSize]
            for _ in range(batchSize)).astype('int32')
        weight = randn(nInputBlock, nOutputBlock,
                       inputSize, outputSize).astype('float32')
        bias = randn(nOutputBlock, outputSize).astype('float32')

        return weight, input, inputIndice, bias, outputIndice

    @staticmethod
    def outer_data():
        nInputBlock = 8
        nOutputBlock = 7
        xSize = 6
        ySize = 5
        xWindowSize = 4
        yWindowSize = 3
        batchSize = 2

        o = randn(nInputBlock, nOutputBlock, xSize, ySize).astype('float32')
        x = randn(batchSize, xWindowSize, xSize).astype('float32')
        y = randn(batchSize, yWindowSize, ySize).astype('float32')
        randint = numpy.random.randint
        xIdx = numpy.vstack(randint(0, xWindowSize, nInputBlock)
                            for _ in range(batchSize)).astype('int32')
        yIdx = numpy.vstack(randint(0, yWindowSize, nOutputBlock)
                            for _ in range(batchSize)).astype('int32')

        return o, x, y, xIdx, yIdx

    @staticmethod
    def gemv_numpy(o, W, h, iIdx, oIdx):
        for b in range(o.shape[0]):
            for j in range(o.shape[1]):
                outputIdx = oIdx[b, j]
                for i in range(h.shape[1]):
                    inputIdx = iIdx[b, i]
                    w = W[inputIdx, outputIdx]
                    o[b, j, :] += numpy.dot(h[b, i], w)
        return o

    @staticmethod
    def outer_numpy(o, x, y, xIdx, yIdx):
        for b in range(x.shape[0]):
            for i in range(xIdx.shape[1]):
                for j in range(yIdx.shape[1]):
                    o[xIdx[b, i], yIdx[b, j]] += numpy.outer(x[b, xIdx[b, i], :],
                                           y[b, yIdx[b, j], :])
        return o

    def test_sparseblockdot(self):
        b = tensor.fmatrix()
        W = tensor.ftensor4()
        h = tensor.ftensor3()
        iIdx = tensor.imatrix()
        oIdx = tensor.imatrix()

        o = sparse_block_dot(W, h, iIdx, b, oIdx)

        f = theano.function([W, h, iIdx, b, oIdx], o, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = \
            BlockSparse_Gemv_and_Outer.gemv_data()

        th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)

        ref_out = BlockSparse_Gemv_and_Outer.gemv_numpy(
             b_val.take(oIdx_val, axis=0), W_val, h_val, iIdx_val, oIdx_val)

        utt.assert_allclose(ref_out, th_out)

    def test_sparseblockgemv(self):

        b = tensor.fmatrix()
        W = tensor.ftensor4()
        h = tensor.ftensor3()
        iIdx = tensor.imatrix()
        oIdx = tensor.imatrix()

        o = self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

        f = theano.function([W, h, iIdx, b, oIdx], o, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = \
            BlockSparse_Gemv_and_Outer.gemv_data()

        th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)
        ref_out = BlockSparse_Gemv_and_Outer.gemv_numpy(
             b_val.take(oIdx_val, axis=0), W_val, h_val, iIdx_val, oIdx_val)

        utt.assert_allclose(ref_out, th_out)

    def test_sparseblockgemvF(self):
        """
            Test the fortan order for W (which can happen in the grad for some
            graphs).
        """
        b = tensor.fmatrix()
        W = tensor.ftensor4()
        h = tensor.ftensor3()
        iIdx = tensor.imatrix()
        oIdx = tensor.imatrix()

        o = self.gemv_op(b.take(oIdx, axis=0),
            tensor.DimShuffle((False, False, False, False),
                              (0, 1, 3, 2))(tensor.as_tensor_variable(W)),
            h, iIdx, oIdx)

        f = theano.function([W, h, iIdx, b, oIdx], o, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = \
            BlockSparse_Gemv_and_Outer.gemv_data()

        th_out = f(numpy.swapaxes(W_val, 2, 3), h_val, iIdx_val, b_val,
                   oIdx_val)
        ref_out = BlockSparse_Gemv_and_Outer.gemv_numpy(
             b_val.take(oIdx_val, axis=0), W_val, h_val, iIdx_val, oIdx_val)

        utt.assert_allclose(ref_out, th_out)

    def test_sparseblockgemv_grad(self):
#        h_val = randn(1, 2, 3).astype('float32')
#        iIdx_val = numpy.random.permutation(3)[:2][None, :]
#        oIdx_val = numpy.random.permutation(3)[:2][None, :]
#        W_val = randn(3, 3, 3, 4).astype('float32')
#        b_val = randn(3, 4).astype('float32')

        W_val, h_val, iIdx_val, b_val, oIdx_val = \
            BlockSparse_Gemv_and_Outer.gemv_data()

        iIdx = theano.tensor.constant(iIdx_val)
        oIdx = theano.tensor.constant(oIdx_val)

        def metaop(b, h, W):
            print b, h, W
            print iIdx.dtype, oIdx.dtype
            return sparse_block_dot(W, h, iIdx, b, oIdx)

        def op(b, h, W):
            return self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

        print W_val.shape
        print h_val.shape
        print b_val.shape
        print iIdx_val.shape
        print oIdx_val.shape

        utt.verify_grad(metaop, [b_val, h_val, W_val], mode=self.mode)
        utt.verify_grad(op, [b_val, h_val, W_val], mode=self.mode)

    def test_sparseblockgemv_grad_1(self):
        """
            Test that we correctly handle cases where dimensions are 1.
        """
        h_val = randn(1, 1, 1).astype('float32')
        iIdx_val = numpy.random.permutation(1)[:1][None, :]
        oIdx_val = numpy.random.permutation(1)[:1][None, :]
        W_val = randn(1, 1, 1, 1).astype('float32')
        b_val = randn(1, 1).astype('float32')

        iIdx = theano.tensor.constant(iIdx_val)
        oIdx = theano.tensor.constant(oIdx_val)

        def metaop(b, h, W):
            return sparse_block_dot(W, h, iIdx, b, oIdx)

        def op(b, h, W):
            return self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

        utt.verify_grad(metaop, [b_val, h_val, W_val], mode=self.mode)
        utt.verify_grad(op, [b_val, h_val, W_val], mode=self.mode)

    def test_sparseblockgemv_grad_shape(self):
        b = tensor.fmatrix()
        W = tensor.ftensor4()
        h = tensor.ftensor3()
        iIdx = tensor.imatrix()
        oIdx = tensor.imatrix()

        o = self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
        go = theano.grad(o.sum(), [b, W, h])

        f = theano.function([W, h, iIdx, b, oIdx], go, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = \
            BlockSparse_Gemv_and_Outer.gemv_data()

        # just make sure that it runs correcly and all the shapes are ok.
        b_g, W_g, h_g = f(W_val, h_val, iIdx_val, b_val, oIdx_val)

        assert b_g.shape == b_val.shape
        assert h_g.shape == h_val.shape
        assert W_g.shape == W_val.shape

    def test_sparse_block_outer(self):
        o = tensor.ftensor4()
        x = tensor.ftensor3()
        y = tensor.ftensor3()
        xIdx = tensor.imatrix()
        yIdx = tensor.imatrix()

        out, updates = self.outer_op(o, x, y, xIdx, yIdx)

        f = theano.function([o, x, y, xIdx, yIdx], out, updates=updates, on_unused_input="warn")

        o_val, x_val, y_val, xIdx_val, yIdx_val = \
            BlockSparse_Gemv_and_Outer.outer_data()

        th_out = f(o_val, x_val, y_val, xIdx_val, yIdx_val)
        ref_out = BlockSparse_Gemv_and_Outer.outer_numpy(
            o_val, x_val, y_val, xIdx_val, yIdx_val)

        print th_out.shape
        print ref_out.shape

        utt.assert_allclose(ref_out, th_out)


# a = BlockSparse_Gemv_and_Outer()
# a.setUp()
# a.test_sparse_block_outer()