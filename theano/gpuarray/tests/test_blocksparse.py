from __future__ import absolute_import, print_function, division
import numpy as np

import theano
from theano import tensor
import theano.tests.unittest_tools as utt
from theano.tensor.nnet.tests import test_blocksparse

from .config import mode_with_gpu, test_ctx_name

from ..type import gpuarray_shared_constructor
from ..blocksparse import (GpuSparseBlockGemv,
                           GpuSparseBlockOuter,
                           gpu_sparse_block_gemv,
                           gpu_sparse_block_outer)


class BlockSparse_Gemv_and_Outer(test_blocksparse.BlockSparse_Gemv_and_Outer):
    def setUp(self):
        utt.seed_rng()
        self.mode = mode_with_gpu.excluding('constant_folding')
        self.gemv_op = gpu_sparse_block_gemv
        self.outer_op = gpu_sparse_block_outer
        self.gemv_class = GpuSparseBlockGemv
        self.outer_class = GpuSparseBlockOuter

    # This test is temporarily disabled since we disabled the output_merge
    # and alpha_merge optimizations for blocksparse due to brokeness.
    # Re-enable when those are re-added.
    def Xtest_blocksparse_grad_merge(self):
        b = tensor.fmatrix()
        h = tensor.ftensor3()
        iIdx = tensor.lmatrix()
        oIdx = tensor.lmatrix()

        W_val, h_val, iIdx_val, b_val, oIdx_val = self.gemv_data()
        W = gpuarray_shared_constructor(W_val, context=test_ctx_name)

        o = gpu_sparse_block_gemv(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
        gW = theano.grad(o.sum(), W)

        lr = np.asarray(0.05, dtype='float32')

        upd = W - lr * gW

        f1 = theano.function([h, iIdx, b, oIdx], updates=[(W, upd)],
                             mode=mode_with_gpu)

        # Make sure the lr update was merged.
        assert isinstance(f1.maker.fgraph.outputs[0].owner.op,
                          GpuSparseBlockOuter)

        # Exclude the merge optimizations.
        mode = mode_with_gpu.excluding('local_merge_blocksparse_alpha')
        mode = mode.excluding('local_merge_blocksparse_output')

        f2 = theano.function([h, iIdx, b, oIdx], updates=[(W, upd)], mode=mode)

        # Make sure the lr update is not merged.
        assert not isinstance(f2.maker.fgraph.outputs[0].owner.op,
                              GpuSparseBlockOuter)

        f2(h_val, iIdx_val, b_val, oIdx_val)
        W_ref = W.get_value()

        # reset the var
        W.set_value(W_val)
        f1(h_val, iIdx_val, b_val, oIdx_val)
        W_opt = W.get_value()

        utt.assert_allclose(W_ref, W_opt)
