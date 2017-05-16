from __future__ import absolute_import, print_function, division

import numpy as np
import unittest

import theano
import theano.tensor as T

try:
    from pygpu.gpuarray import GpuArrayException
    PYGPU_AVAILABLE = True
except ImportError:
    PYGPU_AVAILABLE = False


class TestScanCheckpoint(unittest.TestCase):

    def setUp(self):
        self.k = T.iscalar("k")
        self.A = T.vector("A")
        result, _ = theano.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(self.A),
            non_sequences=self.A,
            n_steps=self.k)
        result_check, _ = theano.scan_checkpoints(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(self.A),
            non_sequences=self.A,
            n_steps=self.k,
            save_every_N=100)
        self.result = result[-1]
        self.result_check = result_check[-1]
        self.grad_A = T.grad(self.result.sum(), self.A)
        self.grad_A_check = T.grad(self.result_check.sum(), self.A)

    def test_forward_pass(self):
        # Test forward computation of A**k.
        f = theano.function(inputs=[self.A, self.k],
                            outputs=[self.result, self.result_check])
        out, out_check = f(range(10), 101)
        assert np.allclose(out, out_check)

    def test_backward_pass(self):
        # Test gradient computation of A**k.
        f = theano.function(inputs=[self.A, self.k],
                            outputs=[self.grad_A, self.grad_A_check])
        out, out_check = f(range(10), 101)
        assert np.allclose(out, out_check)

    @unittest.skipUnless(PYGPU_AVAILABLE, 'Requires pygpu.')
    def test_memory(self):
        # Test that scan_checkpoint reduces memory usage.
        if None not in theano.gpuarray.type.list_contexts():
            return unittest.SkipTest('Requires gpuarray backend.')
        from theano.gpuarray.tests.config import mode_with_gpu  # noqa
        f = theano.function(inputs=[self.A, self.k],
                            outputs=self.grad_A, mode=mode_with_gpu)
        f_check = theano.function(inputs=[self.A, self.k],
                                  outputs=self.grad_A_check,
                                  mode=mode_with_gpu)
        free_gmem = theano.gpuarray.type._context_reg[None].free_gmem
        data = np.ones(free_gmem // 3000, dtype=np.float32)
        # Check that it works with the checkpoints
        size = 1000
        if isinstance(mode_with_gpu, theano.compile.DebugMode):
            size = 100
        f_check(data, size)
        # Check that the basic scan fails in that case
        # Skip that check in DebugMode, as it can fail in different ways
        if not isinstance(mode_with_gpu, theano.compile.DebugMode):
            self.assertRaises(GpuArrayException, f, data, 1000)

    def test_taps_error(self):
        # Test that an error rises if we use taps in outputs_info.
        self.assertRaises(RuntimeError, theano.scan_checkpoints,
                          lambda: None, [], {'initial': self.A, 'taps': [-2]})
