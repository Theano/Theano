from __future__ import absolute_import, print_function, division

import numpy
import unittest

import theano
import theano.tensor as T


class TestScanCheckpoint(unittest.TestCase):

    def setUp(self):
        k = T.iscalar("k")
        A = T.vector("A")
        self.k = k
        self.A = A
        result, _ = theano.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(A),
            non_sequences=A,
            n_steps=k)
        result_check, _ = theano.scan_with_checkpoints(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(A),
            non_sequences=A,
            n_steps=k,
            save_every_N=50)
        self.result = result[-1]
        self.result_check = result_check[-1]
        self.grad_A = T.grad(self.result.sum(), self.A)
        self.grad_A_check = T.grad(self.result_check.sum(), self.A)

    def test_forward_pass(self):
        """Test forward computation of A**k."""
        f = theano.function(inputs=[self.A, self.k],
                            outputs=[self.result, self.result_check])
        out, out_check = f(range(10), 100)
        assert numpy.allclose(out, out_check)

    def test_backward_pass(self):
        """Test gradient computation of A**k."""
        f = theano.function(inputs=[self.A, self.k],
                            outputs=[self.grad_A, self.grad_A_check])
        out, out_check = f(range(10), 100)
        assert numpy.allclose(out, out_check)

    @unittest.skipUnless(theano.gpuarray.type._context_reg[None],
                         'Requires gpuarray backend.')
    def test_memory(self):
        """Test that scan_checkpoint reduces memory usage."""
        k = T.iscalar("k")
        A = T.vector("A")
        result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                                      outputs_info=T.ones_like(A),
                                      non_sequences=A,
                                      n_steps=k)
        result_check, updates_check = theano.scan_with_checkpoints(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(A),
            non_sequences=A,
            n_steps=k,
            save_every_N=10000)
        result = result[-1]
        result_check = result_check[-1]
        grad_A = T.grad(result.sum(), A)
        grad_A_check = T.grad(result_check.sum(), A)
        f = theano.function(inputs=[A, k], outputs=grad_A,
                            updates=updates + updates_check)
        f_check = theano.function(inputs=[A, k], outputs=grad_A_check,
                                  updates=updates + updates_check)
        free_gmem = theano.gpuarray.type._context_reg[None].free_gmem
        data = numpy.ones(free_gmem / 40., dtype=numpy.float32)
        # Check that it works with the checkpoints
        f_check(data, 1000000)
        # Check that the basic scan fails in that case
        self.assertRaises(MemoryError, f, data, 1000000)
