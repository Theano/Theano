import unittest

import theano
from theano import tensor
from theano.sandbox import cuda


class TestGradient(unittest.TestCase):
    verbose = 0

    def test_gpu_out_multiple_clients(self):
        # Test that when the output of gpu_from_host is used by more
        # than one Op, the gradient still works.
        # A problem used to be that GpuFromHost.grad expected the output
        # gradient to be on GPU, but the summation of the different
        # incoming gradients was done on CPU.

        x = tensor.fmatrix('x')
        z = cuda.gpu_from_host(x)

        n1 = tensor.nnet.sigmoid(z)
        n2 = tensor.dot(z, z.T)

        s1 = n1.sum()
        s2 = n2.sum()

        c = s1 + s2

        dc_dx = theano.grad(c, x)
        if self.verbose:
            theano.printing.debugprint(c, print_type=True)
            theano.printing.debugprint(dc_dx, print_type=True)
