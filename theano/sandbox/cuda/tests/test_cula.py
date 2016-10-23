from __future__ import absolute_import, print_function, division
import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
from theano.misc.pycuda_init import pycuda_available
from theano.sandbox.cuda.cula import cula_available

from theano.sandbox.cuda import cula

if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda not available')
if not pycuda_available:
    raise SkipTest('Optional package pycuda not available')
if not cula_available:
    raise SkipTest('Optional package scikits.cuda.cula not available')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestCula(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def run_gpu_solve(self, A_val, x_val):
        b_val = numpy.dot(A_val, x_val)
        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.TensorType(dtype="float32",
                                     broadcastable=[False, ] * x_val.ndim)()
        solver = cula.gpu_solve(A, b)
        fn = theano.function([A, b], [solver])
        res = fn(A_val, b_val)
        x_res = numpy.array(res[0])
        utt.assert_allclose(x_res, x_val)

    def test_invalid_input_fail_1d(self):
        """ Invalid solve input test case with 1D vector as first input. """
        A = theano.tensor.vector("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        self.assertRaises(AssertionError, cula.gpu_solve, A, b)

    def test_invalid_input_fail_3d(self):
        """ Invalid solve input test case with 3D tensor as first input. """
        A = theano.tensor.tensor3("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        self.assertRaises(AssertionError, cula.gpu_solve, A, b)

    def test_diag_solve_2d(self):
        """ Diagonal solve test case with 2D array as second input. """
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve_2d(self):
        """ Symmetric solve test case with 2D array as second input. """
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = (A_val + A_val.T) / 2.0
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_sym, x_val)

    def test_orth_solve_2d(self):
        """ Orthogonal solve test case with 2D array as second input. """
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = numpy.linalg.svd(A_val)[0]
        x_val = numpy.random.uniform(-0.4, 0.4, (A_orth.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve_2d(self):
        """ Uniform random solve test case with 2D array as second input. """
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     (A_val.shape[1], 4)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_diag_solve_1d(self):
        """ Diagonal solve test case with 1D array as second input. """
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     A_val.shape[1]).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve_1d(self):
        """ Symmetric solve test case with 1D array as second input. """
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = (A_val + A_val.T) / 2.0
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     A_val.shape[1]).astype("float32")
        self.run_gpu_solve(A_sym, x_val)

    def test_orth_solve_1d(self):
        """ Orthogonal solve test case with 1D array as second input. """
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = numpy.linalg.svd(A_val)[0]
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     A_orth.shape[1]).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve_1d(self):
        """ Uniform random solve test case with 1D array as second input. """
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     A_val.shape[1]).astype("float32")
        self.run_gpu_solve(A_val, x_val)
