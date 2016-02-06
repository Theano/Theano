import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
from theano.misc.pycuda_init import pycuda_available
from theano.sandbox.cuda.cublas import cublas_available
from theano.sandbox.cuda import cublas

if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda not available')
if not pycuda_available:
    raise SkipTest('Optional package pycuda not available')
if not cublas_available:
    raise SkipTest('Optional package skcuda.cublas not available')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestGpuTriangularSolve(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def check_gpu_triangular_solve(self, A_val, x_val, lower):
        b_val = numpy.dot(A_val, x_val)
        A = theano.tensor.matrix("A", dtype="float32")
        if x_val.ndim == 1:
            b = theano.tensor.vector("b", dtype="float32")
        else:
            b = theano.tensor.matrix("b", dtype="float32")
        if lower:
            x = cublas.gpu_lower_triangular_solve(A, b)
        else:
            x = cublas.gpu_upper_triangular_solve(A, b)
        fn = theano.function([A, b], x)
        x_res = numpy.array(fn(A_val, b_val))
        utt.assert_allclose(x_res, x_val)

    def test_invalid_input_fail_1d(self):
        """ Invalid input test case with 1D vector as first input. """
        def invalid_input_func():
            A = theano.tensor.tensor3("A", dtype="float32")
            b = theano.tensor.matrix("b", dtype="float32")
            cublas.gpu_lower_triangular_solve(A, b)
            cublas.gpu_upper_triangular_solve(A, b)
        self.assertRaises(AssertionError, invalid_input_func)

    def test_invalid_input_fail_3d(self):
        """ Invalid input test case with 3D tensor as first input. """
        def invalid_input_func():
            A = theano.tensor.vector("A", dtype="float32")
            b = theano.tensor.matrix("b", dtype="float32")
            cublas.gpu_lower_triangular_solve(A, b)
            cublas.gpu_upper_triangular_solve(A, b)
        self.assertRaises(AssertionError, invalid_input_func)

    def test_diag_solve_2d(self):
        """ Diagonal solve test case with 2D array as second input. """
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.check_gpu_triangular_solve(A_val, x_val, True)
        self.check_gpu_triangular_solve(A_val, x_val, False)

    def test_tri_solve_2d(self):
        """ Triangular solve test cases with 2D array as second input. """
        A_val = (numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32") +
                 numpy.eye(5, dtype="float32"))
        L_val = numpy.tril(A_val)
        U_val = numpy.triu(A_val)
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.check_gpu_triangular_solve(L_val, x_val, True)
        self.check_gpu_triangular_solve(U_val, x_val, False)

    def test_diag_solve_1d(self):
        """ Diagonal solve test case with 1D array as second input. """
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     A_val.shape[1]).astype("float32")
        self.check_gpu_triangular_solve(A_val, x_val, True)
        self.check_gpu_triangular_solve(A_val, x_val, False)

    def test_tri_solve_1d(self):
        """ Triangular solve test cases with 1D array as second input. """
        A_val = (numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32") +
                 numpy.eye(5, dtype="float32"))
        L_val = numpy.tril(A_val)
        U_val = numpy.triu(A_val)
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     A_val.shape[1]).astype("float32")
        self.check_gpu_triangular_solve(L_val, x_val, True)
        self.check_gpu_triangular_solve(U_val, x_val, False)
