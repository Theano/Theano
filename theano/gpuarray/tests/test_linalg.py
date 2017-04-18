from __future__ import absolute_import, division, print_function

import unittest
import numpy as np
import theano

from theano.tests import unittest_tools as utt
from .config import mode_with_gpu

from numpy.linalg.linalg import LinAlgError

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
from theano.gpuarray.linalg import (cusolver_available, gpu_solve, GpuCholesky)

if not cusolver_available:
    raise SkipTest('Optional package scikits.cuda.cusolver not available')


class TestCusolver(unittest.TestCase):
    def run_gpu_solve(self, A_val, x_val, A_struct=None):
        b_val = np.dot(A_val, x_val)
        b_val_trans = np.dot(A_val.T, x_val)

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        b_trans = theano.tensor.matrix("b", dtype="float32")

        if A_struct is None:
            solver = gpu_solve(A, b)
            solver_trans = gpu_solve(A, b_trans, trans='T')
        else:
            solver = gpu_solve(A, b, A_struct)
            solver_trans = gpu_solve(A, b_trans, A_struct, trans='T')

        fn = theano.function([A, b, b_trans], [solver, solver_trans], mode=mode_with_gpu)
        res = fn(A_val, b_val, b_val_trans)
        x_res = np.array(res[0])
        x_res_trans = np.array(res[1])
        utt.assert_allclose(x_val, x_res)
        utt.assert_allclose(x_val, x_res_trans)

    def test_diag_solve(self):
        np.random.seed(1)
        A_val = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                           dtype="float32")
        x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                  1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_bshape_solve(self):
        """
        Test when shape of b (k, m) is such as m > k
        """
        np.random.seed(1)
        A_val = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                           dtype="float32")
        x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                  A_val.shape[1] + 1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = np.dot(A_val, A_val.T)
        x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                  1)).astype("float32")
        self.run_gpu_solve(A_sym, x_val, 'symmetric')

    def test_orth_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = np.linalg.svd(A_val)[0]
        x_val = np.random.uniform(-0.4, 0.4, (A_orth.shape[1],
                                  1)).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = np.random.uniform(-0.4, 0.4,
                                  (A_val.shape[1], 4)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_linalgerrsym_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = np.random.uniform(-0.4, 0.4,
                                  (A_val.shape[1], 4)).astype("float32")
        A_val = np.dot(A_val.T, A_val)
        # make A singular
        A_val[:, 2] = A_val[:, 1] + A_val[:, 3]

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        solver = gpu_solve(A, b, 'symmetric')

        fn = theano.function([A, b], [solver], mode=mode_with_gpu)
        self.assertRaises(LinAlgError, fn, A_val, x_val)

    def test_linalgerr_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = np.random.uniform(-0.4, 0.4,
                                  (A_val.shape[1], 4)).astype("float32")
        # make A singular
        A_val[:, 2] = 0

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        solver = gpu_solve(A, b, trans='T')

        fn = theano.function([A, b], [solver], mode=mode_with_gpu)
        self.assertRaises(LinAlgError, fn, A_val, x_val)


class TestGpuCholesky(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def get_gpu_cholesky_func(self, lower=True, inplace=False):
        # Helper function to compile function from GPU Cholesky op.
        A = theano.tensor.matrix("A", dtype="float32")
        cholesky_op = GpuCholesky(lower=lower, inplace=inplace)
        chol_A = cholesky_op(A)
        return theano.function([A], chol_A, accept_inplace=inplace, mode=mode_with_gpu)

    def compare_gpu_cholesky_to_np(self, A_val, lower=True, inplace=False):
        # Helper function to compare op output to np.cholesky output.
        chol_A_val = np.linalg.cholesky(A_val)
        if not lower:
            chol_A_val = chol_A_val.T
        fn = self.get_gpu_cholesky_func(lower, inplace)
        res = fn(A_val)
        chol_A_res = np.array(res)
        utt.assert_allclose(chol_A_res, chol_A_val)

    def test_invalid_input_fail_non_square(self):
        # Invalid Cholesky input test with non-square matrix as input.
        A_val = np.random.normal(size=(3, 2)).astype("float32")
        fn = self.get_gpu_cholesky_func(True, False)
        self.assertRaises(ValueError, fn, A_val)

    def test_invalid_input_fail_vector(self):
        # Invalid Cholesky input test with vector as input.
        def invalid_input_func():
            A = theano.tensor.vector("A", dtype="float32")
            GpuCholesky(lower=True, inplace=False)(A)
        self.assertRaises(AssertionError, invalid_input_func)

    def test_invalid_input_fail_tensor3(self):
        # Invalid Cholesky input test with 3D tensor as input.
        def invalid_input_func():
            A = theano.tensor.tensor3("A", dtype="float32")
            GpuCholesky(lower=True, inplace=False)(A)
        self.assertRaises(AssertionError, invalid_input_func)

    def test_diag_chol(self):
        # Diagonal matrix input Cholesky test.
        for lower in [True, False]:
            for inplace in [True, False]:
                # make sure all diagonal elements are positive so positive-definite
                A_val = np.diag(np.random.uniform(size=5).astype("float32") + 1)
                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)

    def test_dense_chol_lower(self):
        # Dense matrix input lower-triangular Cholesky test.
        for lower in [True, False]:
            for inplace in [True, False]:
                M_val = np.random.normal(size=(3, 3)).astype("float32")
                # A = M.dot(M) will be positive definite for all non-singular M
                A_val = M_val.dot(M_val.T)
                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)

    def test_invalid_input_fail_non_symmetric(self):
        # Invalid Cholesky input test with non-symmetric input.
        #    (Non-symmetric real input must also be non-positive definite).
        A_val = None
        while True:
            A_val = np.random.normal(size=(3, 3)).astype("float32")
            if not np.allclose(A_val, A_val.T):
                break
        fn = self.get_gpu_cholesky_func(True, False)
        self.assertRaises(LinAlgError, fn, A_val)

    def test_invalid_input_fail_negative_definite(self):
        # Invalid Cholesky input test with negative-definite input.
        M_val = np.random.normal(size=(3, 3)).astype("float32")
        # A = -M.dot(M) will be negative definite for all non-singular M
        A_val = -M_val.dot(M_val.T)
        fn = self.get_gpu_cholesky_func(True, False)
        self.assertRaises(LinAlgError, fn, A_val)
