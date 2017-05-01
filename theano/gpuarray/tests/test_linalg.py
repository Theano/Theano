from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.linalg.linalg import LinAlgError

import theano
from theano import config
from theano.gpuarray.linalg import (GpuCholesky, GpuMagmaMatrixInverse,
                                    cusolver_available, gpu_matrix_inverse,
                                    gpu_solve, gpu_svd)
from theano.tensor.nlinalg import matrix_inverse
from theano.tests import unittest_tools as utt

from .. import gpuarray_shared_constructor
from .config import mode_with_gpu, mode_without_gpu
from .test_basic_ops import rand


class TestCusolver(unittest.TestCase):

    def setUp(self):
        if not cusolver_available:
            self.skipTest('Optional package scikits.cuda.cusolver not available')

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
        if not cusolver_available:
            self.skipTest('Optional package scikits.cuda.cusolver not available')
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


class TestMagma(unittest.TestCase):

    def setUp(self):
        if not config.magma.enabled:
            self.skipTest('Magma is not enabled, skipping test')

    def test_gpu_matrix_inverse(self):
        A = theano.tensor.fmatrix("A")

        fn = theano.function([A], gpu_matrix_inverse(A), mode=mode_with_gpu)
        N = 1000
        A_val = rand(N, N).astype('float32')
        A_val_inv = fn(A_val)
        utt.assert_allclose(np.dot(A_val_inv, A_val), np.eye(N), atol=1e-3)

    def test_gpu_matrix_inverse_inplace(self):
        N = 1000
        A_val_gpu = gpuarray_shared_constructor(rand(N, N).astype('float32'))
        A_val_copy = A_val_gpu.get_value()
        fn = theano.function([], GpuMagmaMatrixInverse(inplace=True)(A_val_gpu),
                             mode=mode_with_gpu, accept_inplace=True)
        fn()
        utt.assert_allclose(np.dot(A_val_gpu.get_value(), A_val_copy), np.eye(N), atol=1e-3)

    def test_gpu_matrix_inverse_inplace_opt(self):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], matrix_inverse(A), mode=mode_with_gpu)
        assert any([
            node.op.inplace
            for node in fn.maker.fgraph.toposort() if
            isinstance(node.op, GpuMagmaMatrixInverse)
        ])

    def run_gpu_svd(self, A_val, full_matrices=True, compute_uv=True):
        A = theano.tensor.fmatrix("A")
        f = theano.function(
            [A], gpu_svd(A, full_matrices=full_matrices, compute_uv=compute_uv),
            mode=mode_with_gpu)
        return f(A_val)

    def assert_column_orthonormal(self, Ot):
        utt.assert_allclose(np.dot(Ot.T, Ot), np.eye(Ot.shape[1]))

    def check_svd(self, A, U, S, VT, rtol=None, atol=None):
        S_m = np.zeros_like(A)
        np.fill_diagonal(S_m, S)
        utt.assert_allclose(
            np.dot(np.dot(U, S_m), VT), A, rtol=rtol, atol=atol)

    def test_gpu_svd_wide(self):
        A = rand(100, 50).astype('float32')
        M, N = A.shape

        U, S, VT = self.run_gpu_svd(A)
        self.assert_column_orthonormal(U)
        self.assert_column_orthonormal(VT.T)
        self.check_svd(A, U, S, VT)

        U, S, VT = self.run_gpu_svd(A, full_matrices=False)
        self.assertEqual(U.shape[1], min(M, N))
        self.assert_column_orthonormal(U)
        self.assertEqual(VT.shape[0], min(M, N))
        self.assert_column_orthonormal(VT.T)

    def test_gpu_svd_tall(self):
        A = rand(50, 100).astype('float32')
        M, N = A.shape

        U, S, VT = self.run_gpu_svd(A)
        self.assert_column_orthonormal(U)
        self.assert_column_orthonormal(VT.T)
        self.check_svd(A, U, S, VT)

        U, S, VT = self.run_gpu_svd(A, full_matrices=False)
        self.assertEqual(U.shape[1], min(M, N))
        self.assert_column_orthonormal(U)
        self.assertEqual(VT.shape[0], min(M, N))
        self.assert_column_orthonormal(VT.T)

    def test_gpu_singular_values(self):
        A = theano.tensor.fmatrix("A")
        f_cpu = theano.function(
            [A], theano.tensor.nlinalg.svd(A, compute_uv=False),
            mode=mode_without_gpu)
        f_gpu = theano.function(
            [A], gpu_svd(A, compute_uv=False), mode=mode_with_gpu)

        A_val = rand(50, 100).astype('float32')
        utt.assert_allclose(f_cpu(A_val), f_gpu(A_val))

        A_val = rand(100, 50).astype('float32')
        utt.assert_allclose(f_cpu(A_val), f_gpu(A_val))
