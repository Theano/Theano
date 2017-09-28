from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.linalg.linalg import LinAlgError

import theano
from theano import config
from theano.gpuarray.linalg import (GpuCholesky, GpuMagmaCholesky,
                                    GpuMagmaEigh, GpuMagmaMatrixInverse,
                                    GpuMagmaQR, GpuMagmaSVD,
                                    cusolver_available, gpu_matrix_inverse,
                                    gpu_solve, gpu_svd, gpu_qr)
from theano.tensor.nlinalg import (SVD, MatrixInverse, QRFull,
                                   QRIncomplete, eigh, matrix_inverse, qr)
from theano.tensor.slinalg import Cholesky, cholesky, imported_scipy
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
        # Test when shape of b (k, m) is such as m > k

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
        return theano.function([A], chol_A, accept_inplace=inplace,
                               mode=mode_with_gpu)

    def compare_gpu_cholesky_to_np(self, A_val, lower=True, inplace=False):
        # Helper function to compare op output to np.cholesky output.
        chol_A_val = np.linalg.cholesky(A_val)
        if not lower:
            chol_A_val = chol_A_val.T
        fn = self.get_gpu_cholesky_func(lower, inplace)
        res = fn(A_val)
        chol_A_res = np.array(res)
        utt.assert_allclose(chol_A_res, chol_A_val)

    def test_gpu_cholesky_opt(self):
        if not imported_scipy:
            self.skipTest('SciPy is not enabled, skipping test')
        A = theano.tensor.matrix("A", dtype="float32")
        fn = theano.function([A], cholesky(A), mode=mode_with_gpu)
        assert any([isinstance(node.op, GpuCholesky)
                    for node in fn.maker.fgraph.toposort()])

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

    @utt.assertFailure_fast
    def test_diag_chol(self):
        # Diagonal matrix input Cholesky test.
        for lower in [True, False]:
            for inplace in [True, False]:
                # make sure all diagonal elements are positive so positive-definite
                A_val = np.diag(np.random.uniform(size=5).astype("float32") + 1)
                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)

    @utt.assertFailure_fast
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

    def test_magma_opt_float16(self):
        ops_to_gpu = [(MatrixInverse(), GpuMagmaMatrixInverse),
                      (SVD(), GpuMagmaSVD),
                      (QRFull(mode='reduced'), GpuMagmaQR),
                      (QRIncomplete(mode='r'), GpuMagmaQR),
                      # TODO: add support for float16 to Eigh numpy
                      # (Eigh(), GpuMagmaEigh),
                      (Cholesky(), GpuMagmaCholesky)]
        for op, gpu_op in ops_to_gpu:
            A = theano.tensor.matrix("A", dtype="float16")
            fn = theano.function([A], op(A), mode=mode_with_gpu.excluding('cusolver'))
            assert any([isinstance(node.op, gpu_op)
                        for node in fn.maker.fgraph.toposort()])

    def test_gpu_matrix_inverse(self):
        A = theano.tensor.fmatrix("A")

        fn = theano.function([A], gpu_matrix_inverse(A), mode=mode_with_gpu)
        N = 1000
        test_rng = np.random.RandomState(seed=1)
        # Copied from theano.tensor.tests.test_basic.rand.
        A_val = test_rng.rand(N, N).astype('float32') * 2 - 1
        A_val_inv = fn(A_val)
        utt.assert_allclose(np.eye(N), np.dot(A_val_inv, A_val), atol=1e-2)

    @utt.assertFailure_fast
    def test_gpu_matrix_inverse_inplace(self):
        N = 1000
        test_rng = np.random.RandomState(seed=1)
        A_val_gpu = gpuarray_shared_constructor(test_rng.rand(N, N).astype('float32') * 2 - 1)
        A_val_copy = A_val_gpu.get_value()
        A_val_gpu_inv = GpuMagmaMatrixInverse()(A_val_gpu)
        fn = theano.function([], A_val_gpu_inv, mode=mode_with_gpu, updates=[(A_val_gpu, A_val_gpu_inv)])
        assert any([
            node.op.inplace
            for node in fn.maker.fgraph.toposort() if
            isinstance(node.op, GpuMagmaMatrixInverse)
        ])
        fn()
        utt.assert_allclose(np.eye(N), np.dot(A_val_gpu.get_value(), A_val_copy), atol=5e-3)

    @utt.assertFailure_fast
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

    def run_gpu_cholesky(self, A_val, lower=True):
        A = theano.tensor.fmatrix("A")
        f = theano.function([A], GpuMagmaCholesky(lower=lower)(A),
                            mode=mode_with_gpu.excluding('cusolver'))
        return f(A_val)

    def rand_symmetric(self, N):
        A = rand(N, N).astype('float32')
        # ensure that eigenvalues are not too small which sometimes results in
        # magma cholesky failure due to gpu limited numerical precision
        D, W = np.linalg.eigh(A)
        D[D < 1] = 1
        V_m = np.zeros_like(A)
        np.fill_diagonal(V_m, D)
        return np.dot(np.dot(W.T, V_m), W)

    def check_cholesky(self, N, lower=True, rtol=None, atol=None):
        A = self.rand_symmetric(N)
        L = self.run_gpu_cholesky(A, lower=lower)
        if not lower:
            L = L.T
        utt.assert_allclose(np.dot(L, L.T), A, rtol=rtol, atol=atol)

    def test_gpu_cholesky(self):
        self.check_cholesky(1000, atol=1e-3)
        self.check_cholesky(1000, lower=False, atol=1e-3)

    def test_gpu_cholesky_opt(self):
        A = theano.tensor.matrix("A", dtype="float32")
        fn = theano.function([A], cholesky(A), mode=mode_with_gpu.excluding('cusolver'))
        assert any([isinstance(node.op, GpuMagmaCholesky)
                    for node in fn.maker.fgraph.toposort()])

    @utt.assertFailure_fast
    def test_gpu_cholesky_inplace(self):
        A = self.rand_symmetric(1000)
        A_gpu = gpuarray_shared_constructor(A)
        A_copy = A_gpu.get_value()
        C = GpuMagmaCholesky()(A_gpu)
        fn = theano.function([], C, mode=mode_with_gpu, updates=[(A_gpu, C)])
        assert any([
            node.op.inplace
            for node in fn.maker.fgraph.toposort() if
            isinstance(node.op, GpuMagmaCholesky)
        ])
        fn()
        L = A_gpu.get_value()
        utt.assert_allclose(np.dot(L, L.T), A_copy, atol=1e-3)

    @utt.assertFailure_fast
    def test_gpu_cholesky_inplace_opt(self):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], GpuMagmaCholesky()(A), mode=mode_with_gpu)
        assert any([
            node.op.inplace
            for node in fn.maker.fgraph.toposort() if
            isinstance(node.op, GpuMagmaCholesky)
        ])

    def run_gpu_qr(self, A_val, complete=True):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], gpu_qr(A, complete=complete),
                             mode=mode_with_gpu)
        return fn(A_val)

    def check_gpu_qr(self, M, N, complete=True, rtol=None, atol=None):
        A = rand(M, N).astype('float32')
        if complete:
            Q_gpu, R_gpu = self.run_gpu_qr(A, complete=complete)
        else:
            R_gpu = self.run_gpu_qr(A, complete=complete)

        Q_np, R_np = np.linalg.qr(A, mode='reduced')
        utt.assert_allclose(R_np, R_gpu, rtol=rtol, atol=atol)
        if complete:
            utt.assert_allclose(Q_np, Q_gpu, rtol=rtol, atol=atol)

    def test_gpu_qr(self):
        self.check_gpu_qr(1000, 500, atol=1e-3)
        self.check_gpu_qr(1000, 500, complete=False, atol=1e-3)
        self.check_gpu_qr(500, 1000, atol=1e-3)
        self.check_gpu_qr(500, 1000, complete=False, atol=1e-3)

    def test_gpu_qr_opt(self):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], qr(A), mode=mode_with_gpu)
        assert any([
            isinstance(node.op, GpuMagmaQR) and node.op.complete
            for node in fn.maker.fgraph.toposort()
        ])

    def test_gpu_qr_incomplete_opt(self):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], qr(A, mode='r'), mode=mode_with_gpu)
        assert any([
            isinstance(node.op, GpuMagmaQR) and not node.op.complete
            for node in fn.maker.fgraph.toposort()
        ])

    def run_gpu_eigh(self, A_val, UPLO='L', compute_v=True):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], GpuMagmaEigh(UPLO=UPLO, compute_v=compute_v)(A),
                             mode=mode_with_gpu)
        return fn(A_val)

    def check_gpu_eigh(self, N, UPLO='L', compute_v=True, rtol=None, atol=None):
        A = rand(N, N).astype('float32')
        A = np.dot(A.T, A)
        d_np, v_np = np.linalg.eigh(A, UPLO=UPLO)
        if compute_v:
            d_gpu, v_gpu = self.run_gpu_eigh(A, UPLO=UPLO, compute_v=compute_v)
        else:
            d_gpu = self.run_gpu_eigh(A, UPLO=UPLO, compute_v=False)
        utt.assert_allclose(d_np, d_gpu, rtol=rtol, atol=atol)
        if compute_v:
            utt.assert_allclose(
                np.eye(N), np.dot(v_gpu, v_gpu.T), rtol=rtol, atol=atol)
            D_m = np.zeros_like(A)
            np.fill_diagonal(D_m, d_gpu)
            utt.assert_allclose(
                A, np.dot(np.dot(v_gpu, D_m), v_gpu.T), rtol=rtol, atol=atol)

    def test_gpu_eigh(self):
        self.check_gpu_eigh(1000, UPLO='L', atol=1e-3)
        self.check_gpu_eigh(1000, UPLO='U', atol=1e-3)
        self.check_gpu_eigh(1000, UPLO='L', compute_v=False, atol=1e-3)
        self.check_gpu_eigh(1000, UPLO='U', compute_v=False, atol=1e-3)

    def test_gpu_eigh_opt(self):
        A = theano.tensor.fmatrix("A")
        fn = theano.function([A], eigh(A), mode=mode_with_gpu)
        assert any([
            isinstance(node.op, GpuMagmaEigh)
            for node in fn.maker.fgraph.toposort()
        ])
