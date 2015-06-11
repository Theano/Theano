import unittest
import numpy
from numpy.testing import assert_allclose

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
from theano.misc.pycuda_init import pycuda_available
from theano.sandbox.cuda.cula import cula_available

from theano.sandbox import cuda
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
    def run_gpu_solve(self, A_val, x_val):
        b_val = numpy.dot(A_val, x_val)
        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")

        solver = cula.gpu_solve(A, b)
        fn = theano.function([A, b], [solver])
        res = fn(A_val, b_val)
        x_res = numpy.array(res[0])
        utt.assert_allclose(x_res, x_val)

    def test_diag_solve(self):
        numpy.random.seed(1)
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = (A_val + A_val.T) / 2.0
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_sym, x_val)

    def test_orth_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = numpy.linalg.svd(A_val)[0]
        x_val = numpy.random.uniform(-0.4, 0.4, (A_orth.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     (A_val.shape[1], 4)).astype("float32")
        self.run_gpu_solve(A_val, x_val)


def test_syev():
    try:
        from scipy.linalg.lapack import dsyev
    except ImportError:
        raise SkipTest('SciPy unavailable')

    rng = numpy.random.RandomState(utt.fetch_seed())

    def assert_geqrf_allclose(compute_v, lower, n):
        A = rng.rand(n, n).astype(theano.config.floatX)
        A = A + A.T - numpy.diag(numpy.diag(A))
        A_ = theano.shared(cuda.CudaNdarray(A), borrow=True)
        A_T = theano.shared(cuda.CudaNdarray(A.T), borrow=True)
        dsyev_ = theano.function([], cula.gpu_syev(A_, compute_v, lower))
        dsyev_T = theano.function([], cula.gpu_syev(A_T.T, compute_v, lower))
        w, v, _ = dsyev(A, compute_v, lower)
        w_, v_ = dsyev_()
        w_T, v_T = dsyev_T()
        assert_allclose(w, numpy.array(w_), 1e-4)
        assert_allclose(w, numpy.array(w_T), 1e-4)
        if compute_v:
            # Eigenvector signs can be different
            v_ = numpy.sign(v[:1, :]) * numpy.sign(v_[:1, :]) * numpy.array(v_)
            v_T = (numpy.sign(v[:1, :]) * numpy.sign(v_T[:1, :]) *
                   numpy.array(v_T))
            assert_allclose(v, v_, 1e-4)
            assert_allclose(v, v_T, 1e-4)

    yield assert_geqrf_allclose, 1, 0, 5
    yield assert_geqrf_allclose, 1, 1, 7
    yield assert_geqrf_allclose, 0, 1, 50
    yield assert_geqrf_allclose, 0, 0, 15
