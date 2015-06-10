import numpy
from nose.plugins.skip import SkipTest
from numpy.testing import assert_allclose
try:
    from scipy.linalg.lapack import dgeqrf, dormqr
except ImportError:
    raise SkipTest('SciPy not available')

import theano
from theano.sandbox.cuda.cusolver import (gpu_geqrf, gpu_ormqr,
                                          cusolver_available)
from theano.sandbox.cuda import cuda_available, CudaNdarray
from theano.tests.unittest_tools import fetch_seed

if not cuda_available:
    raise SkipTest('CUDA not available')
if not cusolver_available:
    raise SkipTest('cuSOLVER not available')


def test_geqrf():
    rng = numpy.random.RandomState(fetch_seed())

    def assert_geqrf_allclose(m, n):
        A = rng.rand(m, n).astype(theano.config.floatX)
        A_ = theano.shared(CudaNdarray(A), borrow=True)
        A_T = theano.shared(CudaNdarray(A.T), borrow=True)
        dgeqrf_ = theano.function([], gpu_geqrf(A_))
        dgeqrf_T = theano.function([], gpu_geqrf(A_T.T))
        qr, tau, work, info = dgeqrf(A)
        qr_, tau_, work_, info_ = dgeqrf_()
        qr_T, tau_T, work_T, info_T = dgeqrf_T()
        assert_allclose(qr, numpy.array(qr_), 1e-3)
        assert_allclose(qr, numpy.array(qr_T), 1e-3)
        assert_allclose(tau, tau_, 1e-3)
        assert_allclose(tau, tau_T, 1e-3)
        assert not info and not numpy.array(info_)

    yield assert_geqrf_allclose, 10, 10
    yield assert_geqrf_allclose, 20, 10
    yield assert_geqrf_allclose, 10, 20


def test_ormqr():
    rng = numpy.random.RandomState(fetch_seed())

    def assert_ormqr_allclose(side, trans, a, tau, m, n, lwork):
        a = a.astype(theano.config.floatX)
        tau = tau.astype(theano.config.floatX)

        C = rng.rand(m, n).astype(theano.config.floatX)
        a_ = theano.shared(a)
        tau_ = theano.shared(tau)
        C_ = theano.shared(CudaNdarray(C), borrow=True)
        C_T = theano.shared(CudaNdarray(C.T), borrow=True)
        dormqr_ = theano.function([], gpu_ormqr(side, trans, a_, tau_,
                                                C_, lwork))
        dormqr_T = theano.function([], gpu_ormqr(side, trans, a_, tau_,
                                                 C_T.T, lwork))
        cq, work, info = dormqr(side, trans, a, tau, C, lwork)
        cq_, work_, info_ = dormqr_()
        cq_T, work_T, info_T = dormqr_T()
        assert_allclose(cq, numpy.array(cq_), 1e-3)
        assert_allclose(cq, numpy.array(cq_T), 1e-3)
        assert not info and not numpy.array(info_)

    a, tau, _, _ = dgeqrf(numpy.random.rand(10, 10))

    yield assert_ormqr_allclose, 'L', 'N', a, tau, 10, 10, 10
    yield assert_ormqr_allclose, 'R', 'N', a, tau, 10, 10, 10
    yield assert_ormqr_allclose, 'L', 'T', a, tau, 10, 10, 10
    yield assert_ormqr_allclose, 'R', 'T', a, tau, 10, 10, 10

    a, tau, _, _ = dgeqrf(numpy.random.rand(20, 10))

    yield assert_ormqr_allclose, 'L', 'N', a, tau, 10, 10, 10
    yield assert_ormqr_allclose, 'R', 'T', a, tau, 10, 10, 10
