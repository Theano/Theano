import numpy
from nose.plugins.skip import SkipTest
from numpy.testing import assert_allclose
from scipy.linalg.lapack import dgeqrf

import theano
from theano.sandbox.cuda.cusolver import geqrf, cusolver_available
from theano.sandbox.cuda import cuda_available, CudaNdarray

if not cuda_available:
    raise SkipTest('CUDA not available')
if not cusolver_available:
    raise SkipTest('cuSOLVER not available')


def test_geqrf():
    rng = numpy.random.RandomState(1)

    def assert_geqrf_allclose(m, n):
        A = rng.rand(m, n).astype(theano.config.floatX)
        A_ = theano.shared(CudaNdarray(A), borrow=True)
        A_T = theano.shared(CudaNdarray(A.T), borrow=True)
        theano.config.optimizer = 'None'
        dgeqrf_ = theano.function([], geqrf(A_))
        dgeqrf_T = theano.function([], geqrf(A_T.T))
        qr_, tau_, work_, info_ = dgeqrf_()
        qr_T, tau_T, work_T, info_T = dgeqrf_T()
        qr, tau, work, info = dgeqrf(numpy.array(A))
        assert_allclose(qr, qr_.eval(), 1e-3)
        assert_allclose(qr, qr_T.eval(), 1e-3)
        assert_allclose(tau, tau_, 1e-3)
        assert_allclose(tau, tau_T, 1e-3)
        assert not info and not numpy.array(info_)

    yield assert_geqrf_allclose, 10, 10
    yield assert_geqrf_allclose, 20, 10
    yield assert_geqrf_allclose, 10, 20
