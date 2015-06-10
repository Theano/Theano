import numpy
from nose.plugins.skip import SkipTest
from numpy.testing import assert_allclose
from scipy.linalg.lapack import dgeqrf

import theano
from theano import tensor
from theano.sandbox.cuda.cusolver import geqrf, cusolver_available
from theano.sandbox.cuda import cuda_available

if not cuda_available:
    raise SkipTest('CUDA not available')
if not cusolver_available:
    raise SkipTest('cuSOLVER not available')


def test_geqrf():
    rng = numpy.random.RandomState(1)

    def assert_geqrf_allclose(m, n):
        A = rng.rand(m, n).astype(theano.config.floatX)
        A_tensor = tensor.fmatrix('A')
        dgeqrf_ = theano.function([A_tensor], geqrf(A_tensor))
        qr_, tau_, work_, info_ = dgeqrf_(A)
        qr, tau, work, info = dgeqrf(A)
        assert_allclose(qr, qr_.eval(), 1e-3)
        assert_allclose(tau, tau_, 1e-3)
        assert not info and not numpy.array(info_)

    yield assert_geqrf_allclose, 10, 10
    yield assert_geqrf_allclose, 20, 10
    yield assert_geqrf_allclose, 10, 20
