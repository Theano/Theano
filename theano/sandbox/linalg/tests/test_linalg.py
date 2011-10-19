import numpy

import theano
from theano import tensor, function
from theano.tensor.basic import _allclose
from theano.tests import unittest_tools as utt
from theano import config

utt.seed_rng()

try:
    import scipy
    if scipy.__version__ < '0.7':
        raise ImportError()
    use_scipy = True
except ImportError:
    use_scipy = False

# The one in comment are not tested...
from theano.sandbox.linalg.ops import (cholesky,
                                       matrix_inverse,
                                       #solve,
                                       diag,
                                       extract_diag,
                                       alloc_diag,
                                       det,
                                       #PSD_hint,
                                       #trace,
                                       #spectral_radius_bound
                                       )

from nose.plugins.skip import SkipTest


if 0:
    def test_cholesky():
        #TODO: test upper and lower triangular
        #todo: unittest randomseed
        rng = numpy.random.RandomState(1234)

        r = rng.randn(5,5)

        pd = numpy.dot(r,r.T)

        x = tensor.matrix()
        chol = cholesky(x)
        f = function([x], tensor.dot(chol, chol.T)) # an optimization could remove this

        ch_f = function([x], chol)

        # quick check that chol is upper-triangular
        ch = ch_f(pd)
        print ch
        assert ch[0,4] != 0
        assert ch[4,0] == 0
        assert numpy.allclose(numpy.dot(ch.T,ch),pd)
        assert not numpy.allclose(numpy.dot(ch,ch.T),pd)


def test_inverse_correctness():
    #todo: unittest randomseed
    rng = numpy.random.RandomState(12345)

    r = rng.randn(4,4).astype(theano.config.floatX)

    x = tensor.matrix()
    xi = matrix_inverse(x)

    ri = function([x], xi)(r)
    assert ri.shape == r.shape
    assert ri.dtype == r.dtype

    rir = numpy.dot(ri,r)
    rri = numpy.dot(r,ri)

    assert _allclose(numpy.identity(4), rir), rir
    assert _allclose(numpy.identity(4), rri), rri

def test_inverse_grad():

    rng = numpy.random.RandomState(1234)

    r = rng.randn(4,4)
    tensor.verify_grad(matrix_inverse, [r], rng=numpy.random)


def test_det_grad():
    # If scipy is not available, this test will fail, thus we skip it.
    if not use_scipy:
        raise SkipTest('Scipy is not available')
    rng = numpy.random.RandomState(1234)

    r = rng.randn(5,5)
    tensor.verify_grad(det, [r], rng=numpy.random)

def test_extract_diag():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.tensor.matrix()
    g = extract_diag(x)
    f = theano.function([x], g)

    m = rng.rand(3,3).astype(config.floatX)
    v = numpy.diag(m)
    r = f(m)
    # The right diagonal is extracted
    assert (r == v).all()

    m = rng.rand(2, 3).astype(config.floatX)
    ok = False
    try:
        r = f(m)
    except Exception:
        ok = True
    assert ok

    xx = theano.tensor.vector()
    ok = False
    try:
        extract_diag(xx)
    except TypeError:
        ok = True
    assert ok
# not testing the view=True case since it is not used anywhere.
