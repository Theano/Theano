import unittest

import numpy
import numpy.linalg
from numpy.testing import assert_array_almost_equal
from numpy.testing import dec, assert_array_equal, assert_allclose
from numpy import inf

import theano
from theano import tensor, function
from theano.tensor.basic import _allclose
from theano.tests.test_rop import break_op
from theano.tests import unittest_tools as utt
from theano import config

from theano.tensor.slinalg import ( Cholesky,
                                    cholesky,
                                    CholeskyGrad,
                                    Solve,
                                    solve,
                                    Eigvalsh,
                                    EigvalshGrad,
                                    eigvalsh,
                                    expm
                                    )

from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
from nose.tools import assert_raises

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False

def check_lower_triangular(pd, ch_f):
    ch = ch_f(pd)
    assert ch[0, pd.shape[1] - 1] == 0
    assert ch[pd.shape[0] - 1, 0] != 0
    assert numpy.allclose(numpy.dot(ch, ch.T), pd)
    assert not numpy.allclose(numpy.dot(ch.T, ch), pd)


def check_upper_triangular(pd, ch_f):
    ch = ch_f(pd)
    assert ch[4, 0] == 0
    assert ch[0, 4] != 0
    assert numpy.allclose(numpy.dot(ch.T, ch), pd)
    assert not numpy.allclose(numpy.dot(ch, ch.T), pd)


def test_cholesky():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the Cholesky op.")

    rng = numpy.random.RandomState(utt.fetch_seed())
    r = rng.randn(5, 5).astype(config.floatX)
    pd = numpy.dot(r, r.T)
    x = tensor.matrix()
    chol = cholesky(x)
    # Check the default.
    ch_f = function([x], chol)
    yield check_lower_triangular, pd, ch_f
    # Explicit lower-triangular.
    chol = Cholesky(lower=True)(x)
    ch_f = function([x], chol)
    yield check_lower_triangular, pd, ch_f
    # Explicit upper-triangular.
    chol = Cholesky(lower=False)(x)
    ch_f = function([x], chol)
    yield check_upper_triangular, pd, ch_f


def test_cholesky_grad():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the Cholesky op.")
    rng = numpy.random.RandomState(utt.fetch_seed())
    r = rng.randn(5, 5).astype(config.floatX)
    pd = numpy.dot(r, r.T)
    eps = None
    if config.floatX == "float64":
        eps = 2e-8
    # Check the default.
    yield (lambda: utt.verify_grad(cholesky, [pd], 3, rng, eps=eps))
    # Explicit lower-triangular.
    yield (lambda: utt.verify_grad(Cholesky(lower=True), [pd], 3,
                                   rng, eps=eps))
    # Explicit upper-triangular.
    yield (lambda: utt.verify_grad(Cholesky(lower=False), [pd], 3,
                                   rng, eps=eps))


@attr('slow')
def test_cholesky_and_cholesky_grad_shape():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the Cholesky op.")

    rng = numpy.random.RandomState(utt.fetch_seed())
    x = tensor.matrix()
    for l in (cholesky(x), Cholesky(lower=True)(x), Cholesky(lower=False)(x)):
        f_chol = theano.function([x], l.shape)
        g = tensor.grad(l.sum(), x)
        f_cholgrad = theano.function([x], g.shape)
        topo_chol = f_chol.maker.fgraph.toposort()
        topo_cholgrad = f_cholgrad.maker.fgraph.toposort()
        if config.mode != 'FAST_COMPILE':
            assert sum([node.op.__class__ == Cholesky
                        for node in topo_chol]) == 0
            assert sum([node.op.__class__ == CholeskyGrad
                        for node in topo_cholgrad]) == 0
        for shp in [2, 3, 5]:
            m = numpy.cov(rng.randn(shp, shp + 10)).astype(config.floatX)
            yield numpy.testing.assert_equal, f_chol(m), (shp, shp)
            yield numpy.testing.assert_equal, f_cholgrad(m), (shp, shp)



def test_eigvalsh():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the geigvalsh op.")
    import scipy.linalg

    A = theano.tensor.dmatrix('a')
    B = theano.tensor.dmatrix('b')
    f = function([A, B], eigvalsh(A, B))

    rng = numpy.random.RandomState(utt.fetch_seed())
    a = rng.randn(5, 5)
    a = a + a.T
    for b in [10 * numpy.eye(5, 5) + rng.randn(5, 5)]:
        w = f(a, b)
        refw = scipy.linalg.eigvalsh(a, b)
        numpy.testing.assert_array_almost_equal(w, refw)

    # We need to test None separatly, as otherwise DebugMode will
    # complain, as this isn't a valid ndarray.
    b = None
    B = theano.tensor.NoneConst
    f = function([A], eigvalsh(A, B))
    w = f(a)
    refw = scipy.linalg.eigvalsh(a, b)
    numpy.testing.assert_array_almost_equal(w, refw)


def test_eigvalsh_grad():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the geigvalsh op.")
    import scipy.linalg

    rng = numpy.random.RandomState(utt.fetch_seed())
    a = rng.randn(5, 5)
    a = a + a.T
    b = 10 * numpy.eye(5, 5) + rng.randn(5, 5)
    tensor.verify_grad(lambda a, b: eigvalsh(a, b).dot([1, 2, 3, 4, 5]),
                       [a, b], rng=numpy.random)


class test_Solve(utt.InferShapeTester):
    def setUp(self):
        super(test_Solve, self).setUp()
        self.op_class = Solve
        self.op = Solve()

    def test_infer_shape(self):
        if not imported_scipy:
            raise SkipTest("Scipy needed for the Cholesky op.")
        rng = numpy.random.RandomState(utt.fetch_seed())
        A = theano.tensor.matrix()
        b = theano.tensor.matrix()
        self._compile_and_check([A, b],  # theano.function inputs
                                [self.op(A, b)],  # theano.function outputs
                                # A must be square
                                [numpy.asarray(rng.rand(5, 5),
                                               dtype=config.floatX),
                                 numpy.asarray(rng.rand(5, 1),
                                               dtype=config.floatX)],
                                self.op_class,
                                warn=False)
        rng = numpy.random.RandomState(utt.fetch_seed())
        A = theano.tensor.matrix()
        b = theano.tensor.vector()
        self._compile_and_check([A, b],  # theano.function inputs
                                [self.op(A, b)],  # theano.function outputs
                                # A must be square
                                [numpy.asarray(rng.rand(5, 5),
                                               dtype=config.floatX),
                                 numpy.asarray(rng.rand(5),
                                               dtype=config.floatX)],
                                self.op_class,
                                warn=False)


def test_expm():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the expm op.")
    rng = numpy.random.RandomState(utt.fetch_seed())
    A = rng.randn(5, 5).astype(config.floatX)

    ref = scipy.linalg.expm(A)

    x = tensor.matrix()
    m = expm(x)
    expm_f = function([x], m)

    val = expm_f(A)
    numpy.testing.assert_array_almost_equal(val, ref)


def test_expm_grad_1():
    # with symmetric matrix (real eigenvectors)
    if not imported_scipy:
        raise SkipTest("Scipy needed for the expm op.")
    rng = numpy.random.RandomState(utt.fetch_seed())
    A = rng.randn(3, 3).astype(config.floatX)
    A = A + A.T

    tensor.verify_grad(expm, [A,], rng=rng)
