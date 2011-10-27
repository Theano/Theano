from pkg_resources import parse_version as V

import numpy

import theano
from theano import tensor, function
from theano.tensor.basic import _allclose
from theano.tensor.tests.test_rop import break_op
from theano.tests import unittest_tools as utt
from theano import config

try:
    import scipy
    if V(scipy.__version__) < V('0.7'):
        raise ImportError()
    use_scipy = True
except ImportError:
    use_scipy = False

# The one in comment are not tested...
from theano.sandbox.linalg.ops import (cholesky,
                                       matrix_inverse,
                                       #solve,
                                       #diag,
                                       ExtractDiag,
                                       extract_diag,
                                       #alloc_diag,
                                       det,
                                       #PSD_hint,
                                       trace,
                                       matrix_dot,
                                       #spectral_radius_bound
                                       )

from nose.plugins.skip import SkipTest


if 0:
    def test_cholesky():
        #TODO: test upper and lower triangular
        #todo: unittest randomseed
        rng = numpy.random.RandomState(utt.fetch_seed())

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
    rng = numpy.random.RandomState(utt.fetch_seed())

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


def test_matrix_dot():
    rng = numpy.random.RandomState(utt.fetch_seed())
    n = rng.randint(4) + 2
    rs = []
    xs = []
    for k in xrange(n):
        rs += [rng.randn(4, 4).astype(theano.config.floatX)]
        xs += [tensor.matrix()]
    sol = matrix_dot(*xs)

    theano_sol = function(xs, sol)(*rs)
    numpy_sol = rs[0]
    for r in rs[1:]:
        numpy_sol = numpy.dot(numpy_sol, r)

    assert _allclose(numpy_sol, theano_sol)


def test_inverse_singular():
    singular = numpy.array([[1, 0, 0]] + [[0, 1, 0]] * 2,
                           dtype=theano.config.floatX)
    a = tensor.matrix()
    f = function([a], matrix_inverse(a))
    try:
        f(singular)
    except numpy.linalg.LinAlgError:
        return
    assert False


def test_inverse_grad():
    rng = numpy.random.RandomState(utt.fetch_seed())
    r = rng.randn(4, 4)
    tensor.verify_grad(matrix_inverse, [r], rng=numpy.random)

    rng = numpy.random.RandomState(utt.fetch_seed())

    r = rng.randn(4,4)
    tensor.verify_grad(matrix_inverse, [r], rng=numpy.random)


def test_rop_lop():
    mx = tensor.matrix('mx')
    mv = tensor.matrix('mv')
    v  = tensor.vector('v')
    y = matrix_inverse(mx).sum(axis=0)

    yv = tensor.Rop(y, mx, mv)
    rop_f = function([mx, mv], yv)

    sy, _ = theano.scan( lambda i,y,x,v: (tensor.grad(y[i],x)*v).sum(),
                       sequences = tensor.arange(y.shape[0]),
                       non_sequences = [y,mx,mv])
    scan_f = function([mx,mv], sy)

    rng = numpy.random.RandomState(utt.fetch_seed())
    vx = numpy.asarray(rng.randn(4,4), theano.config.floatX)
    vv = numpy.asarray(rng.randn(4,4), theano.config.floatX)

    v1 = rop_f(vx,vv)
    v2 = scan_f(vx,vv)

    assert _allclose(v1, v2), ('ROP mismatch: %s %s' % (v1, v2))

    raised = False
    try:
        tmp = tensor.Rop(theano.clone(y,
                                      replace={mx:break_op(mx)}), mx, mv)
    except ValueError:
        raised = True
    if not raised:
        raise Exception((
            'Op did not raised an error even though the function'
            ' is not differentiable'))

    vv = numpy.asarray(rng.uniform(size=(4,)), theano.config.floatX)
    yv = tensor.Lop(y, mx, v)
    lop_f = function([mx, v], yv)

    sy = tensor.grad((v*y).sum(), mx)
    scan_f = function([mx, v], sy)

    v1 = lop_f(vx, vv)
    v2 = scan_f(vx, vv)
    assert _allclose(v1, v2), ('LOP mismatch: %s %s' % (v1, v2))


def test_det_grad():
    # If scipy is not available, this test will fail, thus we skip it.
    if not use_scipy:
        raise SkipTest('Scipy is not available')
    rng = numpy.random.RandomState(utt.fetch_seed())

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

    f = theano.function([x], g.shape)
    topo = f.maker.env.toposort()
    assert sum([node.op.__class__ == ExtractDiag for node in topo]) == 0
    m = rng.rand(3,3).astype(config.floatX)
    assert f(m) == 3

# not testing the view=True case since it is not used anywhere.


def test_trace():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.tensor.matrix()
    g = trace(x)
    f = theano.function([x], g)

    m = rng.rand(4, 4).astype(config.floatX)
    v = numpy.trace(m)
    assert v == f(m)

    xx = theano.tensor.vector()
    ok = False
    try:
        trace(xx)
    except TypeError:
        ok = True
    assert ok


