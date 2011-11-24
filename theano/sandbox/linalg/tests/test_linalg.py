from pkg_resources import parse_version as V

import numpy

import theano
from theano import tensor, function
from theano.tensor.basic import _allclose
from theano.tensor.tests.test_rop import break_op
from theano.tests import unittest_tools as utt
from theano import config

# The one in comment are not tested...
from theano.sandbox.linalg.ops import (cholesky,
                                       Cholesky, # op class
                                       CholeskyGrad,
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
                                       spectral_radius_bound
                                       )

from nose.plugins.skip import SkipTest


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
    rng = numpy.random.RandomState(utt.fetch_seed())
    r = rng.randn(5, 5).astype(config.floatX)
    pd = numpy.dot(r, r.T)
    # Check the default.
    yield utt.verify_grad, cholesky, [pd], 3, rng
    # Explicit lower-triangular.
    yield utt.verify_grad, Cholesky(lower=True), [pd], 3, rng
    # Explicit upper-triangular.
    yield utt.verify_grad, Cholesky(lower=False), [pd], 3, rng


def test_cholesky_and_cholesky_grad_shape():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = tensor.matrix()
    for l in (cholesky(x), Cholesky(lower=True)(x), Cholesky(lower=False)(x)):
        f_chol = theano.function([x], l.shape)
        g = tensor.grad(l.sum(), x)
        f_cholgrad = theano.function([x], g.shape)
        topo_chol = f_chol.maker.env.toposort()
        topo_cholgrad = f_cholgrad.maker.env.toposort()
        if config.mode != 'FAST_COMPILE':
            assert sum([node.op.__class__ == Cholesky
                        for node in topo_chol]) == 0
            assert sum([node.op.__class__ == CholeskyGrad
                        for node in topo_cholgrad]) == 0
        for shp in [2, 3, 5]:
            m = numpy.cov(rng.randn(shp, shp + 10)).astype(config.floatX)
            yield numpy.testing.assert_equal, f_chol(m), (shp, shp)
            yield numpy.testing.assert_equal, f_cholgrad(m), (shp, shp)


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


def test_det():
    rng = numpy.random.RandomState(utt.fetch_seed())

    r = rng.randn(5, 5).astype(config.floatX)
    x = tensor.matrix()
    f = theano.function([x], det(x))
    assert numpy.allclose(numpy.linalg.det(r), f(r))


def test_det_grad():
    rng = numpy.random.RandomState(utt.fetch_seed())

    r = rng.randn(5, 5).astype(config.floatX)
    tensor.verify_grad(det, [r], rng=numpy.random)


def test_det_shape():
    rng = numpy.random.RandomState(utt.fetch_seed())
    r = rng.randn(5, 5).astype(config.floatX)

    x = tensor.matrix()
    f = theano.function([x], det(x))
    f_shape = theano.function([x], det(x).shape)
    assert numpy.all(f(r).shape == f_shape(r))


def test_extract_diag():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.tensor.matrix()
    g = extract_diag(x)
    f = theano.function([x], g)

    for shp in [(2, 3), (3, 2), (3, 3)]:
        m = rng.rand(*shp).astype(config.floatX)
        v = numpy.diag(m)
        r = f(m)
        # The right diagonal is extracted
        assert (r == v).all()

    # Test we accept only matrix
    xx = theano.tensor.vector()
    ok = False
    try:
        extract_diag(xx)
    except TypeError:
        ok = True
    assert ok

    # Test infer_shape
    f = theano.function([x], g.shape)
    topo = f.maker.env.toposort()
    if config.mode != 'FAST_COMPILE':
        assert sum([node.op.__class__ == ExtractDiag for node in topo]) == 0
    for shp in [(2, 3), (3, 2), (3, 3)]:
        m = rng.rand(*shp).astype(config.floatX)
        assert f(m) == min(shp)

# not testing the view=True case since it is not used anywhere.


def test_trace():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.tensor.matrix()
    g = trace(x)
    f = theano.function([x], g)

    for shp in [(2, 3), (3, 2), (3, 3)]:
        m = rng.rand(*shp).astype(config.floatX)
        v = numpy.trace(m)
        assert v == f(m)

    xx = theano.tensor.vector()
    ok = False
    try:
        trace(xx)
    except TypeError:
        ok = True
    assert ok

def test_spectral_radius_bound():
    tol = 10**(-6)
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.tensor.matrix()
    radius_bound = spectral_radius_bound(x, 5)
    f = theano.function([x], radius_bound)
    
    shp = (3, 3)
    m = rng.rand(*shp).astype(config.floatX)
    m = numpy.cov(m)
    radius_bound_theano = f(m)
    
    # test the approximation
    mm = m
    for i in range(5):
        mm = numpy.dot(mm, mm)
    radius_bound_numpy = numpy.trace(mm)**(2**(-5))
    assert abs(radius_bound_numpy - radius_bound_theano) < tol
    
    # test the bound 
    eigen_val = numpy.linalg.eig(m)
    assert (eigen_val[0].max() - radius_bound_theano) < tol
    
    # test type errors
    xx = theano.tensor.vector()
    ok = False
    try:
        spectral_radius_bound(xx, 5)
    except TypeError:
        ok = True
    assert ok
    ok = False
    try:
        spectral_radius_bound(x, 5.)
    except TypeError:
        ok = True
    assert ok
    
    # test value error
    ok = False
    try:
        spectral_radius_bound(x, -5)
    except ValueError:
        ok = True
    assert ok
