import unittest

import numpy
import numpy.linalg
from numpy.testing import assert_array_almost_equal

import theano
from theano import tensor, function
from theano.tensor.basic import _allclose
from theano.tests.test_rop import break_op
from theano.tests import unittest_tools as utt
from theano import config

# The one in comment are not tested...
from theano.sandbox.linalg.ops import (cholesky,
                                       Cholesky,  # op class
                                       CholeskyGrad,
                                       matrix_inverse,
                                       pinv,
                                       Solve,
                                       diag,
                                       ExtractDiag,
                                       extract_diag,
                                       AllocDiag,
                                       alloc_diag,
                                       det,
                                       #PSD_hint,
                                       trace,
                                       matrix_dot,
                                       spectral_radius_bound,
                                       imported_scipy,
                                       Eig,
                                       inv_as_solve
                                       )
from theano.sandbox.linalg import eig, eigh, eigvalsh
from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr


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


def test_inverse_correctness():
    rng = numpy.random.RandomState(utt.fetch_seed())

    r = rng.randn(4, 4).astype(theano.config.floatX)

    x = tensor.matrix()
    xi = matrix_inverse(x)

    ri = function([x], xi)(r)
    assert ri.shape == r.shape
    assert ri.dtype == r.dtype

    rir = numpy.dot(ri, r)
    rri = numpy.dot(r, ri)

    assert _allclose(numpy.identity(4), rir), rir
    assert _allclose(numpy.identity(4), rri), rri


def test_pseudoinverse_correctness():
    rng = numpy.random.RandomState(utt.fetch_seed())
    d1 = rng.randint(4) + 2
    d2 = rng.randint(4) + 2
    r = rng.randn(d1, d2).astype(theano.config.floatX)

    x = tensor.matrix()
    xi = pinv(x)

    ri = function([x], xi)(r)
    assert ri.shape[0] == r.shape[1]
    assert ri.shape[1] == r.shape[0]
    assert ri.dtype == r.dtype
    # Note that pseudoinverse can be quite unprecise so I prefer to compare
    # the result with what numpy.linalg returns
    assert _allclose(ri, numpy.linalg.pinv(r))


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

    r = rng.randn(4, 4)
    tensor.verify_grad(matrix_inverse, [r], rng=numpy.random)


def test_rop_lop():
    mx = tensor.matrix('mx')
    mv = tensor.matrix('mv')
    v = tensor.vector('v')
    y = matrix_inverse(mx).sum(axis=0)

    yv = tensor.Rop(y, mx, mv)
    rop_f = function([mx, mv], yv)

    sy, _ = theano.scan(lambda i, y, x, v: (tensor.grad(y[i], x) * v).sum(),
                        sequences=tensor.arange(y.shape[0]),
                        non_sequences=[y, mx, mv])
    scan_f = function([mx, mv], sy)

    rng = numpy.random.RandomState(utt.fetch_seed())
    vx = numpy.asarray(rng.randn(4, 4), theano.config.floatX)
    vv = numpy.asarray(rng.randn(4, 4), theano.config.floatX)

    v1 = rop_f(vx, vv)
    v2 = scan_f(vx, vv)

    assert _allclose(v1, v2), ('ROP mismatch: %s %s' % (v1, v2))

    raised = False
    try:
        tensor.Rop(
            theano.clone(y, replace={mx: break_op(mx)}),
            mx,
            mv)
    except ValueError:
        raised = True
    if not raised:
        raise Exception((
            'Op did not raised an error even though the function'
            ' is not differentiable'))

    vv = numpy.asarray(rng.uniform(size=(4,)), theano.config.floatX)
    yv = tensor.Lop(y, mx, v)
    lop_f = function([mx, v], yv)

    sy = tensor.grad((v * y).sum(), mx)
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


class test_diag(unittest.TestCase):
    """
    Test that linalg.diag has the same behavior as numpy.diag.
    numpy.diag has two behaviors:
    (1) when given a vector, it returns a matrix with that vector as the
    diagonal.
    (2) when given a matrix, returns a vector which is the diagonal of the
    matrix.

    (1) and (2) are tested by test_alloc_diag and test_extract_diag
    respectively.

    test_diag test makes sure that linalg.diag instantiates
    the right op based on the dimension of the input.
    """
    def __init__(self, name, mode=None, shared=tensor._shared,
                 floatX=None, type=tensor.TensorType):
        self.mode = mode
        self.shared = shared
        if floatX is None:
            floatX = config.floatX
        self.floatX = floatX
        self.type = type
        super(test_diag, self).__init__(name)

    def test_alloc_diag(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        x = theano.tensor.vector()
        g = alloc_diag(x)
        f = theano.function([x], g)

        # test "normal" scenario (5x5 matrix) and special cases of 0x0 and 1x1
        for shp in [5, 0, 1]:
            m = rng.rand(shp).astype(self.floatX)
            v = numpy.diag(m)
            r = f(m)
            # The right matrix is created
            assert (r == v).all()

        # Test we accept only vectors
        xx = theano.tensor.matrix()
        ok = False
        try:
            alloc_diag(xx)
        except TypeError:
            ok = True
        assert ok

        # Test infer_shape
        f = theano.function([x], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != 'FAST_COMPILE':
            assert sum([node.op.__class__ == AllocDiag for node in topo]) == 0
        for shp in [5, 0, 1]:
            m = rng.rand(shp).astype(self.floatX)
            assert (f(m) == m.shape).all()

    def test_alloc_diag_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        x = rng.rand(5)
        tensor.verify_grad(alloc_diag, [x], rng=rng)

    def test_diag(self):
        # test that it builds a matrix with given diagonal when using
        # vector inputs
        x = theano.tensor.vector()
        y = diag(x)
        assert y.owner.op.__class__ == AllocDiag

        # test that it extracts the diagonal when using matrix input
        x = theano.tensor.matrix()
        y = extract_diag(x)
        assert y.owner.op.__class__ == ExtractDiag

        # other types should raise error
        x = theano.tensor.tensor3()
        ok = False
        try:
            y = extract_diag(x)
        except TypeError:
            ok = True
        assert ok

    # not testing the view=True case since it is not used anywhere.
    def test_extract_diag(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        m = rng.rand(2, 3).astype(self.floatX)
        x = self.shared(m)
        g = extract_diag(x)
        f = theano.function([], g)
        assert [isinstance(node.inputs[0].type, self.type)
                for node in f.maker.fgraph.toposort()
                if isinstance(node.op, ExtractDiag)] == [True]

        for shp in [(2, 3), (3, 2), (3, 3), (1, 1), (0, 0)]:
            m = rng.rand(*shp).astype(self.floatX)
            x.set_value(m)
            v = numpy.diag(m)
            r = f()
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
        f = theano.function([], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != 'FAST_COMPILE':
            assert sum([node.op.__class__ == ExtractDiag
                        for node in topo]) == 0
        for shp in [(2, 3), (3, 2), (3, 3)]:
            m = rng.rand(*shp).astype(self.floatX)
            x.set_value(m)
            assert f() == min(shp)

    def test_extract_diag_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        x = rng.rand(5, 4).astype(self.floatX)
        tensor.verify_grad(extract_diag, [x], rng=rng)

    @attr('slow')
    def test_extract_diag_empty(self):
        c = self.shared(numpy.array([[], []], self.floatX))
        f = theano.function([], extract_diag(c), mode=self.mode)

        assert [isinstance(node.inputs[0].type, self.type)
                for node in f.maker.fgraph.toposort()
                if isinstance(node.op, ExtractDiag)] == [True]


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
    tol = 10 ** (-6)
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.tensor.matrix()
    radius_bound = spectral_radius_bound(x, 5)
    f = theano.function([x], radius_bound)

    shp = (3, 4)
    m = rng.rand(*shp)
    m = numpy.cov(m).astype(config.floatX)
    radius_bound_theano = f(m)

    # test the approximation
    mm = m
    for i in range(5):
        mm = numpy.dot(mm, mm)
    radius_bound_numpy = numpy.trace(mm) ** (2 ** (-5))
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


class test_Eig(utt.InferShapeTester):
    op_class = Eig
    op = eig
    dtype = 'float64'

    def setUp(self):
        super(test_Eig, self).setUp()
        self.rng = numpy.random.RandomState(utt.fetch_seed())
        self.A = theano.tensor.matrix(dtype=self.dtype)
        X = numpy.asarray(self.rng.rand(5, 5),
                          dtype=self.dtype)
        self.S = X.dot(X.T)

    def test_infer_shape(self):
        A = self.A
        S = self.S
        self._compile_and_check([A],  # theano.function inputs
                                self.op(A),  # theano.function outputs
                                # S must be square
                                [S],
                                self.op_class,
                                warn=False)

    def test_eval(self):
        A = theano.tensor.matrix(dtype=self.dtype)
        self.assertEquals([e.eval({A: [[1]]}) for e in self.op(A)],
                          [[1.0], [[1.0]]])
        x = [[0, 1], [1, 0]]
        w, v = [e.eval({A: x}) for e in self.op(A)]
        assert_array_almost_equal(numpy.dot(x, v), w * v)


class test_Eigh(test_Eig):
    op = staticmethod(eigh)

    def test_uplo(self):
        S = self.S
        a = theano.tensor.matrix(dtype=self.dtype)
        wu, vu = [out.eval({a: S}) for out in self.op(a, 'U')]
        wl, vl = [out.eval({a: S}) for out in self.op(a, 'L')]
        assert_array_almost_equal(wu, wl)
        assert_array_almost_equal(vu * numpy.sign(vu[0, :]),
                                  vl * numpy.sign(vl[0, :]))

    def test_grad(self):
        S = self.S
        utt.verify_grad(lambda x: self.op(x)[0], [S], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x)[1], [S], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x, 'U')[0], [S], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x, 'U')[1], [S], rng=self.rng)


class test_Eigh_float32(test_Eigh):
    dtype = 'float32'


def test_matrix_inverse_solve():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the Solve op.")
    A = theano.tensor.dmatrix('A')
    b = theano.tensor.dmatrix('b')
    node = matrix_inverse(A).dot(b).owner
    [out] = inv_as_solve.transform(node)
    assert isinstance(out.owner.op, Solve)


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
    rng = numpy.random.RandomState(utt.fetch_seed())
    a = rng.randn(5, 5)
    a = a + a.T
    b = 10 * numpy.eye(5, 5) + rng.randn(5, 5)
    tensor.verify_grad(lambda a, b: eigvalsh(a, b).dot([1, 2, 3, 4, 5]),
                       [a, b], rng=numpy.random)


class Matrix_power():

    def test_numpy_compare(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        A = tensor.matrix("A", dtype=theano.config.floatX)
        Q = matrix_power(A, 3)
        fn = function([A], [Q])
        a = rng.rand(4, 4).astype(theano.config.floatX)

        n_p = numpy.linalg.matrix_power(a, 3)
        t_p = fn(a)
        assert numpy.allclose(n_p, t_p)

    def test_non_square_matrix(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        A = tensor.matrix("A", dtype=theano.config.floatX)
        Q = matrix_power(A, 3)
        f = function([A], [Q])
        a = rng.rand(4, 3).astype(theano.config.floatX)
        self.assertRaises(ValueError, f, a)
