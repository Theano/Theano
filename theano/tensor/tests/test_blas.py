from nose.plugins.skip import SkipTest
import traceback
import theano.tensor as T
from theano.gof import Env
from theano.printing import pp

import numpy, theano
from numpy.testing import dec
from numpy.testing.noseclasses import KnownFailureTest

from theano.tensor.blas import *
from theano.tensor.blas import (_dot22, _dot22scalar, res_is_a, _as_scalar, _is_real_matrix,
        _gemm_canonicalize, _factor_canonicalized)
from unittest import TestCase
from theano.tests import unittest_tools
from copy import copy

from theano import Param, shared, config
from test_basic import (_approx_eq, as_tensor_variable, inplace_func,
        compile, constant, inplace, eval_outputs)

if config.mode == 'FAST_COMPILE':
    mode_not_fast_compile = 'FAST_RUN'
else: mode_not_fast_compile = config.mode

mode_blas_opt = theano.compile.get_default_mode().including('BlasOpt', 'specialize')

def test_dot_eq():
    assert T.Dot() == T.Dot()

class t_gemm(TestCase):
    """This test suite is supposed to establish that gemm works as it is supposed to."""
    def setUp(self):
        unittest_tools.seed_rng()
        _approx_eq.debug = 0
        Gemm.debug = False

    @staticmethod
    def _gemm(z,a,x,y,b):
        assert a.shape == ()
        assert b.shape == ()
        return b * z + a * numpy.dot(x,y)
    @staticmethod
    def rand(*args):
        return numpy.random.rand(*args)

    def cmp(self, z, a, x, y, b):
        def cmp_linker(z, a, x, y, b, l):
            z,a,x,y,b = [numpy.asarray(p) for p in z,a,x,y,b]
            z_orig = z.copy()
            tz,ta,tx,ty,tb = [as_tensor_variable(p).type() for p in z,a,x,y,b]

            f = inplace_func([tz,ta,tx,ty,tb], gemm_inplace(tz,ta,tx,ty,tb), mode=compile.Mode(optimizer = None, linker = l))
            new_z = f(z,a,x,y,b)
            z_after = self._gemm(z_orig, a, x, y, b)

            #print z_orig, z_after, z, type(z_orig), type(z_after), type(z)
            #_approx_eq.debug = 1
            self.failUnless(_approx_eq(z_after, z))
            if a == 0.0 and b == 1.0:
                return
            else:
                self.failIf(numpy.all(z_orig == z))

        cmp_linker(copy(z), a, x, y, b, 'c|py')
        cmp_linker(copy(z), a, x, y, b, 'py')
        if config.blas.ldflags:
            # If blas.ldflags is equal to '', the C code will not be generated
            cmp_linker(copy(z), a, x, y, b, 'c')

    def test0a(self):
        Gemm.debug = True
        try:
            g = gemm_inplace([1.], 1., [1.], [1.], 1.)
        except ValueError, e:
            if e[0] is Gemm.E_rank:
                return
        self.fail()

    def test0(self):
        try:
            self.cmp(1., 0., 1.0, 1.0, 1.0)
        except ValueError, e:
            if e[0] is Gemm.E_rank:
                return
        self.fail()

    def test2(self):
        try:
            self.cmp(2., 1.0, [3,2,1.], [[1],[2],[3.]], 1.0)
        except ValueError, e:
            self.failUnless(e[0] == Gemm.E_rank)
            return
        self.fail()
    def test4(self):
        self.cmp(self.rand(3,4), 1.0, self.rand(3,5), self.rand(5,4), 0.0)
    def test5(self): self.cmp(self.rand(3,4), 1.0,
            self.rand(3,5), self.rand(5,4), 1.0)
    def test6(self): self.cmp(self.rand(3,4), 1.0,
            self.rand(3,5), self.rand(5,4), -1.0)
    def test7(self): self.cmp(self.rand(3,4), 0.0,
            self.rand(3,5), self.rand(5,4), 0.0)
    def test8(self): self.cmp(self.rand(3,4), 0.0,
            self.rand(3,5), self.rand(5,4), 0.6)
    def test9(self): self.cmp(self.rand(3,4), 0.0,
            self.rand(3,5), self.rand(5,4), -1.0)
    def test10(self):
        _approx_eq.debug = 1
        self.cmp(self.rand(3,4), -1.0, self.rand(3,5), self.rand(5,4), 0.0)
    def test11(self): self.cmp(self.rand(3,4), -1.0,
            self.rand(3,5), self.rand(5,4), 1.0)
    def test12(self): self.cmp(self.rand(3,4), -1.0,
            self.rand(3,5), self.rand(5,4), -1.0)

    def test_factorised_scalar(self):
        a=T.matrix()
        b=T.matrix()
        c=T.matrix()
        s=theano.shared(numpy.zeros((5,5)))

        lr1=T.constant(0.01)
        lr2=T.constant(2)
        l2_reg=T.constant(0.0001)

        #test constant merge with gemm
        f = theano.function([a,b],updates={s:lr1*T.dot(a,b)+l2_reg*lr2*s},mode=mode_not_fast_compile).maker.env.toposort()
        #[Gemm{inplace}(<TensorType(float64, matrix)>, 0.01, <TensorType(float64, matrix)>, <TensorType(float64, matrix)>, 2e-06)]
        assert len(f)==1
        assert f[0].op==gemm_inplace

        #test factored scalar with merge
        f = theano.function([a,b],updates={s:lr1*(T.dot(a,b)-l2_reg*s)},mode=mode_not_fast_compile).maker.env.toposort()
        #[Gemm{inplace}(<TensorType(float64, matrix)>, 0.01, <TensorType(float64, matrix)>, <TensorType(float64, matrix)>, -2e-06)]
        assert len(f)==1
        assert f[0].op==gemm_inplace

        #test factored scalar with merge and neg
        f = theano.function([a,b],updates={s:s-lr1*(s*.0002+T.dot(a,b))},mode=mode_not_fast_compile).maker.env.toposort()
        #[Gemm{inplace}(<TensorType(float64, matrix)>, -0.01, <TensorType(float64, matrix)>, <TensorType(float64, matrix)>, 0.999998)]
        assert len(f)==1
        assert f[0].op==gemm_inplace

    def test_destroy_map0(self):
        """test that only first input can be overwritten"""
        Z = as_tensor_variable(self.rand(2,2))
        try:
            gemm_inplace(Z, 1.0, Z, Z, 1.0)
        except InconsistencyError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map1(self):
        """test that only first input can be overwritten"""
        Z = as_tensor_variable(self.rand(2,2))
        A = as_tensor_variable(self.rand(2,2))
        try:
            gemm_inplace(Z, 1.0, A, inplace.transpose_inplace(Z), 1.0)
        except InconsistencyError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map2(self):
        """test that only first input can be overwritten"""
        Z = as_tensor_variable(self.rand(2,2))
        A = as_tensor_variable(self.rand(2,2))
        try:
            gemm_inplace(Z, 1.0, inplace.transpose_inplace(Z), A, 1.0)
        except InconsistencyError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map3(self):
        """test that only first input can be overwritten"""
        Z = as_tensor_variable(self.rand(2,2))
        A = as_tensor_variable(self.rand(2,2))
        try:
            gemm_inplace(Z, 1.0, Z, A, 1.0)
        except InconsistencyError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()

    def test_destroy_map4(self):
        """test that dot args can be aliased"""
        Z = shared(self.rand(2,2))
        A = shared(self.rand(2,2))
        f = inplace_func([], gemm_inplace(Z, 1.0, A, A, 1.0))
        f()
        f = inplace_func([], gemm_inplace(Z, 1.0, A, A.T, 1.0))
        f()

    def test_transposes(self):
        # three square matrices which are not contiguous
        A = self.rand(4,5)[:,:4]
        B = self.rand(4,5)[:,:4]
        C = self.rand(4,5)[:,:4]

        def t(z,x,y,a=1.0, b=0.0,l='c|py',dt='float64'):
            z,a,x,y,b = [theano._asarray(p,dtype=dt) for p in z,a,x,y,b]
            z_orig = z.copy()
            z_after = self._gemm(z, a, x, y, b)

            tz,ta,tx,ty,tb = [shared(p) for p in z,a,x,y,b]

            #f = inplace_func([tz,ta,tx,ty,tb], gemm_inplace(tz,ta,tx,ty,tb), mode = compile.Mode(optimizer = None, linker=l))
            #f(z, a, x, y, b)
            f = inplace_func([], gemm_inplace(tz,ta,tx,ty,tb), mode = compile.Mode(optimizer = None, linker=l))
            f()
            self.failUnless(_approx_eq(z_after, tz.value), (z_orig, z_after, z, z_after - z))
            f()
            self.failUnless(_approx_eq(z_after, tz.value), (z_orig, z_after, z, z_after - z))
            f()
            self.failUnless(_approx_eq(z_after, tz.value), (z_orig, z_after, z, z_after - z))

            #tz.value *= 0 # clear z's value
            y_T = ty.value.T
            ty.value = tx.value.T
            tx.value = y_T

            f()
            # test that the transposed version of multiplication gives same answer
            self.failUnless(_approx_eq(z_after, tz.value.T))

        t(C,A,B)
        t(C.T, A, B)
        t(C, A.T, B, dt='float32')
        t(C, A, B.T)
        t(C.T, A.T, B)
        t(C, A.T, B.T, dt='float32')
        t(C.T, A, B.T)
        t(C.T, A.T, B.T, dt='float32')

        t(C, A[:,:2], B[:2, :])
        t(C.T, A[:,:2], B[:2, :], dt='float32')
        t(C, A[:2,:].T, B[:2, :])
        t(C.T, A[:2,:].T, B[:2, :], dt='float32')
        t(C, A[:2,:].T, B[:, :2].T)
        t(C.T, A[:2,:].T, B[:, :2].T)

        try:
            t(C.T, A[:2,:], B[:, :2].T)
        except ValueError, e:
            if e[0].find('aligned') >= 0:
                return
        self.fail()

def test_res_is_a():
    X,Y,Z,a,b = XYZab()

    assert not res_is_a(a, T.sqrt)
    assert not res_is_a(a+a, T.sqrt)
    assert res_is_a(T.sqrt(a+a), T.sqrt)

    #leave the maxclients  stuff untested because it requires being in an env.

class t_as_scalar(TestCase):
    def test0(self):
        """Test that it works on scalar constants"""
        a = T.constant(2.5)
        b = T.constant(numpy.asarray([[[0.5]]]))
        b2 = b.dimshuffle()
        assert b2.ndim == 0
        d_a = T.DimShuffle([], [])(a)
        d_b = T.DimShuffle([True, True, True], [0,2,1])(b)
        d_a2 = T.DimShuffle([], ['x', 'x', 'x'])(a)

        self.failUnless(_as_scalar(a) == a)
        self.failUnless(_as_scalar(b) != b)
        self.failUnless(_as_scalar(d_a) != d_a)
        self.failUnless(_as_scalar(d_b) != d_b)
        self.failUnless(_as_scalar(d_a2) != d_a2)

    def test1(self):
        """Test that it fails on nonscalar constants"""
        a = T.constant(numpy.ones(5))
        self.failUnless(None == _as_scalar(a))
        self.failUnless(None == _as_scalar(T.DimShuffle([False], [0,'x'])(a)))

    def test2(self):
        """Test that it works on scalar variables"""
        a = T.dscalar()
        d_a = T.DimShuffle([], [])(a)
        d_a2 = T.DimShuffle([], ['x', 'x'])(a)

        self.failUnless(_as_scalar(a) is a)
        self.failUnless(_as_scalar(d_a) is a)
        self.failUnless(_as_scalar(d_a2) is a)

    def test3(self):
        """Test that it fails on nonscalar variables"""
        a = T.dmatrix()
        self.failUnless(None == _as_scalar(a))
        self.failUnless(None == _as_scalar(T.DimShuffle([False, False], [0,'x', 1])(a)))

class T_real_matrix(TestCase):
    def test0(self):
        self.failUnless(_is_real_matrix(T.DimShuffle([False,False], [1, 0])(T.dmatrix())))
        self.failUnless(not _is_real_matrix(T.DimShuffle([False], ['x', 0])(T.dvector())))

def fail(msg):
    print 'FAIL', msg
    assert False

"""This test suite ensures that Gemm is inserted where it belongs, and that the resulting
functions compute the same things as the originals."""
def XYZab():
    return T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()

class Failure(Exception):
    pass

class Warning(Exception):
    pass

def just_gemm(i, o, ishapes = [(4,3), (3,5), (4,5), (), ()], max_graphlen=0):
    try:
        f = inplace_func(
                [Param(ii, mutable=True, allow_downcast=True) for ii in i],
                o,
                mode='FAST_RUN')
        at_least_one_gemm = False
        for node in f.maker.env.nodes:
            if node.op == T.dot: raise Warning('dot not changed to gemm_inplace in graph')
            if node.op == _dot22: raise Warning('_dot22 not changed to gemm_inplace in graph')
            if node.op == gemm_inplace: at_least_one_gemm = True
        assert at_least_one_gemm
        g = inplace_func(i, o, mode=compile.Mode(linker='py', optimizer=None),
                allow_input_downcast=True)
        for node in g.maker.env.nodes:
            if node.op == gemm_inplace: raise Exception('gemm_inplace in original graph')

        graphlen = len(f.maker.env.toposort())
        if max_graphlen and (graphlen <= max_graphlen):
            theano.printing.debugprint(f)
            assert False, 'graphlen=%i>%i'%(graphlen, max_graphlen)

        rng = numpy.random.RandomState(unittest_tools.fetch_seed(234))
        r0 = f(*[rng.randn(*sh) for sh in ishapes])
        rng = numpy.random.RandomState(unittest_tools.fetch_seed(234))
        r1 = g(*[rng.randn(*sh) for sh in ishapes])
        max_abs_err = numpy.max(numpy.abs(r0[0] - r1[0]))
        if  max_abs_err > 1.0e-8:
            raise Failure('GEMM is computing the wrong output. max_rel_err =', max_abs_err)
    except Failure:
        for node in f.maker.env.toposort():
            print 'GRAPH', node
        raise
    except Warning, e:
        #for node in f.maker.env.toposort():
        #    print 'GRAPH', node
        print 'WARNING:', e
        #traceback.print_exc()


def test_gemm_opt0():
    """Many subgraphs whose dots can be eliminated"""
    X,Y,Z,a,b = XYZab()

    just_gemm([X,Y,Z,a,b], [T.dot(X,Y) * a + Z * b])
    just_gemm([X,Y,Z,a,b], [a * T.dot(X,Y) + b * Z])
    just_gemm([X,Y,Z,a,b], [b * Z + a * T.dot(X,Y)])
    just_gemm([X,Y,Z,a,b], [T.dot(X,Y) * a - Z * b])
    just_gemm([X,Y,Z,a,b], [a * T.dot(X,Y) - b * Z])
    just_gemm([X,Y,Z,a,b], [b * Z - a * T.dot(X,Y)])

    #with transposes (transposes should be pushed through dot in canonicalize)
    just_gemm([X,Y,Z,a,b], [b * Z.T - a * T.dot(Y.T,X.T)])
    just_gemm([X,Y,Z,a,b], [b * Z.T + a * b * T.dot(X,Y).T])

    #with N multiplications instead of just one
    just_gemm([X,Y,Z,a,b], [(b * b) * Z * a + (a * a) * T.dot(X,Y) * b])
    just_gemm([X,Y,Z,a,b], [Z + T.dot(X,Y)])
    just_gemm([X,Y,Z,a,b], [Z*b + T.dot(X,Y)])
    just_gemm([X,Y,Z,a,b], [Z + a*b*a*T.dot(X,Y)])
    just_gemm([X,Y,Z,a,b], [(b * b) * Z * a - (a * a) * T.dot(X,Y) * b])
    just_gemm([X,Y,Z,a,b], [Z - T.dot(X,Y)])
    just_gemm([X,Y,Z,a,b], [Z*b - T.dot(X,Y)])
    just_gemm([X,Y,Z,a,b], [Z - a*b*a*T.dot(X,Y)])


def test_gemm_opt_double_gemm():
    """This is the pattern that shows up in the autoencoder"""
    X,Y,Z,a,b = T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()
    R, S, c = T.dmatrix(), T.dmatrix(), T.dscalar()

    just_gemm([X,Y,Z,a,b, R, S, c], [Z *c + a * T.dot(X,Y) + b * T.dot(R,S).T],
            ishapes=[(4,3), (3,5), (4,5), (), (), (5,9), (9,4), ()])

    ishapes=[(4,3), (3,5), (4,5), (), (), (5,9), (9,4), ()]
    i = [X,Y,Z,a,b, R, S, c]
    o = [a * T.dot(X,Y) + gemm_inplace(Z, b, S.T, R.T, 1.0)]
    try:
        f = inplace_func([Param(ii, mutable=True) for ii in i],o, mode='FAST_RUN')
        for node in f.maker.env.nodes:
            if node.op == T.dot: raise Failure('dot in graph')
            if node.op == _dot22: raise Failure('_dot22 in graph')
        g = inplace_func(i, o, mode=compile.Mode(linker='py', optimizer=None))
        #for node in g.maker.env.nodes:
        #    if node.op == gemm_inplace: raise Failure('gemm_inplace in graph')

        rng = numpy.random.RandomState(unittest_tools.fetch_seed(234))
        r0 = f(*[rng.randn(*sh) for sh in ishapes])
        rng = numpy.random.RandomState(unittest_tools.fetch_seed(234))
        r1 = g(*[rng.randn(*sh) for sh in ishapes])
        max_abs_err = numpy.max(numpy.abs(r0[0] - r1[0]))
        if  max_abs_err > 1.0e-8:
            raise Failure('GEMM is computing the wrong output. max_rel_err =', max_abs_err)
    except Failure:
        for node in f.maker.env.toposort():
            print 'GRAPH', node
        raise


def test_gemm_canonicalize():
    X,Y,Z,a,b = T.dmatrix('X'), T.dmatrix('Y'), T.dmatrix('Z'), T.dscalar('a'), T.dscalar('b')
    R,S,U,c,d = T.dmatrix('R'), T.dmatrix('S'), T.dmatrix('U'), T.dscalar('c'), T.dscalar('d')
    u = T.row('u')
    v = T.vector('v')
    w = T.col('w')

    can = []
    _gemm_canonicalize(X + Y + Z, 1.0, can, 0)
    assert can == [(1.0, X), (1.0, Y), (1.0, Z)]

    can = []
    _gemm_canonicalize(X + Y + u, 1.0, can, 0)
    assert can == [(1.0, X), (1.0, Y), (1.0, u)], can

    can = []
    _gemm_canonicalize(X + Y + v, 1.0, can, 0)
    # [(1.0, X), (1.0, Y), (1.0, InplaceDimShuffle{x,0}(v))]
    assert can[:2] == [(1.0, X), (1.0, Y)]
    assert isinstance(can[2], tuple)
    assert len(can[2]) == 2
    assert can[2][0] == 1.0
    assert can[2][1].owner
    assert isinstance(can[2][1].owner.op, T.DimShuffle)
    assert can[2][1].owner.inputs == [v]

    can = []
    _gemm_canonicalize(X + Y + w, 1.0, can, 0)
    assert can == [(1.0, X), (1.0, Y), (1.0, w)], can

    can = []
    _gemm_canonicalize(a*X + Y - b*Z*c, 1.0, can, 0)
    assert can[0] == (a, X)
    assert can[1] == (1.0, Y)
    assert can[2][0].owner.op == T.mul
    assert can[2][0].owner.inputs[0].owner.op == T.neg
    assert can[2][0].owner.inputs[0].owner.inputs[0] == c
    assert can[2][0].owner.inputs[1] == b

    can = []
    _gemm_canonicalize((-d) * X - (a*X + Y - b*Z*c), 1.0, can, 0)
    print can
    assert can[0][0].owner.op == T.neg
    assert can[0][0].owner.inputs[0] == d
    assert can[0][1] == X
    assert can[1][0].owner.op == T.neg
    assert can[1][0].owner.inputs[0] == a
    assert can[2] == (-1.0, Y)
    assert can[3][0].owner.op == T.mul
    assert can[3][0].owner.inputs == [c,b]

def test_gemm_factor():
    X,Y,Z,a,b = T.dmatrix('X'), T.dmatrix('Y'), T.dmatrix('Z'), T.dscalar('a'), T.dscalar('b')
    R,S,U,c,d = T.dmatrix('R'), T.dmatrix('S'), T.dmatrix('U'), T.dscalar('c'), T.dscalar('d')

    assert [(1.0, X), (1.0, Y)] == _factor_canonicalized([(1.0, X), (1.0, Y)])
    assert [(2.0, X)] == _factor_canonicalized([(1.0, X),(1.0, X)])


def test_gemm_nested():
    X,Y,Z,a,b = T.dmatrix('X'), T.dmatrix('Y'), T.dmatrix('Z'), T.dscalar('a'), T.dscalar('b')
    R,S,U,c,d = T.dmatrix('R'), T.dmatrix('S'), T.dmatrix('U'), T.dscalar('c'), T.dscalar('d')

    just_gemm([X,Y,Z,R,S,U,a,b,c,d],
            [a * Z - b * (c*T.dot(X,Y) + d*Z)],
            ishapes=[(2,3),(3,4),(2,4),(2,3),(3,4),(2,4),(),(),(),()],
            max_graphlen=1)
    print "---------------------"
    just_gemm([X,Y,Z,R,S,U,a,b,c,d],
            [a * Z - b * (c*T.dot(X,Y) + d*Z + c*Z)],
            ishapes=[(2,3),(3,4),(2,4),(2,3),(3,4),(2,4),(),(),(),()],
            max_graphlen=1)
    print "---------------------"
    just_gemm([X,Y,Z,R,S,U,a,b,c,d],
            [a * Z - b * (c*T.dot(X,Y) + d*Z + c*U)],
            ishapes=[(2,3),(3,4),(2,4),(2,3),(3,4),(2,4),(),(),(),()],
            max_graphlen=3)

def test_gemm_opt_wishlist():
    X,Y,Z,a,b = T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()

    #with >2 additions of the same T.dot(X,Y term
    just_gemm([X,Y,Z,a,b], [(b * b) * Z * a + (a * a) * T.dot(X,Y) + b * T.dot(X,Y)])

    just_gemm([X,Y,Z,a,b], [Z + T.dot(X,Y) + T.dot(X,Y)])

def test_gemm_with_vector():
    """Many subgraphs whose dots can be eliminated.
    This adds a vector two the previous test, which triggers the long-sought GEMM bug.
    """
    X,Y,Z,a,b = XYZab()
    v = T.vector()
    def my_just_gemm(o):
        i = [X,Y,Z,a,b,v]
        ishapes = [(4,3), (3,5), (4,5), (), (), (5,)]
        rval = just_gemm(i, o, ishapes=ishapes)

    my_just_gemm([v + T.dot(X,Y) * a + Z * b])
    my_just_gemm([v + a * T.dot(X,Y) + b * Z])
    my_just_gemm([v + b * Z + a * T.dot(X,Y)])
    my_just_gemm([v + T.dot(X,Y) * a - Z * b])
    my_just_gemm([v + a * T.dot(X,Y) - b * Z])
    my_just_gemm([v + b * Z - a * T.dot(X,Y)])

    #with N multiplications instead of just one
    my_just_gemm([v + (b * b) * Z * a + (a * a) * T.dot(X,Y) * b])
    my_just_gemm([v + Z + T.dot(X,Y)])
    my_just_gemm([v + Z*b + T.dot(X,Y)])
    my_just_gemm([v + Z + a*b*a*T.dot(X,Y)])
    my_just_gemm([v + (b * b) * Z * a - (a * a) * T.dot(X,Y) * b])
    my_just_gemm([Z - T.dot(X,Y) + v])
    my_just_gemm([Z*b - T.dot(X,Y) + v])
    my_just_gemm([Z - a*b*a*T.dot(X,Y) + v])

def test_gemm_opt_vector_stuff():
    X,Y,Z,a,b = T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()
    u,v = T.dvector(), T.dvector()

    f = inplace_func([a, u, v], a + T.dot(u,v), mode='FAST_RUN')
    if gemm_inplace in [n.op for n in f.maker.env.nodes]:
        raise Failure('gemm_inplace in graph')

    f = inplace_func([a, u, X,Y], a * u + T.dot(X,Y), mode='FAST_RUN')
    if (gemm_inplace in [n.op for n in f.maker.env.nodes]):
        raise Failure('gemm_inplace in graph')

def test_inplace0():
    #should fail to insert gemm_inplace because gemm_inplace would create cycles
    X,Y,Z,a,b = T.dmatrix('X'), T.dmatrix('Y'), T.dmatrix('Z'), T.dscalar('a'), T.dscalar('b')
    R, S, c = T.dmatrix('R'), T.dmatrix('S'), T.dscalar('c')

    f = inplace_func([X,Y,Z,a,b, R, S, c],
            [Z * (Z + b * T.dot(R,S).T)], mode='FAST_RUN')
    if (gemm_inplace in [n.op for n in f.maker.env.nodes]):
        print pp(f.maker.env.outputs[0])
        raise Failure('gemm_inplace in graph')
    assert _dot22 in [n.op for n in f.maker.env.nodes]

    f = inplace_func([X,Y,Z,a,b, R, S, c],
            [Z * (c*Z + a * T.dot(X,Y) + b * T.dot(R,S).T)], mode='FAST_RUN')
    # gemm_inplace should be inserted here, to work in-place on Z*c
    if (not gemm_inplace in [n.op for n in f.maker.env.nodes]):
        print pp(f.maker.env.outputs[0])
        #raise Failure('no gemm_inplace in graph')
        raise KnownFailureTest("gemm not always inserted, see #415")

def test_inplace1():
    X,Y,Z,a,b = XYZab()
    # with > 2 terms in the overall addition
    f = inplace_func([X,Y,Z,a,b],
            [Z + Z + T.dot(X,Y)], mode='FAST_RUN')
    theano.printing.debugprint(f)
    # it doesn't work inplace because we didn't mark Z as mutable input
    assert [n.op for n in f.maker.env.nodes] == [gemm_no_inplace]

def test_dot22():
    a=T.matrix()
    b=T.matrix()
    f = theano.function([a,b],T.dot(a,b),mode=mode_blas_opt)
    topo = f.maker.env.toposort()
    assert _dot22 in [x.op for x in topo]
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())

    av=rng.uniform(size=(5,5)).astype(config.floatX)
    bv=rng.uniform(size=(5,5)).astype(config.floatX)
    f(av,bv)

def test_dot22scalar():
    ## including does not seem to work for 'local_dot_to_dot22' and
    ## 'local_dot22_to_dot22scalar'
    ## TODO: exclude other optimizations in BlasOpt?
    #m = theano.compile.get_default_mode().including('local_dot_to_dot22','local_dot22_to_dot22scalar','specialize')
    #m = theano.compile.get_default_mode().including('BlasOpt', 'specialize')
    a=T.matrix()
    b=T.matrix()
    c=T.matrix()
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())

    av=rng.uniform(size=(5,5)).astype(config.floatX)
    bv=rng.uniform(size=(5,5)).astype(config.floatX)
    cv=rng.uniform(size=(5,5)).astype(config.floatX)

    if True:
        f = theano.function([a,b],0.2*T.dot(a,b),mode=mode_blas_opt)
        topo = f.maker.env.toposort()
        assert _dot22scalar in [x.op for x in topo]
        assert len(topo)==1
        f(av,bv)

    if True:
        f = theano.function([a,b,c],0.2*c*T.dot(a,b),mode=mode_blas_opt)
        topo = f.maker.env.toposort()
        assert _dot22scalar in [x.op for x in topo]
        assert len(topo)==2
        f(av,bv,cv)

    f = theano.function([a,b,c],c * 0.2*T.dot(a,b),mode=mode_blas_opt)
    topo = f.maker.env.toposort()
    assert _dot22scalar in [x.op for x in topo]
    assert len(topo)==2
    f(av,bv,cv)

    ## Here, canonicalize also seems needed
    ## TODO: add only the optimizations needed?
    m2 = mode_blas_opt.including('canonicalize')
    f = theano.function([a,b,c],0.1*c * 0.2*T.dot(a,b),mode=m2)
    topo = f.maker.env.toposort()
    assert _dot22scalar in [x.op for x in topo]
    assert len(topo)==2
    f(av,bv,cv)

    f = theano.function([a,b,c],c * 0.2*a*T.dot(a,b),mode=m2)
    topo = f.maker.env.toposort()
    assert _dot22scalar in [x.op for x in topo]
    assert len(topo)==2
    f(av,bv,cv)

    f = theano.function([a,b,c],0.2*c *a*T.dot(a,b),mode=mode_blas_opt)
    topo = f.maker.env.toposort()
    #currently the canonizer don't always merge all Mul together...
    # dot22scalar optimizer does not do a recursive search
    # therefore, it doesn't find potential matches of the scalar.
    # TODO: combine with the 'canonicalization' that is part of the Gemm optimizer.
    #
    #    assert _dot22scalar in [x.op for x in topo]
    #    assert len(topo)==2

    f(av,bv,cv)
    f = theano.function([a,b,c],c * a*0.2*T.dot(a,b),mode=m2)
    topo = f.maker.env.toposort()
    assert _dot22scalar in [x.op for x in topo]
    assert len(topo)==2
    f(av,bv,cv)


def test_dot_w_self():
    # This can trigger problems in the optimization because what would normally be a gemm must
    # not be because the output is aliased to one of the inputs.

    A = shared(value = numpy.ones((2,2)))
    B = T.matrix()

    p = T.dot(A,A)*B

    grad = T.grad(T.mean(p),[A])
    f = theano.function([B], p, updates = { A : A - grad[0]} )

    # tests correctness in debugmode
    f(numpy.asarray([[0,1], [2,3]], dtype=config.floatX))


def test_dot_vm():
    ''' Test vector dot matrix '''
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
    m = theano.shared(numpy.array(rng.uniform(size=(2,2)), dtype='float32'))
    f = theano.function([], theano.dot(v,m), mode = mode_blas_opt)

    # Assert they produce the same output
    assert numpy.allclose(f(), numpy.dot(v.value,m.value))

    assert sum([isinstance(node.op, T.Dot) for node in
                f.maker.env.toposort() ]) == 1

def test_dot_mv():
    ''' Test matrix dot vector '''
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
    m = theano.shared(numpy.array(rng.uniform(size=(2,2)),
                                   dtype='float32'))
    f = theano.function([], theano.dot(m,v), mode = mode_blas_opt)

    # Assert they produce the same output
    assert numpy.allclose(f(), numpy.dot(m.value,v.value))

    assert sum([isinstance(node.op, T.Dot) for node in
                f.maker.env.toposort() ]) == 1

def test_gemv1():
    ''' test vector1+dot(matrix,vector2) '''
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    v1 = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
    v2_orig = numpy.array(rng.uniform(size=(2,)), dtype='float32')
    v2 = theano.shared(v2_orig)
    m  = theano.shared(numpy.array(rng.uniform(size=(2,2)), dtype='float32'))

    f = theano.function([], v2+theano.dot(m,v1), mode = mode_blas_opt)

    # Assert they produce the same output
    assert numpy.allclose(f(), numpy.dot(m.value,v1.value)+v2_orig)
    topo = f.maker.env.toposort()
    assert len(topo)==1
    assert isinstance(topo[0].op, Gemv)
    assert topo[0].op.inplace==False

    #test the inplace version
    f = theano.function([], [], updates={v2:v2+theano.dot(m,v1)}
                        , mode = mode_blas_opt)

    # Assert they produce the same output
    f()
    assert numpy.allclose(v2.value, numpy.dot(m.value,v1.value)+v2_orig)
    topo = f.maker.env.toposort()
    assert len(topo)==1
    assert isinstance(topo[0].op, Gemv)
    if config.mode != 'FAST_COMPILE':
        assert topo[0].op.inplace==True

def test_gemv2():
    ''' test vector1+dot(vector2,matrix) '''
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    v1 = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
    v2_orig = numpy.array(rng.uniform(size=(2,)), dtype='float32')
    v2 = theano.shared(v2_orig )
    m  = theano.shared(numpy.array(rng.uniform(size=(2,2)), dtype='float32'))

    f = theano.function([], v2+theano.dot(v1,m), mode = mode_blas_opt)

    # Assert they produce the same output
    assert numpy.allclose(f(), numpy.dot(v1.value,m.value)+v2.value)
    topo = f.maker.env.toposort()
    assert sum(isinstance(node.op, Gemv) for node in topo)==1
    assert topo[-1].op.inplace==False

    #test the inplace version
    f = theano.function([], [], updates={v2:v2+theano.dot(v1,m)}
                        , mode = mode_blas_opt)

    # Assert they produce the same output
    f()
    assert numpy.allclose(v2.value, numpy.dot(v1.value, m.value)+v2_orig)
    topo = f.maker.env.toposort()
    assert sum(isinstance(node.op, Gemv) for node in topo)==1
    if config.mode != 'FAST_COMPILE':
        assert topo[-1].op.inplace==True
