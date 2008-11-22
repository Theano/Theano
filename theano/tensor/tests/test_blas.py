
import theano.tensor as T
from ...gof import Env
import numpy
from theano.tensor.blas import *
from theano.tensor.blas import _as_scalar, _dot22
from unittest import TestCase
from copy import copy

from theano import In, Out
from .test_basic import (_approx_eq, as_tensor, function,
        compile, value, constant, inplace, eval_outputs)

class t_gemm(TestCase):
    """This test suite is supposed to establish that gemm works as it is supposed to."""
    def setUp(self):
        numpy.random.seed(44)
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
            tz,ta,tx,ty,tb = [as_tensor(p).type() for p in z,a,x,y,b]

            f = function([tz,ta,tx,ty,tb], gemm(tz,ta,tx,ty,tb), mode=compile.Mode(optimizer = None, linker = l))
            new_z = f(z,a,x,y,b)
            z_after = self._gemm(z_orig, a, x, y, b)

            self.failUnless(z is new_z)
            #print z_orig, z_after, z, type(z_orig), type(z_after), type(z)
            #_approx_eq.debug = 1
            self.failUnless(_approx_eq(z_after, z))
            if a == 0.0 and b == 1.0:
                return
            else:
                self.failIf(numpy.all(z_orig == z))

        cmp_linker(copy(z), a, x, y, b, 'c|py')
        cmp_linker(copy(z), a, x, y, b, 'c')
        cmp_linker(copy(z), a, x, y, b, 'py')

    def test0a(self): 
        Gemm.debug = True
        try:
            g = gemm([1.], 1., [1.], [1.], 1.)
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

    def test_destroy_map0(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, Z, Z, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map1(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        A = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, A, inplace.transpose_inplace(Z), 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map2(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        A = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, inplace.transpose_inplace(Z), A, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map3(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        A = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, Z, A, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()

    def test_destroy_map4(self):
        """test that dot args can be aliased"""
        Z = value(self.rand(2,2))
        A = value(self.rand(2,2))
        eval_outputs([gemm(Z, 1.0, A, A, 1.0)])
        eval_outputs([gemm(Z, 1.0, A, A.T, 1.0)])


    def test_transposes(self):
        # three square matrices which are not contiguous
        A = self.rand(4,5)[:,:4]
        B = self.rand(4,5)[:,:4]
        C = self.rand(4,5)[:,:4]

        def t(z,x,y,a=1.0, b=0.0,l='c|py',dt='float64'):
            z,a,x,y,b = [numpy.asarray(p,dtype=dt) for p in z,a,x,y,b]
            z_orig = z.copy()
            z_after = self._gemm(z, a, x, y, b)

            tz,ta,tx,ty,tb = [value(p) for p in z,a,x,y,b]

            f = function([tz,ta,tx,ty,tb], gemm(tz,ta,tx,ty,tb), mode = compile.Mode(optimizer = None, linker=l))
            f(z, a, x, y, b)
            self.failUnless(_approx_eq(z_after, z), (z_orig, z_after, z, z_after - z))

            f(z.T, a, y.T, x.T, b)
            self.failUnless(_approx_eq(z_after, z))

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

class t_as_scalar(TestCase):
    def test0(self):
        """Test that it works on scalar constants"""
        a = T.constant(2.5)
        b = T.constant(numpy.asarray([[[0.5]]]))
        d_a = T.DimShuffle([], [])(a)
        d_b = T.DimShuffle([True, True, True], [0,2,1])(b)
        d_a2 = T.DimShuffle([], ['x', 'x', 'x'])(a)

        self.failUnless(numpy.all(_as_scalar(a) == a))
        self.failUnless(numpy.all(_as_scalar(b) == b.data), (b, _as_scalar(b)))
        self.failUnless(numpy.all(_as_scalar(d_a) == a))
        self.failUnless(numpy.all(_as_scalar(d_b) == b.data))
        self.failUnless(numpy.all(_as_scalar(d_a2) == a))

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

class T_gemm_opt(TestCase):
    """This test suite ensures that Gemm is inserted where it belongs, and that the resulting
    functions compute the same things as the originals."""
    def XYZab(self):
        return T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()

    def just_gemm(self, i, o, ishapes = [(4,3), (3,5), (4,5), (), ()]):
        def on_fail():
            for node in f.maker.env.toposort():
                print 'GRAPH', node
            self.fail()

        f = function([In(ii, mutable=True) for ii in i],o, mode='FAST_RUN')
        for node in f.maker.env.nodes:
            if node.op == T.dot: on_fail()
            if node.op == _dot22: on_fail()
        g = function(i, o, mode='FAST_COMPILE')
        for node in g.maker.env.nodes:
            if node.op == gemm: on_fail()

        rng = numpy.random.RandomState(234)
        r0 = f(*[rng.randn(*sh) for sh in ishapes])
        rng = numpy.random.RandomState(234)
        r1 = g(*[rng.randn(*sh) for sh in ishapes])
        if numpy.max(numpy.abs(r0[0] - r1[0])) > 1.0e-8:
            self.fail()

    def test0(self):
        """Many subgraphs whose dots can be eliminated"""
        X,Y,Z,a,b = self.XYZab()

        self.just_gemm([X,Y,Z,a,b], [T.dot(X,Y) * a + Z * b])
        self.just_gemm([X,Y,Z,a,b], [a * T.dot(X,Y) + b * Z])
        self.just_gemm([X,Y,Z,a,b], [b * Z + a * T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [T.dot(X,Y) * a - Z * b])
        self.just_gemm([X,Y,Z,a,b], [a * T.dot(X,Y) - b * Z])
        self.just_gemm([X,Y,Z,a,b], [b * Z - a * T.dot(X,Y)])

        #with transposes (transposes should be pushed through dot in canonicalize)
        self.just_gemm([X,Y,Z,a,b], [b * Z.T - a * T.dot(Y.T,X.T)])
        self.just_gemm([X,Y,Z,a,b], [b * Z.T + a * b * T.dot(X,Y).T])

        #with N multiplications instead of just one
        self.just_gemm([X,Y,Z,a,b], [(b * b) * Z * a + (a * a) * T.dot(X,Y) * b])
        self.just_gemm([X,Y,Z,a,b], [Z + T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [Z*b + T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [Z + a*b*a*T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [(b * b) * Z * a - (a * a) * T.dot(X,Y) * b])
        self.just_gemm([X,Y,Z,a,b], [Z - T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [Z*b - T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [Z - a*b*a*T.dot(X,Y)])

        # with > 2 terms in the overall addition
        self.just_gemm([X,Y,Z,a,b], [Z + Z + T.dot(X,Y) + Z])

    def test_double_gemm(self):
        """This is the pattern that shows up in the autoencoder"""
        X,Y,Z,a,b = T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()
        R, S, c = T.dmatrix(), T.dmatrix(), T.dscalar()

        self.just_gemm([X,Y,Z,a,b, R, S, c], [Z *c + a * T.dot(X,Y) + b * T.dot(R,S).T],
                ishapes=[(4,3), (3,5), (4,5), (), (), (5,9), (9,4), ()])

    def wishlist(self):
        X,Y,Z,a,b = T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()

        #with >2 additions of the same T.dot(X,Y term
        self.just_gemm([X,Y,Z,a,b], [Z + T.dot(X,Y) + T.dot(X,Y)])
        self.just_gemm([X,Y,Z,a,b], [(b * b) * Z * a + (a * a) * T.dot(X,Y) + b * T.dot(X,Y)])

