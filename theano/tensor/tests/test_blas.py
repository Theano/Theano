#from nose.plugins.skip import SkipTest
#import traceback
import sys
import theano.tensor as T
#from theano.gof import Env
from theano.printing import pp

import numpy
import theano
from numpy import (arange, array, common_type, complex64, complex128, float32,
                  float64, newaxis, shape, transpose, zeros)
from numpy.testing import assert_, assert_array_almost_equal
#from numpy.testing import dec
#from numpy.testing.noseclasses import KnownFailureTest

#from theano.tensor.blas import *
from theano.tensor.blas import (_dot22, _dot22scalar, res_is_a, _as_scalar,
                                _is_real_matrix, _gemm_canonicalize,
                                _factor_canonicalized, Gemm, Gemv,
                                gemm_inplace, gemm_no_inplace,
                                InconsistencyError, Ger, ger, ger_destructive)
from unittest import TestCase
from theano.tests import unittest_tools
from copy import copy, deepcopy

from theano import Param, shared, config
from test_basic import (_approx_eq, as_tensor_variable, inplace_func,
        compile, inplace)
        #, constant, eval_outputs)
import theano.tensor.blas_scipy

if config.mode == 'FAST_COMPILE':
    mode_not_fast_compile = 'FAST_RUN'
else:
    mode_not_fast_compile = config.mode

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

    def cmp(self, z_, a_, x_, y_, b_):
        for dtype in ['float32', 'float64', 'complex64', 'complex128']:
            z = numpy.asarray(z_, dtype=dtype)
            a = numpy.asarray(a_, dtype=dtype)
            x = numpy.asarray(x_, dtype=dtype)
            y = numpy.asarray(y_, dtype=dtype)
            b = numpy.asarray(b_, dtype=dtype)
            def cmp_linker(z, a, x, y, b, l):
                z,a,x,y,b = [numpy.asarray(p) for p in z,a,x,y,b]
                z_orig = z.copy()
                tz,ta,tx,ty,tb = [as_tensor_variable(p).type() for p in z,a,x,y,b]

                f = inplace_func([tz,ta,tx,ty,tb], gemm_inplace(tz,ta,tx,ty,tb), mode=compile.Mode(optimizer = None, linker = l))
                new_z = f(z,a,x,y,b)
                z_after = self._gemm(z_orig, a, x, y, b)

                #print z_orig, z_after, z, type(z_orig), type(z_after), type(z)
                #_approx_eq.debug = 1
                self.assertTrue(_approx_eq(z_after, z))
                if a == 0.0 and b == 1.0:
                    return
                elif z_orig.size == 0:
                    self.assertTrue(z.size==0)
                else:
                    self.assertFalse(numpy.all(z_orig == z))

            cmp_linker(copy(z), a, x, y, b, 'c|py')
            cmp_linker(copy(z), a, x, y, b, 'py')
            if config.blas.ldflags and not dtype.startswith("complex"):
                # If blas.ldflags is equal to '', the C code will not be generated
                cmp_linker(copy(z), a, x, y, b, 'c')

    def test0a(self):
        Gemm.debug = True
        try:
            g = gemm_inplace([1.], 1., [1.], [1.], 1.)
        except TypeError, e:
            if e[0] is Gemm.E_rank:
                return
        self.fail()

    def test0(self):
        try:
            self.cmp(1., 0., 1.0, 1.0, 1.0)
        except TypeError, e:
            if e[0] is Gemm.E_rank:
                return
        self.fail()

    def test2(self):
        try:
            self.cmp(2., 1.0, [3,2,1.], [[1],[2],[3.]], 1.0)
        except TypeError, e:
            self.assertTrue(e[0] == Gemm.E_rank)
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

    def test_shape_0(self):
        self.cmp(self.rand(0,4), -1.0, self.rand(0,5), self.rand(5,4), -1.0)
        self.cmp(self.rand(3,0), -1.0, self.rand(3,5), self.rand(5,0), -1.0)
        self.cmp(self.rand(3,4), -1.0, self.rand(3,0), self.rand(0,4), -1.0)
        self.cmp(self.rand(0,0), -1.0, self.rand(0,5), self.rand(5,0), -1.0)
        self.cmp(self.rand(0,0), -1.0, self.rand(0,0), self.rand(0,0), -1.0)

    def test_factorised_scalar(self):
        a=T.dmatrix()
        b=T.dmatrix()
        c=T.dmatrix()
        s=theano.shared(numpy.zeros((5,5)))

        lr1=T.constant(0.01).astype('float64')
        lr2=T.constant(2).astype('float64')
        l2_reg=T.constant(0.0001).astype('float64')

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
        one = T.constant(1.0).astype(Z.dtype)
        f = inplace_func([], gemm_inplace(Z, one, A, A, one))
        f()
        f = inplace_func([], gemm_inplace(Z, one, A, A.T, one))
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
            self.assertTrue(_approx_eq(z_after, tz.get_value(borrow=True)), (z_orig, z_after, z, z_after - z))
            f()
            self.assertTrue(_approx_eq(z_after, tz.get_value(borrow=True)), (z_orig, z_after, z, z_after - z))
            f()
            self.assertTrue(_approx_eq(z_after, tz.get_value(borrow=True)), (z_orig, z_after, z, z_after - z))

            #tz.value *= 0 # clear z's value
            y_T = ty.get_value(borrow=True).T
            ty.set_value(tx.get_value(borrow=True).T, borrow=True)
            tx.set_value(y_T, borrow=True)

            f()
            # test that the transposed version of multiplication gives same answer
            self.assertTrue(_approx_eq(z_after, tz.get_value(borrow=True).T))

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

    def test_non_contiguous(self):
        # Like test_transposes but with matrices without any
        # continuous dimension
        A = self.rand(4,4,3)
        B = self.rand(4,4,3)
        C = self.rand(4,4,3)

        def t(z, x, y, a=1.0, b=0.0, l='c|py', dt='float64'):
            z, a, x, y, b = [theano._asarray(p, dtype=dt) for p in z, a, x, y, b]
            z_orig = z.copy()
            z_after = numpy.zeros_like(z_orig)
            for i in xrange(3):
                z_after[:,:,i] = self._gemm(z[:,:,i], a, x[:,:,i], y[:,:,i], b)

            tz, ta, tx, ty, tb = [shared(p) for p in z, a, x, y, b]
            for i in xrange(3):
                f_i = inplace_func([],
                        gemm_inplace(tz[:,:,i], ta, tx[:,:,i], ty[:,:,i], tb),
                        mode=compile.Mode(optimizer=None, linker=l))
                for j in xrange(3):
                    # tz will not _always_ be overwritten,
                    # and adding update={...} in the call to function()
                    # will create cycles, so we update by hand.
                    z_i = f_i()
                    z = tz.get_value(borrow=True, return_internal_type=True)
                    z[:,:,i] = z_i

                    self.assertTrue(
                            _approx_eq(z_after[:,:,i],
                                       tz.get_value(borrow=True)[:,:,i]),
                            (z_orig[:,:,i], z_after[:,:,i],
                                z[:,:,i], z_after[:,:,i] - z[:,:,i]))

                tz_i = gemm_no_inplace(tz[:,:,i], ta, tx[:,:,i], ty[:,:,i], tb)
                g_i = theano.function([], tz_i,
                        updates={tz:T.set_subtensor(tz[:,:,i], tz_i)},
                        mode=compile.Mode(optimizer=None, linker=l))
                for j in xrange(3):
                    g_i()
                    self.assertTrue(
                            _approx_eq(z_after[:,:,i],
                                       tz.get_value(borrow=True)[:,:,i]),
                            (z_orig[:,:,i], z_after[:,:,i],
                                z[:,:,i], z_after[:,:,i] - z[:,:,i]))

        t(C, A, B)
        t(C.transpose((1,0,2)), A, B)
        t(C, A.transpose((1,0,2)), B, dt='float32')
        t(C, A, B.transpose((1,0,2)))
        t(C.transpose((1,0,2)), A.transpose((1,0,2)), B)
        t(C, A.transpose((1,0,2)), B.transpose((1,0,2)), dt='float32')
        t(C.transpose((1,0,2)), A, B.transpose((1,0,2)))
        t(C.transpose((1,0,2)), A.transpose((1,0,2)), B.transpose((1,0,2)), dt='float32')

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

        self.assertTrue(_as_scalar(a) == a)
        self.assertTrue(_as_scalar(b) != b)
        self.assertTrue(_as_scalar(d_a) != d_a)
        self.assertTrue(_as_scalar(d_b) != d_b)
        self.assertTrue(_as_scalar(d_a2) != d_a2)

    def test1(self):
        """Test that it fails on nonscalar constants"""
        a = T.constant(numpy.ones(5))
        self.assertTrue(None == _as_scalar(a))
        self.assertTrue(None == _as_scalar(T.DimShuffle([False], [0,'x'])(a)))

    def test2(self):
        """Test that it works on scalar variables"""
        a = T.dscalar()
        d_a = T.DimShuffle([], [])(a)
        d_a2 = T.DimShuffle([], ['x', 'x'])(a)

        self.assertTrue(_as_scalar(a) is a)
        self.assertTrue(_as_scalar(d_a) is a)
        self.assertTrue(_as_scalar(d_a2) is a)

    def test3(self):
        """Test that it fails on nonscalar variables"""
        a = T.dmatrix()
        self.assertTrue(None == _as_scalar(a))
        self.assertTrue(None == _as_scalar(T.DimShuffle([False, False], [0,'x', 1])(a)))

class T_real_matrix(TestCase):
    def test0(self):
        self.assertTrue(_is_real_matrix(T.DimShuffle([False,False], [1, 0])(T.dmatrix())))
        self.assertTrue(not _is_real_matrix(T.DimShuffle([False], ['x', 0])(T.dvector())))

def fail(msg):
    print 'FAIL', msg
    assert False

"""This test suite ensures that Gemm is inserted where it belongs, and that the resulting
functions compute the same things as the originals."""
def XYZab():
    return T.dmatrix(), T.dmatrix(), T.dmatrix(), T.dscalar(), T.dscalar()

class Failure(Exception):
    pass

def just_gemm(i, o, ishapes = [(4,3), (3,5), (4,5), (), ()], max_graphlen=0):
    try:
        f = inplace_func(
                [Param(ii, mutable=True, allow_downcast=True) for ii in i],
                o,
                mode='FAST_RUN')
        at_least_one_gemm = False
        for node in f.maker.env.nodes:
            if node.op == T.dot:
                raise Failure('dot not changed to gemm_inplace in graph')
            if node.op == _dot22:
                raise Failure('_dot22 not changed to gemm_inplace in graph')
            if node.op == gemm_inplace:
                at_least_one_gemm = True
        assert at_least_one_gemm
        g = inplace_func(i, o, mode=compile.Mode(linker='py', optimizer=None),
                allow_input_downcast=True)
        for node in g.maker.env.nodes:
            if node.op == gemm_inplace:
                raise Exception('gemm_inplace in original graph')

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
    just_gemm([X,Y,Z,a,b], [b * Z + a * T.dot(X,Y).T],
            ishapes=[(5,3), (3,4), (4,5), (), ()])

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
    o = [(a * T.dot(X,Y)
        + gemm_inplace(Z, b, S.T, R.T, T.constant(1.0).astype('float64')))]
    try:
        f = inplace_func([Param(ii, mutable=True) for ii in i],o,
                mode='FAST_RUN')
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

def test_upcasting_scalar_nogemv():
    # Test that the optimization does not crash when the scale has an incorrect
    # dtype, and forces upcasting of the result
    v = T.fvector('v')
    w = T.fmatrix('w')
    t = T.fvector('t')
    alpha = T.dscalar('a')

    rval = T.dot(w, v) * alpha + t

    f = theano.function([w, v, t, alpha], rval)
    t = f.maker.env.toposort()
    assert numpy.sum([isinstance(n.op, Gemv) for n in t]) == 0
    theano.printing.debugprint(f, print_type=True)

def test_upcasting_scalar_nogemm():
    # Test that the optimization does not crash when the scale has an incorrect
    # dtype, and forces upcasting of the result
    v = T.fmatrix('v')
    w = T.fmatrix('w')
    t = T.fmatrix('t')
    alpha = T.dscalar('a')

    rval = T.dot(w, v) * alpha + t

    f = theano.function([w, v, t, alpha], rval)
    t = f.maker.env.toposort()
    assert numpy.sum([isinstance(n.op, Gemm) for n in t]) == 0
    #theano.printing.debugprint(f, print_type=True)

    v = T.fmatrix('v')
    w = T.fmatrix('w')
    t = T.fmatrix('t')
    alpha = T.cscalar('a')

    on_opt_error = config.on_opt_error
    try:
        config.on_opt_error = 'raise'
        rval = T.dot(w, v) * alpha + t
        f = theano.function([w, v, t, alpha], rval)
    finally:
        config.on_opt_error = on_opt_error

    t = f.maker.env.toposort()
    assert numpy.sum([isinstance(n.op, Gemm) for n in t]) == 0
    #theano.printing.debugprint(f, print_type=True)

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
    assert gemm_no_inplace in [n.op for n in f.maker.env.nodes]

    # gemm_inplace should be inserted here, to work in-place on Z*c
    f = inplace_func([X,Y,Z,a,b, R, S, c],
            [Z * (c*Z + a * T.dot(X,Y) + b * T.dot(R,S).T)],
            mode='FAST_RUN')
    if (not gemm_inplace in [n.op for n in f.maker.env.nodes]):
        theano.printing.debugprint(f)
        raise Failure('no gemm_inplace in graph')

def test_inplace1():
    X,Y,Z,a,b = XYZab()
    # with > 2 terms in the overall addition
    f = inplace_func([X,Y,Z,a,b],
            [Z + Z + T.dot(X,Y)], mode='FAST_RUN')
    theano.printing.debugprint(f)
    # it doesn't work inplace because we didn't mark Z as mutable input
    assert [n.op for n in f.maker.env.nodes] == [gemm_no_inplace]

def test_dot22():
    for dtype1 in ['float32', 'float64', 'complex64', 'complex128']:
        a = T.matrix(dtype=dtype1)
        for dtype2 in ['float32', 'float64', 'complex64', 'complex128']:
            b = T.matrix(dtype=dtype2)
            f = theano.function([a,b],T.dot(a,b),mode=mode_blas_opt)
            topo = f.maker.env.toposort()
            if dtype1 == dtype2:
                assert _dot22 in [x.op for x in topo], (dtype1,dtype2)
            else:
                assert T.dot in [x.op for x in topo], (dtype1,dtype2)
            rng = numpy.random.RandomState(unittest_tools.fetch_seed())

            def cmp(a_shp, b_shp):
                av=rng.uniform(size=a_shp).astype(dtype1)
                bv=rng.uniform(size=b_shp).astype(dtype2)
                f(av,bv)

            cmp((3, 4), (4, 5))
            cmp((0, 4), (4, 5))
            cmp((3, 0), (0, 5))
            cmp((3, 4), (4, 0))
            cmp((0, 4), (4, 0))
            cmp((0, 0), (0, 0))

def test_dot22scalar():
    ## including does not seem to work for 'local_dot_to_dot22' and
    ## 'local_dot22_to_dot22scalar'
    ## TODO: exclude other optimizations in BlasOpt?
    #m = theano.compile.get_default_mode().including('local_dot_to_dot22','local_dot22_to_dot22scalar','specialize')
    #m = theano.compile.get_default_mode().including('BlasOpt', 'specialize')
    rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    for dtype1 in ['complex64', 'complex128']:
        a = T.matrix('a', dtype=dtype1)
        for dtype2 in ['complex64', 'complex128']:
            b = T.matrix('b', dtype=dtype2)
            for dtype3 in ['complex64', 'complex128']:
                c = T.matrix('c', dtype=dtype3)
                for dtype4 in ['complex64', 'complex128']:
                    cst = theano.tensor.basic.constant(.2, dtype=dtype4)
                    cst2 = theano.tensor.basic.constant(.1, dtype=dtype4)

                    def check_dot22scalar(func, len_topo_scalar=-1):
                        topo = func.maker.env.toposort()
                        ops = [x.op for x in topo]
                        dtype4_upcast = theano.scalar.upcast(dtype4, dtype1, dtype2)
                        if dtype1 == dtype2 == dtype3 == dtype4_upcast:
                            if len_topo_scalar>0:
                                assert len(topo) == len_topo_scalar
                            assert _dot22scalar in ops, (dtype1, dtype2, dtype3, dtype4)
                        elif dtype1 == dtype2 == dtype4_upcast:
                            if not (len_topo_scalar > 0):
                                assert len(topo) == len_topo_scalar
                                assert _dot22scalar in ops, (dtype1, dtype2, dtype3, dtype4)
                            else:
                                # Currently there is a problem of optimization order
                                # The constant get upcasted to float64 before we try to merge it
                                # with the dot22 of float32. So this prevent the merge.
                                assert _dot22scalar in ops or _dot22 in ops, (dtype1, dtype2, dtype3, dtype4)

                        elif dtype1 == dtype2:
                            assert _dot22 in ops, (dtype1, dtype2, dtype3, dtype4)
                        else:
                            assert T.dot in ops, (dtype1, dtype2, dtype3, dtype4)


                    def cmp(a_shp, b_shp, c_shp, sqr_shp=(5,5)):
                        av=rng.uniform(size=a_shp).astype(dtype1)
                        bv=rng.uniform(size=b_shp).astype(dtype2)
                        cv=rng.uniform(size=c_shp).astype(dtype3)
                        sv=rng.uniform(size=sqr_shp).astype(dtype1)

                        if False:
                            f = theano.function([a,b],cst*T.dot(a,b),mode=mode_blas_opt)
                            topo = f.maker.env.toposort()
                            check_dot22scalar(f, 1)

                            f(av,bv)

                        if True:
                            f = theano.function([a,b,c],cst*c*T.dot(a,b),mode=mode_blas_opt)
                            topo = f.maker.env.toposort()
                            check_dot22scalar(f, 2)

                            f(av,bv,cv)

                        f = theano.function([a,b,c],c * cst*T.dot(a,b),mode=mode_blas_opt)
                        topo = f.maker.env.toposort()
                        check_dot22scalar(f, 2)
                        f(av,bv,cv)

                        ## Here, canonicalize also seems needed
                        ## TODO: add only the optimizations needed?
                        m2 = mode_blas_opt.including('canonicalize')
                        f = theano.function([a,b,c],cst2 *c * cst*T.dot(a,b),mode=m2)
                        topo = f.maker.env.toposort()
                        check_dot22scalar(f, 2)
                        f(av,bv,cv)

                        if dtype1 == dtype2 == dtype3:
                            f = theano.function([a,b,c],c * cst*a*T.dot(a,b),mode=m2)
                            topo = f.maker.env.toposort()
                            check_dot22scalar(f, 2)
                            f(sv,sv,sv)

                            f = theano.function([a,b,c],cst*c *a*T.dot(a,b),mode=mode_blas_opt)
                            topo = f.maker.env.toposort()
                            #currently the canonizer don't always merge all Mul together...
                            # dot22scalar optimizer does not do a recursive search
                            # therefore, it doesn't find potential matches of the scalar.
                            # TODO: combine with the 'canonicalization' that is part of the Gemm optimizer.
                            #
                            #    assert _dot22scalar in [x.op for x in topo]
                            #    assert len(topo)==2
                            f(sv,sv,sv)

                            f = theano.function([a,b,c],c * a*cst*T.dot(a,b),mode=m2)
                            topo = f.maker.env.toposort()
                            check_dot22scalar(f, 2)
                            f(sv,sv,sv)

                    cmp((3,4),(4,5),(3,5))
                    cmp((0,4),(4,5),(0,5))
                    cmp((3,0),(0,5),(3,5))
                    cmp((3,4),(4,0),(3,0),(0,0))
                    cmp((0,4),(4,0),(0,0))
                    cmp((0,0),(0,0),(0,0))

def test_dot_w_self():
    # This can trigger problems in the optimization because what would normally be a gemm must
    # not be because the output is aliased to one of the inputs.

    A = shared(value=numpy.ones((2,2)))
    B = T.matrix()

    p = T.dot(A,A)*B

    grad = T.grad(T.mean(p), A)
    f = theano.function([B], p, updates={A : A - grad})

    # tests correctness in debugmode
    f(numpy.asarray([[0,1], [2,3]], dtype=config.floatX))


###############################################################################
## Tests for Gemv
###############################################################################

class TestGemv(TestCase):
    def test_dot_vm(self):
        ''' Test vector dot matrix '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        m = theano.shared(numpy.array(rng.uniform(size=(2,3)), dtype='float32'))
        f = theano.function([], theano.dot(v,m), mode = mode_blas_opt)

        # Assert they produce the same output
        assert numpy.allclose(f(), numpy.dot(v.get_value(), m.get_value()))

        # Assert that the dot was optimized somehow
        assert sum([isinstance(node.op, T.Dot) for node in
                    f.maker.env.toposort() ]) == 0
        assert sum([isinstance(node.op, T.blas.Dot22) for node in
                    f.maker.env.toposort() ]) == 1

    def test_dot_mv(self):
        ''' Test matrix dot vector '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        m = theano.shared(numpy.array(rng.uniform(size=(3,2)),
                                       dtype='float32'))
        f = theano.function([], theano.dot(m,v), mode = mode_blas_opt)

        # Assert they produce the same output
        assert numpy.allclose(f(), numpy.dot(m.get_value(), v.get_value()))

        # Assert that the dot was optimized somehow
        assert sum([isinstance(node.op, T.Dot) for node in
                    f.maker.env.toposort() ]) == 0
        assert sum([isinstance(node.op, T.blas.Dot22) for node in
                    f.maker.env.toposort() ]) == 1

    @staticmethod
    def t_gemv1(m_shp):
        ''' test vector2+dot(matrix,vector1) '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v1 = theano.shared(numpy.array(rng.uniform(size=(m_shp[1],)), dtype='float32'))
        v2_orig = numpy.array(rng.uniform(size=(m_shp[0],)), dtype='float32')
        v2 = theano.shared(v2_orig)
        m  = theano.shared(numpy.array(rng.uniform(size=m_shp), dtype='float32'))

        f = theano.function([], v2+theano.dot(m,v1), mode = mode_blas_opt)

        # Assert they produce the same output
        assert numpy.allclose(f(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert isinstance(topo[0].op, Gemv)
        assert topo[0].op.inplace==False

        #test the inplace version
        f = theano.function([], [], updates={v2:v2+theano.dot(m,v1)}
                            , mode = mode_blas_opt)

        # Assert they produce the same output
        f()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert isinstance(topo[0].op, Gemv)
        if config.mode != 'FAST_COMPILE':
            assert topo[0].op.inplace==True

    def test_gemv1(self):
        self.t_gemv1((3,2))
        self.t_gemv1((0,2))
        self.t_gemv1((3,0))
        self.t_gemv1((0,0))

    def test_gemv2(self):
        ''' test vector2+dot(vector1,matrix) '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v1 = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        v2_orig = numpy.array(rng.uniform(size=(3,)), dtype='float32')
        v2 = theano.shared(v2_orig )
        m  = theano.shared(numpy.array(rng.uniform(size=(2,3)), dtype='float32'))

        f = theano.function([], v2+theano.dot(v1,m), mode = mode_blas_opt)

        # Assert they produce the same output
        assert numpy.allclose(f(),
                numpy.dot(v1.get_value(), m.get_value()) + v2.get_value())
        topo = f.maker.env.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo)==1
        assert topo[-1].op.inplace==False

        #test the inplace version
        f = theano.function([], [], updates={v2:v2+theano.dot(v1,m)}
                            , mode = mode_blas_opt)

        # Assert they produce the same output
        f()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(v1.get_value(), m.get_value()) + v2_orig)
        topo = f.maker.env.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo)==1
        if config.mode != 'FAST_COMPILE':
            assert topo[-1].op.inplace==True

    def test_gemv_dimensions(self):
        A = T.matrix('A')
        x, y = T.vectors('x', 'y')
        alpha = theano.shared(theano._asarray(1.0, dtype=config.floatX),
                name='alpha')
        beta = theano.shared(theano._asarray(1.0, dtype=config.floatX),
                name='beta')

        z = beta * y + alpha * T.dot(A, x)
        f = theano.function([A, x, y], z)

        # Matrix value
        A_val = numpy.ones((5,3), dtype=config.floatX)
        # Different vector length
        ones_3 = numpy.ones(3, dtype=config.floatX)
        ones_4 = numpy.ones(4, dtype=config.floatX)
        ones_5 = numpy.ones(5, dtype=config.floatX)
        ones_6 = numpy.ones(6, dtype=config.floatX)

        f(A_val, ones_3, ones_5)
        self.assertRaises(ValueError, f, A_val, ones_4, ones_5)
        self.assertRaises(ValueError, f, A_val, ones_3, ones_6)
        self.assertRaises(ValueError, f, A_val, ones_4, ones_6)

# The following gemv tests were added in March 2011 by Ian Goodfellow
# and are based on the gemv tests from scipy
# http://projects.scipy.org/scipy/browser/trunk/scipy/linalg/tests/test_fblas.py?rev=6803
# NOTE: At the time these tests were written, theano did not have a
# conjugate function. If such a thing is ever added, the tests involving
# conjugate should be ported over as well.


def matrixmultiply(a, b):
    if len(b.shape) == 1:
        b_is_vector = True
        b = b[:,newaxis]
    else:
        b_is_vector = False
    assert_(a.shape[1] == b.shape[0])
    c = zeros((a.shape[0], b.shape[1]), common_type(a, b))
    for i in xrange(a.shape[0]):
        for j in xrange(b.shape[1]):
            s = 0
            for k in xrange(a.shape[1]):
                s += a[i,k] * b[k, j]
            c[i,j] = s
    if b_is_vector:
        c = c.reshape((a.shape[0],))
    return c


class BaseGemv(object):
    def get_data(self,x_stride=1,y_stride=1):
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        mult = array(1, dtype=self.dtype)
        if self.dtype in [complex64,complex128]:
            mult = array(1 + 1j, dtype=self.dtype)
        alpha = array(1., dtype=self.dtype) * mult
        beta = array(1., dtype=self.dtype) * mult
        a = rng.randn(3,3).astype(self.dtype) * mult
        x = arange(shape(a)[0]*x_stride,dtype=self.dtype) * mult
        y = arange(shape(a)[1]*y_stride,dtype=self.dtype) * mult
        return alpha,beta,a,x,y

    def test_simple(self):
        alpha, beta, a, x, y = [ shared(value) for value in self.get_data() ]
        desired_oy = alpha.get_value() * matrixmultiply(a.get_value(),x.get_value()) + beta.get_value() * y.get_value()

        oy    = alpha * T.dot(a,x) + beta * y

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==1

        oy_val = oy_func()

        assert_array_almost_equal(desired_oy, oy_val)

    def test_default_beta_y(self):

        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        a = shared(a_v)
        x = shared(x_v)

        desired_oy = matrixmultiply(a_v, x_v)

        oy = T.dot(a,x)

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        # The only op in the graph is a dot.
        # In the gemm case, we create a dot22 for that case
        # There is no dot21.
        # Creating one is not usefull as this is not faster(in fact it would be slower!
        # as more code would be in python, numpy.dot will call gemv itself)
        # See ticket 594
        """
>>> t0=time.time();x=scipy.linalg.blas.fblas.dgemv(1,a.T,b,1,z.T);t1=time.time();print t1-t0
0.00192999839783
>>> t0=time.time();x=numpy.dot(a,b);t1=time.time();print t1-t0
0.00158381462097
"""
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==0

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)


    def test_simple_transpose(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v)+beta_v*y_v

        oy = alpha * T.dot(a.T,x)+beta*y

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==1

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_x_stride(self):
        vs = self.get_data(x_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(a_v,x_v[::2])+beta_v*y_v

        oy = alpha * T.dot(a,x[::2])+beta*y

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==1

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_x_stride_transpose(self):
        vs = self.get_data(x_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v[::2])+beta_v*y_v

        oy = alpha * T.dot(a.T,x[::2])+beta*y

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==1

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_y_stride(self):
        vs = self.get_data(y_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(a_v,x_v)+beta_v*y_v[::2]

        oy = alpha * T.dot(a,x)+beta*y[::2]

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==1

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_y_stride_transpose(self):
        vs = self.get_data(y_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v)+beta_v*y_v[::2]

        oy = alpha * T.dot(a.T,x)+beta*y[::2]

        oy_func = theano.function([], oy, mode = mode_blas_opt)

        topo = oy_func.maker.env.toposort()
        assert sum([isinstance(node.op, theano.tensor.blas.Gemv) for node in topo])==1

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)



class TestSgemv(TestCase, BaseGemv):
    dtype = float32

class TestDgemv(TestCase, BaseGemv):
    dtype = float64

#The optimization to put Gemv don't work for complex type for now.
# See ticket 653.
#class TestCgemv(TestCase, BaseGemv):
#    dtype = complex64

#class TestZgemv(TestCase, BaseGemv):
#    dtype = complex128

###############################################################################
## Tests for Ger
###############################################################################

class TestGer_make_node(TestCase):
    def setUp(self):
        self.iv = T.tensor(dtype='int32', broadcastable=(False,))
        self.fv = T.tensor(dtype='float32', broadcastable=(False,))
        self.fv1 = T.tensor(dtype='float32', broadcastable=(True,))
        self.dv = T.tensor(dtype='float64', broadcastable=(False,))
        self.dv1 = T.tensor(dtype='float64', broadcastable=(True,))
        self.cv = T.tensor(dtype='complex64', broadcastable=(False,))
        self.zv = T.tensor(dtype='complex128', broadcastable=(False,))

        self.fv_2 = T.tensor(dtype='float32', broadcastable=(False,))
        self.fv1_2 = T.tensor(dtype='float32', broadcastable=(True,))
        self.dv_2 = T.tensor(dtype='float64', broadcastable=(False,))
        self.dv1_2 = T.tensor(dtype='float64', broadcastable=(True,))
        self.cv_2 = T.tensor(dtype='complex64', broadcastable=(False,))
        self.zv_2 = T.tensor(dtype='complex128', broadcastable=(False,))

        self.fm = T.fmatrix()
        self.dm = T.dmatrix()
        self.cm = T.cmatrix()
        self.zm = T.zmatrix()

        self.fa = T.fscalar()
        self.da = T.dscalar()
        self.ca = T.cscalar()
        self.za = T.zscalar()

    def test_works_on_all_valid_dtypes(self):
        self.assertEquals(self.fm.type,
            ger(self.fm, self.fa, self.fv, self.fv_2).type)
        self.assertEquals(self.fm.type,
            ger(self.fm, self.fa, self.fv, self.fv_2).type)
        self.assertEquals(self.fm.type,
            ger(self.fm, self.fa, self.fv, self.fv_2).type)
        self.assertEquals(self.fm.type,
            ger(self.fm, self.fa, self.fv, self.fv_2).type)

    def test_fails_on_invalid_dtypes(self):
        self.assertRaises(TypeError,
                ger, T.imatrix(), T.iscalar(), T.ivector(),
                T.ivector())

    def test_fails_for_nonscalar_alpha(self):
        self.assertRaises(TypeError,
                ger, self.fm, self.fm, self.fv, self.fv_2)
        # boundary case - fv1 has the right dtype and could be dimshuffled to a
        # scalar, but that's not make_node's job.
        self.assertRaises(TypeError,
                ger, self.fm, self.fv1, self.fv, self.fv_2)
        # actually doing the aforementioned dimshuffle makes it work
        self.assertEquals(self.fm.type,
                ger(self.fm, self.fv1.dimshuffle(), self.fv, self.fv_2).type)

    def test_fails_for_nonmatrix_A(self):
        self.assertRaises(TypeError,
                ger, self.fv, self.fa, self.fv, self.fv_2)

    def test_fails_for_nonvector_x_or_y(self):
        self.assertRaises(TypeError,
                ger, self.fm, self.fa, self.fv.dimshuffle('x', 0), self.fv_2)
        self.assertRaises(TypeError,
                ger, self.fm, self.fa, self.fv, self.fv_2.dimshuffle('x', 0))

    def test_fails_for_mixed_dtypes(self):
        self.assertRaises(TypeError, ger, self.dm, self.fa, self.fv, self.fv_2)
        self.assertRaises(TypeError, ger, self.fm, self.da, self.fv, self.fv_2)
        self.assertRaises(TypeError, ger, self.fm, self.fa, self.dv, self.fv_2)
        self.assertRaises(TypeError, ger, self.fm, self.fa, self.fv, self.dv_2)
        self.assertRaises(TypeError, ger, self.cm, self.fa, self.fv, self.dv_2)
        self.assertRaises(TypeError, ger, self.cm, self.fa, self.fv, self.zv_2)

# TODO: refactor this into some place where all OpTesters could use it.
# This object name should not start with Test.
# Otherwise nosetests will execute it!
class T_OpContractMixin(object):
    # self.ops should be a list of instantiations of an Op class to test.
    # self.other_op should be an op which is different from every op
    other_op = T.add

    def copy(self, x):
        return copy(x)

    def deepcopy(self, x):
        return deepcopy(x)

    def clone(self, op):
        raise NotImplementedError('return new instance like `op`')

    def test_eq(self):
        for i, op_i in enumerate(self.ops):
            assert op_i == op_i
            assert op_i == self.copy(op_i)
            assert op_i == self.deepcopy(op_i)
            assert op_i == self.clone(op_i)
            assert op_i != self.other_op
            for j, op_j in enumerate(self.ops):
                if i == j: continue
                assert op_i != op_j

    def test_hash(self):
        for i, op_i in enumerate(self.ops):
            h_i = hash(op_i)
            assert h_i == hash(op_i)
            assert h_i == hash(self.copy(op_i))
            assert h_i == hash(self.deepcopy(op_i))
            assert h_i == hash(self.clone(op_i))
            assert h_i != hash(self.other_op)
            for j, op_j in enumerate(self.ops):
                if i == j: continue
                assert op_i != hash(op_j)

    def test_name(self):
        for op in self.ops:
            s = str(op)    # show that str works
            assert s       # names should not be empty

class TestGer_OpContract(TestCase, T_OpContractMixin):
    #TODO: These tests could be factored into a generic Op-testing base-class
    def setUp(self):
        self.ops = [ger, ger_destructive]

    def clone(self, op):
        return Ger(op.destructive)

class TestGer_make_thunk(TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(unittest_tools.fetch_seed())

    def given_dtype(self, dtype, M, N):
        sA = T.tensor(dtype=dtype, broadcastable=(False, False))
        sa = T.tensor(dtype=dtype, broadcastable=())
        sx = T.tensor(dtype=dtype, broadcastable=(False,))
        sy = T.tensor(dtype=dtype, broadcastable=(False,))

        sZ = ger(sA, sa, sx, sy)
        node = sZ.owner

        storage_map = {sA:[None], sa:[None], sx:[None], sy:[None], sZ:[None]}

        thunk = ger.make_thunk(node, storage_map,
                compute_map={}, no_recycling=[])

        # non-standard for make_thunk to receive node.op != self,
        # but works for now.
        thunk_d = ger_destructive.make_thunk(node, storage_map,
                compute_map={}, no_recycling=[])

        def rand(*shape):
            return numpy.asarray(1 + self.rng.rand(*shape), dtype=dtype)

        storage_map[sA][0] = rand(M, N)
        storage_map[sa][0] = rand()
        storage_map[sx][0] = rand(M)
        storage_map[sy][0] = rand(N)

        storage_map_copy = dict([(k,[deepcopy(v[0])]) for k,v in storage_map.items()])

        # TODO: do some DebugMode-type verifications here
        #       if this can be refactored into a Mixin that does the DebugMode
        #       stuff on just one thunk at a time.  Do it in the style of
        #       TestOpContractMixin?
        #       - Compare with Elemwise testers
        thunk()

        assert numpy.all(storage_map[sZ][0] ==
                storage_map[sA][0] + storage_map[sa][0] *
                numpy.outer(storage_map[sx][0], storage_map[sy][0]))
        assert storage_map[sZ][0].dtype == dtype
        assert storage_map[sZ][0].shape == (M, N)

        thunk_d()
        assert numpy.all(storage_map[sZ][0] !=
                storage_map[sA][0] + storage_map[sa][0] *
                numpy.outer(storage_map[sx][0], storage_map[sy][0]))
        assert numpy.all(storage_map[sZ][0] ==
                storage_map_copy[sA][0] + storage_map[sa][0] *
                numpy.outer(storage_map[sx][0], storage_map[sy][0]))
        assert storage_map[sZ][0].dtype == dtype
        assert storage_map[sZ][0].shape == (M, N)

    def test_f32_0_0(self): return self.given_dtype('float32', 0, 0)
    def test_f32_1_0(self): return self.given_dtype('float32', 1, 0)
    def test_f32_0_1(self): return self.given_dtype('float32', 0, 1)
    def test_f32_1_1(self): return self.given_dtype('float32', 1, 1)

    def test_f32_4_4(self): return self.given_dtype('float32', 4, 4)
    def test_f64_4_5(self): return self.given_dtype('float64', 4, 5)
    def test_c64_7_1(self): return self.given_dtype('complex64', 7, 1)
    def test_c128_1_9(self): return self.given_dtype('complex128', 1, 9)


# TODO: Refactor and add to this base class as we refactor test code.
class TestOptimizationMixin(object):

    def assertFunctionContains(self, f, op, min=1, max=sys.maxint):
        toposort = f.maker.env.toposort()
        matches = [node for node in toposort if node.op == op]
        assert (min <= len(matches) <= max), toposort

    def assertFunctionContains0(self, f, op):
        return self.assertFunctionContains(f, op, min=0, max=0)

    def assertFunctionContains1(self, f, op):
        return self.assertFunctionContains(f, op, min=1, max=1)

    def assertFunctionContainsN(self, f, op, N):
        return self.assertFunctionContains(f, op, min=N, max=N)

    def SkipTest(self):
        raise Exception('how do I skip this test properly?')

class TestGer_local_gemm_to_ger(TestCase, TestOptimizationMixin):

    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('fast_run')
        dtype = self.dtype = 'float64'  # optimization isn't dtype-dependent
        self.A = T.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = T.tensor(dtype=dtype, broadcastable=())
        self.x = T.tensor(dtype=dtype, broadcastable=(False,))
        self.y = T.tensor(dtype=dtype, broadcastable=(False,))
        self.origval = theano.tensor.blas_scipy.optimizations_enabled
        theano.tensor.blas_scipy.optimizations_enabled = False

    def tearDown(self):
        theano.tensor.blas_scipy.optimizations_enabled = self.origval

    def function(self, inputs, outputs):
        return theano.function(inputs, outputs, self.mode)

    def b(self, bval):
        return T.as_tensor_variable(numpy.asarray(bval, dtype=self.dtype))

    def test_b_0_triggers_ger(self):
        assert T.blas.local_gemm_to_ger.transform(
                gemm_no_inplace(
                    self.A, self.a, self.x.dimshuffle(0,'x'),
                    self.y.dimshuffle('x', 0), self.b(0)).owner)
    def test_b_1_triggers_ger(self):
        assert T.blas.local_gemm_to_ger.transform(
                gemm_no_inplace(
                    self.A, self.a, self.x.dimshuffle(0,'x'),
                    self.y.dimshuffle('x', 0), self.b(1)).owner)
    def test_b_other_does_not_triggers_ger(self):
        assert not T.blas.local_gemm_to_ger.transform(
                gemm_no_inplace(
                    self.A, self.a, self.x.dimshuffle(0,'x'),
                    self.y.dimshuffle('x', 0), self.b(1.5)).owner)

    def test_outer(self):
        f = self.function([self.x, self.y], T.outer(self.x, self.y))
        self.assertFunctionContains(f, ger_destructive)

    def test_A_plus_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + T.outer(self.x, self.y))
        self.assertFunctionContains(f, ger)

    def test_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + 0.1 * T.outer(self.x, self.y))
        self.assertFunctionContains(f, ger)

    def test_scaled_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                0.2 * self.A + 0.1 * T.outer(self.x, self.y))
        self.assertFunctionContains(f, gemm_no_inplace)
