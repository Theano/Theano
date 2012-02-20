#from nose.plugins.skip import SkipTest
#import traceback
import itertools, sys
import theano.tensor as T
from theano import tensor
from theano.gof.python25 import product as itertools_product
from theano.printing import pp

import numpy
import theano
from numpy import (arange, array, common_type, complex64, complex128, float32,
                  float64, newaxis, shape, transpose, zeros)
from numpy.testing import assert_array_almost_equal
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

mode_blas_opt = theano.compile.get_default_mode().including(
    'BlasOpt', 'specialize', 'InplaceBlasOpt')
mode_blas_opt = mode_blas_opt.excluding('c_blas')

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


def test_dot22scalar_cast():
    """
    Test that in `dot22_to_dot22scalar` we properly cast integers to floats.
    """
    # Note that this test was failing before d5ff6904.
    A = T.dmatrix()
    for scalar_int_type in T.int_dtypes:
        y = T.scalar(dtype=scalar_int_type)
        f = theano.function([A, y], T.dot(A, A) * y, mode=mode_blas_opt)
        assert _dot22scalar in [x.op for x in f.maker.env.toposort()]
    A = T.fmatrix()
    for scalar_int_type in T.int_dtypes:
        y = T.scalar(dtype=scalar_int_type)
        f = theano.function([A, y], T.dot(A, A) * y, mode=mode_blas_opt)
        if scalar_int_type in ['int32', 'int64']:
            assert _dot22 in [x.op for x in f.maker.env.toposort()]
        else:
            assert _dot22scalar in [x.op for x in f.maker.env.toposort()]


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

class TestGemv(TestCase, unittest_tools.TestOptimizationMixin):
    def test_dot_vv(self):
        ''' Currently we generate a gemv for that case'''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        w = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        f = theano.function([], theano.dot(v, w), mode=mode_blas_opt)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, T.dot)
        self.assertFunctionContains1(f, Gemv(False))

        # Assert they produce the same output
        assert numpy.allclose(f(), numpy.dot(v.get_value(), w.get_value()))

    def test_dot_vm(self):
        ''' Test vector dot matrix '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        m = theano.shared(numpy.array(rng.uniform(size=(2,3)), dtype='float32'))
        f = theano.function([], theano.dot(v,m), mode=mode_blas_opt)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, T.dot)
        self.assertFunctionContains1(f, Gemv(True))

        # Assert they produce the same output
        assert numpy.allclose(f(), numpy.dot(v.get_value(), m.get_value()))
        # Assert it works when m has no contiguous dimension
        m.set_value(
                m.get_value(borrow=True)[::-1, ::-1],
                borrow=True)
        assert numpy.allclose(f(), numpy.dot(v.get_value(), m.get_value()))


    def test_dot_mv(self):
        ''' Test matrix dot vector '''
        rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        v = theano.shared(numpy.array(rng.uniform(size=(2,)), dtype='float32'))
        m = theano.shared(numpy.array(rng.uniform(size=(3,2)),
                                       dtype='float32'))
        f = theano.function([], theano.dot(m,v), mode=mode_blas_opt)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, T.dot)
        self.assertFunctionContains1(f, Gemv(True))

        # Assert they produce the same output
        assert numpy.allclose(f(), numpy.dot(m.get_value(), v.get_value()))
        # Assert it works when m has no contiguous dimension
        m.set_value(
                m.get_value(borrow=True)[::-1, ::-1],
                borrow=True)
        assert numpy.allclose(f(), numpy.dot(m.get_value(), v.get_value()))

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
        g = theano.function([], [], updates={v2:v2+theano.dot(m,v1)}
                            , mode = mode_blas_opt)

        # Assert they produce the same output
        g()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = g.maker.env.toposort()
        assert len(topo)==1
        assert isinstance(topo[0].op, Gemv)
        if config.mode != 'FAST_COMPILE':
            assert topo[0].op.inplace==True

        # Do the same tests with a matrix with strides in both dimensions
        m.set_value(
                m.get_value(borrow=True)[::-1, ::-1],
                borrow=True)
        v2.set_value(v2_orig)
        assert numpy.allclose(f(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)
        g()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(m.get_value(), v1.get_value()) + v2_orig)

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
        g = theano.function([], [], updates={v2:v2+theano.dot(v1,m)}
                            , mode = mode_blas_opt)

        # Assert they produce the same output
        g()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(v1.get_value(), m.get_value()) + v2_orig)
        topo = g.maker.env.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo)==1
        if config.mode != 'FAST_COMPILE':
            assert topo[-1].op.inplace==True

        # Do the same tests with a matrix with strides in both dimensions
        m.set_value(
                m.get_value(borrow=True)[::-1, ::-1],
                borrow=True)
        v2.set_value(v2_orig)
        assert numpy.allclose(f(),
                numpy.dot(v1.get_value(), m.get_value()) + v2.get_value())
        g()
        assert numpy.allclose(v2.get_value(),
                numpy.dot(v1.get_value(), m.get_value()) + v2_orig)

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
        f(A_val[::-1, ::-1], ones_3, ones_5)
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
    assert a.shape[1] == b.shape[0]
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
    mode = mode_blas_opt  # can be overridden with self.mode
    shared = staticmethod(theano.shared)

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
        alpha, beta, a, x, y = [ self.shared(value) for value in self.get_data() ]
        desired_oy = alpha.get_value() * matrixmultiply(a.get_value(),x.get_value()) + beta.get_value() * y.get_value()

        oy    = alpha * T.dot(a,x) + beta * y

        oy_func = theano.function([], oy, mode=self.mode)

        topo = oy_func.maker.env.toposort()
        self.assertFunctionContains1(oy_func, self.gemv)

        oy_val = oy_func()

        assert_array_almost_equal(desired_oy, oy_val)

    def test_default_beta_y(self):

        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        a = self.shared(a_v)
        x = self.shared(x_v)

        desired_oy = matrixmultiply(a_v, x_v)

        oy = T.dot(a,x)

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv_inplace)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)


    def test_simple_transpose(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v)+beta_v*y_v

        oy = alpha * T.dot(a.T,x)+beta*y

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_x_stride(self):
        vs = self.get_data(x_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(a_v,x_v[::2])+beta_v*y_v

        oy = alpha * T.dot(a,x[::2])+beta*y

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_x_stride_transpose(self):
        vs = self.get_data(x_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v[::2])+beta_v*y_v

        oy = alpha * T.dot(a.T,x[::2])+beta*y

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_y_stride(self):
        vs = self.get_data(y_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(a_v,x_v)+beta_v*y_v[::2]

        oy = alpha * T.dot(a,x)+beta*y[::2]

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_y_stride_transpose(self):
        vs = self.get_data(y_stride = 2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v)+beta_v*y_v[::2]

        oy = alpha * T.dot(a.T,x)+beta*y[::2]

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_a_strides(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]
        a_v = a_v[::-1, ::-1]
        a.set_value(
                a.get_value(borrow=True, return_internal_type=True)[::-1, ::-1],
                borrow=True)

        desired_oy = alpha_v * matrixmultiply(a_v,x_v)+beta_v*y_v

        oy = alpha * T.dot(a,x)+beta*y

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_a_strides_transpose(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = [ self.shared(v) for v in vs ]
        a_v = a_v[::-1, ::-1]
        a.set_value(
                a.get_value(borrow=True, return_internal_type=True)[::-1, ::-1],
                borrow=True)

        desired_oy = alpha_v * matrixmultiply(transpose(a_v),x_v)+beta_v*y_v

        oy = alpha * T.dot(a.T,x)+beta*y

        oy_func = theano.function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_upcasting_scalar_nogemv(self):
        # Test that the optimization does not crash when the scale has
        # an incorrect dtype, and forces upcasting of the result
        # We put this test in this class to test it on the gpu too.
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha_v = alpha_v.astype("float64")
        a_v = a_v.astype("float32")
        x_v = x_v.astype("float32")
        y_v = y_v.astype("float32")

        alpha = T.dscalar('a')
        a = T.fmatrix('w')
        x = T.fvector('v')
        y = T.fvector('t')

        rval = T.dot(a, x) * alpha + y

        f = theano.function([a, x, y, alpha], rval, mode=self.mode)
        # this function is currently optimized so that the gemv is
        # done inplace on a temporarily allocated-buffer, which is
        # then scaled by alpha and to t with a fused elemwise.
        n_gemvs = 0
        #theano.printing.debugprint(f, print_type=True)
        for node in f.maker.env.toposort():
            if node.op == self.gemv_inplace:
                n_gemvs += 1
                assert node.outputs[0].dtype == 'float32'
        assert n_gemvs == 1, n_gemvs
        self.assertFunctionContains1(f, self.gemv_inplace)
        f(a_v, x_v, y_v, alpha_v)


class TestSgemv(TestCase, BaseGemv, unittest_tools.TestOptimizationMixin):
    dtype = float32
    gemv = theano.tensor.blas.gemv_no_inplace
    gemv_inplace = theano.tensor.blas.gemv_inplace


class TestDgemv(TestCase, BaseGemv, unittest_tools.TestOptimizationMixin):
    dtype = float64
    gemv = theano.tensor.blas.gemv_no_inplace
    gemv_inplace = theano.tensor.blas.gemv_inplace

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


class TestGer_OpContract(TestCase, unittest_tools.T_OpContractMixin):
    def setUp(self):
        self.ops = [ger, ger_destructive]

    def clone(self, op):
        return Ger(op.destructive)


class TestGer(TestCase, unittest_tools.TestOptimizationMixin):
    shared = staticmethod(theano.shared)

    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('fast_run')
        self.mode = self.mode.excluding('c_blas', 'scipy_blas')
        dtype = self.dtype = 'float64'  # optimization isn't dtype-dependent
        self.A = T.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = T.tensor(dtype=dtype, broadcastable=())
        self.x = T.tensor(dtype=dtype, broadcastable=(False,))
        self.y = T.tensor(dtype=dtype, broadcastable=(False,))
        self.ger = ger
        self.ger_destructive = ger_destructive
        self.gemm = gemm_no_inplace

    def function(self, inputs, outputs, updates={}):
        return theano.function(inputs, outputs, self.mode, updates=updates)

    def b(self, bval):
        return T.as_tensor_variable(numpy.asarray(bval, dtype=self.dtype))

    def test_b_0_triggers_ger(self):
        """ test local_gemm_to_ger opt"""
        assert T.blas.local_gemm_to_ger.transform(
                gemm_no_inplace(
                    self.A, self.a, self.x.dimshuffle(0,'x'),
                    self.y.dimshuffle('x', 0), self.b(0)).owner)
    def test_b_1_triggers_ger(self):
        """ test local_gemm_to_ger opt"""
        assert T.blas.local_gemm_to_ger.transform(
                gemm_no_inplace(
                    self.A, self.a, self.x.dimshuffle(0,'x'),
                    self.y.dimshuffle('x', 0), self.b(1)).owner)
    def test_b_other_does_not_triggers_ger(self):
        """ test local_gemm_to_ger opt"""
        assert not T.blas.local_gemm_to_ger.transform(
                gemm_no_inplace(
                    self.A, self.a, self.x.dimshuffle(0,'x'),
                    self.y.dimshuffle('x', 0), self.b(1.5)).owner)

    def test_outer(self):
        f = self.function([self.x, self.y], T.outer(self.x, self.y))
        self.assertFunctionContains(f, self.ger_destructive)
        f(numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))

    def test_A_plus_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + T.outer(self.x, self.y))
        self.assertFunctionContains(f, self.ger)
        f(numpy.random.rand(5, 4).astype(self.dtype),
          numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))
        f(numpy.random.rand(5, 4).astype(self.dtype)[::-1, ::-1],
          numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))

    def test_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                self.A + 0.1 * T.outer(self.x, self.y))
        self.assertFunctionContains(f, self.ger)
        f(numpy.random.rand(5, 4).astype(self.dtype),
          numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))
        f(numpy.random.rand(5, 4).astype(self.dtype)[::-1, ::-1],
          numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))

    def test_scaled_A_plus_scaled_outer(self):
        f = self.function([self.A, self.x, self.y],
                          numpy.asarray(0.2, self.dtype) * self.A +
                          numpy.asarray(0.1, self.dtype) * T.outer(
                self.x, self.y))
        # Why gemm? This make the graph simpler did we test that it
        # make it faster?
        self.assertFunctionContains(f, self.gemm)
        f(numpy.random.rand(5, 4).astype(self.dtype),
          numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))
        f(numpy.random.rand(5, 4).astype(self.dtype)[::-1, ::-1],
          numpy.random.rand(5).astype(self.dtype),
          numpy.random.rand(4).astype(self.dtype))

    def given_dtype(self, dtype, M, N):
        """ test corner case shape and dtype"""

        f = self.function([self.A, self.x, self.y],
                self.A + 0.1 * T.outer(self.x, self.y))
        self.assertFunctionContains(f, self.ger)
        f(numpy.random.rand(M, N).astype(self.dtype),
          numpy.random.rand(M).astype(self.dtype),
          numpy.random.rand(N).astype(self.dtype))
        f(numpy.random.rand(M, N).astype(self.dtype)[::-1, ::-1],
          numpy.random.rand(M).astype(self.dtype),
          numpy.random.rand(N).astype(self.dtype))

    def test_f32_0_0(self):
        return self.given_dtype('float32', 0, 0)

    def test_f32_1_0(self):
        return self.given_dtype('float32', 1, 0)

    def test_f32_0_1(self):
        return self.given_dtype('float32', 0, 1)

    def test_f32_1_1(self):
        return self.given_dtype('float32', 1, 1)

    def test_f32_4_4(self):
        return self.given_dtype('float32', 4, 4)

    def test_f32_7_1(self):
        return self.given_dtype('float32', 7, 1)

    def test_f32_1_2(self):
        return self.given_dtype('float32', 1, 2)

    def test_f64_4_5(self):
        return self.given_dtype('float64', 4, 5)

    def test_c64_7_1(self):
        return self.given_dtype('complex64', 7, 1)

    def test_c128_1_9(self):
        return self.given_dtype('complex128', 1, 9)

    def test_inplace(self):
        A = self.shared(numpy.random.rand(4, 5).astype(self.dtype))
        f = self.function([self.x, self.y], [],
                          updates={A: A + T.constant(0.1, dtype=self.dtype) *
                                   T.outer(self.x, self.y)})
        self.assertFunctionContains(f, self.ger_destructive)
        f(numpy.random.rand(4).astype(self.dtype),
          numpy.random.rand(5).astype(self.dtype))

        A.set_value(
            A.get_value(borrow=True, return_internal_type=True)[::-1, ::-1],
            borrow=True)
        f(numpy.random.rand(4).astype(self.dtype),
          numpy.random.rand(5).astype(self.dtype))

class TestBlasStrides(TestCase):
    dtype = 'float64'
    shared = staticmethod(tensor._shared)
    mode = theano.compile.get_default_mode()
    mode = mode.including('fast_run').excluding('gpu', 'c_blas', 'scipy_blas')
    rng = numpy.random.RandomState(seed=unittest_tools.fetch_seed())

    def rand(self, *shape):
        return theano._asarray(self.rng.rand(*shape), dtype=self.dtype)

    def cmp_dot22(self, b_shp, c_shp):
        av = numpy.zeros((0, 0), dtype=self.dtype)
        bv = self.rand(*b_shp)
        cv = self.rand(*c_shp)

        a = self.shared(av, 'a')
        b = self.shared(bv, 'b')
        c = self.shared(cv, 'c')

        b_t = self.shared(bv.T, 'b.T')
        c_t = self.shared(cv.T, 'c.T')

        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)
        bt_dev = b_t.get_value(borrow=False, return_internal_type=True)
        ct_dev = c_t.get_value(borrow=False, return_internal_type=True)

        f_nn = theano.function([], [], updates={a: tensor.dot(b, c)},
                mode=self.mode)
        print 'class name:', self.__class__.__name__
        theano.printing.debugprint(f_nn)
        f_nt = theano.function([], [], updates={a: tensor.dot(b, c_t.T)},
                mode=self.mode)
        f_tn = theano.function([], [], updates={a: tensor.dot(b_t.T, c)},
                mode=self.mode)
        f_tt = theano.function([], [], updates={a: tensor.dot(b_t.T, c_t.T)},
                mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in itertools_product((-1, 1), repeat=4):
            for step in (1, 2):
                b_step1, b_step2, c_step1, c_step2 = (s * step
                        for s in step_signs)

                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                c.set_value(c_dev.copy()[::c_step1, ::c_step2], borrow=True)
                b_t.set_value(bt_dev.copy()[::b_step2, ::b_step1], borrow=True)
                c_t.set_value(ct_dev.copy()[::c_step2, ::c_step1], borrow=True)

                # Numpy result
                a_n = numpy.dot(bv[::b_step1, ::b_step2],
                                cv[::c_step1, ::c_step2])

                f_nn()
                assert numpy.allclose(a.get_value(), a_n)

                f_nt()
                assert numpy.allclose(a.get_value(), a_n)

                f_tn()
                assert numpy.allclose(a.get_value(), a_n)

                f_tt()
                assert numpy.allclose(a.get_value(), a_n)

    def test_dot22(self):
        self.cmp_dot22((3, 4), (4, 5))
        self.cmp_dot22((1, 4), (4, 5))
        self.cmp_dot22((3, 4), (4, 1))
        self.cmp_dot22((3, 1), (1, 1))
        self.cmp_dot22((1, 4), (4, 1))
        self.cmp_dot22((3, 1), (1, 5))
        self.cmp_dot22((0, 4), (4, 5))
        self.cmp_dot22((0, 4), (4, 1))
        self.cmp_dot22((0, 1), (1, 5))
        self.cmp_dot22((3, 4), (4, 0))
        self.cmp_dot22((3, 0), (0, 5))
        self.cmp_dot22((0, 4), (4, 0))
        self.cmp_dot22((0, 0), (0, 0))

    def cmp_dot22scalar(self, b_shp, c_shp):
        av = numpy.zeros((0, 0), dtype=self.dtype)
        bv = self.rand(*b_shp)
        cv = self.rand(*c_shp)
        l = numpy.float32(0.2)

        a = self.shared(av, 'a')
        b = self.shared(bv, 'b')
        c = self.shared(cv, 'c')

        b_t = self.shared(bv.T, 'b.T')
        c_t = self.shared(cv.T, 'c.T')

        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)
        bt_dev = b_t.get_value(borrow=False, return_internal_type=True)
        ct_dev = c_t.get_value(borrow=False, return_internal_type=True)

        f_nn = theano.function([], [], updates={a: l * tensor.dot(b, c)},
                mode=self.mode)
        f_nt = theano.function([], [], updates={a: l * tensor.dot(b, c_t.T)},
                mode=self.mode)
        f_tn = theano.function([], [], updates={a: l * tensor.dot(b_t.T, c)},
                mode=self.mode)
        f_tt = theano.function([], [],
                updates={a: l * tensor.dot(b_t.T, c_t.T)},
                mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in itertools_product((-1, 1), repeat=4):
            for step in (1, 2):
                b_step1, b_step2, c_step1, c_step2 = (s * step
                        for s in step_signs)

                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                c.set_value(c_dev.copy()[::c_step1, ::c_step2], borrow=True)
                b_t.set_value(bt_dev.copy()[::b_step2, ::b_step1], borrow=True)
                c_t.set_value(ct_dev.copy()[::c_step2, ::c_step1], borrow=True)

                # Numpy result
                a_n = l * numpy.dot(bv[::b_step1, ::b_step2],
                                    cv[::c_step1, ::c_step2])

                f_nn()
                assert numpy.allclose(a.get_value(), a_n)

                f_nt()
                assert numpy.allclose(a.get_value(), a_n)

                f_tn()
                assert numpy.allclose(a.get_value(), a_n)

                f_tt()
                assert numpy.allclose(a.get_value(), a_n)

    def test_dot22scalar(self):
        self.cmp_dot22scalar((3, 4), (4, 5))
        self.cmp_dot22scalar((1, 4), (4, 5))
        self.cmp_dot22scalar((3, 4), (4, 1))
        self.cmp_dot22scalar((3, 1), (1, 1))
        self.cmp_dot22scalar((1, 4), (4, 1))
        self.cmp_dot22scalar((3, 1), (1, 5))
        self.cmp_dot22scalar((0, 4), (4, 5))
        self.cmp_dot22scalar((0, 4), (4, 1))
        self.cmp_dot22scalar((0, 1), (1, 5))
        self.cmp_dot22scalar((3, 4), (4, 0))
        self.cmp_dot22scalar((3, 0), (0, 5))
        self.cmp_dot22scalar((0, 4), (4, 0))
        self.cmp_dot22scalar((0, 0), (0, 0))

    def cmp_gemm(self, a_shp, b_shp, c_shp):
        av = self.rand(*a_shp)
        bv = self.rand(*b_shp)
        cv = self.rand(*c_shp)
        l = numpy.float32(0.2)

        a = self.shared(av, 'a')
        b = self.shared(bv, 'b')
        c = self.shared(cv, 'c')

        a_t = self.shared(av.T, 'a.T')
        b_t = self.shared(bv.T, 'b.T')
        c_t = self.shared(cv.T, 'c.T')

        a_dev = a.get_value(borrow=False, return_internal_type=True)
        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)
        bt_dev = b_t.get_value(borrow=False, return_internal_type=True)
        ct_dev = c_t.get_value(borrow=False, return_internal_type=True)

        f_nnn = theano.function([], [],
                updates={a: (l * a + tensor.dot(b, c))},
                mode=self.mode)
        f_nnt = theano.function([], [],
                updates={a: (l * a + tensor.dot(b, c_t.T))},
                mode=self.mode)
        f_ntn = theano.function([], [],
                updates={a: (l * a + tensor.dot(b_t.T, c))},
                mode=self.mode)
        f_ntt = theano.function([], [],
                updates={a: (l * a + tensor.dot(b_t.T, c_t.T))},
                mode=self.mode)
        f_tnn = theano.function([], [],
                updates={a_t: (l * a_t + tensor.dot(b, c).T)},
                mode=self.mode)
        f_tnt = theano.function([], [],
                updates={a_t: (l * a_t + tensor.dot(b, c_t.T).T)},
                mode=self.mode)
        f_ttn = theano.function([], [],
                updates={a_t: (l * a_t + tensor.dot(b_t.T, c).T)},
                mode=self.mode)
        f_ttt = theano.function([], [],
                updates={a_t: (l * a_t + tensor.dot(b_t.T, c_t.T).T)},
                mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in itertools_product((-1, 1), repeat=6):
            for step in (1, 2):
                a_step1, a_step2, b_step1, b_step2, c_step1, c_step2 = \
                        (s * step for s in step_signs)

                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                c.set_value(c_dev.copy()[::c_step1, ::c_step2], borrow=True)
                b_t.set_value(bt_dev.copy()[::b_step2, ::b_step1], borrow=True)
                c_t.set_value(ct_dev.copy()[::c_step2, ::c_step1], borrow=True)

                # Numpy results
                a_n = (l * av[::a_step1, ::a_step2]
                       + numpy.dot(bv[::b_step1, ::b_step2],
                                   cv[::c_step1, ::c_step2]))
                at_n = (l * av[::a_step1, ::a_step2].T
                        + numpy.dot(bv[::b_step1, ::b_step2],
                                    cv[::c_step1, ::c_step2]).T)

                # a's value is updated, so we need to reinitialize it each time
                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_nnn()
                assert numpy.allclose(a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_nnt()
                assert numpy.allclose(a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_ntn()
                assert numpy.allclose(a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_ntt()
                assert numpy.allclose(a.get_value(), a_n)

                a_t.set_value(transpose(a_dev.copy())[::a_step2, ::a_step1],
                        borrow=True)
                f_tnn()
                assert numpy.allclose(a_t.get_value(), at_n)

                a_t.set_value(transpose(a_dev.copy())[::a_step2, ::a_step1],
                        borrow=True)
                f_tnt()
                assert numpy.allclose(a_t.get_value(), at_n)

                a_t.set_value(transpose(a_dev.copy())[::a_step2, ::a_step1],
                        borrow=True)
                f_ttn()
                assert numpy.allclose(a_t.get_value(), at_n)

                a_t.set_value(transpose(a_dev.copy())[::a_step2, ::a_step1],
                        borrow=True)
                f_ttt()
                assert numpy.allclose(a_t.get_value(), at_n)

    def test_gemm(self):
        self.cmp_gemm((3, 5), (3, 4), (4, 5))
        self.cmp_gemm((1, 5), (1, 4), (4, 5))
        self.cmp_gemm((3, 1), (3, 4), (4, 1))
        self.cmp_gemm((3, 1), (3, 1), (1, 1))
        self.cmp_gemm((1, 1), (1, 4), (4, 1))
        self.cmp_gemm((3, 5), (3, 1), (1, 5))
        self.cmp_gemm((0, 5), (0, 4), (4, 5))
        self.cmp_gemm((0, 1), (0, 4), (4, 1))
        self.cmp_gemm((0, 5), (0, 1), (1, 5))
        self.cmp_gemm((3, 0), (3, 4), (4, 0))
        self.cmp_gemm((3, 5), (3, 0), (0, 5))
        self.cmp_gemm((0, 0), (0, 4), (4, 0))
        self.cmp_gemm((0, 0), (0, 0), (0, 0))

    def cmp_gemv(self, a_shp, b_shp, c_shp):
        av = self.rand(a_shp)
        bv = self.rand(*b_shp)
        cv = self.rand(c_shp)
        l = numpy.float32(0.2)

        a = self.shared(av, 'a')
        b = self.shared(bv, 'b')
        c = self.shared(cv, 'c')
        b_t = self.shared(bv.T, 'b.T')

        a_dev = a.get_value(borrow=False, return_internal_type=True)
        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)

        f_n = theano.function([], [], updates={a: (a + l * tensor.dot(b, c))},
                mode=self.mode)

        f_t = theano.function([], [],
                updates={a: (a + l * tensor.dot(b_t.T, c))},
                mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in itertools_product((1, -1), repeat=4):
            for step in (1, 2):
                a_step, b_step1, b_step2, c_step = (s * step
                        for s in step_signs)

                a.set_value(a_dev.copy()[::a_step], borrow=True)
                b.set_value(b_dev.copy()[::b_step1, ::b_step2],
                        borrow=True)
                b_t.set_value(transpose(b_dev.copy())[::b_step2, ::b_step1],
                        borrow=True)
                c.set_value(c_dev.copy()[::c_step], borrow=True)

                a_n = (av[::a_step]
                        + l * numpy.dot(bv[::b_step1, ::b_step2], cv[::c_step]))
                f_n()
                assert numpy.allclose(a.get_value(), a_n), (a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step], borrow=True)
                f_t()
                assert numpy.allclose(a.get_value(), a_n), (a.get_value(), a_n)

    def test_gemv(self):
        self.cmp_gemv(3, (3, 5), 5)
        self.cmp_gemv(1, (1, 5), 5)
        self.cmp_gemv(3, (3, 1), 1)
        self.cmp_gemv(0, (0, 5), 5)
        self.cmp_gemv(3, (3, 0), 0)
        self.cmp_gemv(0, (0, 1), 1)
        self.cmp_gemv(1, (1, 0), 0)
        self.cmp_gemv(0, (0, 0), 0)


    def cmp_ger(self, a_shp, b_shp, c_shp):
        av = self.rand(*a_shp)
        bv = self.rand(b_shp)
        cv = self.rand(c_shp)
        l = numpy.float32(0.2)

        a = self.shared(av, 'a')
        b = self.shared(bv, 'b')
        c = self.shared(cv, 'c')
        a_t = self.shared(av.T, 'a.T')

        a_dev = a.get_value(borrow=False, return_internal_type=True)
        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)

        f_n = theano.function([], [],
                updates={a: (a + l * tensor.outer(b, c))},
                mode=self.mode)

        f_t = theano.function([], [],
                updates={a_t: (a_t + l * tensor.outer(b, c).T)},
                mode=self.mode)

        # Try with all stride patterns, and all transposed patterns
        for step_signs in itertools_product((1, -1), repeat=4):
            for step in (1, 2):
                a_step1, a_step2, b_step, c_step = (s * step
                        for s in step_signs)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                a_t.set_value(transpose(a_dev.copy())[::a_step1, ::a_step2],
                        borrow=True)
                b.set_value(b_dev.copy()[::b_step], borrow=True)
                c.set_value(c_dev.copy()[::c_step], borrow=True)

                f_n()
                n_n = (av[::a_step1, ::a_step2]
                        + l * numpy.outer(bv[::b_step], cv[::c_step]))
                assert numpy.allclose(a.get_value(), n_n), (a.get_value(), n_n)

                f_t()
                n_t = (av.T[::a_step1, ::a_step2]
                        + l * numpy.outer(bv[::b_step], cv[::c_step]).T)
                assert numpy.allclose(a_t.get_value(), n_t),\
                        (a_t.get_value(), n_t)

    def test_ger_strides(self):
        self.cmp_ger((3, 5), 3, 5)
        self.cmp_ger((1, 5), 1, 5)
        self.cmp_ger((3, 1), 3, 1)
        self.cmp_ger((0, 5), 0, 5)
        self.cmp_ger((3, 0), 3, 0)
        self.cmp_ger((0, 1), 0, 1)
        self.cmp_ger((1, 0), 1, 0)
        self.cmp_ger((0, 0), 0, 0)
