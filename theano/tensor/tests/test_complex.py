from __future__ import absolute_import, print_function, division
import unittest
from six.moves import xrange
import theano
import numpy as np
from theano.tensor import *
from theano.tests import unittest_tools as utt

from numpy.testing import dec


class TestRealImag(unittest.TestCase):

    def test0(self):
        x = zvector()
        rng = np.random.RandomState(23)
        xval = np.asarray(list(np.complex(rng.randn(), rng.randn())
                                  for i in xrange(10)))
        assert np.all(xval.real == theano.function([x], real(x))(xval))
        assert np.all(xval.imag == theano.function([x], imag(x))(xval))

    def test_on_real_input(self):
        x = dvector()
        rng = np.random.RandomState(23)
        xval = rng.randn(10)
        np.all(0 == theano.function([x], imag(x))(xval))
        np.all(xval == theano.function([x], real(x))(xval))

        x = imatrix()
        xval = np.asarray(rng.randn(3, 3) * 100, dtype='int32')
        np.all(0 == theano.function([x], imag(x))(xval))
        np.all(xval == theano.function([x], real(x))(xval))

    def test_cast(self):
        x = zvector()
        self.assertRaises(TypeError, cast, x, 'int32')

    def test_complex(self):
        rng = np.random.RandomState(2333)
        m = fmatrix()
        c = complex(m[0], m[1])
        assert c.type == cvector
        r, i = [real(c), imag(c)]
        assert r.type == fvector
        assert i.type == fvector
        f = theano.function([m], [r, i])

        mval = np.asarray(rng.randn(2, 5), dtype='float32')
        rval, ival = f(mval)
        assert np.all(rval == mval[0]), (rval, mval[0])
        assert np.all(ival == mval[1]), (ival, mval[1])

    @dec.skipif(True, "Complex grads not enabled, see #178")
    def test_complex_grads(self):
        def f(m):
            c = complex(m[0], m[1])
            return .5 * real(c) + .9 * imag(c)

        rng = np.random.RandomState(9333)
        mval = np.asarray(rng.randn(2, 5))
        utt.verify_grad(f, [mval])

    @dec.skipif(True, "Complex grads not enabled, see #178")
    def test_mul_mixed0(self):

        def f(a):
            ac = complex(a[0], a[1])
            return abs((ac)**2).sum()

        rng = np.random.RandomState(9333)
        aval = np.asarray(rng.randn(2, 5))
        try:
            utt.verify_grad(f, [aval])
        except utt.verify_grad.E_grad as e:
            print(e.num_grad.gf)
            print(e.analytic_grad)
            raise

    @dec.skipif(True, "Complex grads not enabled, see #178")
    def test_mul_mixed1(self):

        def f(a):
            ac = complex(a[0], a[1])
            return abs(ac).sum()

        rng = np.random.RandomState(9333)
        aval = np.asarray(rng.randn(2, 5))
        try:
            utt.verify_grad(f, [aval])
        except utt.verify_grad.E_grad as e:
            print(e.num_grad.gf)
            print(e.analytic_grad)
            raise

    @dec.skipif(True, "Complex grads not enabled, see #178")
    def test_mul_mixed(self):

        def f(a, b):
            ac = complex(a[0], a[1])
            return abs((ac*b)**2).sum()

        rng = np.random.RandomState(9333)
        aval = np.asarray(rng.randn(2, 5))
        bval = rng.randn(5)
        try:
            utt.verify_grad(f, [aval, bval])
        except utt.verify_grad.E_grad as e:
            print(e.num_grad.gf)
            print(e.analytic_grad)
            raise

    @dec.skipif(True, "Complex grads not enabled, see #178")
    def test_polar_grads(self):
        def f(m):
            c = complex_from_polar(abs(m[0]), m[1])
            return .5 * real(c) + .9 * imag(c)

        rng = np.random.RandomState(9333)
        mval = np.asarray(rng.randn(2, 5))
        utt.verify_grad(f, [mval])

    @dec.skipif(True, "Complex grads not enabled, see #178")
    def test_abs_grad(self):
        def f(m):
            c = complex(m[0], m[1])
            return .5 * abs(c)

        rng = np.random.RandomState(9333)
        mval = np.asarray(rng.randn(2, 5))
        utt.verify_grad(f, [mval])
