import unittest
import theano
from theano.tensor import *
from theano.tests import unittest_tools as utt

class TestRealImag(unittest.TestCase):

    def test0(self):
        x= zvector()
        rng = numpy.random.RandomState(23)
        xval = numpy.asarray(list(numpy.complex(rng.randn(), rng.randn()) for i in xrange(10)))
        assert numpy.all( xval.real == theano.function([x], real(x))(xval))
        assert numpy.all( xval.imag == theano.function([x], imag(x))(xval))

    def test_on_real_input(self):
        x= dvector()
        rng = numpy.random.RandomState(23)
        xval = rng.randn(10)
        try:
            numpy.all( 0 == theano.function([x], imag(x))(xval))
            assert 0
        except TypeError:
            pass
        try:
            numpy.all( xval == theano.function([x], real(x))(xval))
            assert 0
        except TypeError:
            pass

    def test_cast(self):
        x= zvector()
        self.failUnlessRaises(TypeError, cast, x, 'int32')

    def test_complex(self):
        rng = numpy.random.RandomState(2333)
        m = fmatrix()
        c = complex(m[0], m[1])
        assert c.type == cvector
        r,i = [real(c), imag(c)]
        assert r.type == fvector
        assert i.type == fvector
        f = theano.function([m], [r,i] )

        mval = numpy.asarray(rng.randn(2,5), dtype='float32')
        rval, ival = f(mval)
        assert numpy.all(rval == mval[0]), (rval,mval[0])
        assert numpy.all(ival == mval[1]), (ival, mval[1])

    def test_complex_grads(self):
        def f(m):
            c = complex(m[0], m[1])
            return .5 * real(c) + .9 * imag(c)

        rng = numpy.random.RandomState(9333)
        mval = numpy.asarray(rng.randn(2,5))
        utt.verify_grad(f, [mval])

    def test_polar_grads(self):
        def f(m):
            c = complex_from_polar(abs(m[0]), m[1])
            return .5 * real(c) + .9 * imag(c)

        rng = numpy.random.RandomState(9333)
        mval = numpy.asarray(rng.randn(2,5))
        utt.verify_grad(f, [mval])

