import unittest
import theano
from theano.tensor import *

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
        assert numpy.all( 0 == theano.function([x], imag(x))(xval))
        assert numpy.all( xval == theano.function([x], real(x))(xval))

    def test_cast(self):
        x= zvector()
        self.failUnlessRaises(TypeError, cast, x, 'int32')
