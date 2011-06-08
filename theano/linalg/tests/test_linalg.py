
import numpy
from theano import tensor, function

from .ops import *

if 0:
    def test_cholesky():
        #TODO: test upper and lower triangular
        #todo: unittest randomseed 
        rng = numpy.random.RandomState(1234)

        r = rng.randn(5,5)

        pd = numpy.dot(r,r.T)

        x = tensor.matrix()
        chol = Cholesky()(x)
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
    #todo: unittest randomseed
    rng = numpy.random.RandomState(12345)

    r = rng.randn(4,4)

    x = tensor.matrix()
    xi = matrix_inverse(x)

    ri = function([x], xi)(r)
    assert ri.shape == r.shape
    assert ri.dtype == r.dtype

    rir = numpy.dot(ri,r)
    rri = numpy.dot(r,ri)

    assert numpy.allclose(numpy.identity(4), rir), rir
    assert numpy.allclose(numpy.identity(4), rri), rri

def test_inverse_grad():

    rng = numpy.random.RandomState(1234)

    r = rng.randn(4,4)
    tensor.verify_grad(matrix_inverse, [r], rng=numpy.random)


def test_det_grad():
    rng = numpy.random.RandomState(1234)

    r = rng.randn(5,5)
    tensor.verify_grad(det, [r], rng=numpy.random)

