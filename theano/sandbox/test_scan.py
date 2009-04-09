import numpy
import theano
from theano.tensor import dscalar, dvector, dmatrix
from scan import scan1_lambda

RUN_TESTS = False
def run(TF):
    def deco(f):
        if TF and RUN_TESTS:
            print 'running test', f.__name__
            f()
        return f if RUN_TESTS else None
    return deco

@run(True)
def test_extra_inputs():
    u = dscalar('u')
    c = dscalar('c')
    x = dvector('x')

    y = scan1_lambda(
            lambda x_i, y_prev, c: (x_i + y_prev) * c,
            x, u, c)

    sum_y = theano.tensor.sum(y)

    f = theano.function([x,u, c], y)

    xval = numpy.asarray([1., 1, 1. , 1, 1])
    uval = numpy.asarray(2.)

    yval = f(xval, uval, 2.0)
    assert numpy.all(yval == [2.,    6.,   14.,   30.,   62.,  126.])



    g_x = theano.tensor.grad(sum_y, x)
    g_u = theano.tensor.grad(sum_y, u)

    gf = theano.function([x, u, c], [g_x, g_u])

    gxval, guval = gf(xval, uval, 2.0)

    #print gxval
    #print guval
    assert numpy.all(gxval == [ 62.,  30.,  14.,   6.,   2.])
    assert numpy.all(guval == 63)


@run(True)
def test_verify_scan_grad():
    def scanxx(x, u, c):
        # u = dvector('u')
        # c = dvector('c')
        # x = dmatrix('x')
        y = scan1_lambda(
                lambda x_i, y_prev, c: (x_i + y_prev) * c,
                x, u, c)
        return y

    rng = numpy.random.RandomState(456)

    xval = rng.rand(4, 3)
    uval = rng.rand(3)
    cval = rng.rand(3)

    theano.tensor.verify_grad(scanxx, (xval, uval, cval), rng=rng)

