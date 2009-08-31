import sys, time
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

import theano_cuda_ndarray as tcn


def test_dot():

    a = tcn.shared_constructor(numpy.random.rand(4,4), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, tensor.dot(a,b))])

    a0 = a.value * 1.0
    print a0
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    bval = numpy.random.rand(4,4)
    f(bval)
    print a.value

    assert numpy.allclose(numpy.dot(a0, bval), a.value)

def test_gemm():

    a = tcn.shared_constructor(numpy.random.rand(4,4), 'a')

    b = tensor.fmatrix('b')
    c = tensor.fmatrix('c')

    f = pfunc([b,c], [], updates=[(a, tensor.dot(a,b) + tensor.exp(c))])

    a0 = a.value * 1.0
    print a0
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    bval = numpy.random.rand(4,4)
    cval = numpy.random.rand(4,4)
    f(bval,cval)
    print a.value

    assert numpy.allclose(numpy.dot(a0, bval)+numpy.exp(cval), a.value)

def test_maxpool():
    """TODO: test the gpu version!!! """
    for d0, d1, r_true, r_false in [(4,4,[[[[5,7],[13,15]]]],[[[[5,7],[13,15]]]]),
                                    (5,5,[[[[6, 8],[ 16, 18], [ 21, 23]]]],
                                     [[[[6, 8, 9],[ 16, 18, 19], [ 21, 23, 24]]]])]:
        for border,ret in [(True,r_true),(False, r_false)]:
            ret=numpy.array(ret)
            a = tcn.blas.DownsampleFactorMax((2,2),border)
            dmatrix4 = tensor.TensorType("float32", (False, False, False, False))
            b = dmatrix4()
            f = pfunc([b], [a(b)])
            
            bval = numpy.arange(0,d0*d1).reshape(1,1,d0,d1)
            r = f(bval)[0]
#            print bval, bval.shape, border
            print r, r.shape
            assert (ret==r).all()
