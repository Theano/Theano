import sys, time
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
try:
    import cuda_ndarray
except ImportError:
    raise SkipTest('Optional package cuda_ndarray not available')

import theano.sandbox.cuda as tcn

from theano.sandbox.downsample import DownsampleFactorMax

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

if 0:
    # This is commented out because it doesn't make sense...
    # tcn.blas has no op called DownsampleFactorMax
    # tcn.blas has an op called GpuDownsampleFactorMax, but that op requires arguments that are
    # CudaNdarrayType variables... so rethink this test?
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

def test_downsample():

    for shp in [
            (1, 1, 1, 12),
            (1, 1, 2, 2), 
            #(1, 1, 1, 1), #### Commented out because it makes FP-exception that I don't understand
            (1,1,4,4),
            (1, 1, 10, 11),
            (1, 2, 2, 2),
            (3,5,4,4),
            (1, 1, 12, 12),
            (1, 1, 2, 14),
            (1, 1, 12, 14),
            (1, 1, 14, 14),
            (1, 1, 16, 16),
            (1, 1, 18, 18),
            (1, 1, 24, 24),
            (1, 6, 24, 24),
            (10, 1, 24, 24),
            (10, 6, 24, 24),
            (30, 6, 12, 12),
            (30, 2, 24, 24),
            (30, 6, 24, 24),
            (10, 10, 10, 11)]:
        for ds in (1,1), (2, 2):
            if ds[0] > shp[2]: continue
            if ds[1] > shp[3]: continue
            for ignore_border in (True, False):
                print 'test_downsample', shp, ds, ignore_border
                ds_op = DownsampleFactorMax(ds, ignore_border=ignore_border)

                a = tcn.shared_constructor(numpy.random.rand(*shp), 'a')
                f = pfunc([], ds_op(tensor.as_tensor_variable(a)))
                worked = False
                for i, node in enumerate(f.maker.env.toposort()):
                    print i, node
                    if isinstance(node.op, tcn.blas.GpuDownsampleFactorMax):
                        f()  # let debugmode do the testing
                        worked = True
                assert worked
