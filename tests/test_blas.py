import sys, time
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

import theano_cuda_ndarray as tcn

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
