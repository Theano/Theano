from theano.compile.pfunc import pfunc
from theano import tensor
from theano.tests import unittest_tools

import numpy

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda as tcn

from theano.tensor.signal.downsample import DownsampleFactorMax

import theano.compile.mode


if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape),dtype='float32')

def test_dot22():

    a = tcn.shared_constructor(my_rand(4,4), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, tensor.dot(a,b))], mode=mode_with_gpu)

    a0 = a.value * 1.0
    print a0
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    bval = my_rand(4,4)
    f(bval)
    print a.value

    assert numpy.allclose(numpy.dot(a0, bval), a.value)

def test_dot22scalar():
    a = tensor.fmatrix()
    b = tensor.fmatrix()
    scalar = tensor.fscalar()
    av = my_rand(4,4)
    bv = my_rand(4,4)

    f = theano.function([a,b], tensor.dot(a,b)*numpy.asarray(4, 'float32'), mode=mode_with_gpu)
    f2 = theano.function([a,b], tensor.dot(a,b)*numpy.asarray(4, 'float32'))
    t=f.maker.env.toposort()
    assert len(t)==4
    assert isinstance(t[0].op,tcn.GpuFromHost)
    assert isinstance(t[1].op,tcn.GpuFromHost)
    assert isinstance(t[2].op,tcn.blas.GpuDot22Scalar)
    assert isinstance(t[3].op,tcn.HostFromGpu)
    assert numpy.allclose(f(av,bv),f2(av,bv))

    f = theano.function([a,b,scalar], tensor.dot(a,b)*scalar, mode=mode_with_gpu)
    f2 = theano.function([a,b,scalar], tensor.dot(a,b)*scalar)
    t=f.maker.env.toposort()
    assert len(t)==4
    assert isinstance(t[0].op,tcn.GpuFromHost)
    assert isinstance(t[1].op,tcn.GpuFromHost)
    assert isinstance(t[2].op,tcn.blas.GpuDot22Scalar)
    assert isinstance(t[3].op,tcn.HostFromGpu)
    assert numpy.allclose(f(av,bv,0.5),f2(av,bv,0.5))

def test_gemm():

    a = tcn.shared_constructor(my_rand(4,4), 'a')

    b = tensor.fmatrix('b')
    c = tensor.fmatrix('c')

    f = pfunc([b,c], [], updates=[(a, tensor.dot(a,b) + tensor.exp(c))], mode=mode_with_gpu)
    assert any([node.op == tcn.blas.gpu_gemm_inplace for node in f.maker.env.toposort()])

    a0 = a.value * 1.0
    print a0
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    bval = my_rand(4,4)
    cval = my_rand(4,4)
    f(bval,cval)
    print a.value

    assert numpy.allclose(numpy.dot(a0, bval)+numpy.exp(cval), a.value)

def test_gemm_no_inplace():

    a = tcn.shared_constructor(my_rand(4,4), 'a')
    cval = my_rand(4,4)
    c = tcn.shared_constructor(cval.copy(), 'c')

    b = tcn.fmatrix('b')
    b2 = tcn.fmatrix('b2')

    f = pfunc([b,b2], [tensor.dot(a,b2) + c], updates=[(a, tensor.dot(a,b) + c)], mode=mode_with_gpu)

    a0 = a.value * 1.0
    #print a0
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    assert any([node.op == tcn.blas.gpu_gemm_no_inplace for node in f.maker.env.toposort()])
    bval = my_rand(4,4)
    bval2 = my_rand(4,4)
    rval = f(bval,bval2)
    #print a.value

    assert numpy.allclose(numpy.dot(a0, bval)+cval, a.value)
    assert numpy.allclose(numpy.dot(a0, bval2)+cval, rval)

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
                f = pfunc([b], [a(b)], mode=mode_with_gpu)
                
                bval = numpy.arange(0,d0*d1).reshape(1,1,d0,d1)
                r = f(bval)[0]
    #            print bval, bval.shape, border
                print r, r.shape
                assert (ret==r).all()

def test_downsample():
    import random
    shps = [ (1, 1, 1, 12),
            (1, 1, 2, 2), 
            (1, 1, 1, 1),
            (1,1,4,4),
            (1, 1, 10, 11),
            (1, 2, 2, 2),
            (3,5,4,4),
            (25, 1, 7, 7),
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
            (10, 10, 10, 11)]

    numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)

    for shp in shps:
        for ds in (2, 2), (3,2), (1,1):
            if ds[0] > shp[2]: continue
            if ds[1] > shp[3]: continue
            for ignore_border in (True, False):
                print 'test_downsample', shp, ds, ignore_border
                ds_op = DownsampleFactorMax(ds, ignore_border=ignore_border)

                a = tcn.shared_constructor(my_rand(*shp), 'a')
                f = pfunc([], ds_op(tensor.as_tensor_variable(a)), mode=mode_with_gpu)
                f2 = pfunc([], ds_op(tensor.as_tensor_variable(a)), mode=mode_without_gpu)
                assert any([isinstance(node.op, tcn.blas.GpuDownsampleFactorMax) for node in
                            f.maker.env.toposort()])
                assert numpy.allclose(f(),f2())
                
                g = pfunc([], tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),a), mode=mode_with_gpu)
                g2 = pfunc([], tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),a), mode=mode_without_gpu)
                assert any([isinstance(node.op, tcn.blas.GpuDownsampleFactorMaxGrad)
                            for node in g.maker.env.toposort()])
                assert numpy.allclose(g(),g2())

                #We already check that the gpu version return the same value as the gpu version
                #for GpuDownsampleFactorMaxGrad. So no need to call verify_grad here.



