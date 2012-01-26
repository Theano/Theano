from unittest import TestCase

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

from theano.tensor.signal.downsample import DownsampleFactorMax, DownsampleFactorMaxGrad

import theano.compile.mode
from theano.tensor.tests.test_blas import BaseGemv, TestGer
from theano.sandbox.cuda.blas import gpu_gemv_no_inplace, gpu_gemv_inplace
from theano.sandbox.cuda.blas import gpu_ger_inplace, gpu_ger_no_inplace


if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape),dtype='float32')

def test_dot22():
    def cmp(a_shp, b_shp):
        a = tcn.shared_constructor(my_rand(*a_shp), 'a')

        b = tensor.fmatrix()

        f = pfunc([b], [], updates=[(a, tensor.dot(a,b))], mode=mode_with_gpu)

        a0 = a.get_value() * 1.0
        bval = my_rand(*b_shp)
        f(bval)

        assert numpy.allclose(numpy.dot(a0, bval), a.get_value())

    cmp((3,4),(4,5))
    cmp((0,4),(4,5))
    cmp((3,4),(4,0))
    cmp((3,0),(0,5))
    cmp((0,4),(4,0))
    cmp((0,0),(0,0))

def test_dot22scalar():
    def cmp(a_shp, b_shp):
        a = tensor.fmatrix()
        b = tensor.fmatrix()
        scalar = tensor.fscalar()
        av = my_rand(*a_shp)
        bv = my_rand(*b_shp)

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

    cmp((3,4),(4,5))
    cmp((0,4),(4,5))
    cmp((3,4),(4,0))
    cmp((3,0),(0,5))
    cmp((0,4),(4,0))
    cmp((0,0),(0,0))

def test_gemm():
    def cmp(a_shp, b_shp):
        a = tcn.shared_constructor(my_rand(*a_shp), 'a')

        b = tensor.fmatrix('b')
        c = tensor.fmatrix('c')

        f = pfunc([b,c], [], updates=[(a, tensor.dot(a,b) + tensor.exp(c))], mode=mode_with_gpu)
        assert any([node.op == tcn.blas.gpu_gemm_inplace for node in f.maker.env.toposort()])

        a0 = a.get_value() * 1.0
        bval = my_rand(*b_shp)
        cval = my_rand(a_shp[0],b_shp[1])
        f(bval,cval)

        assert numpy.allclose(numpy.dot(a0, bval)+numpy.exp(cval), a.get_value())
    cmp((3,4),(4,5))
    cmp((0,4),(4,5))
    cmp((3,4),(4,0))
    cmp((3,0),(0,5))
    cmp((0,4),(4,0))
    cmp((0,0),(0,0))

def test_gemm_no_inplace():

    def cmp(a_shp, b_shp):
        a = tcn.shared_constructor(my_rand(*a_shp), 'a')
        cval = my_rand(a_shp[0], b_shp[1])
        c = tcn.shared_constructor(cval.copy(), 'c')

        b = tcn.fmatrix('b')
        b2 = tcn.fmatrix('b2')

        f = pfunc([b,b2], [tensor.dot(a,b2) + c], updates=[(a, tensor.dot(a,b) + c)], mode=mode_with_gpu)

        a0 = a.get_value() * 1.0
        assert any([node.op == tcn.blas.gpu_gemm_no_inplace for node in f.maker.env.toposort()])
        bval = my_rand(*b_shp)
        bval2 = my_rand(*b_shp)
        rval = f(bval,bval2)

        assert numpy.allclose(numpy.dot(a0, bval)+cval, a.get_value())
        assert numpy.allclose(numpy.dot(a0, bval2)+cval, rval)

    cmp((3,4),(4,5))
    cmp((0,4),(4,5))
    cmp((3,4),(4,0))
    cmp((3,0),(0,5))
    cmp((0,4),(4,0))
    cmp((0,0),(0,0))

def test_outer():
    x = tcn.shared_constructor(my_rand(8,), 'x')
    y = tcn.shared_constructor(my_rand(6,), 'y')

    x_val = x.get_value().copy()
    y_val = y.get_value().copy()

    f = pfunc([], tensor.outer(x, y), mode=mode_with_gpu)
    assert numpy.allclose(numpy.outer(x_val, y_val), f())

    f = pfunc([], tensor.outer(x[::2], y), mode=mode_with_gpu)
    assert numpy.allclose(numpy.outer(x_val[::2], y_val), f())

    f = pfunc([], tensor.outer(x, y[::3]), mode=mode_with_gpu)
    assert numpy.allclose(numpy.outer(x_val, y_val[::3]), f())

    f = pfunc([], tensor.outer(x[::2], y[::3]), mode=mode_with_gpu)
    assert numpy.allclose(numpy.outer(x_val[::2], y_val[::3]), f())

    f = pfunc([], tensor.outer(x[::-1], y), mode=mode_with_gpu)
    assert numpy.allclose(numpy.outer(x_val[::-1], y_val), f())

    f = pfunc([], tensor.outer(x, y[::-1]), mode=mode_with_gpu)
    assert numpy.allclose(numpy.outer(x_val, y_val[::-1]), f())

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
            (10, 10, 10, 11),
            (1,1,10,1025),
            (1,1,10,1023),
            (1,1,1025,10),
            (1,1,1023,10),
             ]

    numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)

    for shp in shps:
        for ds in (2, 2), (3,2), (1,1):
            if ds[0] > shp[2]: continue
            if ds[1] > shp[3]: continue
            #GpuDownsampleFactorMax don't having more then 512 columns in the output tensor
            if float(shp[3])/ds[1]>512: continue
            for ignore_border in (True, False):
                print 'test_downsample', shp, ds, ignore_border
                ds_op = DownsampleFactorMax(ds, ignore_border=ignore_border)

                a = tcn.shared_constructor(my_rand(*shp), 'a')
                f = pfunc([], ds_op(tensor.as_tensor_variable(a)), mode=mode_with_gpu)
                f2 = pfunc([], ds_op(tensor.as_tensor_variable(a)), mode=mode_without_gpu)
                assert any([isinstance(node.op, tcn.blas.GpuDownsampleFactorMax) for node in
                            f.maker.env.toposort()])
                assert any([isinstance(node.op, DownsampleFactorMax) for node in
                            f2.maker.env.toposort()])
                assert numpy.allclose(f(),f2())

                g = pfunc([], tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),a), mode=mode_with_gpu)
                g2 = pfunc([], tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),a), mode=mode_without_gpu)
                assert any([isinstance(node.op, tcn.blas.GpuDownsampleFactorMaxGrad)
                            for node in g.maker.env.toposort()])
                assert any([isinstance(node.op, DownsampleFactorMaxGrad)
                            for node in g2.maker.env.toposort()])
                assert numpy.allclose(g(),g2())

                #We already check that the gpu version return the same value as the gpu version
                #for GpuDownsampleFactorMaxGrad. So no need to call verify_grad here.


class TestGpuGemv(TestCase, BaseGemv,
                  unittest_tools.TestOptimizationMixin):
    mode = mode_with_gpu
    dtype = 'float32'

    # As all input are transfered to the gpu, this allow to make all
    # the gemv inplace.
    gemv = gpu_gemv_inplace
    gemv_inplace = gpu_gemv_inplace


class TestVectorMatrixDot(TestCase):
    ### Tolerance factor used in this tests
    atol = 1e-6
    ##########################

    def test_dot_vm(self):
        ''' Test vector dot matrix '''
        v = theano.shared( numpy.array(numpy.random.rand(2), dtype='float32'))
        m = theano.shared( numpy.array(numpy.random.rand(2,5),
                                       dtype='float32'))
        no_gpu_f = theano.function([], theano.dot(v,m), mode = mode_without_gpu)
        gpu_f    = theano.function([], theano.dot(v,m), mode = mode_with_gpu)
        #gpu_f2 is needed to test the case when the input is not on the gpu
        #but the output is moved to the gpu.
        gpu_f2   = theano.function([], tcn.gpu_from_host(theano.dot(v,m)), mode = mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.env.toposort() ]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.env.toposort() ]) == 1

    def test_dot_mv(self):
        ''' Test matrix dot vector '''
        v = theano.shared( numpy.array(numpy.random.rand(2), dtype='float32'))
        m = theano.shared( numpy.array(numpy.random.rand(5,2),
                                       dtype='float32'))
        no_gpu_f = theano.function([], theano.dot(m,v), mode = mode_without_gpu)
        gpu_f    = theano.function([], theano.dot(m,v), mode = mode_with_gpu)
        #gpu_f2 is needed to test the case when the input is not on the gpu
        #but the output is moved to the gpu.
        gpu_f2   = theano.function([], tcn.gpu_from_host(theano.dot(m,v)), mode = mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.env.toposort() ]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.env.toposort() ]) == 1

    def test_gemv1(self):
        ''' test vector1+dot(matrix,vector2) '''
        v1 = theano.tensor._shared( numpy.array(numpy.random.rand(2)  , dtype='float32'))
        v2 = theano.tensor._shared( numpy.array(numpy.random.rand(5)  , dtype='float32'))
        m  = theano.tensor._shared( numpy.array(numpy.random.rand(5,2), dtype='float32'))

        no_gpu_f = theano.function([], v2+theano.dot(m,v1), mode = mode_without_gpu)
        gpu_f    = theano.function([], v2+theano.dot(m,v1), mode = mode_with_gpu)
        #gpu_f2 is needed to test the case when the input is not on the gpu
        #but the output is moved to the gpu.
        gpu_f2    = theano.function([], tcn.gpu_from_host(v2+theano.dot(m,v1)), mode = mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.env.toposort()]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.env.toposort()]) == 1

    def test_gemv2(self):
        ''' test vector1+dot(vector2,matrix) '''
        v1 = theano.shared( numpy.array(numpy.random.rand(5)  , dtype='float32'))
        v2 = theano.shared( numpy.array(numpy.random.rand(2)  , dtype='float32'))
        m  = theano.shared( numpy.array(numpy.random.rand(5,2), dtype='float32'))

        no_gpu_f = theano.function([], v2+theano.dot(v1,m), mode = mode_without_gpu)
        gpu_f    = theano.function([], v2+theano.dot(v1,m), mode = mode_with_gpu)
        #gpu_f2 is needed to test the case when the input is not on the gpu
        #but the output is moved to the gpu.
        gpu_f2    = theano.function([], tcn.gpu_from_host(v2+theano.dot(v1,m)), mode = mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.env.toposort()]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.env.toposort()]) == 1


class TestGpuGer(TestGer):
    def setUp(self):
        self.mode = mode_with_gpu
        dtype = self.dtype = 'float32'  # optimization isn't dtype-dependent
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor.tensor(dtype=dtype, broadcastable=())
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.ger = gpu_ger_no_inplace
        self.ger_destructive = gpu_ger_inplace
        self.gemm = tcn.blas.gpu_gemm_no_inplace

        # data on the gpu make the op always inplace
        self.ger = gpu_ger_inplace
        self.gemm = tcn.blas.gpu_gemm_inplace


class TestGpuGer_OpContract(TestCase, unittest_tools.T_OpContractMixin):
    def setUp(self):
        self.ops = [gpu_ger_no_inplace, gpu_ger_inplace]

    def clone(self, op):
        return tcn.blas.GpuGer(op.inplace)
