from __future__ import absolute_import, print_function, division
import copy
from unittest import TestCase

from theano.compile.pfunc import pfunc
from theano import gradient
from theano import tensor
from theano.tests import unittest_tools

import numpy

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda as tcn

from theano.tensor.signal.pool import (Pool,
        PoolGrad, DownsampleFactorMaxGradGrad)

import theano.compile.mode
from theano.tensor.tests.test_blas import BaseGemv, TestBlasStrides, TestGer
from theano.sandbox.cuda.blas import gpu_gemv_no_inplace, gpu_gemv_inplace
from theano.sandbox.cuda.blas import gpu_ger_inplace, gpu_ger_no_inplace
from theano.sandbox.cuda.blas import batched_dot, GpuBatchedDot

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
            'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

# The CPU tests already compare C/Py, so we only check C/GPU
mode_with_gpu = copy.copy(mode_with_gpu)
mode_without_gpu = copy.copy(mode_without_gpu)
mode_with_gpu.check_py_code = False
mode_without_gpu.check_py_code = False


def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape), dtype='float32')


class TestBatchedDot(unittest_tools.InferShapeTester):
    mode = mode_with_gpu

    def test_batched_dot_correctness(self):
        # test both implementations
        for threshold in [0, 100]:
            batched_dot = GpuBatchedDot(stream_threshold=threshold)

            def cmp(a_shp, b_shp):

                a=numpy.random.randn(*a_shp).astype(numpy.float32)
                b=numpy.random.randn(*b_shp).astype(numpy.float32)

                x=tensor.ftensor3()
                y=tensor.ftensor3()

                f=theano.function([x,y], batched_dot(x,y), mode=mode_with_gpu)

                z0=numpy.asarray(f(a,b))

                ga = cuda_ndarray.CudaNdarray(a)
                gb = cuda_ndarray.CudaNdarray(b)

                z1=numpy.asarray(f(ga,gb))

                z_test = numpy.sum(a[:,:,:,None]*b[:,None,:,:],axis=-2)

                unittest_tools.assert_allclose(z0, z_test)
                unittest_tools.assert_allclose(z1, z_test)

            cmp((5,4,3), (5,3,2))
            cmp((5,3,3), (5,3,3))
            cmp((5,2,6), (5,6,3))

            # Test dimensions of 0
            cmp((0,2,6), (0,6,3))
            cmp((5,0,3), (5,3,2))
            cmp((5,4,0), (5,0,2))
            cmp((5,4,3), (5,3,0))
            cmp((0,0,0), (0,0,0))

            # Test dimensions of 1
            cmp((1,2,6), (1,6,3))
            cmp((5,1,3), (5,3,2))
            cmp((5,4,1), (5,1,2))
            cmp((5,4,3), (5,3,1))

    def test_batched_dot_errors(self):

        def fail(a_shp, b_shp):

            a=numpy.random.randn(*a_shp).astype(numpy.float32)
            b=numpy.random.randn(*b_shp).astype(numpy.float32)

            x=tensor.ftensor3()
            y=tensor.ftensor3()

            f=theano.function([x,y], batched_dot(x,y), mode=mode_with_gpu)

            z = f(a,b)

        # Different batch size
        self.assertRaises(RuntimeError, fail, (5,4,3), (6,3,2))

        # Shape mismatch
        self.assertRaises(RuntimeError, fail, (5,4,3), (5,2,2))

    def test_batched_dot_gradient(self):
        for threshold in [0, 100]:
            unittest_tools.verify_grad(
                GpuBatchedDot(stream_threshold=threshold),
                [numpy.random.randn(5,7,2).astype(numpy.float32),
                 numpy.random.randn(5,2,6).astype(numpy.float32)],
                mode=mode_with_gpu)

    def test_infer_shape(self):
        # only matrix/matrix is supported
        admat = tensor.ftensor3()
        bdmat = tensor.ftensor3()
        admat_val = my_rand(7, 4, 5)
        bdmat_val = my_rand(7, 5, 3)
        self._compile_and_check([admat, bdmat],
                                [GpuBatchedDot()(admat, bdmat)],
                                [admat_val, bdmat_val],
                                GpuBatchedDot)


def test_dot22():
    def cmp(a_shp, b_shp):
        a0 = my_rand(*a_shp)
        a = tcn.shared_constructor(a0, 'a')

        b = tensor.fmatrix()

        f = pfunc([b], [], updates=[(a, tensor.dot(a, b))], mode=mode_with_gpu)

        bval = my_rand(*b_shp)
        f(bval)

        assert numpy.allclose(numpy.dot(a0, bval), a.get_value())

        # Try with a matrix equal to a0, but with strides in both dims
        a.set_value(a0)
        a.set_value(
                a.get_value(borrow=True,
                    return_internal_type=True)[::-1, ::-1],
                borrow=True)
        f(bval)

    cmp((3, 4), (4, 5))
    cmp((0, 4), (4, 5))
    cmp((3, 4), (4, 0))
    cmp((3, 0), (0, 5))
    cmp((0, 4), (4, 0))
    cmp((0, 0), (0, 0))


def test_dot22scalar():
    def cmp(a_shp, b_shp):
        a = tensor.fmatrix()
        b = tensor.fmatrix()
        scalar = tensor.fscalar()
        av = my_rand(*a_shp)
        bv = my_rand(*b_shp)

        f = theano.function(
                [a, b],
                tensor.dot(a, b) * numpy.asarray(4, 'float32'),
                mode=mode_with_gpu)
        f2 = theano.function(
                [a, b],
                tensor.dot(a, b) * numpy.asarray(4, 'float32'))
        t = f.maker.fgraph.toposort()
        assert any([isinstance(n.op, tcn.blas.GpuDot22Scalar) for n in t])
#        assert any([isinstance(n.op, tcn.basic_ops.GpuAllocEmpty)
#                    for n in t])
        assert numpy.allclose(f(av, bv), f2(av, bv))

        f = theano.function([a, b, scalar], tensor.dot(a, b) * scalar,
                            mode=mode_with_gpu)
        f2 = theano.function([a, b, scalar], tensor.dot(a, b) * scalar)
        t = f.maker.fgraph.toposort()
        assert any([isinstance(n.op, tcn.blas.GpuDot22Scalar) for n in t])
#        assert any([isinstance(n.op, tcn.basic_ops.GpuAllocEmpty)
#                    for n in t])
        assert numpy.allclose(f(av, bv, 0.5), f2(av, bv, 0.5))

        f = theano.function([a, b, scalar],
                            tensor.blas._dot22scalar(a, b, scalar),
                            mode=mode_with_gpu)
        f2 = theano.function([a, b, scalar], tensor.dot(a, b) * scalar)
        t = f.maker.fgraph.toposort()
        assert len(t) == 4
        assert isinstance(t[0].op, tcn.GpuFromHost)
        assert isinstance(t[1].op, tcn.GpuFromHost)
        assert isinstance(t[2].op, tcn.blas.GpuDot22Scalar)
        assert isinstance(t[3].op, tcn.HostFromGpu)
        assert numpy.allclose(f(av, bv, 0.5), f2(av, bv, 0.5))
    cmp((3, 4), (4, 5))
    cmp((0, 4), (4, 5))
    cmp((3, 4), (4, 0))
    cmp((3, 0), (0, 5))
    cmp((0, 4), (4, 0))
    cmp((0, 0), (0, 0))


def test_gemm():
    def cmp(a_shp, b_shp):
        a0 = my_rand(*a_shp)
        a = tcn.shared_constructor(a0, 'a')

        b = tensor.fmatrix('b')
        c = tensor.fmatrix('c')

        f = pfunc([b, c], [], updates=[(a, tensor.dot(a, b) + tensor.exp(c))],
                mode=mode_with_gpu)
        assert any([node.op == tcn.blas.gpu_gemm_inplace
            for node in f.maker.fgraph.toposort()])

        bval = my_rand(*b_shp)
        cval = my_rand(a_shp[0], b_shp[1])
        f(bval, cval)

        assert numpy.allclose(numpy.dot(a0, bval) + numpy.exp(cval),
                a.get_value())

        # Try with a matrix equal to a0, but with strides in both dims
        a.set_value(a0)
        a.set_value(
                a.get_value(borrow=True,
                    return_internal_type=True)[::-1, ::-1],
                borrow=True)
        f(bval, cval)

    cmp((3, 4), (4, 5))
    cmp((0, 4), (4, 5))
    cmp((3, 4), (4, 0))
    cmp((3, 0), (0, 5))
    cmp((0, 4), (4, 0))
    cmp((0, 0), (0, 0))


def test_gemm_no_inplace():

    def cmp(a_shp, b_shp):
        a0 = my_rand(*a_shp)
        a = tcn.shared_constructor(a0, 'a')
        cval = my_rand(a_shp[0], b_shp[1])
        c = tcn.shared_constructor(cval.copy(), 'c')

        b = tcn.fmatrix('b')
        b2 = tcn.fmatrix('b2')

        f = pfunc(
                [b, b2],
                [tensor.dot(a, b2) + c],
                updates=[(a, tensor.dot(a, b) + c)],
                mode=mode_with_gpu)

        assert any([node.op == tcn.blas.gpu_gemm_no_inplace
            for node in f.maker.fgraph.toposort()])
        bval = my_rand(*b_shp)
        bval2 = my_rand(*b_shp)
        rval = f(bval, bval2)

        assert numpy.allclose(numpy.dot(a0, bval) + cval, a.get_value())
        assert numpy.allclose(numpy.dot(a0, bval2) + cval, rval)

        # Try with a matrix equal to a0, but with strides in both dims
        a.set_value(a0)
        a.set_value(
                a.get_value(borrow=True,
                    return_internal_type=True)[::-1, ::-1],
                borrow=True)
        f(bval, bval2)

    cmp((3, 4), (4, 5))
    cmp((0, 4), (4, 5))
    cmp((3, 4), (4, 0))
    cmp((3, 0), (0, 5))
    cmp((0, 4), (4, 0))
    cmp((0, 0), (0, 0))


class TestBlasStridesGpu(TestBlasStrides):
    dtype = 'float32'
    shared = staticmethod(tcn.shared_constructor)
    mode = mode_with_gpu


if 0:
    # This is commented out because it doesn't make sense...
    # tcn.blas has no op called Pool
    # tcn.blas has an op called GpuDownsampleFactorMax, but that op requires arguments that are
    # CudaNdarrayType variables... so rethink this test?
    def test_maxpool():
        """TODO: test the gpu version!!! """
        for d0, d1, r_true, r_false in [(4, 4, [[[[5, 7], [13, 15]]]], [[[[5, 7], [13, 15]]]]),
                                        (5, 5, [[[[6, 8], [ 16, 18], [ 21, 23]]]],
                                         [[[[6, 8, 9], [ 16, 18, 19], [ 21, 23, 24]]]])]:
            for border, ret in [(True, r_true), (False, r_false)]:
                ret = numpy.array(ret)
                a = tcn.blas.Pool((2, 2), border)
                dmatrix4 = tensor.TensorType("float32", (False, False, False, False))
                b = dmatrix4()
                f = pfunc([b], [a(b)], mode=mode_with_gpu)

                bval = numpy.arange(0, d0*d1).reshape(1, 1, d0, d1)
                r = f(bval)[0]
    #            print bval, bval.shape, border
                # print r, r.shape
                assert (ret == r).all()


def test_downsample():
    shps = [(1, 1, 1, 12),
            (1, 1, 2, 2),
            (1, 1, 1, 1),
            (1, 1, 4, 4),
            (1, 1, 10, 11),
            (1, 2, 2, 2),
            (3, 5, 4, 4),
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
            (1, 1, 10, 1025),
            (1, 1, 10, 1023),
            (1, 1, 1025, 10),
            (1, 1, 1023, 10),
            (65536, 1, 10, 10),
            (1, 65536, 10, 10),
             ]

    numpy.random.RandomState(unittest_tools.fetch_seed()).shuffle(shps)

    for shp in shps:
        for ds in (2, 2), (3, 2), (1, 1):
            if ds[0] > shp[2]:
                continue
            if ds[1] > shp[3]:
                continue
            # GpuDownsampleFactorMax doesn't like having more than 512 columns
            # in the output tensor.
            if float(shp[3]) / ds[1] > 512:
                continue
            for ignore_border in (True, False):
                # print 'test_downsample', shp, ds, ignore_border
                ds_op = Pool(ds, ignore_border=ignore_border)

                a = tcn.shared_constructor(my_rand(*shp), 'a')
                f = pfunc([], ds_op(tensor.as_tensor_variable(a)),
                        mode=mode_with_gpu.excluding('cudnn'))
                f2 = pfunc([], ds_op(tensor.as_tensor_variable(a)),
                        mode=mode_without_gpu)
                assert any([isinstance(node.op,
                                       tcn.blas.GpuDownsampleFactorMax)
                    for node in f.maker.fgraph.toposort()])
                assert any([isinstance(node.op, Pool)
                    for node in f2.maker.fgraph.toposort()])
                assert numpy.allclose(f(), f2())

                # The grad is too slow on GT220 GPU
                # This cause the computer to freeze...
                # Remove this when it gets optimized enough
                # This only bypass the last 2 checks
                # Those tests where passing in all Mode on a GTX470
                if shp[0] > 30000 or shp[1] > 30000:
                    continue

                g = pfunc(
                        [],
                        tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),
                            a),
                        mode=mode_with_gpu.excluding('cudnn'))
                g2 = pfunc(
                        [],
                        tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),
                            a),
                        mode=mode_without_gpu)
                assert any([isinstance(node.op,
                                       tcn.blas.GpuDownsampleFactorMaxGrad)
                            for node in g.maker.fgraph.toposort()])
                assert any([isinstance(node.op, PoolGrad)
                            for node in g2.maker.fgraph.toposort()])
                assert numpy.allclose(g(), g2()), shp

                ggf = gradient.Lop(tensor.grad((ds_op(
                    tensor.as_tensor_variable(a))**2).sum(), a), a, a)

                ref_mode = copy.copy(mode_without_gpu)
                ref_mode.check_py_code = False
                gpu_mode = copy.copy(mode_with_gpu)
                gpu_mode.check_py_code = False
                gg = pfunc([], ggf, mode=gpu_mode)
                gg2 = pfunc([], ggf, mode=ref_mode)

                assert any([isinstance(node.op,
                                       tcn.blas.GpuDownsampleFactorMaxGradGrad)
                            for node in gg.maker.fgraph.toposort()])
                assert any([isinstance(node.op, DownsampleFactorMaxGradGrad)
                            for node in gg2.maker.fgraph.toposort()])
                assert numpy.allclose(gg(), gg2()), shp

                # We already check that the gpu version return
                # the same value as the gpu version for
                # GpuDownsampleFactorMaxGrad. So no need to call
                # verify_grad here.


class TestGpuGemv(TestCase, BaseGemv,
                  unittest_tools.TestOptimizationMixin):
    mode = mode_with_gpu
    dtype = 'float32'

    gemv = gpu_gemv_no_inplace
    gemv_inplace = gpu_gemv_inplace
    # Mimic shared constructors registry
    @staticmethod
    def shared(val):
        # If we don't put shared on the GPU, we won't be able to test
        # the no inplace version as the added transfer will make them inplace.
        try:
            return tcn.shared_constructor(val)
        except TypeError:
            return theano.shared(val)


class TestGpuGemvNoTransfer(TestCase, BaseGemv,
                  unittest_tools.TestOptimizationMixin):
    mode = mode_with_gpu
    dtype = 'float32'

    # Mimic shared constructors registry
    @staticmethod
    def shared(val):
        try:
            return tcn.shared_constructor(val)
        except TypeError:
            return theano.shared(val)

    # In this test, inputs are not always transfered to GPU
    gemv = gpu_gemv_no_inplace
    gemv_inplace = gpu_gemv_inplace


class TestVectorMatrixDot(TestCase):
    # Tolerance factor used in this tests
    atol = 1e-6
    ##########################

    def test_dot_vm(self):
        ''' Test vector dot matrix '''
        v = theano.shared(numpy.array(numpy.random.rand(2), dtype='float32'))
        m = theano.shared(numpy.array(numpy.random.rand(2, 5),
                                       dtype='float32'))
        no_gpu_f = theano.function([], theano.dot(v, m), mode=mode_without_gpu)
        gpu_f = theano.function([], theano.dot(v, m), mode=mode_with_gpu)
        # gpu_f2 is needed to test the case when the input is not on the gpu
        # but the output is moved to the gpu.
        gpu_f2 = theano.function([], tcn.gpu_from_host(theano.dot(v, m)),
                mode=mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.fgraph.toposort()]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.fgraph.toposort()]) == 1

        # Check double-strided m
        m.set_value(
                m.get_value(borrow=True,
                    return_internal_type=True)[::-1, ::-1],
                borrow=True)
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)

    def test_dot_mv(self):
        ''' Test matrix dot vector '''
        v = theano.shared(numpy.array(numpy.random.rand(2), dtype='float32'))
        m = theano.shared(numpy.array(numpy.random.rand(5, 2),
                                       dtype='float32'))
        no_gpu_f = theano.function([], theano.dot(m, v), mode=mode_without_gpu)
        gpu_f = theano.function([], theano.dot(m, v), mode=mode_with_gpu)
        # gpu_f2 is needed to test the case when the input is not on the gpu
        # but the output is moved to the gpu.
        gpu_f2 = theano.function([], tcn.gpu_from_host(theano.dot(m, v)),
                mode=mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.fgraph.toposort()]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.fgraph.toposort()]) == 1

    def test_gemv1(self):
        ''' test vector1+dot(matrix,vector2) '''
        v1 = theano.tensor._shared(numpy.array(numpy.random.rand(2),
            dtype='float32'))
        v2 = theano.tensor._shared(numpy.array(numpy.random.rand(5),
            dtype='float32'))
        m = theano.tensor._shared(numpy.array(numpy.random.rand(5, 2),
            dtype='float32'))

        no_gpu_f = theano.function([], v2 + theano.dot(m, v1),
                mode=mode_without_gpu)
        gpu_f = theano.function([], v2 + theano.dot(m, v1), mode=mode_with_gpu)
        # gpu_f2 is needed to test the case when the input is not on the gpu
        # but the output is moved to the gpu.
        gpu_f2 = theano.function([], tcn.gpu_from_host(v2 + theano.dot(m, v1)),
                mode=mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.fgraph.toposort()]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.fgraph.toposort()]) == 1

    def test_gemv2(self):
        ''' test vector1+dot(vector2,matrix) '''
        v1 = theano.shared(numpy.array(numpy.random.rand(5), dtype='float32'))
        v2 = tensor._shared(numpy.array(numpy.random.rand(2), dtype='float32'))
        m = theano.shared(numpy.array(numpy.random.rand(5, 2),
            dtype='float32'))

        no_gpu_f = theano.function([], v2 + theano.dot(v1, m),
                mode=mode_without_gpu)
        gpu_f = theano.function([], v2 + theano.dot(v1, m),
                mode=mode_with_gpu)
        # gpu_f2 is needed to test the case when the input is not on the gpu
        # but the output is moved to the gpu.
        gpu_f2 = theano.function([], tcn.gpu_from_host(v2 + theano.dot(v1, m)),
                mode=mode_with_gpu)

        # Assert they produce the same output
        assert numpy.allclose(no_gpu_f(), gpu_f(), atol=self.atol)
        assert numpy.allclose(no_gpu_f(), gpu_f2(), atol=self.atol)
        # Assert that the gpu version actually uses gpu
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f2.maker.fgraph.toposort()]) == 1
        assert sum([node.op is gpu_gemv_inplace for node in
                    gpu_f.maker.fgraph.toposort()]) == 1


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


class TestGpuGerNoTransfer(TestGer):
    @staticmethod
    def shared(val):
        try:
            return tcn.shared_constructor(val)
        except TypeError:
            return theano.shared(val)

    def setUp(self):
        self.mode = mode_with_gpu
        dtype = self.dtype = 'float32'  # optimization isn't dtype-dependent
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor.tensor(dtype=dtype, broadcastable=())
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        # data on the gpu make the op always inplace
        self.ger = gpu_ger_inplace
        self.ger_destructive = gpu_ger_inplace
        self.gemm = tcn.blas.gpu_gemm_inplace


class TestGpuGer_OpContract(TestCase, unittest_tools.T_OpContractMixin):
    def setUp(self):
        self.ops = [gpu_ger_no_inplace, gpu_ger_inplace]

    def clone(self, op):
        return tcn.blas.GpuGer(op.inplace)
