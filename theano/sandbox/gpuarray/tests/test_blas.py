from unittest import TestCase
from nose.plugins.skip import SkipTest

import numpy

import theano
from theano import tensor
from theano.tests import unittest_tools
from theano.tensor.blas import (gemv_inplace, gemm_inplace, ger_destructive,
                                _dot22)
from theano.tensor.tests.test_blas import TestGer, BaseGemv

from theano.sandbox.gpuarray import gpuarray_shared_constructor
from theano.sandbox.gpuarray.tests.test_basic_ops import (makeTester, rand,
                                                          mode_with_gpu,
                                                          mode_without_gpu)

from theano.sandbox.gpuarray.blas import (gpugemv_inplace, gpugemv_no_inplace,
                                          gpugemm_inplace, gpugemm_no_inplace,
                                          gpuger_inplace, gpuger_no_inplace,
                                          GpuGer, gpu_dot22, )
from theano.tensor.signal.downsample import (DownsampleFactorMax,
                                             DownsampleFactorMaxGrad)


def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape), dtype='float32')

GpuGemvTester = makeTester('GpuGemvTester',
                           op=gemv_inplace, gpu_op=gpugemv_inplace,
                           cases=dict(
        dot_vv=[rand(1), 1, rand(1, 2), rand(2), 0],
        dot_vm=[rand(3), 1, rand(3, 2), rand(2), 0],
#        test_02=[rand(0), 1, rand(0, 2), rand(2), 0],
#        test_30=[rand(3), 1, rand(3, 0), rand(0), 0],
#        test_00=[rand(0), 1, rand(0, 0), rand(0), 0],
        test_stride=[rand(3)[::-1], 1, rand(3, 2)[::-1], rand(2)[::-1], 0],
        )
)

class TestGpuSgemv(TestCase, BaseGemv, unittest_tools.TestOptimizationMixin):
    mode = mode_with_gpu
    dtype = 'float32'

    gemv = gpugemv_no_inplace
    gemv_inplace = gpugemv_inplace

    @staticmethod
    def shared(val):
        try:
            return gpuarray_shared_constructor(val)
        except TypeError:
            return theano.shared(val)


GpuGemmTester = makeTester('GpuGemmTester',
                           op=gemm_inplace, gpu_op=gpugemm_inplace,
                           cases=dict(
        test1=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 0.0],
        test2=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 1.0],
        test3=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), -1.0],
        test4=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.0],
        test5=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.6],
        test6=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), -1.0],
        test7=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 0.0],
        test8=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 1.1],
        test9=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), -1.1],
 #       test10=[rand(0, 4), -1.0, rand(0, 5), rand(5, 4), 0.0],
 #       test11=[rand(3, 0), -1.0, rand(3, 5), rand(5, 0), 1.1],
 #       test12=[rand(3, 4), -1.0, rand(3, 0), rand(0, 4), -1.1],
 #       test13=[rand(0, 0), -1.0, rand(0, 0), rand(0, 0), -1.1],
        )
)

class TestGpuSger(TestGer):
    def setUp(self):
        self.mode = mode_with_gpu
        dtype = self.dtype = 'float32'  # optimization isn't dtype-dependent
        self.A = tensor.tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor.tensor(dtype=dtype, broadcastable=())
        self.x = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor.tensor(dtype=dtype, broadcastable=(False,))
        self.ger_destructive = gpuger_inplace

        # data on the gpu make the op always inplace
        self.ger = gpuger_inplace
        self.gemm = gpugemm_inplace

    def test_f32_0_0(self):
        raise SkipTest('0-sized objects not supported')
    def test_f32_1_0(self):
        raise SkipTest('0-sized objects not supported')
    def test_f32_0_1(self):
        raise SkipTest('0-sized objects not supported')

class TestGpuSgerNoTransfer(TestGpuSger):
    shared = staticmethod(gpuarray_shared_constructor)

class TestGpuGer_OpContract(TestCase, unittest_tools.T_OpContractMixin):
    def setUp(self):
        self.ops = [gpuger_no_inplace, gpuger_inplace]

    def clone(self, op):
        return GpuGer(destructive=op.destructive)


GpuDot22Tester = makeTester(
    'GpuGemmTester',
    op=_dot22, gpu_op=gpu_dot22,
    cases=dict(
        test1=[rand(3, 4), rand(4, 5)],
        test2=[rand(1, 4), rand(4, 5)],
        test3=[rand(3, 1), rand(1, 5)],
        test4=[rand(3, 4), rand(4, 1)],
#        test5=[rand(0, 4), rand(4, 5)],
#        test6=[rand(3, 0), rand(0, 5)],
#        test7=[rand(3, 4), rand(4, 0)],
#        test8=[rand(0, 4), rand(4, 0)],
#        test9=[rand(0, 0), rand(0, 0)],
    )
)


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
             ] # (28, 2)

    

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
                #print '### test_downsample ###', shp, ds, ignore_border
                ds_op = DownsampleFactorMax(ds, ignore_border=ignore_border)

                a = gpuarray_shared_constructor(my_rand(*shp), 'a')
                f = theano.function([], ds_op(tensor.as_tensor_variable(a)),
                        mode=mode_with_gpu)
                f2 = theano.function([], ds_op(tensor.as_tensor_variable(a)),
                        mode=mode_without_gpu)
                assert any([isinstance(node.op,
                                       theano.sandbox.gpuarray.blas.GpuDownsampleFactorMax)
                    for node in f.maker.fgraph.toposort()])
                assert any([isinstance(node.op, DownsampleFactorMax)
                    for node in f2.maker.fgraph.toposort()])
                assert numpy.allclose(f(), f2())

                # The grad is too slow on GT220 GPU
                # This cause the computer to freeze...
                # Remove this when it gets optimized enough
                # This only bypass the last 2 checks
                # Those tests where passing in all Mode on a GTX470
                if shp[0] > 30000 or shp[1] > 30000:
                    continue

                g = theano.function(
                        [],
                        tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),
                            a),
                        mode=mode_with_gpu)
                g2 = theano.function(
                        [],
                        tensor.grad(ds_op(tensor.as_tensor_variable(a)).sum(),
                            a),
                        mode=mode_without_gpu)
                assert any([isinstance(node.op,
                                       theano.sandbox.gpuarray.blas.GpuDownsampleFactorMaxGrad)
                            for node in g.maker.fgraph.toposort()])
                assert any([isinstance(node.op, DownsampleFactorMaxGrad)
                            for node in g2.maker.fgraph.toposort()])
                assert numpy.allclose(g(), g2()), shp

                # We already check that the gpu version return
                # the same value as the gpu version for
                # GpuDownsampleFactorMaxGrad. So no need to call
                # verify_grad here.






