from __future__ import absolute_import, print_function, division
from unittest import TestCase
from nose.plugins.skip import SkipTest
import itertools
import copy

import numpy

import theano
from theano import gradient
from theano import tensor
from theano.tests import unittest_tools as utt
from theano.tensor.blas import gemv_inplace, gemm_inplace, _dot22, batched_dot
from theano.tensor.tests.test_blas import TestGer, BaseGemv
from theano.tensor.signal.pool import Pool, DownsampleFactorMaxGradGrad

from .. import gpuarray_shared_constructor
from .config import mode_with_gpu, mode_without_gpu
from .test_basic_ops import makeTester, rand

from ..blas import (gpugemv_inplace, gpugemv_no_inplace,
                    gpugemm_inplace, gpugemmbatch_no_inplace,
                    gpuger_inplace, gpuger_no_inplace,
                    GpuGer, gpu_dot22, GpuDownsampleFactorMaxGradGrad)


GpuGemvTester = makeTester(
    'GpuGemvTester',
    op=gemv_inplace, gpu_op=gpugemv_inplace,
    cases=dict(dot_vv=[rand(1), 1, rand(1, 2), rand(2), 0],
               dot_vm=[rand(3), 1, rand(3, 2), rand(2), 0],
               # test_02=[rand(0), 1, rand(0, 2), rand(2), 0],
               # test_30=[rand(3), 1, rand(3, 0), rand(0), 0],
               # test_00=[rand(0), 1, rand(0, 0), rand(0), 0],
               test_stride=[rand(3)[::-1], 1, rand(3, 2)[::-1], rand(2)[::-1], 0],
               )
    )


class TestGpuSgemv(TestCase, BaseGemv, utt.TestOptimizationMixin):
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


GpuGemmTester = makeTester(
    'GpuGemmTester',
    op=gemm_inplace, gpu_op=gpugemm_inplace,
    cases=dict(test1=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 0.0],
               test2=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 1.0],
               test3=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), -1.0],
               test4=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.0],
               test5=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.6],
               test6=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), -1.0],
               test7=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 0.0],
               test8=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 1.1],
               test9=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), -1.1],
               # test10=[rand(0, 4), -1.0, rand(0, 5), rand(5, 4), 0.0],
               # test11=[rand(3, 0), -1.0, rand(3, 5), rand(5, 0), 1.1],
               # test12=[rand(3, 4), -1.0, rand(3, 0), rand(0, 4), -1.1],
               # test13=[rand(0, 0), -1.0, rand(0, 0), rand(0, 0), -1.1],
               )
    )


GpuGemmBatchTester = makeTester(
    'GpuGemmBatchTester',
    op=lambda z, alpha, x, y, beta: alpha * batched_dot(x, y) + beta * z,
    gpu_op=gpugemmbatch_no_inplace,
    cases=dict(
        ("test_b%im%ik%in%i" % (b, m, k, n),
         [rand(b, m, n), rand(), rand(b, m, k), rand(b, k, n), rand()])
        for b, m, k, n in itertools.combinations([2, 3, 5, 7, 11, 13], 4)))


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


class TestGpuGer_OpContract(TestCase, utt.T_OpContractMixin):
    def setUp(self):
        self.ops = [gpuger_no_inplace, gpuger_inplace]

    def clone(self, op):
        return GpuGer(inplace=op.inplace)


GpuDot22Tester = makeTester(
    'GpuDot22Tester',
    op=_dot22, gpu_op=gpu_dot22,
    cases=dict(
        test1=[rand(3, 4), rand(4, 5)],
        test2=[rand(1, 4), rand(4, 5)],
        test3=[rand(3, 1), rand(1, 5)],
        test4=[rand(3, 4), rand(4, 1)],
        # test5=[rand(0, 4), rand(4, 5)],
        # test6=[rand(3, 0), rand(0, 5)],
        # test7=[rand(3, 4), rand(4, 0)],
        # test8=[rand(0, 4), rand(4, 0)],
        # test9=[rand(0, 0), rand(0, 0)],
    )
)


def test_max_pool2d_grad_grad():
    shps = [(1, 12),
            (1, 1, 12),
            (1, 1, 1, 12),
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
            (1, 1, 1023, 10), ]

    numpy.random.RandomState(utt.fetch_seed()).shuffle(shps)
    test_ds = (2, 2), (3, 2), (1, 1)
    test_st = (2, 2), (3, 2), (1, 1)

    for shp in shps:
        for ds, st in itertools.product(test_ds, test_st):
            if ds[0] > shp[-2] or ds[1] > shp[-1]:
                continue
            for ignore_border, pad in zip((True, False), [(1, 1), (0, 0)]):
                if pad[0] >= ds[0] or pad[1] >= ds[1]:
                    continue
                # print('test_downsample', shp, ds, st, pad, ignore_border)
                ds_op = Pool(ndim=len(ds), ignore_border=ignore_border)

                a = theano.shared(rand(*shp), 'a')

                ggf = gradient.Lop(tensor.grad((ds_op(
                    tensor.as_tensor_variable(a), ds, st, pad)**2).sum(), a), a, a)

                ref_mode = copy.copy(mode_without_gpu)
                ref_mode.check_py_code = False
                gpu_mode = copy.copy(mode_with_gpu)
                gpu_mode.check_py_code = False
                gg = theano.function([], ggf, mode=gpu_mode)
                gg2 = theano.function([], ggf, mode=ref_mode)

                assert any([
                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
                    for node in gg.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, DownsampleFactorMaxGradGrad)
                    for node in gg2.maker.fgraph.toposort()
                ])
                assert numpy.allclose(gg(), gg2()), (shp, ds, st,
                                                     ignore_border)


def test_max_pool3d_grad_grad():
    shps = [(1, 1, 12),
            (1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1025),
            (1, 1, 2, 2, 2),
            (1, 1, 7, 7, 7),
            (1, 1, 9, 10, 11),
            (1, 6, 18, 18, 18),
            (1, 1, 6, 24, 24),
            (1, 10, 1, 24, 24),
            (1, 10, 6, 24, 24),
            (1, 30, 6, 12, 12),
            (1, 30, 2, 24, 24),
            (1, 30, 6, 24, 24),
            (1, 10, 10, 10, 11),
            (1, 1, 10, 10, 1025),
            (1, 1, 10, 10, 1023),
            (1, 1, 10, 1025, 10),
            (1, 1, 10, 1023, 10), ]

    numpy.random.RandomState(utt.fetch_seed()).shuffle(shps)
    test_ds = (2, 2, 2), (3, 2, 3), (1, 1, 1)
    test_st = (2, 2, 2), (2, 3, 2), (1, 1, 1)

    for shp in shps:
        for ds, st in itertools.product(test_ds, test_st):
            if ds[0] > shp[-3] or ds[1] > shp[-2] or ds[2] > shp[-1]:
                continue
            for ignore_border, pad in zip((True, False), [(1, 1, 1), (0, 0, 0)]):
                if pad[0] >= ds[0] or pad[1] >= ds[1] or pad[2] >= ds[2]:
                    continue
                # print('test_downsample', shp, ds, st, pad, ignore_border)
                ds_op = Pool(ndim=len(ds), ignore_border=ignore_border)

                a = theano.shared(rand(*shp), 'a')

                ggf = gradient.Lop(tensor.grad((ds_op(
                    tensor.as_tensor_variable(a), ds, st, pad)**2).sum(), a), a, a)

                ref_mode = copy.copy(mode_without_gpu)
                ref_mode.check_py_code = False
                gpu_mode = copy.copy(mode_with_gpu)
                gpu_mode.check_py_code = False
                gg = theano.function([], ggf, mode=gpu_mode)
                gg2 = theano.function([], ggf, mode=ref_mode)

                assert any([
                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
                    for node in gg.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, DownsampleFactorMaxGradGrad)
                    for node in gg2.maker.fgraph.toposort()
                ])
                assert numpy.allclose(gg(), gg2()), (shp, ds, st,
                                                     ignore_border)
