from __future__ import absolute_import, print_function, division
from unittest import TestCase
from nose.plugins.skip import SkipTest
import itertools
import numpy as np

import theano
from theano import config
from theano import tensor
from theano.tests import unittest_tools as utt
from theano.tensor.blas import gemv, gemv_inplace, gemm_inplace, _dot22, batched_dot
from theano.tensor.tests.test_blas import TestGer, BaseGemv

from .. import gpuarray_shared_constructor
from .config import mode_with_gpu, test_ctx_name
from .test_basic_ops import makeTester, rand
from ..blas import (gpugemv_inplace, gpugemv_no_inplace,
                    gpugemm_inplace, gpugemm_no_inplace,
                    gpugemmbatch_inplace,
                    gpuger_inplace, gpuger_no_inplace,
                    GpuGer, GpuGemm, gpu_dot22)


GpuGemvTester = makeTester(
    'GpuGemvTester',
    op=gemv_inplace, gpu_op=gpugemv_inplace,
    # It doesn't support float16
    cases=dict(dot_vv=[rand(1), 1., rand(1, 2), rand(2), 0.],
               dot_vm=[rand(3), 1., rand(3, 2), rand(2), 0.],
               float32=[rand(3).astype('float32'), np.float32(1),
                        rand(3, 2).astype('float32'),
                        rand(2).astype('float32'), np.float32(0)],
               float64=[rand(3).astype('float64'), np.float64(1),
                        rand(3, 2).astype('float64'),
                        rand(2).astype('float64'), np.float64(0)],
               # test_02=[rand(0), 1, rand(0, 2), rand(2), 0],
               # test_30=[rand(3), 1, rand(3, 0), rand(0), 0],
               # test_00=[rand(0), 1, rand(0, 0), rand(0), 0],
               test_stride=[rand(3)[::-1], 1., rand(3, 2)[::-1], rand(2)[::-1], 0.],
               )
    )


def test_float16():
    # gemv (gemm called)
    float16_data = [rand(3).astype('float16'),
                    np.asarray(1, dtype=np.float32),
                    rand(3, 3).astype('float16'),
                    rand(3).astype('float16'),
                    np.asarray(0.5, dtype=np.float32)]
    float16_shared = [gpuarray_shared_constructor(val, target=test_ctx_name)
                      for val in float16_data]
    o = gemv(*float16_shared)
    f = theano.function([], o, mode=mode_with_gpu)
    y, alpha, A, x, beta = float16_data
    out = f()
    utt.assert_allclose(np.asarray(out), alpha * np.dot(A, x) + beta * y)
    topo = f.maker.fgraph.toposort()
    assert any([isinstance(n.op, GpuGemm) for n in topo])

    # gemm
    float16_data = [rand(3, 3).astype('float16'),
                    np.asarray(1, dtype=np.float32),
                    rand(3, 3).astype('float16'),
                    rand(3, 3).astype('float16'),
                    np.asarray(0.5, dtype=np.float32)]
    float16_shared = [gpuarray_shared_constructor(val, target=test_ctx_name)
                      for val in float16_data]
    o = gpugemm_no_inplace(*float16_shared)
    f = theano.function([], o)
    y, alpha, A, x, beta = float16_data
    out = f()
    utt.assert_allclose(np.asarray(out), alpha * np.dot(A, x) + beta * y)

    # dot22
    float16_data = [rand(3, 3).astype('float16'),
                    rand(3, 3).astype('float16')]

    float16_shared = [gpuarray_shared_constructor(val)
                      for val in float16_data]
    o = gpu_dot22(*float16_shared)
    f = theano.function([], o)
    x, y = float16_data
    out = f()
    utt.assert_allclose(np.asarray(out), np.dot(x, y))


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
    # float16 tested in test_float16
    cases=dict(test1=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 0.0],
               test2=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 1.0],
               test3=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), -1.0],
               test4=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.0],
               test5=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.6],
               test6=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), -1.0],
               test7=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 0.0],
               test8=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 1.1],
               float32=[rand(3, 4).astype('float32'), np.float32(-1.0),
                        rand(3, 5).astype('float32'),
                        rand(5, 4).astype('float32'), np.float32(-1.1)],
               float64=[rand(3, 4).astype('float64'), np.float64(-1.0),
                        rand(3, 5).astype('float64'),
                        rand(5, 4).astype('float64'), np.float64(-1.1)],
               # test10=[rand(0, 4), -1.0, rand(0, 5), rand(5, 4), 0.0],
               # test11=[rand(3, 0), -1.0, rand(3, 5), rand(5, 0), 1.1],
               # test12=[rand(3, 4), -1.0, rand(3, 0), rand(0, 4), -1.1],
               # test13=[rand(0, 0), -1.0, rand(0, 0), rand(0, 0), -1.1],
               )
    )


gemm_batched_tests = dict(
    ("test_b%im%ik%in%i" % (b, m, k, n),
     [rand(b, m, n), rand(), rand(b, m, k), rand(b, k, n), rand()])
    for b, m, k, n in itertools.combinations([2, 3, 5, 7, 11, 13], 4))

gemm_batched_tests['float16'] = [rand(3, 4, 7).astype('float16'),
                                 rand().astype('float16'),
                                 rand(3, 4, 4).astype('float16'),
                                 rand(3, 4, 7).astype('float16'),
                                 rand().astype('float16')]
gemm_batched_tests['float32'] = [rand(3, 4, 7).astype('float32'),
                                 rand().astype('float32'),
                                 rand(3, 4, 4).astype('float32'),
                                 rand(3, 4, 7).astype('float32'),
                                 rand().astype('float32')]
gemm_batched_tests['float64'] = [rand(3, 4, 7).astype('float64'),
                                 rand().astype('float64'),
                                 rand(3, 4, 4).astype('float64'),
                                 rand(3, 4, 7).astype('float64'),
                                 rand().astype('float64')]


GpuGemmBatchTester = makeTester(
    'GpuGemmBatchTester',
    op=lambda z, alpha, x, y, beta: alpha * batched_dot(x, y) + beta * z,
    gpu_op=gpugemmbatch_inplace,
    cases=gemm_batched_tests
    )


class TestGpuGemmBatchStrided(TestCase):
    def test0(self):
        # Reported in https://github.com/Theano/Theano/issues/5730
        x = tensor.tensor3()
        y = tensor.tensor3()
        z = tensor.batched_dot(x, y[:, 0, :, np.newaxis])
        f = theano.function([x, y], z, mode=mode_with_gpu)
        x_num = np.arange(32 * 19 * 600, dtype=config.floatX).reshape((32, 19, 600))
        y_num = np.arange(7 * 32 * 600, dtype=config.floatX).reshape((32, 7, 600))
        f(x_num, y_num)
        assert f.maker.fgraph.toposort()[-2].op.inplace


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


def test_gemv_zeros():
    W = tensor.matrix()
    v = tensor.vector()
    f = theano.function([W, v], W.dot(v), mode=mode_with_gpu)

    # Apply to an empty matrix shape (5,0) and an empty vector shape (0,)
    dim = 1000
    A = np.zeros((dim, 0), dtype=theano.config.floatX)
    b = np.zeros((0,), dtype=theano.config.floatX)
    tmp = f(A, b)
    assert np.allclose(tmp,
                       np.zeros((dim,)))


def test_gemv_dot_strides():
    # Reported in https://github.com/Theano/Theano/issues/6142
    xv = rand(5)
    yv = rand(5, 1)
    x = gpuarray_shared_constructor(xv)
    y = gpuarray_shared_constructor(yv, broadcastable=(False, True))
    f = theano.function([], tensor.dot(x, y[::-1]), mode=mode_with_gpu)
    out = f()
    utt.assert_allclose(out, np.dot(xv, yv[::-1]))
