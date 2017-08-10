from __future__ import absolute_import, print_function, division
import unittest

import copy
import itertools

import numpy as np
import theano
from theano import config
from theano import gradient
from theano import tensor
from theano.tensor.signal.pool import (Pool, MaxPoolGrad, AveragePoolGrad,
                                       DownsampleFactorMaxGradGrad)
from theano.tests import unittest_tools as utt

from .config import mode_with_gpu, mode_without_gpu
from .test_basic_ops import rand
from ..dnn import GpuDnnPool, GpuDnnPoolGrad
from ..pool import (GpuPool, GpuMaxPoolGrad, GpuAveragePoolGrad,
                    GpuDownsampleFactorMaxGradGrad)


class TestPool(unittest.TestCase):

    def test_pool_py_interface(self):
        shp = (2, 2, 2, 2)
        inp = theano.shared(rand(*shp), 'a')
        inp = tensor.as_tensor_variable(inp)
        with self.assertRaises(ValueError):
            # test when pad >= ws
            ds_op = GpuPool(ignore_border=True, ndim=2)
            ds_op(inp, [2, 2], pad=[3, 3])
        with self.assertRaises(ValueError):
            # test when ignore_border and pad >= 0
            ds_op = GpuPool(ignore_border=False, ndim=2)
            ds_op(inp, [2, 2], pad=[1, 1])

    def test_pool_c_interface(self):
        gpu_mode = mode_with_gpu.excluding("cudnn")
        gpu_mode.check_py_code = False

        shp = (2, 2, 2, 2)
        inp = theano.shared(rand(*shp), 'a')
        inp = tensor.as_tensor_variable(inp)
        with self.assertRaises(ValueError):
            # test when ignore_border and pad >= 0
            ds_op = GpuPool(ignore_border=False, ndim=2)
            pad = tensor.as_tensor_variable([1, 1])
            f = theano.function([], ds_op(inp, [2, 2], pad=pad), mode=gpu_mode)
            f()

    def test_pool_big_ws(self):
        gpu_mode = mode_with_gpu.excluding("cudnn")
        gpu_mode.check_py_code = False

        shp = (2, 2, 2, 2)
        inp = theano.shared(rand(*shp), 'a')
        inp = tensor.as_tensor_variable(inp)
        ds_op = GpuPool(ignore_border=False, mode='average_exc_pad', ndim=2)
        pad = tensor.as_tensor_variable([0, 0])
        f = theano.function([], ds_op(inp, [5, 5], stride=[1, 1], pad=pad),
                            mode=gpu_mode)
        f()

    def test_corner_cases_compatibility(self):
        ref_mode = copy.copy(mode_with_gpu).including("cudnn")
        ref_mode.check_py_code = False

        gpu_mode = copy.copy(mode_with_gpu).excluding("cudnn")
        gpu_mode.check_py_code = False

        test_values = (
            ((1, 1), (1, 1), (0, 0), (1, 2, 16, 16)),
            ((1, 1), (3, 3), (0, 0), (1, 2, 16, 16)),
            ((1, 1), (5, 7), (0, 0), (1, 2, 16, 16)),
            ((3, 3), (1, 1), (0, 0), (1, 2, 16, 16)),
            ((3, 3), (3, 3), (1, 1), (1, 2, 16, 16)),
            ((3, 3), (5, 7), (2, 2), (1, 2, 16, 16)),
            ((5, 3), (1, 1), (4, 2), (1, 2, 16, 16)),
            ((5, 3), (3, 3), (2, 1), (1, 2, 16, 16)),
            ((5, 3), (5, 7), (3, 2), (1, 2, 16, 16)),
            ((5, 1, 2), (1, 1, 1), (4, 0, 1), (16, 3, 16)),
            ((5, 1, 2), (3, 1, 2), (3, 0, 0), (1, 16, 3, 16)),
            ((5, 1, 2), (5, 1, 4), (2, 0, 1), (1, 2, 16, 3, 16)),
            ((5, 3), (3, 2), (4, 2), (1, 2, 16, 16)),
            ((5, 3), (7, 5), (4, 2), (1, 2, 16, 16)),
            ((5, 3), (10, 6), (2, 2), (1, 2, 16, 16)),
            ((5, 5), (1, 1), (4, 4), (1, 2, 8, 5)),
            ((3, 2), (2, 3), (2, 1), (1, 2, 8, 5)),
            ((7, 7), (10, 10), (5, 5), (1, 2, 8, 5)),
            ((9, 9), (1, 1), (3, 4), (1, 2, 8, 5)),
            ((3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 2, 16, 16, 16)),
            ((3, 3, 3), (3, 3, 2), (0, 1, 2), (1, 2, 16, 16, 16)),
            ((3, 3, 3), (5, 7, 4), (2, 1, 0), (1, 2, 16, 16, 16)),)
        for ws, st, pad, shp in test_values:
            for init_fn in [np.zeros, np.ones]:
                ds_op = Pool(ndim=len(ws), mode='max', ignore_border=True)
                a_v = init_fn(shp, dtype=config.floatX)
                a = tensor.as_tensor_variable(theano.shared(a_v, 'a'))
                a_p = ds_op(a, ws, st, pad)
                f_ref = theano.function([], a_p, mode=ref_mode)
                f_gpu = theano.function([], a_p, mode=gpu_mode)
                print(f_ref.maker.fgraph.toposort())
                print(f_gpu.maker.fgraph.toposort())
                assert any([
                    isinstance(node.op, GpuDnnPool)
                    for node in f_ref.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, GpuPool)
                    for node in f_gpu.maker.fgraph.toposort()
                ])
                assert np.allclose(f_ref(), f_gpu())

                a_pg = tensor.grad(a_p.sum(), a)
                g_ref = theano.function([], a_pg, mode=ref_mode)
                g_gpu = theano.function([], a_pg, mode=gpu_mode)
                assert any([
                    isinstance(node.op, GpuDnnPoolGrad)
                    for node in g_ref.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, GpuMaxPoolGrad)
                    for node in g_gpu.maker.fgraph.toposort()
                ])
                assert np.allclose(g_ref(), g_gpu())


def test_pool2d():
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
            (1, 1, 1023, 10),
            (3, 2, 16, 16, 16),
            (3, 2, 6, 6, 6, 5),
            (3, 2, 6, 6, 6, 5, 7), ]

    np.random.RandomState(utt.fetch_seed()).shuffle(shps)
    test_ws = (2, 2), (3, 2), (1, 1)
    test_st = (2, 2), (3, 2), (1, 1)
    test_mode = ['max', 'sum', 'average_inc_pad', 'average_exc_pad']

    ref_mode = copy.copy(mode_without_gpu)
    ref_mode.check_py_code = False
    gpu_mode = mode_with_gpu.excluding("cudnn")
    gpu_mode.check_py_code = False

    for shp in shps:
        for mode, ws, st in itertools.product(test_mode, test_ws, test_st):
            if ws[0] > shp[-2] or ws[1] > shp[-1]:
                continue
            for ignore_border, pad in zip((True, False), [(1, 1), (0, 0)]):
                if pad[0] >= ws[0] or pad[1] >= ws[1]:
                    continue
                if mode == 'average_exc_pad' and (pad[0] > 0 or pad[1] > 0):
                    continue
                # print('test_pool2d', shp, ws, st, pad, mode, ignore_border)
                ds_op = Pool(ndim=len(ws), mode=mode, ignore_border=ignore_border)

                a = theano.shared(rand(*shp), 'a')
                a_pooled = ds_op(tensor.as_tensor_variable(a), ws, st, pad)

                f = theano.function([], a_pooled, mode=gpu_mode)
                f2 = theano.function([], a_pooled, mode=ref_mode)

                assert any([isinstance(node.op, GpuPool)
                            for node in f.maker.fgraph.toposort()])
                assert any([isinstance(node.op, Pool)
                            for node in f2.maker.fgraph.toposort()])
                assert np.allclose(f(), f2()), (shp, ws, st, pad, mode, ignore_border)

                a_pooled_grad = tensor.grad(a_pooled.sum(), a)

                g = theano.function([], a_pooled_grad, mode=gpu_mode)
                g2 = theano.function([], a_pooled_grad, mode=ref_mode)

                if mode == 'max':
                    gop = GpuMaxPoolGrad
                    gop2 = MaxPoolGrad
                else:
                    gop = GpuAveragePoolGrad
                    gop2 = AveragePoolGrad
                assert any([isinstance(node.op, gop)
                            for node in g.maker.fgraph.toposort()])
                assert any([isinstance(node.op, gop2)
                            for node in g2.maker.fgraph.toposort()])

                assert np.allclose(g(), g2()), (shp, ws, st, pad, mode, ignore_border)

                # test rop and grad grad for max pooling
                # for average pooling grad grad is just average pooling grad
                if mode != 'max':
                    continue

                ea = theano.shared(rand(*shp), 'ea')

                gr = theano.function([], tensor.Rop(a_pooled, a, ea), mode=gpu_mode)
                gr2 = theano.function([], tensor.Rop(a_pooled, a, ea), mode=ref_mode)

                assert any([
                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
                    for node in gr.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, DownsampleFactorMaxGradGrad)
                    for node in gr2.maker.fgraph.toposort()
                ])
                assert np.allclose(gr(), gr2()), (shp, ws, st, pad, mode, ignore_border)

                ggf = gradient.Lop(tensor.grad((a_pooled**2).sum(), a), a, a)

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
                assert np.allclose(gg(), gg2()), (shp, ws, st, pad, mode, ignore_border)


def test_pool3d():
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
            (1, 1, 10, 1023, 10),
            (3, 2, 6, 6, 6, 5),
            (3, 2, 6, 6, 6, 5, 7), ]

    np.random.RandomState(utt.fetch_seed()).shuffle(shps)
    test_ws = (2, 2, 2), (3, 2, 3), (1, 1, 1)
    test_st = (2, 2, 2), (2, 3, 2), (1, 1, 1)
    test_mode = ['max', 'sum', 'average_inc_pad', 'average_exc_pad']

    ref_mode = copy.copy(mode_without_gpu)
    ref_mode.check_py_code = False
    gpu_mode = mode_with_gpu.excluding("cudnn")
    gpu_mode.check_py_code = False

    for shp in shps:
        for mode, ws, st in itertools.product(test_mode, test_ws, test_st):
            if ws[0] > shp[-3] or ws[1] > shp[-2] or ws[2] > shp[-1]:
                continue
            for ignore_border, pad in zip((True, False), [(1, 1, 1), (0, 0, 0)]):
                if pad[0] >= ws[0] or pad[1] >= ws[1] or pad[2] >= ws[2]:
                    continue
                if mode == 'average_exc_pad' and (pad[0] > 0 or pad[1] > 0 or pad[2] > 0):
                    continue
                # print('test_pool3d', shp, ws, st, pad, mode, ignore_border)
                ds_op = Pool(ndim=len(ws), mode=mode, ignore_border=ignore_border)

                a = theano.shared(rand(*shp), 'a')
                a_pooled = ds_op(tensor.as_tensor_variable(a), ws, st, pad)

                f = theano.function([], a_pooled, mode=gpu_mode)
                f2 = theano.function([], a_pooled, mode=ref_mode)

                assert any([isinstance(node.op, GpuPool)
                            for node in f.maker.fgraph.toposort()])
                assert any([isinstance(node.op, Pool)
                            for node in f2.maker.fgraph.toposort()])
                assert np.allclose(f(), f2()), (shp, ws, st, pad, mode, ignore_border)

                a_pooled_grad = tensor.grad(a_pooled.sum(), a)

                g = theano.function([], a_pooled_grad, mode=gpu_mode)
                g2 = theano.function([], a_pooled_grad, mode=ref_mode)

                if mode == 'max':
                    gop = GpuMaxPoolGrad
                    gop2 = MaxPoolGrad
                else:
                    gop = GpuAveragePoolGrad
                    gop2 = AveragePoolGrad
                assert any([isinstance(node.op, gop)
                            for node in g.maker.fgraph.toposort()])
                assert any([isinstance(node.op, gop2)
                            for node in g2.maker.fgraph.toposort()])

                assert np.allclose(g(), g2()), (shp, ws, st, pad, mode, ignore_border)

                # test rop and grad grad for max pooling
                # for average pooling grad grad is just average pooling grad
                if mode != 'max':
                    continue

                ea = theano.shared(rand(*shp), 'ea')

                gr = theano.function([], tensor.Rop(a_pooled, a, ea), mode=gpu_mode)
                gr2 = theano.function([], tensor.Rop(a_pooled, a, ea), mode=ref_mode)

                assert any([
                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
                    for node in gr.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, DownsampleFactorMaxGradGrad)
                    for node in gr2.maker.fgraph.toposort()
                ])
                assert np.allclose(gr(), gr2()), (shp, ws, st, pad, mode, ignore_border)

                ggf = gradient.Lop(tensor.grad((a_pooled**2).sum(), a), a, a)

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
                assert np.allclose(gg(), gg2()), (shp, ws, st, pad, mode, ignore_border)
