from __future__ import absolute_import, print_function, division
import copy
import itertools

from theano import gradient
from theano import tensor
from theano.tests import unittest_tools as utt

import numpy
import theano.compile.mode
from theano.tensor.signal.pool import (Pool, MaxPoolGrad, AveragePoolGrad,
                                       DownsampleFactorMaxGradGrad)
from theano.sandbox.cuda.pool import (GpuPool, GpuMaxPoolGrad, GpuAveragePoolGrad,
                                      GpuMaxPoolGradGrad)


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode(
        ).including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode(
        ).excluding('gpu')


def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape), dtype='float32')


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
            (65536, 1, 10, 10),
            (1, 65536, 10, 10), ]

    numpy.random.RandomState(utt.fetch_seed()).shuffle(shps)
    test_ws = (2, 2), (3, 2), (1, 1)
    test_st = (2, 2), (3, 2), (1, 1)
    test_mode = ['max', 'sum', 'average_inc_pad', 'average_exc_pad']

    ref_mode = copy.copy(mode_without_gpu)
    ref_mode.check_py_code = False
    gpu_mode = copy.copy(mode_with_gpu)
    gpu_mode = gpu_mode.excluding('cudnn')
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

                a = theano.shared(my_rand(*shp), 'a')
                a_pooled = ds_op(tensor.as_tensor_variable(a), ws, st, pad)

                f = theano.function([], a_pooled, mode=gpu_mode)
                f2 = theano.function([], a_pooled, mode=ref_mode)

                assert any([isinstance(node.op, GpuPool)
                            for node in f.maker.fgraph.toposort()])
                assert any([isinstance(node.op, Pool)
                            for node in f2.maker.fgraph.toposort()])
                assert numpy.allclose(f(), f2()), (shp, ws, st, pad, mode, ignore_border)

                # The grad is too slow on GT220 GPU
                # This cause the computer to freeze...
                # Remove this when it gets optimized enough
                # This only bypass the last 2 checks
                # Those tests where passing in all Mode on a GTX470
                if shp[0] > 30000 or shp[1] > 30000:
                    continue

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

                assert numpy.allclose(g(), g2()), (shp, ws, st, pad, mode, ignore_border)

                # test grad grad for max pooling
                # for average pooling grad grad is just average pooling grad
                if mode != 'max':
                    continue

                ggf = gradient.Lop(tensor.grad((a_pooled**2).sum(), a), a, a)

                gg = theano.function([], ggf, mode=gpu_mode)
                gg2 = theano.function([], ggf, mode=ref_mode)

                assert any([
                    isinstance(node.op, GpuMaxPoolGrad)
                    for node in gg.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, DownsampleFactorMaxGradGrad)
                    for node in gg2.maker.fgraph.toposort()
                ])
                assert numpy.allclose(gg(), gg2()), (shp, ws, st, pad, mode, ignore_border)


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
            (1, 1, 10, 1023, 10), ]

    numpy.random.RandomState(utt.fetch_seed()).shuffle(shps)
    test_ws = (2, 2, 2), (3, 2, 3), (1, 1, 1)
    test_st = (2, 2, 2), (2, 3, 2), (1, 1, 1)
    test_mode = ['max', 'sum', 'average_inc_pad', 'average_exc_pad']

    ref_mode = copy.copy(mode_without_gpu)
    ref_mode.check_py_code = False
    gpu_mode = copy.copy(mode_with_gpu)
    gpu_mode.check_py_code = False
    gpu_mode = gpu_mode.excluding('cudnn')

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

                a = theano.shared(my_rand(*shp), 'a')
                a_pooled = ds_op(tensor.as_tensor_variable(a), ws, st, pad)

                f = theano.function([], a_pooled, mode=gpu_mode)
                f2 = theano.function([], a_pooled, mode=ref_mode)

                assert any([isinstance(node.op, GpuPool)
                            for node in f.maker.fgraph.toposort()])
                assert any([isinstance(node.op, Pool)
                            for node in f2.maker.fgraph.toposort()])
                assert numpy.allclose(f(), f2()), (shp, ws, st, pad, mode, ignore_border)

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

                assert numpy.allclose(g(), g2()), (shp, ws, st, pad, mode, ignore_border)

                # test grad grad for max pooling
                # for average pooling grad grad is just average pooling grad
                if mode != 'max':
                    continue

                ggf = gradient.Lop(tensor.grad((a_pooled**2).sum(), a), a, a)

                gg = theano.function([], ggf, mode=gpu_mode)
                gg2 = theano.function([], ggf, mode=ref_mode)

                assert any([
                    isinstance(node.op, GpuMaxPoolGradGrad)
                    for node in gg.maker.fgraph.toposort()
                ])
                assert any([
                    isinstance(node.op, DownsampleFactorMaxGradGrad)
                    for node in gg2.maker.fgraph.toposort()
                ])
                assert numpy.allclose(gg(), gg2()), (shp, ws, st, pad, mode, ignore_border)
