from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
from itertools import product
import os
import unittest
from six import reraise
from six.moves import cPickle
import six.moves.builtins as builtins
import sys

import numpy

import theano
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.signal.pool import (Pool, pool_2d,
                                       MaxPoolGrad, AveragePoolGrad,
                                       max_pool_2d_same_size,
                                       DownsampleFactorMaxGradGrad)

from theano import function

theano.config.dnn.enabled = "False"


class TestDownsampleFactorMax(utt.InferShapeTester):

    @staticmethod
    def numpy_max_pool_2d(input, ds, ignore_border=False, mode='max'):
        '''Helper function, implementing pool_2d in pure numpy'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))
        xi = 0
        yi = 0
        if not ignore_border:
            if input.shape[-2] % ds[0]:
                xi += 1
            if input.shape[-1] % ds[1]:
                yi += 1
        out_shp = list(input.shape[:-2])
        out_shp.append(input.shape[-2] // ds[0] + xi)
        out_shp.append(input.shape[-1] // ds[1] + yi)
        output_val = numpy.zeros(out_shp)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii = i * ds[0]
                for j in range(output_val.shape[-1]):
                    jj = j * ds[1]
                    patch = input[k][ii:ii + ds[0], jj:jj + ds[1]]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride_padding(
            x, ds, ignore_border=True, st=None, padding=(0, 0), mode='max'):
        assert ignore_border
        pad_h = padding[0]
        pad_w = padding[1]
        h = x.shape[-2]
        w = x.shape[-1]
        assert ds[0] > pad_h
        assert ds[1] > pad_w

        def pad_img(x):
            y = numpy.zeros(
                (x.shape[0], x.shape[1],
                 x.shape[2] + pad_h * 2, x.shape[3] + pad_w * 2),
                dtype=x.dtype)
            y[:, :, pad_h:(x.shape[2] + pad_h), pad_w:(x.shape[3] + pad_w)] = x

            return y
        img_rows = h + 2 * pad_h
        img_cols = w + 2 * pad_w
        out_r = (img_rows - ds[0]) // st[0] + 1
        out_c = (img_cols - ds[1]) // st[1] + 1
        out_shp = list(x.shape[:-2])
        out_shp.append(out_r)
        out_shp.append(out_c)
        ds0, ds1 = ds
        st0, st1 = st
        output_val = numpy.zeros(out_shp)
        y = pad_img(x)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average
        inc_pad = mode == 'average_inc_pad'

        for k in numpy.ndindex(*x.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii_st = i * st[0]
                ii_end = builtins.min(ii_st + ds[0], img_rows)
                if not inc_pad:
                    ii_st = builtins.max(ii_st, pad_h)
                    ii_end = builtins.min(ii_end, h + pad_h)
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    jj_end = builtins.min(jj_st + ds[1], img_cols)
                    if not inc_pad:
                        jj_st = builtins.max(jj_st, pad_w)
                        jj_end = builtins.min(jj_end, w + pad_w)
                    patch = y[k][ii_st:ii_end, jj_st:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride(input, ds, ignore_border=False, st=None,
                                 mode='max'):
        '''Helper function, implementing pool_2d in pure numpy
           this function provides st input to indicate the stide size
           for the pooling regions. if not indicated, st == sd.'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))

        if st is None:
            st = ds
        img_rows = input.shape[-2]
        img_cols = input.shape[-1]

        out_r = 0
        out_c = 0
        if img_rows - ds[0] >= 0:
            out_r = (img_rows - ds[0]) // st[0] + 1
        if img_cols - ds[1] >= 0:
            out_c = (img_cols - ds[1]) // st[1] + 1

        if not ignore_border:
            if out_r > 0:
                if img_rows - ((out_r - 1) * st[0] + ds[0]) > 0:
                    rr = img_rows - out_r * st[0]
                    if rr > 0:
                        out_r += 1
            else:
                if img_rows > 0:
                        out_r += 1
            if out_c > 0:
                if img_cols - ((out_c - 1) * st[1] + ds[1]) > 0:
                    cr = img_cols - out_c * st[1]
                    if cr > 0:
                        out_c += 1
            else:
                if img_cols > 0:
                        out_c += 1

        out_shp = list(input.shape[:-2])
        out_shp.append(out_r)
        out_shp.append(out_c)

        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        output_val = numpy.zeros(out_shp)
        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii_st = i * st[0]
                ii_end = builtins.min(ii_st + ds[0], img_rows)
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    jj_end = builtins.min(jj_st + ds[1], img_cols)
                    patch = input[k][ii_st:ii_end, jj_st:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    def test_DownsampleFactorMax(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # generate random images
        maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
        imval = rng.rand(4, 2, 16, 16)
        images = tensor.dtensor4()
        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max',
                                                        'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border

                # Pure Numpy computation
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                f = function([images, ], [output, ])
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                # Pool op
                maxpool_op = Pool(ignore_border=ignore_border,
                                  mode=mode)(images, maxpoolshp)

                output_shape = Pool.out_shape(imval.shape, maxpoolshp,
                                              ignore_border=ignore_border)
                utt.assert_allclose(numpy.asarray(output_shape), numpy_output_val.shape)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3), (16, 16))
        stridesizes = ((1, 1), (3, 3), (5, 7),)
        # generate random images
        imval = rng.rand(4, 10, 16, 16)
        # The same for each mode
        outputshps = (
            (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
            (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
            (4, 10, 14, 14), (4, 10, 5, 5), (4, 10, 3, 2),
            (4, 10, 14, 14), (4, 10, 6, 6), (4, 10, 4, 3),
            (4, 10, 12, 14), (4, 10, 4, 5), (4, 10, 3, 2),
            (4, 10, 12, 14), (4, 10, 5, 6), (4, 10, 4, 3),
            (4, 10, 1, 1), (4, 10, 1, 1), (4, 10, 1, 1),
            (4, 10, 1, 1), (4, 10, 1, 1), (4, 10, 1, 1),)

        images = tensor.dtensor4()
        indx = 0
        for mode, maxpoolshp, ignore_border in product(['max',
                                                        'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad'],
                                                       maxpoolshps,
                                                       [True, False]):
                for stride in stridesizes:
                    outputshp = outputshps[indx % len(outputshps)]
                    indx += 1
                    # Pool op
                    numpy_output_val = \
                        self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                      ignore_border, stride,
                                                      mode)
                    assert numpy_output_val.shape == outputshp, (
                        "outshape is %s, calculated shape is %s"
                        % (outputshp, numpy_output_val.shape))
                    maxpool_op = \
                        Pool(ignore_border=ignore_border, mode=mode)(
                            images, maxpoolshp, stride)
                    f = function([images], maxpool_op)
                    output_val = f(imval)
                    utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStrideExtra(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((5, 3), (5, 3), (5, 3), (5, 5), (3, 2), (7, 7), (9, 9))
        stridesizes = ((3, 2), (7, 5), (10, 6), (1, 1),
                       (2, 3), (10, 10), (1, 1))
        imvsizs = ((16, 16), (16, 16), (16, 16), (8, 5),
                   (8, 5), (8, 5), (8, 5))
        outputshps = ((4, 10, 4, 7), (4, 10, 5, 8), (4, 10, 2, 3),
                      (4, 10, 3, 4), (4, 10, 2, 3), (4, 10, 2, 3),
                      (4, 10, 4, 1), (4, 10, 4, 1), (4, 10, 3, 2),
                      (4, 10, 4, 2), (4, 10, 1, 0), (4, 10, 1, 1),
                      (4, 10, 0, 0), (4, 10, 1, 1))
        images = tensor.dtensor4()
        for indx in numpy.arange(len(maxpoolshps)):
            imvsize = imvsizs[indx]
            imval = rng.rand(4, 10, imvsize[0], imvsize[1])
            stride = stridesizes[indx]
            maxpoolshp = maxpoolshps[indx]
            for ignore_border, mode in product([True, False],
                                               ['max', 'sum',
                                                'average_inc_pad',
                                                'average_exc_pad']):
                indx_out = indx * 2
                if not ignore_border:
                    indx_out += 1
                outputshp = outputshps[indx_out]
                # Pool op
                numpy_output_val = \
                    self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                  ignore_border, stride, mode)
                assert numpy_output_val.shape == outputshp, (
                    "outshape is %s, calculated shape is %s"
                    % (outputshp, numpy_output_val.shape))
                maxpool_op = \
                    Pool(ignore_border=ignore_border, mode=mode)(
                        images, maxpoolshp, stride)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxPaddingStride(self):
        ignore_border = True  # padding does not support ignore_border=False
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolsizes = [(3, 3), (4, 4), (3, 4), (4, 3), (2, 2)]
        stridesizes = [(2, 2), (2, 2), (1, 1), (1, 2), (2, 2)]
        paddingsizes = [(2, 2), (1, 2), (2, 1), (0, 0), (1, 1)]
        imgsizes = [(5, 5), (5, 5), (5, 6), (6, 5), (5, 5)]
        m = 4  # minibatch
        c = 2  # channel size
        images = tensor.dtensor4()
        for indx, mode in product(numpy.arange(len(maxpoolsizes)),
                                  ['max', 'sum', 'average_inc_pad',
                                   'average_exc_pad']):
            imgsize = imgsizes[indx]
            imval = rng.rand(m, c, imgsize[0], imgsize[1]) - 0.5

            stridesize = stridesizes[indx]
            maxpoolsize = maxpoolsizes[indx]
            paddingsize = paddingsizes[indx]
            numpy_output_val = self.numpy_max_pool_2d_stride_padding(
                imval, maxpoolsize, ignore_border,
                stridesize, paddingsize, mode)
            maxpool_op = Pool(ignore_border=ignore_border, mode=mode)(
                images, maxpoolsize, stridesize, paddingsize)
            f = function([images], maxpool_op)
            output_val = f(imval)
            utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxPaddingStride_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        imgsizes = ((10, 10), (10, 5), (5, 5))
        maxpoolsizes = ((5, 3), (3, 5), (3, 3))
        stridesizes = ((3, 2), (2, 3), (3, 3))
        paddingsizes = ((2, 2), (2, 1), (2, 2))
        # average_inc_pad and average_exc_pad do not
        # support grad with padding
        for mode in ['max', 'sum']:
            for i in range(len(imgsizes)):
                imgsize = imgsizes[i]
                imval = rng.rand(1, 1, imgsize[0], imgsize[1]) * 10.0
                maxpoolsize = maxpoolsizes[i]
                stridesize = stridesizes[i]
                paddingsize = paddingsizes[i]

                def mp(input):
                    return Pool(ignore_border=True, mode=mode)(
                        input, maxpoolsize, stridesize, paddingsize)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMax_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 2), (2, 3))
        imval = rng.rand(2, 3, 3, 4) * 10.0
        # more variance means numeric gradient will be more accurate

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max',
                                                        'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
            def mp(input):
                return Pool(ignore_border=ignore_border, mode=mode)(
                    input, maxpoolshp)
            utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMax_grad_st(self):
        """checks the gradient for the case that stride is used"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3))
        stridesizes = ((1, 1), (3, 3), (5, 7))
        imval = rng.rand(1, 2, 16, 16)

        for maxpoolshp, ignore_border, mode, stride in product(maxpoolshps,
                                                               [True, False],
                                                               ['max',
                                                                'sum',
                                                                'average_inc_pad',
                                                                'average_exc_pad'],
                                                               stridesizes):
            def mp(input):
                return Pool(ignore_border=ignore_border, mode=mode)(
                    input, maxpoolshp, stride)
            utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMax_grad_st_extra(self):
        """checks the gradient for the case
        that stride is used for extra examples"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((5, 3), (5, 3), (5, 3), (5, 5), (3, 2), (7, 7), (9, 9))
        stridesizes = ((3, 2), (7, 5), (10, 6), (1, 1),
                       (2, 3), (10, 10), (1, 1))
        imvsizs = ((16, 16), (16, 16), (16, 16), (8, 5),
                   (8, 5), (8, 5), (8, 5))

        for mode in ['max', 'sum', 'average_inc_pad', 'average_exc_pad']:
            for indx in numpy.arange(len(maxpoolshps)):
                imvsize = imvsizs[indx]
                imval = rng.rand(1, 2, imvsize[0], imvsize[1])
                stride = stridesizes[indx]
                maxpoolshp = maxpoolshps[indx]
                for ignore_border in [True, False]:
                    def mp(input):
                        return Pool(ignore_border=ignore_border, mode=mode)(
                            input, maxpoolshp, stride)
                    utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMaxGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 2), (2, 3))
        imval = rng.rand(2, 3, 3, 4) * 10.0
        # more variance means numeric gradient will be more accurate

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                # The shape of the gradient will be the shape of the output
                grad_shape = Pool.out_shape(
                    imval.shape, maxpoolshp, ignore_border=ignore_border)
                grad_val = rng.rand(*grad_shape) * 10.0

                def mp(input, grad):
                    out = Pool(ignore_border=ignore_border)(input, maxpoolshp)
                    grad_op = MaxPoolGrad(ignore_border=ignore_border)
                    return grad_op(input, out, grad, maxpoolshp)

                utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_AveragePoolGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        avgpoolshps = ((1, 1), (3, 2), (2, 3))
        imval = rng.rand(2, 3, 3, 4) * 10.0
        # more variance means numeric gradient will be more accurate

        for avgpoolshp in avgpoolshps:
            for ignore_border in [True, False]:
                for mode in ['sum', 'average_inc_pad', 'average_exc_pad']:
                    # print 'maxpoolshp =', maxpoolshp
                    # print 'ignore_border =', ignore_border
                    # The shape of the gradient will be the shape of the output
                    grad_shape = Pool.out_shape(
                        imval.shape, avgpoolshp, ignore_border=ignore_border)
                    grad_val = rng.rand(*grad_shape) * 10.0

                    def mp(input, grad):
                        grad_op = AveragePoolGrad(ignore_border=ignore_border,
                                                  mode=mode)
                        return grad_op(input, grad, avgpoolshp)

                    utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_DownsampleFactorMaxGrad_grad_st(self):
        """checks the gradient of the gradient for
        the case that stride is used"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3))
        stridesizes = ((1, 1), (3, 3), (5, 7))
        imval = rng.rand(1, 2, 16, 16)

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                for stride in stridesizes:
                    grad_shape = Pool.out_shape(
                        imval.shape, maxpoolshp,
                        ignore_border=ignore_border, st=stride)
                    grad_val = rng.rand(*grad_shape)

                    def mp(input, grad):
                        out = Pool(ignore_border=ignore_border)(
                            input, maxpoolshp, stride)
                        grad_op = MaxPoolGrad(ignore_border=ignore_border)
                        return grad_op(input, out, grad, maxpoolshp, stride)

                    utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_AveragePoolGrad_grad_st(self):
        """checks the gradient of the gradient for
        the case that stride is used"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        avgpoolshps = ((1, 1), (3, 3), (5, 3))
        stridesizes = ((1, 1), (3, 3), (5, 7))
        imval = rng.rand(1, 2, 16, 16)

        for avgpoolshp in avgpoolshps:
            for ignore_border in [True, False]:
                for mode in ['sum', 'average_inc_pad', 'average_exc_pad']:
                    for stride in stridesizes:
                        grad_shape = Pool.out_shape(
                            imval.shape, avgpoolshp,
                            ignore_border=ignore_border, st=stride)
                        grad_val = rng.rand(*grad_shape)

                        def mp(input, grad):
                            grad_op = AveragePoolGrad(
                                ignore_border=ignore_border, mode=mode)
                            return grad_op(input, grad, avgpoolshp, stride)

                        utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_DownsampleFactorMaxGrad_grad_st_extra(self):
        """checks the gradient of the gradient for the case that
        stride is used for extra examples"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((5, 3), (5, 3), (5, 3), (5, 5), (3, 2), (7, 7), (9, 9))
        stridesizes = ((3, 2), (7, 5), (10, 6), (1, 1),
                       (2, 3), (10, 10), (1, 1))
        imvsizs = ((16, 16), (16, 16), (16, 16), (8, 5),
                   (8, 5), (8, 5), (8, 5))

        for indx in numpy.arange(len(maxpoolshps)):
            imvsize = imvsizs[indx]
            imval = rng.rand(1, 2, imvsize[0], imvsize[1])
            stride = stridesizes[indx]
            maxpoolshp = maxpoolshps[indx]
            for ignore_border in [True, False]:
                grad_shape = Pool.out_shape(
                    imval.shape, maxpoolshp,
                    ignore_border=ignore_border, st=stride)
                grad_val = rng.rand(*grad_shape)

                def mp(input, grad):
                    out = Pool(ignore_border=ignore_border)(input, maxpoolshp,
                                                            stride)
                    grad_op = MaxPoolGrad(ignore_border=ignore_border)
                    return grad_op(input, out, grad, maxpoolshp, stride)

                # skip the grad verification when the output is empty
                if numpy.prod(grad_shape) == 0:
                    continue
                utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_AveragePoolGrad_grad_st_extra(self):
        """checks the gradient of the gradient for the case that
        stride is used for extra examples"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        avgpoolshps = ((5, 3), (5, 3), (5, 3), (5, 5), (3, 2), (7, 7), (9, 9))
        stridesizes = ((3, 2), (7, 5), (10, 6), (1, 1),
                       (2, 3), (10, 10), (1, 1))
        imvsizs = ((16, 16), (16, 16), (16, 16), (8, 5),
                   (8, 5), (8, 5), (8, 5))

        for indx in numpy.arange(len(avgpoolshps)):
            imvsize = imvsizs[indx]
            imval = rng.rand(1, 2, imvsize[0], imvsize[1])
            stride = stridesizes[indx]
            avgpoolshp = avgpoolshps[indx]
            for ignore_border in [True, False]:
                for mode in ['sum', 'average_inc_pad', 'average_exc_pad']:
                    grad_shape = Pool.out_shape(
                        imval.shape, avgpoolshp,
                        ignore_border=ignore_border, st=stride)
                    grad_val = rng.rand(*grad_shape)

                    def mp(input, grad):
                        grad_op = AveragePoolGrad(ignore_border=ignore_border,
                                                  mode=mode)
                        return grad_op(input, grad, avgpoolshp, stride)

                    # skip the grad verification when the output is empty
                    if numpy.prod(grad_shape) == 0:
                        continue
                    utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_DownsampleFactorMaxPaddingStride_grad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        imgsizes = ((10, 10), (10, 5), (5, 5))
        maxpoolsizes = ((5, 3), (3, 5), (3, 3))
        stridesizes = ((3, 2), (2, 3), (3, 3))
        paddingsizes = ((2, 2), (2, 1), (2, 2))

        for i in range(len(imgsizes)):
            imgsize = imgsizes[i]
            imval = rng.rand(1, 1, imgsize[0], imgsize[1]) * 10.0
            maxpoolsize = maxpoolsizes[i]
            stridesize = stridesizes[i]
            paddingsize = paddingsizes[i]

            grad_shape = Pool.out_shape(imval.shape,
                                        maxpoolsize, st=stridesize,
                                        ignore_border=True,
                                        padding=paddingsize)
            grad_val = rng.rand(*grad_shape) * 10.0

            def mp(input, grad):
                out = Pool(ignore_border=True)(input, maxpoolsize, stridesize,
                                               paddingsize)
                grad_op = MaxPoolGrad(ignore_border=True)
                return grad_op(input, out, grad, maxpoolsize, stridesize,
                               paddingsize)
            utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_AveragePoolPaddingStride_grad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        imgsizes = ((10, 10), (10, 5), (5, 5))
        avgpoolsizes = ((5, 3), (3, 5), (3, 3))
        stridesizes = ((3, 2), (2, 3), (3, 3))
        paddingsizes = ((2, 2), (2, 1), (2, 2))

        for i in range(len(imgsizes)):
            imgsize = imgsizes[i]
            imval = rng.rand(1, 1, imgsize[0], imgsize[1]) * 10.0
            avgpoolsize = avgpoolsizes[i]
            stridesize = stridesizes[i]
            paddingsize = paddingsizes[i]

            # 'average_exc_pad' with non-zero padding is not implemented
            for mode in ['sum', 'average_inc_pad']:
                grad_shape = Pool.out_shape(imval.shape,
                                            avgpoolsize, st=stridesize,
                                            ignore_border=True, padding=paddingsize)
                grad_val = rng.rand(*grad_shape) * 10.0

                def mp(input, grad):
                    grad_op = AveragePoolGrad(ignore_border=True, mode=mode)
                    return grad_op(input, grad, avgpoolsize, stridesize, paddingsize)
                utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_DownsampleFactorMax_hessian(self):
        # Example provided by Frans Cronje, see
        # https://groups.google.com/d/msg/theano-users/qpqUy_3glhw/JMwIvlN5wX4J
        x_vec = tensor.vector('x')
        z = tensor.dot(x_vec.dimshuffle(0, 'x'),
                       x_vec.dimshuffle('x', 0))
        y = pool_2d(input=z, ds=(2, 2), ignore_border=True)
        C = tensor.exp(tensor.sum(y))

        grad_hess = tensor.hessian(cost=C, wrt=x_vec)
        fn_hess = function(inputs=[x_vec], outputs=grad_hess)

        # The value has been manually computed from the theoretical gradient,
        # and confirmed by the implementation.

        assert numpy.allclose(fn_hess([1, 2]), [[0., 0.], [0., 982.7667]])

    def test_DownsampleFactorMaxGradGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        imgsizes = ((10, 10), (10, 5), (5, 5))
        maxpoolsizes = ((5, 3), (3, 5), (3, 3))
        stridesizes = ((3, 2), (2, 3), (3, 3))
        paddingsizes = ((2, 2), (2, 1), (2, 2))

        for i in range(len(imgsizes)):
            imgsize = imgsizes[i]
            imval1 = rng.rand(1, 1, imgsize[0], imgsize[1]) * 10.0
            imval2 = rng.rand(1, 1, imgsize[0], imgsize[1]) * 10.0
            maxpoolsize = maxpoolsizes[i]
            stridesize = stridesizes[i]
            paddingsize = paddingsizes[i]

            def mp(input1, input2):
                op1 = Pool(ignore_border=True)
                pooled_out = op1(input1, maxpoolsize, stride=stridesize,
                                 pad=paddingsize)
                op2 = DownsampleFactorMaxGradGrad(ignore_border=True)
                out = op2(input1, pooled_out, input2, ws=maxpoolsize,
                          stride=stridesize, pad=paddingsize)
                return out
            utt.verify_grad(mp, [imval1, imval2], rng=rng)

    def test_max_pool_2d_2D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 2))
        imval = rng.rand(4, 5)
        images = tensor.dmatrix()

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max', 'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                output_val = function([images], output)(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                def mp(input):
                    return pool_2d(input, maxpoolshp, ignore_border,
                                   mode=mode)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_2d_2D_same_size(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        test_input_array = numpy.array([[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ]]]).astype(theano.config.floatX)
        test_answer_array = numpy.array([[[
            [0., 0., 0., 0.],
            [0., 6., 0., 8.]
        ]]]).astype(theano.config.floatX)
        input = tensor.tensor4(name='input')
        patch_size = (2, 2)
        op = max_pool_2d_same_size(input, patch_size)
        op_output = function([input], op)(test_input_array)
        utt.assert_allclose(op_output, test_answer_array)

        def mp(input):
            return max_pool_2d_same_size(input, patch_size)
        utt.verify_grad(mp, [test_input_array], rng=rng)

    def test_max_pool_2d_3D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = [(1, 2)]
        imval = rng.rand(2, 3, 4)
        images = tensor.dtensor3()

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max', 'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode)
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                output_val = function([images], output)(imval)
                utt.assert_allclose(output_val, numpy_output_val)

# removed as already tested in test_max_pool_2d_2D
# This make test in debug mode too slow.
#                def mp(input):
#                    return pool_2d(input, maxpoolshp, ignore_border)
#                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_2d_6D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = [(3, 2)]
        imval = rng.rand(2, 1, 1, 1, 3, 4)
        images = tensor.TensorType('float64', [False] * 6)()

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max', 'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                output_val = function([images], output)(imval)
                utt.assert_allclose(output_val, numpy_output_val)

# removed as already tested in test_max_pool_2d_2D
# This make test in debug mode too slow.
#                def mp(input):
#                    return pool_2d(input, maxpoolshp, ignore_border)
#                utt.verify_grad(mp, [imval], rng=rng)

    def test_infer_shape(self):
        image = tensor.dtensor4()
        maxout = tensor.dtensor4()
        gz = tensor.dtensor4()
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 2))

        image_val = rng.rand(4, 6, 7, 9)
        out_shapes = [[[[4, 6, 7, 9], [4, 6, 7, 9]],
                       [[4, 6, 3, 4], [4, 6, 4, 5]],
                       [[4, 6, 2, 3], [4, 6, 3, 3]],
                       [[4, 6, 3, 3], [4, 6, 4, 3]],
                       [[4, 6, 2, 4], [4, 6, 3, 5]]],
                      [[None, None],
                       [[4, 6, 4, 5], None],
                       [[4, 6, 3, 3], None],
                       [[4, 6, 4, 3], None],
                       [[4, 6, 3, 5], None]],
                      [[None, None],
                       [None, None],
                       [[4, 6, 3, 4], None],
                       [[4, 6, 4, 4], None],
                       [None, None]]]

        for i, maxpoolshp in enumerate(maxpoolshps):
            for j, ignore_border in enumerate([True, False]):
                for k, padding in enumerate([(0, 0), (1, 1), (1, 2)]):
                    if out_shapes[k][i][j] is None:
                        continue
                    # checking shapes generated by Pool
                    self._compile_and_check([image],
                                            [Pool(ignore_border=ignore_border)
                                             (image, maxpoolshp, pad=padding)],
                                            [image_val], Pool)

                    # checking shapes generated by MaxPoolGrad
                    maxout_val = rng.rand(*out_shapes[k][i][j])
                    gz_val = rng.rand(*out_shapes[k][i][j])
                    self._compile_and_check([image, maxout, gz],
                                            [MaxPoolGrad(
                                                ignore_border=ignore_border)
                                             (image, maxout, gz, maxpoolshp,
                                              pad=padding)],
                                            [image_val, maxout_val, gz_val],
                                            MaxPoolGrad,
                                            warn=False)
        # checking with broadcastable input
        image = tensor.tensor(dtype='float64',
                              broadcastable=(False, False, True, True))
        image_val = rng.rand(4, 6, 1, 1)
        self._compile_and_check(
            [image],
            [Pool(ignore_border=True)(image, (2, 2), pad=(0, 0))],
            [image_val], Pool)

    def test_pooling_with_tensor_vars(self):
        x = tensor.ftensor4()
        window_size = tensor.ivector()
        stride = tensor.ivector()
        padding = tensor.ivector()
        data = numpy.random.normal(0, 1, (1, 1, 5, 5)).astype('float32')

        # checking variable params vs fixed params
        for ignore_border in [True, False]:
            for mode in ['max', 'sum', 'average_inc_pad', 'average_exc_pad']:
                y = pool_2d(x, window_size, ignore_border, stride, padding,
                            mode)
                dx = theano.gradient.grad(y.sum(), x)
                var_fct = theano.function([x, window_size, stride, padding],
                                          [y, dx])
                for ws in (4, 2, 5):
                    for st in (2, 3):
                        for pad in (0, 1):
                            if (pad > st or st > ws or
                                    (pad != 0 and not ignore_border) or
                                    (mode == 'average_exc_pad' and pad != 0)):
                                continue
                            y = pool_2d(x, (ws, ws), ignore_border, (st, st),
                                        (pad, pad), mode)
                            dx = theano.gradient.grad(y.sum(), x)
                            fix_fct = theano.function([x], [y, dx])
                            var_y, var_dx = var_fct(data, (ws, ws), (st, st),
                                                    (pad, pad))
                            fix_y, fix_dx = fix_fct(data)
                            utt.assert_allclose(var_y, fix_y)
                            utt.assert_allclose(var_dx, fix_dx)

    def test_old_pool_interface(self):
        if sys.version_info[0] != 3:
            # Only tested with python 3 because of pickling issues.
            raise SkipTest('Skip old pool interface with python 2.x')
        # 1. Load the old version
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = 'old_pool_interface.pkl'
        with open(os.path.join(testfile_dir, fname), 'rb') as fp:
            try:
                old_fct = cPickle.load(fp, encoding='latin1')
            except ImportError:
                # Windows sometimes fail with nonsensical errors like:
                #   ImportError: No module named type
                #   ImportError: No module named copy_reg
                # when "type" and "copy_reg" are builtin modules.
                if sys.platform == 'win32':
                    exc_type, exc_value, exc_trace = sys.exc_info()
                    reraise(SkipTest, exc_value, exc_trace)
                raise
        # 2. Create the new version
        x = theano.tensor.ftensor4()
        y = pool_2d(x, (2, 2), mode='max', ignore_border=True)
        z = pool_2d(x, (2, 2), mode='average_exc_pad', ignore_border=True)
        dy_dx = theano.gradient.grad(y.sum(), x)
        dz_dx = theano.gradient.grad(z.sum(), x)
        new_fct = theano.function([x], [y, z, dy_dx, dz_dx])
        # 3. Assert that the answer is the same
        rng = numpy.random.RandomState(utt.fetch_seed())
        image_val = rng.rand(4, 6, 7, 9).astype(numpy.float32)
        old_out = old_fct(image_val)
        new_out = new_fct(image_val)
        for o, n in zip(old_out, new_out):
            utt.assert_allclose(o, n)


if __name__ == '__main__':
    unittest.main()
