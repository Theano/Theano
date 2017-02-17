from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
from nose_parameterized import parameterized
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
from theano.tensor.signal.pool import (Pool, pool_2d, pool_3d,
                                       MaxPoolGrad, AveragePoolGrad,
                                       max_pool_2d_same_size,
                                       DownsampleFactorMaxGradGrad)

from theano import function


class TestDownsampleFactorMax(utt.InferShapeTester):

    def test_out_shape(self):
        assert Pool.out_shape((9, 8, 6), (2, 2)) == [9, 4, 3]
        assert Pool.out_shape((8, 6), (2, 2)) == [4, 3]

    @staticmethod
    def numpy_max_pool_2d(input, ws, ignore_border=False, mode='max'):
        '''Helper function, implementing pool_2d in pure numpy'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))
        xi = 0
        yi = 0
        if not ignore_border:
            if input.shape[-2] % ws[0]:
                xi += 1
            if input.shape[-1] % ws[1]:
                yi += 1
        out_shp = list(input.shape[:-2])
        out_shp.append(input.shape[-2] // ws[0] + xi)
        out_shp.append(input.shape[-1] // ws[1] + yi)
        output_val = numpy.zeros(out_shp)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii = i * ws[0]
                for j in range(output_val.shape[-1]):
                    jj = j * ws[1]
                    patch = input[k][ii:ii + ws[0], jj:jj + ws[1]]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_nd(input, ws, ignore_border=False, mode='max'):
        '''Helper function, implementing pool_nd in pure numpy'''
        if len(input.shape) < len(ws):
            raise NotImplementedError('input should have at least %s dim,'
                                      ' shape is %s'
                                      % (str(ws), str(input.shape)))
        nd = len(ws)
        si = [0] * nd
        if not ignore_border:
            for i in range(nd):
                if input.shape[-nd + i] % ws[i]:
                    si[i] += 1
        out_shp = list(input.shape[:-nd])
        for i in range(nd):
            out_shp.append(input.shape[-nd + i] // ws[i] + si[i])
        output_val = numpy.zeros(out_shp)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        for l in numpy.ndindex(*input.shape[:-nd]):
            for r in numpy.ndindex(*output_val.shape[-nd:]):
                patch = input[l][tuple(slice(r[i] * ws[i], (r[i] + 1) * ws[i])
                                       for i in range(nd))]
                output_val[l][r] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride_pad(
            x, ws, ignore_border=True, stride=None, pad=(0, 0), mode='max'):
        assert ignore_border
        pad_h = pad[0]
        pad_w = pad[1]
        h = x.shape[-2]
        w = x.shape[-1]
        assert ws[0] > pad_h
        assert ws[1] > pad_w

        def pad_img(x):
            y = numpy.zeros(
                (x.shape[0], x.shape[1],
                 x.shape[2] + pad_h * 2, x.shape[3] + pad_w * 2),
                dtype=x.dtype)
            y[:, :, pad_h:(x.shape[2] + pad_h), pad_w:(x.shape[3] + pad_w)] = x

            return y
        img_rows = h + 2 * pad_h
        img_cols = w + 2 * pad_w
        out_r = (img_rows - ws[0]) // stride[0] + 1
        out_c = (img_cols - ws[1]) // stride[1] + 1
        out_shp = list(x.shape[:-2])
        out_shp.append(out_r)
        out_shp.append(out_c)
        ws0, ws1 = ws
        stride0, stride1 = stride
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
                ii_stride = i * stride[0]
                ii_end = builtins.min(ii_stride + ws[0], img_rows)
                if not inc_pad:
                    ii_stride = builtins.max(ii_stride, pad_h)
                    ii_end = builtins.min(ii_end, h + pad_h)
                for j in range(output_val.shape[-1]):
                    jj_stride = j * stride[1]
                    jj_end = builtins.min(jj_stride + ws[1], img_cols)
                    if not inc_pad:
                        jj_stride = builtins.max(jj_stride, pad_w)
                        jj_end = builtins.min(jj_end, w + pad_w)
                    patch = y[k][ii_stride:ii_end, jj_stride:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_nd_stride_pad(
            input, ws, ignore_border=True, stride=None, pad=None, mode='max'):
        assert ignore_border
        nd = len(ws)
        if pad is None:
            pad = (0,) * nd
        if stride is None:
            stride = (0,) * nd
        assert len(pad) == len(ws) == len(stride)
        assert all(ws[i] > pad[i] for i in range(nd))

        def pad_img(x):
            # initialize padded input
            y = numpy.zeros(
                x.shape[0:-nd] +
                tuple(x.shape[-nd + i] + pad[i] * 2 for i in range(nd)),
                dtype=x.dtype)
            # place the unpadded input in the center
            block = ((slice(None),) * (len(x.shape) - nd) +
                     tuple(slice(pad[i], x.shape[-nd + i] + pad[i])
                           for i in range(nd)))
            y[block] = x
            return y

        pad_img_shp = list(input.shape[:-nd])
        out_shp = list(input.shape[:-nd])
        for i in range(nd):
            padded_size = input.shape[-nd + i] + 2 * pad[i]
            pad_img_shp.append(padded_size)
            out_shp.append((padded_size - ws[i]) // stride[i] + 1)
        output_val = numpy.zeros(out_shp)
        padded_input = pad_img(input)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average
        inc_pad = mode == 'average_inc_pad'

        for l in numpy.ndindex(*input.shape[:-nd]):
            for r in numpy.ndindex(*output_val.shape[-nd:]):
                region = []
                for i in range(nd):
                    r_stride = r[i] * stride[i]
                    r_end = builtins.min(r_stride + ws[i], pad_img_shp[-nd + i])
                    if not inc_pad:
                        r_stride = builtins.max(r_stride, pad[i])
                        r_end = builtins.min(r_end, input.shape[-nd + i] + pad[i])
                    region.append(slice(r_stride, r_end))
                patch = padded_input[l][region]
                output_val[l][r] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride(input, ws, ignore_border=False, stride=None,
                                 mode='max'):
        '''Helper function, implementing pool_2d in pure numpy
           this function provides stride input to indicate the stide size
           for the pooling regions. if not indicated, stride == ws.'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))

        if stride is None:
            stride = ws
        img_rows = input.shape[-2]
        img_cols = input.shape[-1]

        out_r = 0
        out_c = 0
        if img_rows - ws[0] >= 0:
            out_r = (img_rows - ws[0]) // stride[0] + 1
        if img_cols - ws[1] >= 0:
            out_c = (img_cols - ws[1]) // stride[1] + 1

        if not ignore_border:
            if out_r > 0:
                if img_rows - ((out_r - 1) * stride[0] + ws[0]) > 0:
                    rr = img_rows - out_r * stride[0]
                    if rr > 0:
                        out_r += 1
            else:
                if img_rows > 0:
                        out_r += 1
            if out_c > 0:
                if img_cols - ((out_c - 1) * stride[1] + ws[1]) > 0:
                    cr = img_cols - out_c * stride[1]
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
                ii_stride = i * stride[0]
                ii_end = builtins.min(ii_stride + ws[0], img_rows)
                for j in range(output_val.shape[-1]):
                    jj_stride = j * stride[1]
                    jj_end = builtins.min(jj_stride + ws[1], img_cols)
                    patch = input[k][ii_stride:ii_end, jj_stride:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_nd_stride(input, ws, ignore_border=False, stride=None,
                                 mode='max'):
        '''Helper function, implementing pooling in pure numpy
           this function provides stride input to indicate the stide size
           for the pooling regions. if not indicated, stride == ws.'''
        nd = len(ws)
        if stride is None:
            stride = ws
        assert len(stride) == len(ws)

        out_shp = list(input.shape[:-nd])
        for i in range(nd):
            out = 0
            if input.shape[-nd + i] - ws[i] >= 0:
                out = (input.shape[-nd + i] - ws[i]) // stride[i] + 1
            if not ignore_border:
                if out > 0:
                    if input.shape[-nd + i] - ((out - 1) * stride[i] + ws[i]) > 0:
                        if input.shape[-nd + i] - out * stride[i] > 0:
                            out += 1
                else:
                    if input.shape[-nd + i] > 0:
                        out += 1
            out_shp.append(out)

        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        output_val = numpy.zeros(out_shp)
        for l in numpy.ndindex(*input.shape[:-nd]):
            for r in numpy.ndindex(*output_val.shape[-nd:]):
                region = []
                for i in range(nd):
                    r_stride = r[i] * stride[i]
                    r_end = builtins.min(r_stride + ws[i], input.shape[-nd + i])
                    region.append(slice(r_stride, r_end))
                patch = input[l][region]
                output_val[l][r] = func(patch)
        return output_val

    def test_DownsampleFactorMax(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, input size
        examples = (
            ((2,), (16,)),
            ((2,), (4, 16,)),
            ((2,), (4, 2, 16,)),
            ((1, 1), (4, 2, 16, 16)),
            ((2, 2), (4, 2, 16, 16)),
            ((3, 3), (4, 2, 16, 16)),
            ((3, 2), (4, 2, 16, 16)),
            ((3, 2, 2), (3, 2, 16, 16, 16)),
            ((2, 3, 2), (3, 2, 16, 16, 16)),
            ((2, 2, 3), (3, 2, 16, 16, 16)),
            ((2, 2, 3, 2), (3, 2, 6, 6, 6, 5)),
        )

        for example, ignore_border, mode in product(examples,
                                                    [True, False],
                                                    ['max',
                                                     'sum',
                                                     'average_inc_pad',
                                                     'average_exc_pad']):
            (maxpoolshp, inputsize) = example
            imval = rng.rand(*inputsize)
            images = theano.shared(imval)

            # Pure Numpy computation
            numpy_output_val = self.numpy_max_pool_nd(imval, maxpoolshp,
                                                      ignore_border,
                                                      mode=mode)

            # The pool_2d or pool_3d helper methods
            if len(maxpoolshp) == 2:
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                f = function([], [output, ])
                output_val = f()
                utt.assert_allclose(output_val, numpy_output_val)
            elif len(maxpoolshp) == 3:
                output = pool_3d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                f = function([], [output, ])
                output_val = f()
                utt.assert_allclose(output_val, numpy_output_val)

            # Pool op
            maxpool_op = Pool(ndim=len(maxpoolshp),
                              ignore_border=ignore_border,
                              mode=mode)(images, maxpoolshp)

            output_shape = Pool.out_shape(imval.shape, maxpoolshp,
                                          ndim=len(maxpoolshp),
                                          ignore_border=ignore_border)
            utt.assert_allclose(numpy.asarray(output_shape), numpy_output_val.shape)
            f = function([], maxpool_op)
            output_val = f()
            utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, stride, ignore_border, input, output sizes
        examples = (
            ((1, 1), (1, 1), True, (4, 10, 16, 16), (4, 10, 16, 16)),
            ((1, 1), (3, 3), True, (4, 10, 16, 16), (4, 10, 6, 6)),
            ((1, 1), (5, 7), True, (4, 10, 16, 16), (4, 10, 4, 3)),
            ((1, 1), (1, 1), False, (4, 10, 16, 16), (4, 10, 16, 16)),
            ((1, 1), (3, 3), False, (4, 10, 16, 16), (4, 10, 6, 6)),
            ((1, 1), (5, 7), False, (4, 10, 16, 16), (4, 10, 4, 3)),
            ((3, 3), (1, 1), True, (4, 10, 16, 16), (4, 10, 14, 14)),
            ((3, 3), (3, 3), True, (4, 10, 16, 16), (4, 10, 5, 5)),
            ((3, 3), (5, 7), True, (4, 10, 16, 16), (4, 10, 3, 2)),
            ((3, 3), (1, 1), False, (4, 10, 16, 16), (4, 10, 14, 14)),
            ((3, 3), (3, 3), False, (4, 10, 16, 16), (4, 10, 6, 6)),
            ((3, 3), (5, 7), False, (4, 10, 16, 16), (4, 10, 4, 3)),
            ((5, 3), (1, 1), True, (4, 10, 16, 16), (4, 10, 12, 14)),
            ((5, 3), (3, 3), True, (4, 10, 16, 16), (4, 10, 4, 5)),
            ((5, 3), (5, 7), True, (4, 10, 16, 16), (4, 10, 3, 2)),
            ((5, 3), (1, 1), False, (4, 10, 16, 16), (4, 10, 12, 14)),
            ((5, 3), (3, 3), False, (4, 10, 16, 16), (4, 10, 5, 6)),
            ((5, 3), (5, 7), False, (4, 10, 16, 16), (4, 10, 4, 3)),
            ((16, 16), (1, 1), True, (4, 10, 16, 16), (4, 10, 1, 1)),
            ((16, 16), (3, 3), True, (4, 10, 16, 16), (4, 10, 1, 1)),
            ((16, 16), (5, 7), True, (4, 10, 16, 16), (4, 10, 1, 1)),
            ((16, 16), (1, 1), False, (4, 10, 16, 16), (4, 10, 1, 1)),
            ((16, 16), (3, 3), False, (4, 10, 16, 16), (4, 10, 1, 1)),
            ((16, 16), (5, 7), False, (4, 10, 16, 16), (4, 10, 1, 1)),
            ((3,), (5,), True, (16,), (3,)),
            ((3,), (5,), True, (2, 16,), (2, 3,)),
            ((5,), (3,), True, (2, 3, 16,), (2, 3, 4,)),
            ((5, 1, 3), (3, 3, 3), True, (2, 16, 16, 16), (2, 4, 6, 5)),
            ((5, 1, 3), (3, 3, 3), True, (4, 2, 16, 16, 16), (4, 2, 4, 6, 5)),
        )

        for example, mode in product(examples, ['max',
                                                'sum',
                                                'average_inc_pad',
                                                'average_exc_pad']):
            (maxpoolshp, stride, ignore_border, inputshp, outputshp) = example
            # generate random images
            imval = rng.rand(*inputshp)
            images = theano.shared(imval)
            # Pool op
            numpy_output_val = \
                self.numpy_max_pool_nd_stride(imval, maxpoolshp,
                                              ignore_border, stride,
                                              mode)
            assert numpy_output_val.shape == outputshp, (
                "outshape is %s, calculated shape is %s"
                % (outputshp, numpy_output_val.shape))
            maxpool_op = \
                Pool(ndim=len(maxpoolshp),
                     ignore_border=ignore_border,
                     mode=mode)(images, maxpoolshp, stride)
            f = function([], maxpool_op)
            output_val = f()
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
                    Pool(ignore_border=ignore_border,
                         ndim=len(maxpoolshp), mode=mode)(
                        images, maxpoolshp, stride)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxPaddingStride(self):
        ignore_border = True  # padding does not support ignore_border=False
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, stride, pad, input sizes
        examples = (
            ((3,), (2,), (2,), (5,)),
            ((3,), (2,), (2,), (4, 5)),
            ((3,), (2,), (2,), (4, 2, 5)),
            ((3,), (2,), (2,), (4, 2, 5, 5)),
            ((3, 3), (2, 2), (2, 2), (4, 2, 5, 5)),
            ((4, 4), (2, 2), (1, 2), (4, 2, 5, 5)),
            ((3, 4), (1, 1), (2, 1), (4, 2, 5, 6)),
            ((4, 3), (1, 2), (0, 0), (4, 2, 6, 5)),
            ((2, 2), (2, 2), (1, 1), (4, 2, 5, 5)),
            ((4, 3, 2), (1, 2, 2), (0, 2, 1), (4, 6, 6, 5)),
            ((4, 3, 2), (1, 2, 2), (0, 2, 1), (4, 2, 6, 5, 5)),
        )
        for example, mode in product(examples,
                                     ['max', 'sum', 'average_inc_pad',
                                      'average_exc_pad']):
            (maxpoolshp, stridesize, padsize, inputsize) = example
            imval = rng.rand(*inputsize) - 0.5
            images = theano.shared(imval)

            numpy_output_val = self.numpy_max_pool_nd_stride_pad(
                imval, maxpoolshp, ignore_border,
                stridesize, padsize, mode)
            maxpool_op = Pool(
                ndim=len(maxpoolshp),
                ignore_border=ignore_border,
                mode=mode
                )(images, maxpoolshp, stridesize, padsize)
            f = function([], maxpool_op)
            output_val = f()
            utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxPaddingStride_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, stride, pad, input sizes
        examples = (
            ((10,), (5,), (3,), (2,)),
            ((10,), (5,), (3,), (2, 2)),
            ((10,), (5,), (3,), (1, 1, 2)),
            ((10, 10), (5, 3), (3, 2), (1, 1, 2, 2)),
            ((10, 5), (3, 5), (2, 3), (1, 1, 2, 1)),
            ((5, 5), (3, 3), (3, 3), (1, 1, 2, 2)),
            ((5, 5, 5), (3, 3, 3), (3, 3, 3), (1, 1, 2, 2, 2)),
        )
        # average_inc_pad and average_exc_pad do not
        # support grad with padding
        for mode in ['max', 'sum']:
            for example in examples:
                (maxpoolshp, stridesize, padsize, inputsize) = example
                imval = rng.rand(*inputsize) * 10.0

                def mp(input):
                    return Pool(
                        ndim=len(maxpoolshp),
                        ignore_border=True,
                        mode=mode,
                        )(input, maxpoolshp, stridesize, padsize)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMax_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, input sizes
        examples = (
            ((2,), (3,)),
            ((2,), (2, 3)),
            ((2,), (2, 3, 3)),
            ((1, 1), (2, 3, 3, 4)),
            ((3, 2), (2, 3, 3, 4)),
            ((2, 3), (2, 3, 3, 4)),
            ((1, 1, 1), (2, 3, 3)),
            ((3, 2, 2), (2, 3, 3, 4)),
            ((2, 3, 2), (2, 3, 3, 4, 4)),
            ((2, 2, 3), (2, 3, 3, 4, 4)),
        )

        for example, ignore_border, mode in product(examples,
                                                    [True, False],
                                                    ['max',
                                                     'sum',
                                                     'average_inc_pad',
                                                     'average_exc_pad']):
            (maxpoolshp, inputsize) = example
            imval = rng.rand(*inputsize) * 10.0

            # more variance means numeric gradient will be more accurate
            def mp(input):
                return Pool(ndim=len(maxpoolshp),
                            ignore_border=ignore_border,
                            mode=mode)(input, maxpoolshp)
            utt.verify_grad(mp, [imval], rng=rng)

    # pool, stride, input sizes
    pool_grad_stride_examples = (
        ((1,), (1,), (16,)),
        ((1,), (3,), (1, 16)),
        ((1,), (5,), (1, 2, 16)),
        ((2,), (1,), (16,)),
        ((2,), (3,), (1, 16)),
        ((2,), (5,), (1, 2, 16)),
        ((1, 1), (1, 1), (1, 2, 16, 16)),
        ((1, 1), (3, 3), (1, 2, 16, 16)),
        ((1, 1), (5, 7), (1, 2, 16, 16)),
        ((3, 3), (1, 1), (1, 2, 16, 16)),
        ((3, 3), (3, 3), (1, 2, 16, 16)),
        ((3, 3), (5, 7), (1, 2, 16, 16)),
        ((5, 3), (1, 1), (1, 2, 16, 16)),
        ((5, 3), (3, 3), (1, 2, 16, 16)),
        ((5, 3), (5, 7), (1, 2, 16, 16)),
        ((5, 1, 2), (1, 1, 1), (16, 3, 16)),
        ((5, 1, 2), (3, 1, 2), (1, 16, 3, 16)),
        ((5, 1, 2), (5, 1, 4), (1, 2, 16, 3, 16)),
        ((5, 3), (3, 2), (1, 2, 16, 16)),
        ((5, 3), (7, 5), (1, 2, 16, 16)),
        ((5, 3), (10, 6), (1, 2, 16, 16)),
        ((5, 5), (1, 1), (1, 2, 8, 5)),
        ((3, 2), (2, 3), (1, 2, 8, 5)),
        ((7, 7), (10, 10), (1, 2, 8, 5)),
        ((9, 9), (1, 1), (1, 2, 8, 5)),
    )

    @parameterized.expand(product(pool_grad_stride_examples,
                                  [True, False],
                                  ['max',
                                   'sum',
                                   'average_inc_pad',
                                   'average_exc_pad']),
                          testcase_func_name=utt.custom_name_func)
    def test_DownsampleFactorMax_grad_stride(self, example, ignore_border, mode):
        # checks the gradient for the case that stride is used
        rng = numpy.random.RandomState(utt.fetch_seed())

        (maxpoolshp, stridesize, inputsize) = example
        imval = rng.rand(*inputsize)

        def mp(input):
            return Pool(ndim=len(maxpoolshp),
                        ignore_border=ignore_border,
                        mode=mode)(input, maxpoolshp, stridesize)
        utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMaxGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, input sizes
        examples = (
            ((2,), (2,)),
            ((2,), (2, 3)),
            ((1, 1), (2, 3, 3, 4)),
            ((3, 2), (2, 3, 3, 4)),
            ((2, 3), (2, 3, 3, 4)),
            ((1, 1, 1), (2, 3, 3, 4)),
            ((3, 2, 2), (2, 3, 3, 4)),
            ((2, 3, 2), (2, 3, 3, 4)),
            ((2, 2, 3), (2, 3, 3, 4)),
            ((2, 2, 3), (2, 1, 3, 3, 4)),
        )

        for (maxpoolshp, inputsize) in examples:
            imval = rng.rand(*inputsize) * 10.0
            # more variance means numeric gradient will be more accurate
            for ignore_border in [True, False]:
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                # The shape of the gradient will be the shape of the output
                grad_shape = Pool.out_shape(
                    imval.shape, maxpoolshp, ndim=len(maxpoolshp), ignore_border=ignore_border)
                grad_val = rng.rand(*grad_shape) * 10.0

                def mp(input, grad):
                    out = Pool(
                        ndim=len(maxpoolshp),
                        ignore_border=ignore_border)(input, maxpoolshp)
                    grad_op = MaxPoolGrad(
                        ndim=len(maxpoolshp),
                        ignore_border=ignore_border)
                    return grad_op(input, out, grad, maxpoolshp)

                utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_AveragePoolGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # avgpool, input sizes
        examples = (
            ((2,), (2,)),
            ((2,), (2, 3)),
            ((1, 1), (2, 3, 3, 4)),
            ((3, 2), (2, 3, 3, 4)),
            ((2, 3), (2, 3, 3, 4)),
            ((1, 1, 1), (2, 3, 3, 4)),
            ((3, 2, 2), (2, 3, 3, 4)),
            ((2, 3, 2), (2, 3, 3, 4)),
            ((2, 2, 3), (2, 3, 3, 4)),
            ((2, 2, 3), (2, 1, 3, 3, 4)),
        )

        for (avgpoolshp, inputsize) in examples:
            imval = rng.rand(*inputsize) * 10.0
            # more variance means numeric gradient will be more accurate
            for ignore_border in [True, False]:
                for mode in ['sum', 'average_inc_pad', 'average_exc_pad']:
                    # print 'maxpoolshp =', maxpoolshp
                    # print 'ignore_border =', ignore_border
                    # The shape of the gradient will be the shape of the output
                    grad_shape = Pool.out_shape(
                        imval.shape, avgpoolshp, ndim=len(avgpoolshp),
                        ignore_border=ignore_border)
                    grad_val = rng.rand(*grad_shape) * 10.0

                    def mp(input, grad):
                        grad_op = AveragePoolGrad(
                            ndim=len(avgpoolshp),
                            ignore_border=ignore_border, mode=mode)
                        return grad_op(input, grad, avgpoolshp)

                    utt.verify_grad(mp, [imval, grad_val], rng=rng)

    @parameterized.expand(product(pool_grad_stride_examples,
                                  [True, False]),
                          testcase_func_name=utt.custom_name_func)
    def test_DownsampleFactorMaxGrad_grad_stride(self, example, ignore_border):
        # checks the gradient of the gradient for
        # the case that stride is used
        rng = numpy.random.RandomState(utt.fetch_seed())
        (maxpoolshp, stride, inputsize) = example
        imval = rng.rand(*inputsize)
        grad_shape = Pool.out_shape(
            imval.shape, maxpoolshp, ndim=len(maxpoolshp),
            ignore_border=ignore_border, stride=stride)

        # skip the grad verification when the output is empty
        if numpy.prod(grad_shape) != 0:
            grad_val = rng.rand(*grad_shape)

            def mp(input, grad):
                out = Pool(
                    ndim=len(maxpoolshp),
                    ignore_border=ignore_border)(input, maxpoolshp, stride)
                grad_op = MaxPoolGrad(
                    ndim=len(maxpoolshp),
                    ignore_border=ignore_border)
                return grad_op(input, out, grad, maxpoolshp, stride)

                utt.verify_grad(mp, [imval, grad_val], rng=rng)

    @parameterized.expand(product(pool_grad_stride_examples,
                                  [True, False],
                                  ['sum',
                                   'average_inc_pad',
                                   'average_exc_pad']),
                          testcase_func_name=utt.custom_name_func)
    def test_AveragePoolGrad_grad_stride(self, example, ignore_border, mode):
        # checks the gradient of the gradient for
        # the case that stride is used
        rng = numpy.random.RandomState(utt.fetch_seed())
        (avgpoolshp, stride, inputsize) = example
        imval = rng.rand(*inputsize)
        grad_shape = Pool.out_shape(
            imval.shape, avgpoolshp,
            ndim=len(avgpoolshp),
            ignore_border=ignore_border, stride=stride)

        # skip the grad verification when the output is empty
        if numpy.prod(grad_shape) != 0:
            grad_val = rng.rand(*grad_shape)

            def mp(input, grad):
                grad_op = AveragePoolGrad(
                    ndim=len(avgpoolshp),
                    ignore_border=ignore_border,
                    mode=mode)
                return grad_op(input, grad, avgpoolshp, stride)

            utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_DownsampleFactorMaxPaddingStride_grad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, stride, pad, input sizes
        examples = (
            ((3,), (2,), (2,), (10,)),
            ((3,), (2,), (2,), (2, 10,)),
            ((3,), (2,), (2,), (2, 1, 10,)),
            ((5, 3), (3, 2), (2, 2), (1, 1, 10, 10)),
            ((5, 3), (3, 2), (2, 2), (1, 1, 10, 10)),
            ((3, 5), (2, 3), (2, 1), (1, 1, 10, 5)),
            ((3, 3), (3, 3), (2, 2), (1, 1, 5, 5)),
            ((5, 3, 3), (3, 2, 2), (2, 2, 2), (1, 1, 10, 5, 5)),
            ((3, 5, 3), (2, 3, 2), (2, 1, 2), (1, 1, 5, 10, 5)),
            ((3, 3, 5), (2, 2, 3), (2, 2, 1), (1, 1, 5, 5, 10)),
        )

        for (maxpoolshp, stridesize, padsize, inputsize) in examples:
            imval = rng.rand(*inputsize) * 10.0

            grad_shape = Pool.out_shape(imval.shape,
                                        maxpoolshp,
                                        ndim=len(maxpoolshp),
                                        stride=stridesize,
                                        ignore_border=True,
                                        pad=padsize)
            grad_val = rng.rand(*grad_shape) * 10.0

            def mp(input, grad):
                out = Pool(
                    ndim=len(maxpoolshp),
                    ignore_border=True,
                    )(input, maxpoolshp, stridesize, padsize)
                grad_op = MaxPoolGrad(ndim=len(maxpoolshp),
                                      ignore_border=True)
                return grad_op(input, out, grad, maxpoolshp, stridesize, padsize)
            utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_AveragePoolPaddingStride_grad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # avgpool, stride, pad, input sizes
        examples = (
            ((3,), (2,), (2,), (10,)),
            ((3,), (2,), (2,), (2, 10,)),
            ((3,), (2,), (2,), (2, 1, 10,)),
            ((5, 3), (3, 2), (2, 2), (1, 1, 10, 10)),
            ((5, 3), (3, 2), (2, 2), (1, 1, 10, 10)),
            ((3, 5), (2, 3), (2, 1), (1, 1, 10, 5)),
            ((3, 3), (3, 3), (2, 2), (1, 1, 5, 5)),
            ((5, 3, 3), (3, 2, 2), (2, 2, 2), (1, 1, 10, 5, 5)),
            ((3, 5, 3), (2, 3, 2), (2, 1, 2), (1, 1, 5, 10, 5)),
            ((3, 3, 5), (2, 2, 3), (2, 2, 1), (1, 1, 5, 5, 10)),
        )

        for (avgpoolshp, stridesize, padsize, inputsize) in examples:
            imval = rng.rand(*inputsize) * 10.0

            # 'average_exc_pad' with non-zero padding is not implemented
            for mode in ['sum', 'average_inc_pad']:
                grad_shape = Pool.out_shape(imval.shape,
                                            avgpoolshp,
                                            ndim=len(avgpoolshp),
                                            stride=stridesize,
                                            ignore_border=True,
                                            pad=padsize)
                grad_val = rng.rand(*grad_shape) * 10.0

                def mp(input, grad):
                    grad_op = AveragePoolGrad(ndim=len(avgpoolshp),
                                              ignore_border=True,
                                              mode=mode)
                    return grad_op(input, grad, avgpoolshp, stridesize, padsize)
                utt.verify_grad(mp, [imval, grad_val], rng=rng)

    def test_DownsampleFactorMax_hessian(self):
        # Example provided by Frans Cronje, see
        # https://groups.google.com/d/msg/theano-users/qpqUy_3glhw/JMwIvlN5wX4J
        x_vec = tensor.vector('x')
        z = tensor.dot(x_vec.dimshuffle(0, 'x'),
                       x_vec.dimshuffle('x', 0))
        y = pool_2d(input=z, ws=(2, 2), ignore_border=True)
        C = tensor.exp(tensor.sum(y))

        grad_hess = tensor.hessian(cost=C, wrt=x_vec)
        fn_hess = function(inputs=[x_vec], outputs=grad_hess)

        # The value has been manually computed from the theoretical gradient,
        # and confirmed by the implementation.

        assert numpy.allclose(fn_hess([1, 2]), [[0., 0.], [0., 982.7667]])

    def test_DownsampleFactorMaxGradGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # maxpool, stride, pad, input sizes
        examples = (
            ((3,), (2,), (2,), (10,)),
            ((3,), (2,), (2,), (2, 10,)),
            ((3,), (2,), (2,), (2, 1, 10,)),
            ((5, 3), (3, 2), (2, 2), (1, 1, 10, 10)),
            ((5, 3), (3, 2), (2, 2), (1, 1, 10, 10)),
            ((3, 5), (2, 3), (2, 1), (1, 1, 10, 5)),
            ((3, 3), (3, 3), (2, 2), (1, 1, 5, 5)),
            ((5, 3, 3), (3, 2, 2), (2, 2, 2), (1, 1, 10, 5, 5)),
            ((3, 5, 3), (2, 3, 2), (2, 1, 2), (1, 1, 5, 10, 5)),
            ((3, 3, 5), (2, 2, 3), (2, 2, 1), (1, 1, 5, 5, 10)),
        )

        for (maxpoolshp, stridesize, padsize, inputsize) in examples:
            imval1 = rng.rand(*inputsize) * 10.0
            imval2 = rng.rand(*inputsize) * 10.0

            def mp(input1, input2):
                op1 = Pool(ndim=len(maxpoolshp), ignore_border=True)
                pooled_out = op1(input1, maxpoolshp, stridesize, padsize)
                op2 = DownsampleFactorMaxGradGrad(
                    ndim=len(maxpoolshp),
                    ignore_border=True)
                out = op2(input1, pooled_out, input2, maxpoolshp, stridesize, padsize)
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

    def test_max_pool_3d_3D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1, 1), (3, 2, 1))
        imval = rng.rand(4, 5, 6)
        images = tensor.dtensor3()

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max', 'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_nd(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_3d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                output_val = function([images], output)(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                def mp(input):
                    return pool_3d(input, maxpoolshp, ignore_border,
                                   mode=mode)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_3d_3D_deprecated_interface(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1, 1), (3, 2, 1))
        imval = rng.rand(4, 5, 6)
        images = tensor.dtensor3()

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max', 'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_nd(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_3d(input=images,
                                 ds=maxpoolshp,
                                 ignore_border=ignore_border,
                                 st=maxpoolshp,
                                 padding=(0, 0, 0),
                                 mode=mode)
                output_val = function([images], output)(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                def mp(input):
                    return pool_3d(input, maxpoolshp, ignore_border,
                                   mode=mode)

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
                for k, pad in enumerate([(0, 0), (1, 1), (1, 2)]):
                    if out_shapes[k][i][j] is None:
                        continue
                    # checking shapes generated by Pool
                    self._compile_and_check([image],
                                            [Pool(ignore_border=ignore_border)
                                             (image, maxpoolshp, pad=pad)],
                                            [image_val], Pool)

                    # checking shapes generated by MaxPoolGrad
                    maxout_val = rng.rand(*out_shapes[k][i][j])
                    gz_val = rng.rand(*out_shapes[k][i][j])
                    self._compile_and_check([image, maxout, gz],
                                            [MaxPoolGrad(
                                                ignore_border=ignore_border)
                                             (image, maxout, gz, maxpoolshp,
                                              pad=pad)],
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

    def test_pooling_with_tensor_vars_deprecated_interface(self):
        x = tensor.ftensor4()
        window_size = tensor.ivector()
        stride = tensor.ivector()
        padding = tensor.ivector()
        data = numpy.random.normal(0, 1, (1, 1, 5, 5)).astype('float32')

        # checking variable params vs fixed params
        for ignore_border in [True, False]:
            for mode in ['max', 'sum', 'average_inc_pad', 'average_exc_pad']:
                y = pool_2d(input=x,
                            ds=window_size,
                            ignore_border=ignore_border,
                            st=stride,
                            padding=padding,
                            mode=mode)
                dx = theano.gradient.grad(y.sum(), x)
                var_fct = theano.function([x, window_size, stride, padding],
                                          [y, dx])
                ws = 5
                st = 3
                pad = 1
                if (pad > st or st > ws or
                        (pad != 0 and not ignore_border) or
                        (mode == 'average_exc_pad' and pad != 0)):
                    continue
                y = pool_2d(input=x,
                            ds=(ws, ws),
                            ignore_border=ignore_border,
                            st=(st, st),
                            padding=(pad, pad),
                            mode=mode)
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
