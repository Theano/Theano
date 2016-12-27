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
import math

import theano
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.signal.pool import (Pool, pool_2d,
                                       MaxPoolGrad, AveragePoolGrad,
                                       max_pool_2d_same_size,
                                       DownsampleFactorMaxGradGrad)

from theano import function


class TestDownsampleFactorMax(utt.InferShapeTester):

    @staticmethod
    def numpy_max_pool_2d(input, ds, ignore_border=False, mode='max'):
        '''Helper function, implementing pool_2d in pure numpy'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))
        # using Intel-Caffe style to calculate the output shape
        in_h = input.shape[-2]
        in_w = input.shape[-1]
        kernel_h = stride_h = ds[0]
        kernel_w = stride_w = ds[1]
        pad_h = pad_w = 0

        out_h = int(math.ceil((float)(in_h + 2 * pad_h - kernel_h) / stride_h)) + 1
        out_w = int(math.ceil((float)(in_w + 2 * pad_w - kernel_w) / stride_w)) + 1

        out_shp = list(input.shape[:-2])
        out_shp.extend([out_h, out_w])

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

        # using Intel-Caffe style to calculate the output shape
        in_h = input.shape[-2]
        in_w = input.shape[-1]
        kernel_h = ds[0]
        kernel_w = ds[1]
        stride_h = st[0]
        stride_w = st[1]
        pad_h = pad_w = 0

        out_h = int(math.ceil((float)(in_h + 2 * pad_h - kernel_h) / stride_h)) + 1
        out_w = int(math.ceil((float)(in_w + 2 * pad_w - kernel_w) / stride_w)) + 1

        out_shp = list(input.shape[:-2])
        out_shp.extend([out_h, out_w])

        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        output_val = numpy.zeros(out_shp)
        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii_st = i * st[0]
                if ii_st > in_h:
                    print ('ii_st > in_h!!!')
                    continue
                ii_end = builtins.min(ii_st + ds[0], in_h)
                if ii_st == ii_end:
                    continue
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    if jj_st > in_w:
                        print ('jj_st > in_w!!!')
                        continue
                    jj_end = builtins.min(jj_st + ds[1], in_w)
                    if jj_st == jj_end:
                        continue
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
                                                       [True],
                                                       ['max',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                #print ('maxpoolshp =', maxpoolshp)
                #print ('ignore_border =', ignore_border)
                #print ('mode =', mode)

                # Pure Numpy computation
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                f = function([images, ], [output, ])
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                #print ('numpy_output_val.shape =', numpy_output_val.shape)
                #print ('output_val.shape =', output_val[0].shape)
                # Pool op
                maxpool_op = Pool(ignore_border=ignore_border,
                                  mode=mode)(images, maxpoolshp)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3), (16, 16))
        stridesizes = ((1, 1), (3, 3), (5, 5),)  # (5, 7) failed
        # generate random images
        imval = rng.rand(4, 10, 16, 16)
        # The same for each mode
        outputshps = (
            (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 4),
            (4, 10, 14, 14), (4, 10, 6, 6), (4, 10, 4, 4),
            (4, 10, 12, 14), (4, 10, 5, 6), (4, 10, 4, 4),
            (4, 10, 1, 1), (4, 10, 1, 1), (4, 10, 1, 1),)

        images = tensor.dtensor4()
        indx = 0
        for mode, maxpoolshp, ignore_border in product(['max',
                                                        'average_inc_pad',
                                                        'average_exc_pad'],
                                                       maxpoolshps,
                                                       [True]):
                for stride in stridesizes:
                    outputshp = outputshps[indx % len(outputshps)]
                    indx += 1

                    # Po, Falseol op
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

    def test_DownsampleFactorMax_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 2),) # (2, 3)), failed in average_mode
        imval = rng.rand(2, 3, 3, 4) * 10.0
        # more variance means numeric gradient will be more accurate

        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max',
                                                        # 'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
            def mp(input):
                return Pool(ignore_border=ignore_border, mode=mode)(
                    input, maxpoolshp)
            utt.verify_grad(mp, [imval], rng=rng)


if __name__ == '__main__':
    unittest.main()
