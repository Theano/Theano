from __future__ import absolute_import, print_function, division

from itertools import product
import unittest
import six.moves.builtins as builtins

import numpy
import math

import theano.tensor as T
from theano.tests import unittest_tools as utt

from theano import function
from theano.sandbox.mkl.mkl_pool import Pool
from theano.sandbox.mkl.basic_ops import (U2IPool, I2U)


class TestMKLPool(unittest.TestCase):

    @staticmethod
    def numpy_pool_2d(input, ds, ignore_border=False, mode='max'):
        '''Helper function, implementing pool_2d in pure numpy'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))
        in_h = input.shape[-2]
        in_w = input.shape[-1]
        kernel_h = stride_h = ds[0]
        kernel_w = stride_w = ds[1]
        pad_h = pad_w = 0

        # using Intel MKL style to calculate the output shape
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
    def numpy_pool_2d_stride(input, ds, ignore_border=False, st=None,
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

        in_h = input.shape[-2]
        in_w = input.shape[-1]
        kernel_h = ds[0]
        kernel_w = ds[1]
        stride_h = st[0]
        stride_w = st[1]
        pad_h = pad_w = 0

        # using Intel MKL style to calculate the output shape
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

    @staticmethod
    def numpy_pool_2d_stride_padding(
            x, ds, ignore_border=True, st=None, padding=(0, 0), mode='max'):
        assert ignore_border

        in_h = x.shape[-2]
        in_w = x.shape[-1]
        kernel_h = ds[0]
        kernel_w = ds[1]
        stride_h = st[0]
        stride_w = st[1]
        pad_h = padding[0]
        pad_w = padding[1]

        assert ds[0] > pad_h
        assert ds[1] > pad_w

        def pad_img(x):
            y = numpy.zeros(
                (x.shape[0], x.shape[1],
                 x.shape[2] + pad_h * 2, x.shape[3] + pad_w * 2),
                dtype=x.dtype)
            y[:, :, pad_h:(x.shape[2] + pad_h), pad_w:(x.shape[3] + pad_w)] = x

            return y

        h = in_h + 2 * pad_h
        w = in_w + 2 * pad_w

        # using Intel MKL style to calculate the output shape
        out_h = int(math.ceil((float)(h - kernel_h) / stride_h)) + 1
        out_w = int(math.ceil((float)(w - kernel_w) / stride_w)) + 1

        out_shp = list(x.shape[:-2])
        out_shp.extend([out_h, out_w])

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
                if ii_st > h:
                    print ('ii_st > h!!!')
                    continue
                ii_end = builtins.min(ii_st + ds[0], h)
                if not inc_pad:
                    ii_st = builtins.max(ii_st, pad_h)
                    ii_end = builtins.min(ii_end, in_h + pad_h)
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    if jj_st > w:
                        print ('jj_st > w!!!')
                        continue
                    jj_end = builtins.min(jj_st + ds[1], w)
                    if not inc_pad:
                        jj_st = builtins.max(jj_st, pad_w)
                        jj_end = builtins.min(jj_end, in_w + pad_w)
                    patch = y[k][ii_st:ii_end, jj_st:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    def mkl_pool_func(*inputs):
        if len(inputs) == 5:
            # self, images, ignore_border, mode, ds
            _, images, ignore_border, mode, ds, = inputs
            x_internal = U2IPool(ignore_border=ignore_border,
                                 mode=mode)(images, ds)
            poolOut = Pool(ignore_border=ignore_border,
                           mode=mode)(x_internal, ds)
            output = I2U()(poolOut)
        elif len(inputs) == 6:
            # self, images, ignore_border, mode, ds, st,
            _, images, ignore_border, mode, ds, st, = inputs
            x_internal = U2IPool(ignore_border=ignore_border,
                                 mode=mode)(images, ds, st)
            poolOut = Pool(ignore_border=ignore_border,
                           mode=mode)(x_internal, ds, st)
            output = I2U()(poolOut)
        elif len(inputs) == 7:
            # self, images, ignore_border, mode, ds, st, pad
            _, images, ignore_border, mode, ds, st, pad = inputs
            x_internal = U2IPool(ignore_border=ignore_border,
                                 mode=mode)(images, ds, st, pad)
            poolOut = Pool(ignore_border=ignore_border,
                           mode=mode)(x_internal, ds, st, pad)
            output = I2U()(poolOut)
        else:
            raise ValueError("incorrect inputs list, should be 4 ~ 6 parameters!")

        return output

    def test_pool(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        images = T.dtensor4()

        ds_list = ((1, 1), (2, 2), (3, 3), (2, 3))
        # generate random images
        imval = rng.rand(4, 2, 16, 16)
        for ds, ignore_border, mode in product(ds_list,
                                               [True],
                                               ['max',
                                                'average_exc_pad']):
            # Pure Numpy computation
            numpy_output_val = self.numpy_pool_2d(imval, ds,
                                                  ignore_border,
                                                  mode=mode)

            # MKL Ops
            output = self.mkl_pool_func(images, ignore_border, mode, ds)

            f = function([images, ], [output, ])
            output_val = f(imval)

            utt.assert_allclose(output_val, numpy_output_val)

    def test_pool_stride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # generate random images
        ds_list = ((1, 1), (2, 2), (3, 3), (2, 3), (5, 3))
        st_list = ((1, 1), (3, 3), (5, 3))
        imval = rng.rand(4, 2, 16, 16)
        images = T.dtensor4()
        for ds, st, ignore_border, mode in product(ds_list,
                                                   st_list,
                                                   [True],
                                                   ['max',
                                                    'average_exc_pad']):
            # Pure Numpy computation
            numpy_output_val = self.numpy_pool_2d_stride(imval, ds,
                                                         ignore_border, st, mode)

            # MKL Ops
            output = self.mkl_pool_func(images, ignore_border, mode, ds, st)

            f = function([images, ], [output, ])
            output_val = f(imval)

            utt.assert_allclose(output_val, numpy_output_val)

    def test_pool_stride_padding(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # generate random images
        ds_list = ((3, 3), (4, 4), (3, 4), (5, 5))
        st_list = ((1, 1), (2, 2), (3, 3), (1, 2))
        pad_list = ((1, 1), (0, 0), (1, 1), (1, 1))
        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        n = 4
        c = 2

        images = T.dtensor4()

        for idx, ignore_border, mode in product(numpy.arange(len(ds_list)),
                                                [True],
                                                ['max',
                                                 'average_exc_pad']):
            imgsize = imgsize_list[idx]
            imval = rng.rand(n, c, imgsize[0], imgsize[1])
            ds = ds_list[idx]
            st = st_list[idx]
            pad = pad_list[idx]

            # Pure Numpy computation
            numpy_output_val = self.numpy_pool_2d_stride_padding(imval, ds,
                                                                 ignore_border, st,
                                                                 pad, mode)

            # MKL Ops
            output = self.mkl_pool_func(images, ignore_border, mode, ds, st, pad)

            f = function([images, ], [output, ])
            output_val = f(imval)

            utt.assert_allclose(output_val, numpy_output_val)

    def test_pool_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        ds_list = ((1, 1), (3, 2), (2, 3))
        imval = rng.rand(2, 3, 3, 4) * 10.0

        for ds, ignore_border, mode in product(ds_list,
                                               [True],
                                               ['max',
                                                'average_exc_pad']):
            def mp(input):
                return self.mkl_pool_func(input, ignore_border, mode, ds)

            utt.verify_grad(mp, [imval])

    def test_pool_stride_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        ds_list = ((1, 1), (2, 2), (3, 3), (2, 3), (5, 3))
        st_list = ((1, 1), (3, 3), (5, 3))
        imval = rng.rand(4, 2, 16, 16)

        for ds, st, ignore_border, mode in product(ds_list,
                                                   st_list,
                                                   [True],
                                                   ['max',
                                                    'average_exc_pad']):
            def mp(input):
                return self.mkl_pool_func(input, ignore_border, mode, ds, st)

            utt.verify_grad(mp, [imval])

    def test_pool_stride_pad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        ds_list = ((3, 3), (4, 4), (3, 4), (5, 5))
        st_list = ((1, 1), (2, 2), (3, 3), (1, 2))
        pad_list = ((1, 1), (0, 0), (1, 1), (1, 1))
        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        n = 4
        c = 3

        for idx, ignore_border, mode in product(numpy.arange(len(ds_list)),
                                                [True],
                                                ['max',
                                                 'average_exc_pad']):
            imgsize = imgsize_list[idx]
            imval = rng.rand(n, c, imgsize[0], imgsize[1])
            ds = ds_list[idx]
            st = st_list[idx]
            pad = pad_list[idx]

            def mp(input):
                return self.mkl_pool_func(input, ignore_border, mode, ds, st, pad)

            utt.verify_grad(mp, [imval])


if __name__ == '__main__':
    unittest.main()
