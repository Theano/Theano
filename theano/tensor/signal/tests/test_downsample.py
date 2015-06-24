from itertools import product
import unittest
import six.moves.builtins as builtins

import numpy

import theano
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.signal.downsample import (DownsampleFactorMax, max_pool_2d,
                                             DownsampleFactorMaxGrad,
                                             DownsampleFactorMaxGradGrad,
                                             max_pool_2d_same_size)
from theano import function


class TestDownsampleFactorMax(utt.InferShapeTester):

    @staticmethod
    def numpy_max_pool_2d(input, ds, ignore_border=False, mode='max'):
        '''Helper function, implementing max_pool_2d in pure numpy'''
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
        out_shp.append(input.shape[-2] / ds[0] + xi)
        out_shp.append(input.shape[-1] / ds[1] + yi)
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
        pad_h = padding[0]
        pad_w = padding[1]
        h = x.shape[-2]
        w = x.shape[-1]
        assert ds[0] > pad_h
        assert ds[1] > pad_w

        def pad_img(x):
            y = numpy.zeros(
                (x.shape[0], x.shape[1],
                 x.shape[2]+pad_h*2, x.shape[3]+pad_w*2),
                dtype=x.dtype)
            y[:, :, pad_h:(x.shape[2]+pad_h), pad_w:(x.shape[3]+pad_w)] = x

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
        tt = []
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
        '''Helper function, implementing max_pool_2d in pure numpy
           this function provides st input to indicate the stide size
           for the pooling regions. if not indicated, st == sd.'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))

        if st is None:
            st = ds
        xi = 0
        yi = 0
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
                output = max_pool_2d(images, maxpoolshp, ignore_border,
                                     mode=mode)
                f = function([images, ], [output, ])
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                # DownsampleFactorMax op
                maxpool_op = DownsampleFactorMax(maxpoolshp,
                                                 ignore_border=ignore_border,
                                                 mode=mode)(images)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3))
        stridesizes = ((1, 1), (3, 3), (5, 7))
        # generate random images
        imval = rng.rand(4, 10, 16, 16)
        # The same for each mode
        outputshps = ((4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
                      (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
                      (4, 10, 14, 14), (4, 10, 5, 5), (4, 10, 3, 2),
                      (4, 10, 14, 14), (4, 10, 6, 6), (4, 10, 4, 3),
                      (4, 10, 12, 14), (4, 10, 4, 5), (4, 10, 3, 2),
                      (4, 10, 12, 14), (4, 10, 5, 6), (4, 10, 4, 3))
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
                    # DownsampleFactorMax op
                    numpy_output_val = \
                        self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                      ignore_border, stride,
                                                      mode)
                    assert numpy_output_val.shape == outputshp, (
                        "outshape is %s, calculated shape is %s"
                        % (outputshp, numpy_output_val.shape))
                    maxpool_op = \
                        DownsampleFactorMax(maxpoolshp,
                                            ignore_border=ignore_border,
                                            st=stride, mode=mode)(images)
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
                # DownsampleFactorMax op
                numpy_output_val = \
                    self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                  ignore_border, stride, mode)
                assert numpy_output_val.shape == outputshp, (
                    "outshape is %s, calculated shape is %s"
                    % (outputshp, numpy_output_val.shape))
                maxpool_op = \
                    DownsampleFactorMax(maxpoolshp,
                                        ignore_border=ignore_border,
                                        st=stride, mode=mode)(images)
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
            maxpool_op = DownsampleFactorMax(
                maxpoolsize,
                ignore_border=ignore_border,
                st=stridesize, padding=paddingsize, mode=mode)(images)
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
                    return DownsampleFactorMax(
                        maxpoolsize, ignore_border=True,
                        st=stridesize,
                        padding=paddingsize,
                        mode=mode,
                        )(input)
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
                return DownsampleFactorMax(maxpoolshp,
                                           ignore_border=ignore_border,
                                           mode=mode)(input)
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
                return DownsampleFactorMax(maxpoolshp,
                                           ignore_border=ignore_border,
                                           st=stride, mode=mode)(input)
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
                        return DownsampleFactorMax(maxpoolshp,
                                                   ignore_border=ignore_border,
                                                   st=stride,
                                                   mode=mode)(input)
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
                grad_shape = DownsampleFactorMax.out_shape(
                    imval.shape, maxpoolshp, ignore_border=ignore_border)
                grad_val = rng.rand(*grad_shape) * 10.0

                def mp(input, grad):
                    out = DownsampleFactorMax(
                        maxpoolshp, ignore_border=ignore_border)(input)
                    grad_op = DownsampleFactorMaxGrad(
                        maxpoolshp, ignore_border=ignore_border)
                    return grad_op(input, out, grad)

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
                    grad_shape = DownsampleFactorMax.out_shape(
                        imval.shape, maxpoolshp,
                        ignore_border=ignore_border, st=stride)
                    grad_val = rng.rand(*grad_shape)

                    def mp(input, grad):
                        out = DownsampleFactorMax(
                            maxpoolshp, ignore_border=ignore_border,
                            st=stride)(input)
                        grad_op = DownsampleFactorMaxGrad(
                            maxpoolshp, ignore_border=ignore_border,
                            st=stride)
                        return grad_op(input, out, grad)

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
                grad_shape = DownsampleFactorMax.out_shape(
                    imval.shape, maxpoolshp,
                    ignore_border=ignore_border, st=stride)
                grad_val = rng.rand(*grad_shape)

                def mp(input, grad):
                    out = DownsampleFactorMax(
                        maxpoolshp, ignore_border=ignore_border,
                        st=stride)(input)
                    grad_op = DownsampleFactorMaxGrad(
                        maxpoolshp, ignore_border=ignore_border,
                        st=stride)
                    return grad_op(input, out, grad)

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

            grad_shape = DownsampleFactorMaxGradGrad.out_shape(
                    imval.shape, maxpoolsize, st=stridesize,
                ignore_border=True, padding=paddingsize)
            grad_val = rng.rand(*grad_shape) * 10.0
            def mp(input, grad):
                out = DownsampleFactorMax(
                    maxpoolsize, ignore_border=True,
                    st=stridesize,
                    padding=paddingsize,
                    )(input)
                grad_op = DownsampleFactorMaxGrad(maxpoolsize, ignore_border=True,
                                                  st=stridesize, padding=paddingsize)
                return grad_op(input, out, grad)
            utt.verify_grad(mp, [imval, grad_val], rng=rng)
            
    def test_DownsampleFactorMax_hessian(self):
        # Example provided by Frans Cronje, see
        # https://groups.google.com/d/msg/theano-users/qpqUy_3glhw/JMwIvlN5wX4J
        x_vec = tensor.vector('x')
        z = tensor.dot(x_vec.dimshuffle(0, 'x'),
                       x_vec.dimshuffle('x', 0))
        y = max_pool_2d(input=z, ds=(2, 2))
        C = tensor.exp(tensor.sum(y))

        grad_hess = tensor.hessian(cost=C, wrt=x_vec)
        fn_hess = function(inputs=[x_vec], outputs=grad_hess)

        # The value has been manually computed from the theoretical gradient,
        # and confirmed by the implementation.
        assert numpy.allclose(fn_hess([1, 2]), [[0., 0.], [0., 982.7667]])

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
                output = max_pool_2d(images, maxpoolshp, ignore_border,
                                     mode=mode)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val), (
                    "output_val is %s, numpy_output_val is %s"
                    % (output_val, numpy_output_val))

                def mp(input):
                    return max_pool_2d(input, maxpoolshp, ignore_border,
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
        assert numpy.all(op_output == test_answer_array), (
            "op_output is %s, test_answer_array is %s" % (
                op_output, numpy_output_val
            )
        )
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
                output = max_pool_2d(images, maxpoolshp, ignore_border,
                                     mode=mode)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val), (
                    "output_val is %s, numpy_output_val is %s"
                    % (output_val, numpy_output_val))
                c = tensor.sum(output)
                c_val = function([images], c)(imval)
                g = tensor.grad(c, images)
                g_val = function([images],
                                 [g.shape,
                                 tensor.min(g, axis=(0, 1, 2)),
                                 tensor.max(g, axis=(0, 1, 2))]
                                 )(imval)

# removed as already tested in test_max_pool_2d_2D
# This make test in debug mode too slow.
#                def mp(input):
#                    return max_pool_2d(input, maxpoolshp, ignore_border)
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
                output = max_pool_2d(images, maxpoolshp, ignore_border,
                                     mode=mode)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val)

# removed as already tested in test_max_pool_2d_2D
# This make test in debug mode too slow.
#                def mp(input):
#                    return max_pool_2d(input, maxpoolshp, ignore_border)
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
                for k, padding in enumerate([(0,0), (1,1), (1,2)]):
                    if out_shapes[k][i][j] == None:
                        continue
                    # checking shapes generated by DownsampleFactorMax
                    self._compile_and_check([image],
                                            [DownsampleFactorMax(maxpoolshp,
                                            ignore_border=ignore_border,
                                            padding=padding)(image)],
                                            [image_val], DownsampleFactorMax)

                    # checking shapes generated by DownsampleFactorMaxGrad
                    maxout_val = rng.rand(*out_shapes[k][i][j])
                    gz_val = rng.rand(*out_shapes[k][i][j])
                    self._compile_and_check([image, maxout, gz],
                                            [DownsampleFactorMaxGrad(maxpoolshp,
                                            ignore_border=ignore_border,
                                            padding=padding)
                                            (image, maxout, gz)],
                                            [image_val, maxout_val, gz_val],
                                            DownsampleFactorMaxGrad,
                                            warn=False)


if __name__ == '__main__':
    unittest.main()
