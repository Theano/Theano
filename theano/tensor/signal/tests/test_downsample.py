import unittest
import __builtin__
import numpy
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.signal.downsample import (DownsampleFactorMax, max_pool_2d,
                                             DownsampleFactorMaxGrad)
from theano import function


class TestDownsampleFactorMax(utt.InferShapeTester):

    @staticmethod
    def numpy_max_pool_2d(input, ds, ignore_border=False):
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

        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii = i * ds[0]
                for j in range(output_val.shape[-1]):
                    jj = j * ds[1]
                    patch = input[k][ii:ii + ds[0], jj:jj + ds[1]]
                    output_val[k][i, j] = numpy.max(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride(input, ds, ignore_border=False, st=None):
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

        output_val = numpy.zeros(out_shp)
        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii_st = i * st[0]
                ii_end = __builtin__.min(ii_st + ds[0], img_rows)
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    jj_end = __builtin__.min(jj_st + ds[1], img_cols)
                    patch = input[k][ii_st:ii_end, jj_st:jj_end]
                    output_val[k][i, j] = numpy.max(patch)
        return output_val

    def test_DownsampleFactorMax(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # generate random images
        maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
        imval = rng.rand(4, 10, 64, 64)
        images = tensor.dtensor4()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                #print 'maxpoolshp =', maxpoolshp
                #print 'ignore_border =', ignore_border

                # Pure Numpy computation
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border)
                output = max_pool_2d(images, maxpoolshp, ignore_border)
                f = function([images, ], [output, ])
                output_val = f(imval)
                assert numpy.all(output_val == numpy_output_val)

                #DownsampleFactorMax op
                maxpool_op = DownsampleFactorMax(maxpoolshp,
                                                 ignore_border=
                                                 ignore_border)(images)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3))
        stridesizes = ((1, 1), (3, 3), (5, 7))
        # generate random images
        imval = rng.rand(4, 10, 16, 16)
        outputshps = ((4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
                      (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
                      (4, 10, 14, 14), (4, 10, 5, 5), (4, 10, 3, 2),
                      (4, 10, 14, 14), (4, 10, 6, 6), (4, 10, 4, 3),
                      (4, 10, 12, 14), (4, 10, 4, 5), (4, 10, 3, 2),
                      (4, 10, 12, 14), (4, 10, 5, 6), (4, 10, 4, 3))
        images = tensor.dtensor4()
        indx = 0
        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                for stride in stridesizes:
                    outputshp = outputshps[indx]
                    indx += 1
                    #DownsampleFactorMax op
                    numpy_output_val = \
                        self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                      ignore_border, stride)
                    assert numpy_output_val.shape == outputshp, (
                        "outshape is %s, calculated shape is %s"
                        % (outputshp, numpy_output_val.shape))
                    maxpool_op = \
                        DownsampleFactorMax(maxpoolshp,
                                            ignore_border=ignore_border,
                                            st=stride)(images)
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
            for ignore_border in [True, False]:
                indx_out = indx * 2
                if not ignore_border:
                    indx_out += 1
                outputshp = outputshps[indx_out]
                #DownsampleFactorMax op
                numpy_output_val = \
                    self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                  ignore_border, stride)
                assert numpy_output_val.shape == outputshp, (
                    "outshape is %s, calculated shape is %s"
                    % (outputshp, numpy_output_val.shape))
                maxpool_op = \
                    DownsampleFactorMax(maxpoolshp,
                                        ignore_border=ignore_border,
                                        st=stride)(images)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMax_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 2), (2, 3))
        imval = rng.rand(2, 3, 3, 4) * 10.0
        #more variance means numeric gradient will be more accurate

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                #print 'maxpoolshp =', maxpoolshp
                #print 'ignore_border =', ignore_border
                def mp(input):
                    return DownsampleFactorMax(maxpoolshp,
                                               ignore_border=
                                               ignore_border)(input)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMax_grad_st(self):
        """checks the gradient for the case that stride is used"""
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3))
        stridesizes = ((1, 1), (3, 3), (5, 7))
        imval = rng.rand(1, 2, 16, 16)

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                for stride in stridesizes:
                    def mp(input):
                        return DownsampleFactorMax(maxpoolshp,
                                                   ignore_border=ignore_border,
                                                   st=stride)(input)
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

        for indx in numpy.arange(len(maxpoolshps)):
            imvsize = imvsizs[indx]
            imval = rng.rand(1, 2, imvsize[0], imvsize[1])
            stride = stridesizes[indx]
            maxpoolshp = maxpoolshps[indx]
            for ignore_border in [True, False]:
                def mp(input):
                    return DownsampleFactorMax(maxpoolshp,
                                               ignore_border=ignore_border,
                                               st=stride)(input)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_DownsampleFactorMaxGrad_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 2), (2, 3))
        imval = rng.rand(2, 3, 3, 4) * 10.0
        #more variance means numeric gradient will be more accurate

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                #print 'maxpoolshp =', maxpoolshp
                #print 'ignore_border =', ignore_border
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

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                #print 'maxpoolshp =', maxpoolshp
                #print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border)
                output = max_pool_2d(images, maxpoolshp, ignore_border)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val), (
                    "output_val is %s, numpy_output_val is %s"
                    % (output_val, numpy_output_val))

                def mp(input):
                    return max_pool_2d(input, maxpoolshp, ignore_border)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_2d_3D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = [(1, 2)]
        imval = rng.rand(2, 3, 4)
        images = tensor.dtensor3()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                #print 'maxpoolshp =', maxpoolshp
                #print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border)
                output = max_pool_2d(images, maxpoolshp, ignore_border)
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

#removed as already tested in test_max_pool_2d_2D
#This make test in debug mode too slow.
#                def mp(input):
#                    return max_pool_2d(input, maxpoolshp, ignore_border)
#                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_2d_6D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = [(3, 2)]
        imval = rng.rand(2, 1, 1, 1, 3, 4)
        images = tensor.TensorType('float64', [False] * 6)()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True, False]:
                #print 'maxpoolshp =', maxpoolshp
                #print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border)
                output = max_pool_2d(images, maxpoolshp, ignore_border)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val)

#removed as already tested in test_max_pool_2d_2D
#This make test in debug mode too slow.
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
        out_shapes = [[[4, 6, 7, 9], [4, 6, 7, 9]],
                      [[4, 6, 3, 4], [4, 6, 4, 5]],
                      [[4, 6, 2, 3], [4, 6, 3, 3]],
                      [[4, 6, 3, 3], [4, 6, 4, 3]],
                      [[4, 6, 2, 4], [4, 6, 3, 5]]]

        for i, maxpoolshp in enumerate(maxpoolshps):
            for j, ignore_border in enumerate([True, False]):

                # checking shapes generated by DownsampleFactorMax
                self._compile_and_check([image],
                                        [DownsampleFactorMax(maxpoolshp,
                                        ignore_border=ignore_border)(image)],
                                        [image_val], DownsampleFactorMax)

                # checking shapes generated by DownsampleFactorMaxGrad
                maxout_val = rng.rand(*out_shapes[i][j])
                gz_val = rng.rand(*out_shapes[i][j])
                self._compile_and_check([image, maxout, gz],
                                        [DownsampleFactorMaxGrad(maxpoolshp,
                                        ignore_border=ignore_border)
                                        (image, maxout, gz)],
                                        [image_val, maxout_val, gz_val],
                                        DownsampleFactorMaxGrad,
                                        warn=False)


if __name__ == '__main__':
    unittest.main()
