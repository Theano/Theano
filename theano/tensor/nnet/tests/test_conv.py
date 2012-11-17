import sys
import time
import unittest
import numpy

import theano
import theano.tensor as T
from theano import function, Mode
from theano.tests import unittest_tools as utt

from theano.tensor.nnet import conv

from theano.tensor.basic import _allclose


class TestConv2D(utt.InferShapeTester):

    def setUp(self):
        super (TestConv2D, self).setUp()
        self.input = T.dtensor4('input')
        self.input.name = 'default_V'
        self.filters = T.dtensor4('filters')
        self.filters.name = 'default_filters'

    def validate(self, image_shape, filter_shape,
                 border_mode='valid', subsample=(1, 1),
                 N_image_shape=None, N_filter_shape=None,
                 input=None, filters=None,
                 unroll_batch=None, unroll_kern=None, unroll_patch=None,
                 verify_grad=True, should_raise=False):

        if N_image_shape is None:
            N_image_shape = [T.get_scalar_constant_value(T.
                as_tensor_variable(x)) for x in image_shape]
        if N_filter_shape is None:
            N_filter_shape = [T.get_scalar_constant_value(T.
                as_tensor_variable(x)) for x in filter_shape]

        if input is None:
            input = self.input
        if not filters:
            filters = self.filters

        ############# THEANO IMPLEMENTATION ############

        # we create a symbolic function so that verify_grad can work
        def sym_conv2d(input, filters):
            # define theano graph and function
            input.name = 'input'
            filters.name = 'filters'
            rval =  conv.conv2d(input, filters, image_shape, filter_shape,
                          border_mode, subsample, unroll_batch=unroll_batch,
                          unroll_kern=unroll_kern, unroll_patch=unroll_patch)
            rval.name = 'conv_output'
            return rval

        output = sym_conv2d(input, filters)
        output.name = 'conv2d(%s,%s)' % (input.name, filters.name)
        theano_conv = theano.function([input, filters], output)

        # initialize input and compute result
        image_data = numpy.random.random(N_image_shape)
        filter_data = numpy.random.random(N_filter_shape)
        try:
            theano_output = theano_conv(image_data, filter_data)
        except ValueError:
            if not should_raise:
                raise
            return
        else:
            if should_raise:
                raise Exception(
                "ConvOp should have generated an error")

        ############# REFERENCE IMPLEMENTATION ############
        s = 1.
        orig_image_data = image_data
        if border_mode is not 'full':
            s = -1.
        out_shape2d = numpy.array(N_image_shape[-2:]) +\
                      s * numpy.array(N_filter_shape[-2:]) - s
        out_shape2d = numpy.ceil(out_shape2d / numpy.array(subsample))
        out_shape = (N_image_shape[0], N_filter_shape[0]) + tuple(out_shape2d)
        ref_output = numpy.zeros(out_shape)

        # loop over output feature maps
        ref_output.fill(0)
        if border_mode == 'full':
            image_data2 = numpy.zeros((N_image_shape[0], N_image_shape[1],
                                      N_image_shape[2] + 2 * N_filter_shape[2] - 2,
                                      N_image_shape[3] + 2 * N_filter_shape[3] - 2))
            image_data2[:, :, N_filter_shape[2] - 1:N_filter_shape[2] - 1 + N_image_shape[2],
                              N_filter_shape[3] - 1:N_filter_shape[3] - 1 + N_image_shape[3]] = image_data
            image_data = image_data2
            N_image_shape = image_data.shape
        for bb in range(N_image_shape[0]):
            for nn in range(N_filter_shape[0]):
                for im0 in range(N_image_shape[1]):
                    filter2d = filter_data[nn, im0, :, :]
                    image2d = image_data[bb, im0, :, :]
                    for row in range(ref_output.shape[2]):
                        irow = row * subsample[0]  # image row
                        for col in range(ref_output.shape[3]):
                            icol = col * subsample[1]  # image col
                            ref_output[bb, nn, row, col] += (image2d[
                                irow:irow + N_filter_shape[2],
                                icol:icol + N_filter_shape[3]] * filter2d[::-1,::-1]
                            ).sum()

        self.assertTrue(_allclose(theano_output, ref_output))

        ############# TEST GRADIENT ############
        if verify_grad:
            utt.verify_grad(sym_conv2d, [orig_image_data, filter_data])

    def test_basic1(self):
        """Tests that basic convolutions work for odd and even
        dimensions of image and filter shapes, as well as rectangular
        images and filters.

        """
        self.validate((2, 2, 3, 3), (2, 2, 2, 2), 'valid', verify_grad=False)

    def test_basic(self):
        """Tests that basic convolutions work for odd and even
        dimensions of image and filter shapes, as well as rectangular
        images and filters.

        """
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'valid', verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid')
        self.validate((3, 2, 7, 5), (5, 2, 3, 2), 'valid', verify_grad=False)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'full', verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full')
        # test filter same size as input

    def test_img_kernel_same_shape(self):
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'full')
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid')

    def test_unroll_patch_true(self):
        """
        Test basic convs with True.
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', unroll_patch=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', unroll_patch=True)
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid',
             unroll_patch=True, verify_grad=False)

    def test_unroll_patch_false(self):
        """
        Test basic convs with False.
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', unroll_patch=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', unroll_patch=False)
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid',
             unroll_patch=False, verify_grad=False)

    def test_unroll_patch_true_fail(self):
        """
        Test basic convs with True.
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', unroll_patch=True,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', unroll_patch=True,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid', unroll_patch=True,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)

    def test_unroll_special(self):
        """
        (unroll_kern, unroll_batch) in (0,1),(1,0) is special case.
        """
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid', unroll_batch=1)

    def test_unroll_batch(self):
        """
        Test mini-batch unrolling for various legal values.
        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=2, verify_grad=False)
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=3, verify_grad=False)

    def test_unroll_kern(self):
        """
        Test kernel unrolling for various legal values.
        """
        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid', unroll_kern=2,
             verify_grad=False)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid', unroll_kern=3,
             verify_grad=False)

    def test_unroll_batch_kern(self):
        """Test mini-batch unrolling with kernel unrolling for various
        legal values.

        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=2, unroll_kern=3, verify_grad=False)
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=3, unroll_kern=3, verify_grad=False)
        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
             unroll_batch=2, unroll_kern=2, verify_grad=False)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
             unroll_batch=2, unroll_kern=3, verify_grad=False)

    def test_unroll_batch_kern_fail(self):
        """Test mini-batch unrolling with kernel unrolling for various
        legal values, but pass bad input.  All those test must
        generate errors

        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
                      unroll_batch=2, unroll_kern=3,
                      N_image_shape=(7, 2, 3, 3), N_filter_shape=(3, 2, 2, 2),
                      should_raise=True)
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
                      unroll_batch=3, unroll_kern=3,
                      N_image_shape=(6, 2, 3, 3), N_filter_shape=(4, 2, 2, 2),
                      should_raise=True)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
                      unroll_batch=2, unroll_kern=2,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
                      unroll_batch=2, unroll_kern=3,
                      N_image_shape=(2, 3, 3, 3),  N_filter_shape=(5, 3, 2, 2),
                      should_raise=True)

    def test_subsample(self):
        """
        Tests convolution where subsampling != (1,1)
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'valid', subsample=(3, 3))

        # Fails as of 2012-07-11
        self.assertRaises(NotImplementedError, self.validate, (1, 1, 6, 6),
                          (1, 1, 3, 3), 'full', subsample=(3, 3))

    def test_shape_Constant_tensor(self):
        """
        Tests convolution where the {image,filter}_shape is a Constant tensor.
        """
        as_t = T.as_tensor_variable
        self.validate((as_t(3), as_t(2), as_t(7), as_t(5)), (5, 2,
             2, 3), 'valid')
        self.validate(as_t([3, 2, 7, 5]), (5, 2, 2, 3), 'valid')
        self.validate(as_t((3, 2, 7, 5)), (5, 2, 2, 3), 'valid')
        self.validate((3, 2, 7, 5), (as_t(5), as_t(2), as_t(2),
             as_t(3)), 'valid')
        self.validate((3, 2, 7, 5), as_t([5, 2, 2, 3]), 'valid')
        self.validate((3, 2, 7, 5), as_t((5, 2, 2, 3)), 'valid')
        self.validate(as_t([3, 2, 7, 5]), as_t([5, 2, 2, 3]), 'full')

    def test_invalid_filter_shape(self):
        """
        Tests scenario where filter_shape[1] != input_shape[1]
        """
        self.assertRaises(AssertionError, self.validate,
                          (3, 2, 8, 8), (4, 3, 5, 5),
                          'valid')

    def test_invalid_input_shape(self):
        """
        Tests that when the shape gived at build time is not the same as
        run time we raise an error
        """
        for unroll_batch in [None, 1, 3]:
            for unroll_kern in [None, 2, 4]:
                for unroll_patch in [None, True, False]:
                    for mode in ['valid', 'full']:
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(2, 2, 8, 8),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(3, 1, 8, 8),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(3, 2, 7, 8),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(3, 2, 8, 7),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)

                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(3, 2, 5, 5),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(4, 1, 5, 5),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(4, 2, 6, 5),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(4, 2, 5, 6),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)

    def test_missing_info(self):
        """
        Test convolutions for various pieces of missing info.
        """
        self.validate(None, None,
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))
        self.validate((3, 2, None, None), None,
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))
        self.validate((None, 2, None, None), (None, 2, 5, 5),
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))

    def test_full_mode(self):
        """
        Tests basic convolution in full mode and case where filter
        is larger than the input image.
        """
        self.validate((3, 2, 5, 5), (4, 2, 8, 8), 'full')

        def f():
            self.validate((3, 2, 5, 5), (4, 2, 8, 8), 'valid')
        self.assertRaises(Exception, f)

    def test_wrong_input(self):
        """
        Make sure errors are raised when image and kernel are not 4D tensors
        """
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', input=T.dmatrix())
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', filters=T.dvector())
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', input=T.dtensor3())

    def test_gcc_crash(self):
        """
        gcc 4.3.0 20080428 (Red Hat 4.3.0-8)

        crashed in this following case. I changed the c code to don't hit
        gcc bug. So it should not crash anymore
        """
        self.validate((1, 10, 213, 129), (46, 10, 212, 1), 'valid',
                      verify_grad=False)

    def speed(self):
        n_calls = 20000
        print "n_calls", n_calls
        for border_mode in ['valid', 'full']:
            print
            print border_mode
            for openmp in [False, True]:
                print "OpenMP", openmp
                image_shapes = [(1, 5, 6, 6),
                                (10, 5, 6, 6),
                                #(10, 10, 16, 16),
                                #(10, 10, 32, 32)
                ]
                print "image_shape", image_shapes
                for image_shape in image_shapes:
                    filter_shapes = [(1, 5, 4, 4), (2, 5, 4, 4), (5, 5, 4, 4)]
                    print "filter_shapes", filter_shapes
                    for filter_shape in filter_shapes:

                        input = theano.shared(numpy.random.random(image_shape))
                        filters = theano.shared(numpy.random.random(filter_shape))

                        output = conv.conv2d(input, filters,
                                             image_shape, filter_shape,
                                             border_mode,
                                             unroll_patch=True,
                                             openmp=openmp)
                        mode = theano.Mode(linker=theano.gof.vm.VM_Linker(
                            allow_gc=False,
                            use_cloop=True))
                        theano_conv = theano.function([], output, mode=mode)
                        t1 = time.time()
                        theano_conv.fn(n_calls=n_calls)
                        t2 = time.time()
                        print t2 - t1,
                    print

    def test_infer_shape(self):
    # Note: infer_shape is incomplete and thus input and filter shapes
    # must be provided explicitly

        def rand(*shape):
            r = numpy.asarray(numpy.random.rand(*shape), dtype='float64')
            return r * 2 - 1

        adtens = T.dtensor4()
        bdtens = T.dtensor4()
        aivec_val = [2, 2, 3, 3]
        bivec_val = [2, 2, 2, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='valid')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [2, 2, 3, 3]
        bivec_val = [2, 2, 2, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='full')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 8, 8]
        bivec_val = [4, 2, 5, 5]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='valid')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 8, 8]
        bivec_val = [4, 2, 5, 5]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='full')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 7, 5]
        bivec_val = [5, 2, 3, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='valid')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 7, 5]
        bivec_val = [5, 2, 3, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='full')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 7, 5]
        bivec_val = [5, 2, 2, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='valid')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 7, 5]
        bivec_val = [5, 2, 2, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='full')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 3, 3]
        bivec_val = [4, 2, 3, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='valid')], [adtens_val, bdtens_val], conv.ConvOp)

        aivec_val = [3, 2, 3, 3]
        bivec_val = [4, 2, 3, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val,
                border_mode='full')], [adtens_val, bdtens_val], conv.ConvOp)


if __name__ == '__main__':

    t = TestConv2D('setUp')
    t.setUp()
    t.test_infer_shape()
