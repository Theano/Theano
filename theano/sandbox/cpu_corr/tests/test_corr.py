from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
import numpy

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import conv
from theano.tensor.basic import _allclose


class TestCorr2D(utt.InferShapeTester):
    mode = None
    dtype = theano.config.floatX

    def setUp(self):
        super(TestCorr2D, self).setUp()
        self.input = T.tensor4('input', dtype=self.dtype)
        self.input.name = 'default_V'
        self.filters = T.tensor4('filters', dtype=self.dtype)
        self.filters.name = 'default_filters'
        if not conv.imported_scipy_signal and theano.config.cxx == "":
            raise SkipTest("conv2d tests need SciPy or a c++ compiler")

    def validate(self, image_shape, filter_shape,
                 border_mode='valid', subsample=(1, 1),
                 input=None, filters=None,
                 verify_grad=True, should_raise=False):
        """
        :param image_shape: The constant shape info passed to corrMM.
        :param filter_shape: The constant shape info passed to corrMM.
        """
        N_image_shape = [T.get_scalar_constant_value(T.as_tensor_variable(x))
                         for x in image_shape]
        N_filter_shape = [T.get_scalar_constant_value(T.as_tensor_variable(x))
                          for x in filter_shape]

        if input is None:
            input = self.input
        if filters is None:
            filters = self.filters

        # THEANO IMPLEMENTATION

        # we create a symbolic function so that verify_grad can work
        def sym_CpuCorrMM(input, filters):
            # define theano graph and function
            input.name = 'input'
            filters.name = 'filters'
            rval = conv.CpuCorrMM(border_mode, subsample)(input, filters)
            rval.name = 'corr_output'
            return rval

        output = sym_CpuCorrMM(input, filters)
        output.name = 'CpuCorrMM()(%s,%s)' % (input.name, filters.name)
        theano_corr = theano.function([input, filters], output, mode=self.mode)

        # initialize input and compute result
        image_data = numpy.random.random(N_image_shape).astype(self.dtype)
        filter_data = numpy.random.random(N_filter_shape).astype(self.dtype)
        try:
            theano_output = theano_corr(image_data, filter_data)
        except ValueError:
            if not should_raise:
                raise
            return
        else:
            if should_raise:
                raise Exception("CorrOp should have generated an error")

        # REFERENCE IMPLEMENTATION
        # Testing correlation, not convolution. Reverse filters.
        filter_data_corr = numpy.array(filter_data[:, :, ::-1, ::-1],
                                       copy=True,
                                       order='C')
        s = 1.
        orig_image_data = image_data
        if border_mode is not 'full':
            s = -1.
        out_shape2d = (numpy.array(N_image_shape[-2:]) +
                       s * numpy.array(N_filter_shape[-2:]) - s)
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
                    filter2d = filter_data_corr[nn, im0, :, :]
                    image2d = image_data[bb, im0, :, :]
                    for row in range(ref_output.shape[2]):
                        irow = row * subsample[0]  # image row
                        for col in range(ref_output.shape[3]):
                            icol = col * subsample[1]  # image col
                            ref_output[bb, nn, row, col] += (image2d[
                                irow:irow + N_filter_shape[2],
                                icol:icol + N_filter_shape[3]] * filter2d[::-1, ::-1]
                            ).sum()

        self.assertTrue(_allclose(theano_output, ref_output))

        # TEST GRADIENT
        if verify_grad:
            utt.verify_grad(sym_CpuCorrMM, [orig_image_data, filter_data])

    def test_basic(self):
        """
        Tests that basic correlations work for odd and even
        dimensions of image and filter shapes, as well as rectangular
        images and filters.
        """
        self.validate((2, 2, 3, 3), (2, 2, 2, 2), 'valid', verify_grad=False)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'valid', verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid')
        self.validate((3, 2, 7, 5), (5, 2, 3, 2), 'valid', verify_grad=False)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'full', verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full')
        self.validate((1, 10, 213, 129), (46, 10, 212, 1), 'valid',
                      verify_grad=False)

    def test_img_kernel_same_shape(self):
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'full')
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid')

    @attr('slow')
    def test_subsample(self):
        """
        Tests correlation where subsampling != (1,1)
        TODO
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'valid', subsample=(3, 3))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'full', subsample=(3, 3))

    def test_shape_Constant_tensor(self):
        """
        Tests correlation where the {image,filter}_shape is a Constant tensor.
        """
        as_t = T.as_tensor_variable
        self.validate((as_t(3), as_t(2), as_t(7), as_t(5)),
                      (5, 2, 2, 3), 'valid')
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
        self.assertRaises(ValueError, self.validate,
                          (3, 2, 8, 8), (4, 3, 5, 5),
                          'valid')

    def test_full_mode(self):
        """
        Tests basic correlation in full mode and case where filter
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

    def test_infer_shape(self):

        def rand(*shape):
            r = numpy.asarray(numpy.random.rand(*shape), dtype='float64')
            return r * 2 - 1
        corr = conv.CpuCorrMM

        adtens = T.dtensor4()
        bdtens = T.dtensor4()
        aivec_val = [4, 5, 6, 3]
        bivec_val = [7, 5, 3, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='valid')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='full')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        aivec_val = [6, 2, 8, 3]
        bivec_val = [4, 2, 5, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='valid')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='full')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        aivec_val = [3, 6, 7, 5]
        bivec_val = [5, 6, 3, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='valid')(adtens,
                                                           bdtens)],
                                [adtens_val, bdtens_val], corr)

        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='full')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        aivec_val = [3, 6, 7, 5]
        bivec_val = [5, 6, 2, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='valid')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='full')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        aivec_val = [5, 2, 4, 3]
        bivec_val = [6, 2, 4, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='valid')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)

        self._compile_and_check([adtens, bdtens],
                                [corr(border_mode='full')(adtens, bdtens)],
                                [adtens_val, bdtens_val], corr)


if __name__ == '__main__':

    t = TestCorr2D('setUp')
    t.setUp()
    t.test_infer_shape()
