from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
from nose.tools import assert_equals
import numpy as np
from six import integer_types

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import corr, conv
from theano.tensor.nnet.tests.test_abstract_conv import Grouped_conv_noOptim, TestUnsharedConv
from theano.tensor.nnet.tests.test_abstract_conv import TestAsymmetricPadding, TestCausalConv


class TestCorr2D(utt.InferShapeTester):
    if theano.config.mode == "FAST_COMPILE":
        mode = theano.compile.get_mode("FAST_RUN")
    else:
        mode = None
    dtype = theano.config.floatX

    def setUp(self):
        super(TestCorr2D, self).setUp()
        self.input = T.tensor4('input', dtype=self.dtype)
        self.input.name = 'default_V'
        self.filters = T.tensor4('filters', dtype=self.dtype)
        self.filters.name = 'default_filters'
        if not conv.imported_scipy_signal and theano.config.cxx == "":
            raise SkipTest("CorrMM tests need SciPy or a c++ compiler")
        # This tests can run even when theano.config.blas.ldflags is empty.

    def validate(self, image_shape, filter_shape,
                 border_mode='valid', subsample=(1, 1),
                 input=None, filters=None, verify_grad=True,
                 non_contiguous=False, filter_dilation=(1, 1)):
        """
        :param image_shape: The constant shape info passed to corrMM.
        :param filter_shape: The constant shape info passed to corrMM.
        """
        if not theano.config.cxx:
            raise SkipTest("Need cxx to test conv2d")
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
        def sym_CorrMM(input, filters):
            # define theano graph and function
            input.name = 'input'
            filters.name = 'filters'
            rval = corr.CorrMM(border_mode, subsample,
                               filter_dilation)(input, filters)
            rval.name = 'corr_output'
            return rval

        output = sym_CorrMM(input, filters)
        output.name = 'CorrMM()(%s,%s)' % (input.name, filters.name)
        theano_corr = theano.function([input, filters], output, mode=self.mode)

        # initialize input and compute result
        image_data = np.random.random(N_image_shape).astype(self.dtype)
        filter_data = np.random.random(N_filter_shape).astype(self.dtype)
        if non_contiguous:
            image_data = np.transpose(image_data, axes=(0, 1, 3, 2))
            image_data = image_data.copy()
            image_data = np.transpose(image_data, axes=(0, 1, 3, 2))
            filter_data = np.transpose(filter_data, axes=(0, 1, 3, 2))
            filter_data = filter_data.copy()
            filter_data = np.transpose(filter_data, axes=(0, 1, 3, 2))
            assert not image_data.flags['CONTIGUOUS']
            assert not filter_data.flags['CONTIGUOUS']

        theano_output = theano_corr(image_data, filter_data)

        # REFERENCE IMPLEMENTATION
        # Testing correlation, not convolution. Reverse filters.
        filter_data_corr = np.array(filter_data[:, :, ::-1, ::-1],
                                    copy=True,
                                    order='C')
        orig_image_data = image_data
        img_shape2d = np.array(N_image_shape[-2:])
        fil_shape2d = np.array(N_filter_shape[-2:])
        dil_shape2d = np.array(filter_dilation)
        dil_fil_shape2d = (fil_shape2d - 1) * dil_shape2d + 1
        subsample2d = np.array(subsample)
        if border_mode == 'full':
            padHW = (dil_fil_shape2d - 1)
        elif border_mode == 'valid':
            padHW = np.array([0, 0])
        elif border_mode == 'half':
            padHW = np.floor(dil_fil_shape2d / 2).astype('int32')
        elif isinstance(border_mode, tuple):
            padHW = np.array(border_mode)
        elif isinstance(border_mode, integer_types):
            padHW = np.array([border_mode, border_mode])
        else:
            raise NotImplementedError('Unsupported border_mode {}'.format(border_mode))
        out_shape2d = np.floor((img_shape2d + 2 * (padHW) - dil_fil_shape2d) / subsample2d) + 1
        # avoid numpy deprecation
        out_shape2d = out_shape2d.astype('int32')
        out_shape = (N_image_shape[0], N_filter_shape[0]) + tuple(out_shape2d)
        ref_output = np.zeros(out_shape)

        # loop over output feature maps
        ref_output.fill(0)
        image_data2 = np.zeros((N_image_shape[0], N_image_shape[1],
                                N_image_shape[2] + 2 * padHW[0],
                                N_image_shape[3] + 2 * padHW[1]))
        image_data2[:, :, padHW[0]:padHW[0] + N_image_shape[2],
                    padHW[1]:padHW[1] + N_image_shape[3]] = image_data
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
                                irow:irow + dil_fil_shape2d[0]:filter_dilation[0],
                                icol:icol + dil_fil_shape2d[1]:filter_dilation[1]] * filter2d[::-1, ::-1]
                            ).sum()

        utt.assert_allclose(ref_output, theano_output)

        # TEST GRADIENT
        if verify_grad:
            utt.verify_grad(sym_CorrMM, [orig_image_data, filter_data],
                            mode=self.mode)

    @attr('slow')
    def test_basic(self):
        # Tests that basic correlations work for odd and even
        # dimensions of image and filter shapes, as well as rectangular
        # images and filters.

        border_modes = ['valid', 'full', 'half', (1, 1), (2, 1), (1, 2),
                        (3, 3), 1]
        img_shapes = [(2, 2, 3, 3), (3, 2, 8, 8), (3, 2, 7, 5), (3, 2, 7, 5),
                      (3, 2, 8, 8), (3, 2, 7, 5)]
        fil_shapes = [(2, 2, 2, 2), (4, 2, 5, 5), (5, 2, 2, 3), (5, 2, 3, 2),
                      (4, 2, 5, 5), (5, 2, 2, 3)]

        for border_mode in border_modes:
            for img, fil in zip(img_shapes, fil_shapes):
                self.validate(img, fil, border_mode, verify_grad=False)

        # Very slow on with 'full' or 'half'
        self.validate((1, 10, 213, 129), (46, 10, 212, 1), 'valid', verify_grad=False)

    def test_img_kernel_same_shape(self):
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'full')
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid')
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'half')
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), (1, 1))
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 1)

    @attr('slow')
    def test_subsample(self):
        # Tests correlation where subsampling != (1,1)

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'valid', subsample=(3, 3))

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'full', subsample=(3, 3))

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'half', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'half', subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'half', subsample=(3, 3))

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), (1, 1), subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), (2, 1), subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), (1, 2), subsample=(3, 3))

        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 1, subsample=(3, 3))

    def test_filter_dilation(self):
        # Tests correlation where filter dilation != (1,1)

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', filter_dilation=(2, 2))
        self.validate((3, 2, 14, 10), (5, 2, 2, 3), 'valid', filter_dilation=(3, 1))
        self.validate((1, 1, 14, 14), (1, 1, 3, 3), 'valid', filter_dilation=(2, 3))

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', filter_dilation=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', filter_dilation=(3, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'full', filter_dilation=(2, 3))

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'half', filter_dilation=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'half', filter_dilation=(3, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'half', filter_dilation=(2, 3))

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), (1, 1), filter_dilation=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), (2, 1), filter_dilation=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), (1, 2), filter_dilation=(1, 2))

        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 1, subsample=(3, 3), filter_dilation=(2, 2))

    @attr('slow')
    def test_shape_Constant_tensor(self):
        # Tests correlation where the {image,filter}_shape is a Constant tensor.

        as_t = T.as_tensor_variable
        border_modes = ['valid', 'full', 'half', (1, 1), (2, 1), (1, 2), (3, 3), 1]

        for border_mode in border_modes:
            self.validate((as_t(3), as_t(2), as_t(7), as_t(5)),
                          (5, 2, 2, 3), border_mode)
            self.validate(as_t([3, 2, 7, 5]), (5, 2, 2, 3), border_mode)
            self.validate(as_t((3, 2, 7, 5)), (5, 2, 2, 3), border_mode)
            self.validate((3, 2, 7, 5), (as_t(5), as_t(2), as_t(2),
                          as_t(3)), 'valid')
            self.validate((3, 2, 7, 5), as_t([5, 2, 2, 3]), border_mode)
            self.validate(as_t([3, 2, 7, 5]), as_t([5, 2, 2, 3]), border_mode)

    def test_invalid_filter_shape(self):
        # Tests scenario where filter_shape[1] != input_shape[1]

        self.assertRaises(ValueError, self.validate,
                          (3, 2, 8, 8), (4, 3, 5, 5),
                          'valid')

    def test_full_mode(self):
        # Tests basic correlation in full mode and case where filter
        # is larger than the input image.

        self.validate((3, 2, 5, 5), (4, 2, 8, 8), 'full')

        def f():
            self.validate((3, 2, 5, 5), (4, 2, 8, 8), 'valid')
        self.assertRaises(Exception, f)

    def test_wrong_input(self):
        # Make sure errors are raised when image and kernel are not 4D tensors

        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', input=T.dmatrix())
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', filters=T.dvector())
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', input=T.dtensor3())

    def test_dtype_upcast(self):
        # Checks dtype upcast for CorrMM methods.

        def rand(shape, dtype='float64'):
            r = np.asarray(np.random.rand(*shape), dtype=dtype)
            return r * 2 - 1
        if not theano.config.cxx:
            raise SkipTest("Need cxx to test conv2d")

        ops = [corr.CorrMM, corr.CorrMM_gradWeights, corr.CorrMM_gradInputs]
        a_shapes = [[4, 5, 6, 3], [1, 5, 6, 3], [1, 5, 6, 3]]
        b_shapes = [[7, 5, 3, 2], [1, 5, 3, 1], [7, 1, 3, 1]]
        dtypes = ['float32', 'float64']

        for op, a_shape, b_shape in zip(ops, a_shapes, b_shapes):
            for a_dtype in dtypes:
                for b_dtype in dtypes:
                    c_dtype = theano.scalar.upcast(a_dtype, b_dtype)
                    a_tens = T.tensor4(dtype=a_dtype)
                    b_tens = T.tensor4(dtype=b_dtype)
                    a_tens_val = rand(a_shape, dtype=a_dtype)
                    b_tens_val = rand(b_shape, dtype=b_dtype)

                    c_tens = op()(a_tens, b_tens)
                    f = theano.function([a_tens, b_tens], c_tens, mode=self.mode)
                    assert_equals(f(a_tens_val, b_tens_val).dtype, c_dtype)

    @attr('slow')
    def test_infer_shape_forward(self):
        if theano.config.mode == "FAST_COMPILE":
            raise SkipTest("CorrMM don't work in FAST_COMPILE")
        if not theano.config.cxx:
            raise SkipTest("Need cxx for this test")

        def rand(*shape):
            r = np.asarray(np.random.rand(*shape), dtype='float64')
            return r * 2 - 1
        corrMM = corr.CorrMM

        adtens = T.dtensor4()
        bdtens = T.dtensor4()
        aivec_vals = [[4, 5, 6, 3], [6, 2, 8, 3], [3, 6, 7, 5],
                      [3, 6, 7, 5], [5, 2, 4, 3]]
        bivec_vals = [[7, 5, 3, 2], [4, 2, 5, 3], [5, 6, 3, 2],
                      [5, 6, 2, 3], [6, 2, 4, 3]]
        modes = ['valid', 'full', 'half', (1, 1), (2, 1), (1, 2), 1]
        subsamples = [(1, 1), (2, 1), (1, 2)]

        for aivec_val, bivec_val in zip(aivec_vals, bivec_vals):
            adtens_val = rand(*aivec_val)
            bdtens_val = rand(*bivec_val)
            for mode in modes:
                for subsample in subsamples:
                    # CorrMM
                    cdtens = corrMM(border_mode=mode, subsample=subsample)(adtens, bdtens)
                    self._compile_and_check([adtens, bdtens],
                                            [cdtens],
                                            [adtens_val, bdtens_val], corrMM,
                                            warn=False)

    @attr('slow')
    def test_infer_shape_gradW(self):
        if theano.config.mode == "FAST_COMPILE":
            raise SkipTest("CorrMM don't work in FAST_COMPILE")
        if not theano.config.cxx:
            raise SkipTest("Need cxx for this test")

        def rand(*shape):
            r = np.asarray(np.random.rand(*shape), dtype='float64')
            return r * 2 - 1
        corrMM = corr.CorrMM
        gradW = corr.CorrMM_gradWeights

        adtens = T.dtensor4()
        bdtens = T.dtensor4()
        aivec_vals = [[1, 5, 6, 3], [8, 2, 7, 3], [1, 6, 9, 4],
                      [9, 6, 8, 5], [9, 1, 6, 8]]
        bivec_vals = [[7, 5, 3, 1], [4, 2, 5, 3], [12, 6, 3, 2],
                      [5, 6, 1, 3], [11, 1, 3, 3]]
        modes = ['valid', 'full', 'half', (1, 1), (2, 1), (1, 2), 1]
        subsamples = [(1, 1), (2, 1), (1, 2)]

        for aivec_val, bivec_val in zip(aivec_vals, bivec_vals):
            adtens_val = rand(*aivec_val)
            bdtens_val = rand(*bivec_val)
            for mode in modes:
                for subsample in subsamples:
                    # CorrMM
                    cdtens = corrMM(border_mode=mode, subsample=subsample)(adtens, bdtens)
                    f = theano.function([adtens, bdtens], cdtens)
                    cdtens_val = f(adtens_val, bdtens_val)
                    # CorrMM_gradWeights
                    shape = (theano.shared(bivec_val[2]), theano.shared(bivec_val[3]))
                    bdtens_g = gradW(border_mode=mode,
                                     subsample=subsample)(adtens, cdtens, shape=shape)
                    self._compile_and_check([adtens, cdtens],
                                            [bdtens_g],
                                            [adtens_val, cdtens_val], gradW,
                                            warn=False)

    @attr('slow')
    def test_infer_shape_gradI(self):
        if theano.config.mode == "FAST_COMPILE":
            raise SkipTest("CorrMM don't work in FAST_COMPILE")
        if not theano.config.cxx:
            raise SkipTest("Need cxx for this test")

        def rand(*shape):
            r = np.asarray(np.random.rand(*shape), dtype='float64')
            return r * 2 - 1
        corrMM = corr.CorrMM
        gradI = corr.CorrMM_gradInputs

        adtens = T.dtensor4()
        bdtens = T.dtensor4()
        aivec_vals = [[1, 5, 6, 3], [8, 2, 7, 3], [1, 6, 9, 4],
                      [9, 6, 8, 5], [9, 1, 6, 8]]
        bivec_vals = [[7, 5, 3, 1], [4, 2, 5, 3], [12, 6, 3, 2],
                      [5, 6, 1, 3], [7, 1, 3, 4]]
        modes = ['valid', 'full', 'half', (1, 1), (2, 1), (1, 2), 1]
        subsamples = [(1, 1), (2, 1), (1, 2)]

        for aivec_val, bivec_val in zip(aivec_vals, bivec_vals):
            adtens_val = rand(*aivec_val)
            bdtens_val = rand(*bivec_val)
            for mode in modes:
                for subsample in subsamples:
                    # CorrMM
                    cdtens = corrMM(border_mode=mode, subsample=subsample)(adtens, bdtens)
                    f = theano.function([adtens, bdtens], cdtens)
                    cdtens_val = f(adtens_val, bdtens_val)
                    # CorrMM_gradInputs
                    shape = (theano.shared(aivec_val[2]), theano.shared(aivec_val[3]))
                    adtens_g = gradI(border_mode=mode,
                                     subsample=subsample)(bdtens, cdtens, shape=shape)
                    self._compile_and_check([bdtens, cdtens],
                                            [adtens_g],
                                            [bdtens_val, cdtens_val], gradI,
                                            warn=False)

    def test_non_contiguous(self):
        self.validate((2, 2, 3, 3), (2, 2, 2, 2), 'valid', non_contiguous=True)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'valid', non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 3, 2), 'valid', non_contiguous=True)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'full', non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', non_contiguous=True)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'half', non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'half', non_contiguous=True)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), (1, 1), non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), (1, 2), non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), (2, 1), non_contiguous=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 2, non_contiguous=True)


class TestGroupCorr2d(Grouped_conv_noOptim):
    mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
    conv_op = corr.CorrMM
    conv_gradw_op = corr.CorrMM_gradWeights
    conv_gradi_op = corr.CorrMM_gradInputs

    def test_graph(self):
        # define common values  first
        groups = 3
        bottom = np.random.rand(3, 6, 5, 5).astype(theano.config.floatX)
        kern = np.random.rand(9, 2, 3, 3).astype(theano.config.floatX)
        bottom_sym = T.tensor4('bottom')
        kern_sym = T.tensor4('kern')

        # grouped convolution graph
        conv_group = self.conv(num_groups=groups)(bottom_sym, kern_sym)
        gconv_func = theano.function([bottom_sym, kern_sym], conv_group, mode=self.mode)

        # Graph for the normal hard way
        kern_offset = kern_sym.shape[0] // groups
        bottom_offset = bottom_sym.shape[1] // groups
        split_conv_output = [self.conv()(bottom_sym[:, i * bottom_offset:(i + 1) * bottom_offset, :, :],
                             kern_sym[i * kern_offset:(i + 1) * kern_offset, :, :, :])
                             for i in range(groups)]
        concatenated_output = T.concatenate(split_conv_output, axis=1)
        conv_func = theano.function([bottom_sym, kern_sym], concatenated_output, mode=self.mode)

        # calculate outputs for each graph
        gconv_output = gconv_func(bottom, kern)
        conv_output = conv_func(bottom, kern)

        # compare values
        utt.assert_allclose(gconv_output, conv_output)


class TestUnsharedCorr2d(TestUnsharedConv):
    if theano.config.mode == "FAST_COMPILE":
        mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
    else:
        mode = None
    conv2d_op = corr.CorrMM
    conv2d_gradw_op = corr.CorrMM_gradWeights
    conv2d_gradi_op = corr.CorrMM_gradInputs


class TestAsymmetricCorr(TestAsymmetricPadding):
    if theano.config.mode == "FAST_COMPILE":
        mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
    else:
        mode = None
    conv2d_op = corr.CorrMM
    conv2d_gradw_op = corr.CorrMM_gradWeights
    conv2d_gradi_op = corr.CorrMM_gradInputs


class TestCausalCorr(TestCausalConv):
    if theano.config.mode == "FAST_COMPILE":
        mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
    else:
        mode = None


if __name__ == '__main__':

    t = TestCorr2D('setUp')
    t.setUp()
    t.test_infer_shape()
