from __future__ import absolute_import, print_function, division
import unittest
import numpy
import numpy as np
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises, assert_true

import theano
from theano import tensor
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import corr, abstract_conv as conv
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.nnet.abstract_conv import AbstractConv2d
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradWeights
from theano.tensor.nnet.abstract_conv import bilinear_kernel_1D
from theano.tensor.nnet.abstract_conv import bilinear_kernel_2D
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_weights
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs
from theano.tensor.nnet.conv import ConvOp
from theano.tensor.nnet.corr import (CorrMM, CorrMM_gradWeights,
                                     CorrMM_gradInputs)
from theano.tensor.nnet.ConvGrad3D import ConvGrad3D
from theano.tensor.nnet.ConvTransp3D import ConvTransp3D


def conv_corr(inputs, filters, border_mode="valid",
              subsample=(1, 1), conv_mode='conv',
              filter_dilation=(1, 1)):
    if conv_mode == 'conv':
        filters = filters[:, :, ::-1, ::-1]
    return corr.CorrMM(border_mode,
                       subsample,
                       filter_dilation)(inputs, filters)


def conv_corr_gw(inputs, topgrad, filters_shape,
                 border_mode="valid", subsample=(1, 1),
                 conv_mode='conv', filter_dilation=(1, 1)):
    rval = corr.CorrMM_gradWeights(border_mode,
                                   subsample,
                                   filter_dilation)(inputs, topgrad,
                                                    filters_shape[2:])
    if conv_mode == 'conv':
        rval = rval[:, :, ::-1, ::-1]
    return rval


def conv_corr_gi(filters, topgrad, inputs_shape,
                 border_mode="valid", subsample=(1, 1),
                 conv_mode='conv', filter_dilation=(1, 1)):
    if conv_mode == 'conv':
        filters = filters[:, :, ::-1, ::-1]
    return corr.CorrMM_gradInputs(border_mode,
                                  subsample,
                                  filter_dilation)(filters,
                                                   topgrad,
                                                   inputs_shape[2:])


class TestGetConvOutShape(unittest.TestCase):
    def test_basic(self):
        image_shape, kernel_shape = (3, 2, 12, 9), (4, 2, 5, 6)
        sub_sample = (1, 2)
        filter_dilation = (2, 1)
        test1_params = get_conv_output_shape(
            image_shape, kernel_shape, 'valid', sub_sample, filter_dilation)
        test2_params = get_conv_output_shape(
            image_shape, kernel_shape, 'half', sub_sample, filter_dilation)
        test3_params = get_conv_output_shape(
            image_shape, kernel_shape, 'full', sub_sample, filter_dilation)
        test4_params = get_conv_output_shape(
            image_shape, kernel_shape, (1, 2), sub_sample, filter_dilation)

        self.assertTrue(test1_params == (3, 4, 4, 2))
        self.assertTrue(test2_params == (3, 4, 12, 5))
        self.assertTrue(test3_params == (3, 4, 20, 7))
        self.assertTrue(test4_params == (3, 4, 6, 4))


class BaseTestConv2d:
    @classmethod
    def setup_class(cls):
        if theano.config.blas.ldflags == '':
            raise SkipTest("BLAS required for reference")
        cls.inputs_shapes = [(8, 1, 6, 6), (8, 1, 8, 8), (2, 1, 7, 7),
                             (6, 1, 10, 11), (2, 1, 6, 5), (1, 5, 9, 9)]
        cls.filters_shapes = [(5, 1, 2, 2), (4, 1, 3, 3), (2, 1, 3, 3),
                              (1, 1, 2, 3), (4, 1, 1, 3), (4, 5, 3, 2)]
        cls.subsamples = [(1, 1), (2, 2), (2, 4)]
        cls.filters_dilations = [(1, 1), (1, 2), (2, 1)]
        cls.border_modes = ["valid", "full", (0, 0), (1, 1), (5, 5), (5, 2)]
        cls.filter_flip = [True, False]
        cls.provide_shape = [True, False]
        cls.shared = staticmethod(theano.compile.shared)

    def get_output_shape(self, inputs_shape, filters_shape,
                         subsample, border_mode, filter_dilation):
        dil_filters = ((filters_shape[2] - 1) * filter_dilation[0] + 1,
                       (filters_shape[3] - 1) * filter_dilation[1] + 1)
        if border_mode == "valid":
            border_mode = (0, 0)
        if border_mode == "full":
            border_mode = (dil_filters[0] - 1,
                           dil_filters[1] - 1)
        batch_size = inputs_shape[0]
        num_filters = filters_shape[0]
        return ((batch_size, num_filters,) +
                tuple(None if i is None or k is None
                      else ((i + 2 * pad - ((k - 1) * fd + 1)) // d + 1)
                      for i, k, d, pad, fd in zip(inputs_shape[2:],
                                                  filters_shape[2:],
                                                  subsample, border_mode,
                                                  filter_dilation)))

    def run_fwd(self, inputs_shape, filters_shape, ref=conv_corr,
                subsample=(1, 1), verify_grad=True, mode=None,
                border_mode='valid', filter_flip=True,
                provide_shape=False, target_op=None,
                check_trace=False, filter_dilation=(1, 1)):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = self.shared(inputs_val)
        filters = self.shared(filters_val)

        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None
            kshp = None
        if filter_flip:
            conv_mode = 'conv'
        else:
            conv_mode = 'cross'

        c_ref = ref(inputs, filters,
                    border_mode=border_mode,
                    subsample=subsample,
                    conv_mode=conv_mode,
                    filter_dilation=filter_dilation)
        c = conv.conv2d(inputs, filters,
                        border_mode=border_mode,
                        subsample=subsample,
                        filter_flip=filter_flip,
                        input_shape=imshp,
                        filter_shape=kshp,
                        filter_dilation=filter_dilation)

        f_ref = theano.function([], c_ref, mode='FAST_RUN')
        f = theano.function([], c, mode=mode)

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])
            if check_trace:
                assert_true(check_stack_trace(f, ops_to_check=target_op))

        res_ref = numpy.array(f_ref())
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode=border_mode,
                                                imshp=imshp, kshp=kshp,
                                                subsample=subsample,
                                                filter_dilation=filter_dilation),
                            [inputs_val, filters_val],
                            mode=mode)

    def run_gradweight(self, inputs_shape, filters_shape, output_shape,
                       ref=conv_corr_gw, subsample=(1, 1),
                       filter_flip=True, verify_grad=True, mode=None,
                       border_mode='valid', provide_shape=False,
                       target_op=None, check_trace=False,
                       filter_dilation=(1, 1)):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        output_val = numpy.random.random(output_shape).astype('float32')

        inputs = self.shared(inputs_val)
        output = self.shared(output_val)

        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None
            kshp = None
        if filter_flip:
            conv_mode = 'conv'
        else:
            conv_mode = 'cross'
        c = conv.AbstractConv2d_gradWeights(border_mode=border_mode,
                                            filter_flip=filter_flip,
                                            subsample=subsample,
                                            imshp=imshp, kshp=kshp,
                                            filter_dilation=filter_dilation)
        c = c(inputs, output, filters_shape[-2:])
        c_ref = ref(inputs, output,
                    filters_shape,
                    border_mode=border_mode,
                    subsample=subsample,
                    conv_mode=conv_mode,
                    filter_dilation=filter_dilation)
        f = theano.function([], c, mode=mode)
        f_ref = theano.function([], c_ref, mode='FAST_RUN')

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])
            if check_trace:
                assert_true(check_stack_trace(f, ops_to_check=target_op))

        res_ref = numpy.array(f_ref())
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)

        def abstract_conv2d_gradweight(inputs_val, output_val):
            conv_op = conv.AbstractConv2d_gradWeights(border_mode=border_mode,
                                                      subsample=subsample,
                                                      filter_dilation=filter_dilation)
            return conv_op(inputs_val, output_val, filters_shape[-2:])

        if verify_grad:
            utt.verify_grad(abstract_conv2d_gradweight,
                            [inputs_val, output_val],
                            mode=mode, eps=1)

    def run_gradinput(self, inputs_shape, filters_shape, output_shape,
                      ref=conv_corr_gi, subsample=(1, 1), filter_flip=True,
                      verify_grad=True, mode=None, border_mode='valid',
                      provide_shape=False, target_op=None,
                      check_trace=False, filter_dilation=(1, 1)):
        output_val = numpy.random.random(output_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')
        output = self.shared(output_val)
        filters = self.shared(filters_val)

        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None
            kshp = None
        if filter_flip:
            conv_mode = 'conv'
        else:
            conv_mode = 'cross'
        c = conv.AbstractConv2d_gradInputs(border_mode=border_mode,
                                           subsample=subsample,
                                           filter_flip=filter_flip,
                                           imshp=imshp, kshp=kshp,
                                           filter_dilation=filter_dilation)
        c = c(filters, output, inputs_shape[-2:])
        c_ref = ref(filters, output, inputs_shape,
                    border_mode=border_mode, subsample=subsample,
                    conv_mode=conv_mode, filter_dilation=filter_dilation)
        f = theano.function([], c, mode=mode)
        f_ref = theano.function([], c_ref, mode='FAST_RUN')

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])
            if check_trace:
                assert_true(check_stack_trace(f, ops_to_check=target_op))

        res_ref = numpy.array(f_ref())
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)

        def abstract_conv2d_gradinputs(filters_val, output_val):
            conv_op = conv.AbstractConv2d_gradInputs(border_mode=border_mode,
                                                     subsample=subsample,
                                                     filter_dilation=filter_dilation)
            return conv_op(filters_val, output_val, inputs_shape[-2:])

        if verify_grad:
            utt.verify_grad(abstract_conv2d_gradinputs,
                            [filters_val, output_val],
                            mode=mode, eps=1)

    def test_all(self):
        if type(self) is BaseTestConv2d:
            raise SkipTest("base class")
        ds = [1, 1]
        db = (0, 0)
        dflip = True in self.filter_flip
        dprovide_shape = True in self.provide_shape
        for (i, f) in zip(self.inputs_shapes, self.filters_shapes):
            for provide_shape in self.provide_shape:
                yield (self.tcase, i, f, ds, db, dflip, provide_shape)
            for fd in self.filters_dilations:
                for s in self.subsamples:
                    for b in self.border_modes:
                        yield (self.tcase, i, f, s, db, dflip,
                               dprovide_shape, fd)
            for flip in self.filter_flip:
                yield (self.tcase, i, f, ds, db, flip, dprovide_shape)


class TestCorrConv2d(BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        if theano.config.blas.ldflags == "":
            raise SkipTest()
        BaseTestConv2d.setup_class()

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        o = self.get_output_shape(i, f, s, b, fd)
        if (not theano.config.blas.ldflags or
                not theano.config.cxx or
                theano.config.mode == "FAST_COMPILE"):
            raise SkipTest("Need blas to test conv2d")
        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, provide_shape=provide_shape,
                     border_mode=b, filter_flip=flip,
                     target_op=CorrMM, check_trace=True,
                     filter_dilation=fd)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s, verify_grad=True,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip, target_op=CorrMM_gradWeights,
                            check_trace=True, filter_dilation=fd)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s, verify_grad=True,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip, target_op=CorrMM_gradInputs,
                           check_trace=True, filter_dilation=fd)


class TestCpuConv2d(BaseTestConv2d):
    @classmethod
    def setup(cls):
        BaseTestConv2d.setup_class()
        cls.mode = theano.compile.mode.get_default_mode().excluding('conv_gemm')
        cls.opt_err = theano.config.on_opt_error
        theano.config.on_opt_error = 'ignore'

    @classmethod
    def tearDown(cls):
        theano.config.on_opt_error = cls.opt_err

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        if fd != (1, 1):
            raise SkipTest("No dilation implementation for basic cpu ConvOp.")
        mode = self.mode
        o = self.get_output_shape(i, f, s, b, fd)
        fwd_OK = True
        gradweight_OK = True
        gradinput_OK = True

        if not flip:
            fwd_OK = False
            gradweight_OK = False
            gradinput_OK = False

        if b not in ((0, 0), 'valid', 'full'):
            fwd_OK = False
            gradweight_OK = False
            gradinput_OK = False

        if (not provide_shape) and (s != (1, 1)) and (b == 'full'):
            gradweight_OK = False
            gradinput_OK = False

        if ((s[0] not in (1, 2)) or (s[1] not in (1, 2))) and (b == 'full'):
            gradweight_OK = False
            gradinput_OK = False

        if fwd_OK:
            if not theano.config.blas.ldflags:
                raise SkipTest("Need blas to test conv2d")
            self.run_fwd(inputs_shape=i, filters_shape=f,
                         subsample=s, verify_grad=(gradweight_OK and gradinput_OK),
                         mode=mode, provide_shape=provide_shape,
                         border_mode=b, filter_flip=flip, target_op=ConvOp,
                         check_trace=True, filter_dilation=fd)

        else:
            assert_raises(AssertionError,
                          self.run_fwd,
                          inputs_shape=i,
                          filters_shape=f,
                          subsample=s,
                          verify_grad=False,
                          mode=mode,
                          provide_shape=provide_shape,
                          border_mode=b,
                          filter_flip=flip,
                          check_trace=True,
                          filter_dilation=fd)

        if gradweight_OK:
            if not theano.config.blas.ldflags:
                raise SkipTest("Need blas to test conv2d")
            self.run_gradweight(inputs_shape=i, filters_shape=f,
                                output_shape=o, subsample=s,
                                verify_grad=False, mode=mode,
                                provide_shape=provide_shape, border_mode=b,
                                filter_flip=flip,
                                target_op=(ConvOp, ConvGrad3D),
                                check_trace=True,
                                filter_dilation=fd)
        else:
            assert_raises(AssertionError,
                          self.run_gradweight,
                          inputs_shape=i,
                          filters_shape=f,
                          output_shape=o,
                          subsample=s,
                          verify_grad=False,
                          mode=mode,
                          provide_shape=provide_shape,
                          border_mode=b,
                          filter_flip=flip,
                          check_trace=True,
                          filter_dilation=fd)

        if gradinput_OK:
            if not theano.config.blas.ldflags:
                raise SkipTest("Need blas to test conv2d")
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=False, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=(ConvOp, ConvTransp3D),
                               check_trace=True,
                               filter_dilation=fd)
        else:
            assert_raises(AssertionError,
                          self.run_gradinput,
                          inputs_shape=i,
                          filters_shape=f,
                          output_shape=o,
                          subsample=s,
                          verify_grad=False,
                          mode=mode,
                          provide_shape=provide_shape,
                          border_mode=b,
                          filter_flip=flip,
                          check_trace=True,
                          filter_dilation=fd)


def test_constant_shapes():
    # Check that the `imshp` and `kshp` parameters of the AbstractConv Ops
    # are rejected if not constant or None
    dummy_t4 = tensor.ftensor4()
    alloc_dummy_t4 = tensor.zeros((3, 5, 7, 11), dtype='float32')

    dummy_shape = tensor.lvector()
    dummy_one_shape = tensor.ones(4, dtype='int64')
    constant_vec_shape = tensor.constant([3, 5, 7, 11])

    tuple_shape = (3, 5, 7, 11)
    list_shape = list(tuple_shape)
    constant_list_shape = [tensor.constant(i, dtype='int64')
                           for i in tuple_shape]
    constant_tuple_shape = tuple(constant_list_shape)

    bad_shapes = (
        dummy_shape,
        dummy_one_shape,
        dummy_t4.shape,
        alloc_dummy_t4.shape,
        constant_vec_shape,
    )

    good_shapes = (
        constant_list_shape,
        constant_tuple_shape,
        tuple_shape,
        list_shape
    )

    ops_to_test = (
        AbstractConv2d,
        AbstractConv2d_gradInputs,
        AbstractConv2d_gradWeights
    )

    for op in ops_to_test:
        for shp in bad_shapes:
            assert_raises(ValueError, op, imshp=shp)
            assert_raises(ValueError, op, kshp=shp)

        for shp in good_shapes:
            op(imshp=shp)
            op(kshp=shp)


class TestConvTypes(unittest.TestCase):
    def setUp(self):
        self.input = tensor.ftensor4()
        self.filters = tensor.ftensor4()
        self.topgrad = tensor.ftensor4()

        self.constant_tensor = numpy.zeros((3, 5, 7, 11), dtype='float32')

    def test_grad_types(self):
        # This function simply tests the behaviour of the AbstractConv
        # Ops, not their optimizations
        input = self.input
        filters = self.filters
        topgrad = self.topgrad

        out_shape = tensor.lvector()

        output = conv.conv2d(input, filters)
        grad_input, grad_filters = theano.grad(output.sum(),
                                               wrt=(input, filters))
        assert grad_input.type == input.type, (
            grad_input, grad_input.type, input, input.type)
        assert grad_filters.type == filters.type, (
            grad_filters, grad_filters.type, filters, filters.type)

        grad_filters = conv.AbstractConv2d_gradWeights()(
            input, topgrad, out_shape)
        grad_input, grad_topgrad = theano.grad(grad_filters.sum(),
                                               wrt=(input, topgrad))

        assert grad_input.type == input.type, (
            grad_input, grad_input.type, input, input.type)
        assert grad_topgrad.type == topgrad.type, (
            grad_topgrad, grad_topgrad.type, topgrad, topgrad.type)

        grad_input = conv.AbstractConv2d_gradInputs()(
            filters, topgrad, out_shape)
        grad_filters, grad_topgrad = theano.grad(grad_input.sum(),
                                                 wrt=(filters, topgrad))

        assert grad_filters.type == filters.type, (
            grad_filters, grad_filters.type, filters, filters.type)
        assert grad_topgrad.type == topgrad.type, (
            grad_topgrad, grad_topgrad.type, topgrad, topgrad.type)

    def test_constant_input(self):
        # Check the AbstractConv Ops for constant inputs
        input = self.input
        filters = self.filters
        topgrad = self.topgrad
        constant_tensor = self.constant_tensor
        out_shape = tensor.lvector()

        # Check the forward Op
        output = conv.conv2d(constant_tensor, filters)
        grad_filters = theano.grad(output.sum(), wrt=filters)
        assert grad_filters.type == filters.type, (
            grad_filters, grad_filters.type, filters, filters.type)

        output = conv.conv2d(input, constant_tensor)
        grad_input = theano.grad(output.sum(), wrt=input)
        assert grad_input.type == input.type, (
            grad_input, grad_input.type, input, input.type)

        # Check grad wrt weights
        grad_filters = conv.AbstractConv2d_gradWeights()(
            constant_tensor, topgrad, out_shape)
        grad_topgrad = theano.grad(grad_filters.sum(), wrt=topgrad)
        assert grad_topgrad.type == topgrad.type, (
            grad_topgrad, grad_topgrad.type, topgrad, topgrad.type)

        grad_filters = conv.AbstractConv2d_gradWeights()(
            input, constant_tensor, out_shape)
        grad_input = theano.grad(grad_filters.sum(), wrt=input)
        assert grad_input.type == input.type, (
            grad_input, grad_input.type, input, input.type)

        # Check grad wrt inputs
        grad_input = conv.AbstractConv2d_gradInputs()(
            constant_tensor, topgrad, out_shape)
        grad_topgrad = theano.grad(grad_input.sum(), wrt=topgrad)
        assert grad_topgrad.type == topgrad.type, (
            grad_topgrad, grad_topgrad.type, topgrad, topgrad.type)

        grad_input = conv.AbstractConv2d_gradInputs()(
            filters, constant_tensor, out_shape)
        grad_filters = theano.grad(grad_input.sum(), wrt=filters)
        assert grad_filters.type == filters.type, (
            grad_filters, grad_filters.type, filters, filters.type)


class TestBilinearUpsampling(unittest.TestCase):
    # If BLAS is not available on CPU, then we accept the fallback to the
    # slow Python implementation for that test.
    compile_mode = theano.compile.mode.get_default_mode()
    if theano.config.mode == "FAST_COMPILE":
        compile_mode = compile_mode.excluding("conv_gemm")
        compile_mode = compile_mode.excluding('AbstractConvCheck')
    elif not theano.config.blas.ldflags or not theano.config.cxx:
        compile_mode = compile_mode.excluding('AbstractConvCheck')

    def numerical_kernel_1D(self, ratio):
        """Gets numerical 1D kernel for bilinear upsampling"""
        return np.array(list(range(1, ratio + 1)) +
                        list(range(ratio - 1, 0, -1)))

    def numerical_kernel_2D(self, ratio):
        """Gets numerical 2D kernel for bilinear upsampling"""
        return np.array([i * j for i in self.numerical_kernel_1D(ratio) for j
                         in self.numerical_kernel_1D(ratio)]).\
            reshape(2 * ratio - 1, 2 * ratio - 1)

    def test_bilinear_kernel_2D(self):
        """Test 2D kernels used in bilinear upsampling

        This method tests the correctness of the
        2D kernel values used in bilinear upsampling
        for some upsampling ratios.

        """
        for ratio in [2, 3, 4, 5, 6, 7, 8, 9]:
            # getting the un-normalized kernel
            kernel = bilinear_kernel_2D(ratio=ratio, normalize=False)
            f = theano.function([], kernel)
            kernel_2D = self.numerical_kernel_2D(ratio)
            utt.assert_allclose(kernel_2D, f())

            # getting the normalized kernel
            kernel = bilinear_kernel_2D(ratio=ratio, normalize=True)
            f = theano.function([], kernel)
            kernel_2D = kernel_2D / float(ratio**2)
            utt.assert_allclose(kernel_2D, f())

    def test_bilinear_kernel_1D(self):
        """Test 1D kernels used in bilinear upsampling

        This method tests the correctness of the
        1D kernel values used in bilinear upsampling
        for some upsampling ratios.

        """
        rat = tensor.iscalar()
        kernel_ten = bilinear_kernel_1D(ratio=rat, normalize=False)
        f_ten = theano.function([rat], kernel_ten)

        kernel_ten_norm = bilinear_kernel_1D(ratio=rat, normalize=True)
        f_ten_norm = theano.function([rat], kernel_ten_norm)

        for ratio in [2, 3, 4, 5, 6, 7, 8, 9]:
            # getting the un-normalized kernel
            kernel = bilinear_kernel_1D(ratio=ratio, normalize=False)
            f = theano.function([], kernel)
            kernel_1D = self.numerical_kernel_1D(ratio)
            utt.assert_allclose(kernel_1D, f())
            utt.assert_allclose(kernel_1D, f_ten(ratio))

            # getting the normalized kernel
            kernel = bilinear_kernel_1D(ratio=ratio, normalize=True)
            f = theano.function([], kernel)
            kernel_1D = kernel_1D / float(ratio)
            utt.assert_allclose(kernel_1D, f())
            utt.assert_allclose(kernel_1D, f_ten_norm(ratio))

    def numerical_upsampling_multiplier(self, ratio):
        """Compute upsampling multiplier

        This method computes the multipliers of an array
        that will be upsampled using bilinear interpolation.

        Parameters
        ----------
        ratio: int
            the ratio by which the array will be upsampled.

        Returns
        -------
        1D numpy array
            The multiplers that can be used in bilinear interpolation
            to upsample an array.

        int
            The size of the multipliers array

        """
        kern = np.arange(ratio + 1)
        return kern, kern.shape[0]

    def get_upsampled_twobytwo_mat(self, two_by_two, ratio):
        """Upsample 4D array with two rows and two columns

        This method gets a 4D numpy array with two rows and two columns
        and computes its upsampled array by using bilinear interpolation

        Parameters
        ----------
        two_by_two: numpy 4D array
            The array that will be upsampled by bilinear interpolation.
            Array is of shape (batch size, num channels, 2, 2)

        ratio: int
            The ratio by which two_by_two's last
            two dimensions (row and col) will be upsampled.

        Returns
        -------
        4D numpy array
            The array upsampled by using bilinear interpolation. Array
            is of shape (batch size, num channels, 2*ratio, 2*ratio).

        """
        kern, shp = self.numerical_upsampling_multiplier(ratio)
        up_1D = two_by_two[:, :, :, :1] * kern[::-1] + \
            two_by_two[:, :, :, 1:] * kern
        up_2D = up_1D[:, :, :1, :] * kern[::-1][:, np.newaxis] + \
            up_1D[:, :, 1:, :] * kern[:, np.newaxis]
        num_concat = (ratio - 1) // 2
        for i in range(num_concat):
            up_2D = np.concatenate([up_2D[:, :, :1, :], up_2D], axis=2)
            up_2D = np.concatenate([up_2D, up_2D[:, :, -1:, :]], axis=2)
            up_2D = np.concatenate([up_2D[:, :, :, :1], up_2D], axis=3)
            up_2D = np.concatenate([up_2D, up_2D[:, :, :, -1:]], axis=3)
        if ratio % 2 == 0:
            up_2D = np.concatenate([up_2D, up_2D[:, :, -1:, :]], axis=2)
            up_2D = np.concatenate([up_2D, up_2D[:, :, :, -1:]], axis=3)
        return up_2D / float(ratio)**2

    def test_bilinear_upsampling_1D(self):
        """Test bilinear upsampling using 1D kernels

        This method tests the bilinear_upsampling method
        when using 1D kernels for some upsampling ratios.

        """
        # upsampling for a ratio of two
        input_x = np.array([[[[1, 2], [3, 4]]]], dtype=theano.config.floatX)

        for ratio in [2, 3, 4, 5, 6, 7, 8, 9]:
            bilin_mat = bilinear_upsampling(input=input_x, ratio=ratio,
                                            batch_size=1, num_input_channels=1,
                                            use_1D_kernel=True)
            f = theano.function([], bilin_mat, mode=self.compile_mode)
            up_mat_2d = self.get_upsampled_twobytwo_mat(input_x, ratio)
            utt.assert_allclose(f(), up_mat_2d, rtol=1e-06)

    def test_bilinear_upsampling_reshaping(self):
        # Test bilinear upsampling without giving shape information
        # This method tests the bilinear_upsampling method
        # without giving batch_size and num_input_channels
        # upsampling for a ratio of two
        input_x = np.array([[[[1, 2], [3, 4]]]], dtype=theano.config.floatX)

        for ratio in [2, 3]:
            for use_1D_kernel in [True, False]:
                bilin_mat = bilinear_upsampling(input=input_x, ratio=ratio,
                                                batch_size=None,
                                                num_input_channels=None,
                                                use_1D_kernel=use_1D_kernel)
                f = theano.function([], bilin_mat, mode=self.compile_mode)
                up_mat_2d = self.get_upsampled_twobytwo_mat(input_x, ratio)
                utt.assert_allclose(f(), up_mat_2d, rtol=1e-06)

    def test_compare_1D_and_2D_upsampling_values(self):
        """Compare 1D and 2D upsampling

        This method verifies the bilinear upsampling done by using
        1D and 2D kernels will generate the same result.

        """
        # checking upsampling with ratio 5
        input_x = np.random.rand(5, 4, 6, 7).astype(theano.config.floatX)
        mat_1D = bilinear_upsampling(input=input_x, ratio=5,
                                     batch_size=5, num_input_channels=4,
                                     use_1D_kernel=True)
        mat_2D = bilinear_upsampling(input=input_x, ratio=5,
                                     batch_size=5, num_input_channels=4,
                                     use_1D_kernel=False)
        f_1D = theano.function([], mat_1D, mode=self.compile_mode)
        f_2D = theano.function([], mat_2D, mode=self.compile_mode)
        utt.assert_allclose(f_1D(), f_2D(), rtol=1e-06)

        # checking upsampling with ratio 8
        input_x = np.random.rand(12, 11, 10, 7).astype(theano.config.floatX)
        mat_1D = bilinear_upsampling(input=input_x, ratio=8,
                                     batch_size=12, num_input_channels=11,
                                     use_1D_kernel=True)
        mat_2D = bilinear_upsampling(input=input_x, ratio=8,
                                     batch_size=12, num_input_channels=11,
                                     use_1D_kernel=False)
        f_1D = theano.function([], mat_1D, mode=self.compile_mode)
        f_2D = theano.function([], mat_2D, mode=self.compile_mode)
        utt.assert_allclose(f_1D(), f_2D(), rtol=1e-06)


def test_conv2d_grad_wrt_inputs():

    inputs_shapes = [(8, 1, 12, 12), (8, 1, 18, 18), (2, 1, 4, 4),
                     (6, 1, 10, 11), (2, 1, 6, 5), (1, 5, 9, 9)]
    filters_shapes = [(5, 1, 2, 2), (4, 1, 3, 3), (2, 1, 3, 3),
                      (1, 1, 2, 5), (4, 1, 2, 2), (4, 5, 2, 2)]
    subsamples = [(1, 1), (2, 2)]
    border_modes = ["valid", "full"]
    filter_flip = [True, False]
    x = T.cast(T.ftensor4('x'), theano.config.floatX)
    for (in_shape, fltr_shape) in zip(inputs_shapes, filters_shapes):
        input_val = np.random.rand(*in_shape).astype(theano.config.floatX)

        filter_val = np.random.rand(*fltr_shape).astype(theano.config.floatX)
        for bm in border_modes:
            for ss in subsamples:
                for ff in filter_flip:
                    cgi = conv2d_grad_wrt_inputs(output_grad=input_val,
                                                 filters=filter_val,
                                                 filter_shape=fltr_shape,
                                                 input_shape=in_shape,
                                                 border_mode=bm,
                                                 subsample=ss,
                                                 filter_flip=ff)

                    conv_func = T.nnet.conv.conv2d(x,
                                                   filters=filter_val,
                                                   border_mode=bm,
                                                   filter_shape=fltr_shape,
                                                   image_shape=in_shape,
                                                   subsample=ss)

                    f_prime = theano.grad(conv_func.sum(), x)
                    f = theano.function([x], f_prime)
                    T.allclose(cgi, f(input_val))


def test_conv2d_grad_wrt_weights():

    inputs_shapes = [(8, 1, 12, 12), (8, 1, 18, 18), (2, 1, 4, 4),
                     (6, 1, 10, 11), (2, 1, 6, 5), (1, 5, 9, 9)]
    filters_shapes = [(5, 1, 2, 2), (4, 1, 3, 3), (2, 1, 3, 3),
                      (1, 1, 2, 5), (4, 1, 2, 2), (4, 5, 2, 2)]
    subsamples = [(1, 1), (2, 2)]
    border_modes = ["valid", "full"]
    filter_flip = [True, False]
    w = T.cast(T.ftensor4('w'), theano.config.floatX)
    for (in_shape, fltr_shape) in zip(inputs_shapes, filters_shapes):
        input_val = np.random.rand(*in_shape).astype(theano.config.floatX)
        filter_val = np.random.rand(*fltr_shape).astype(theano.config.floatX)

        for bm in border_modes:
            for ss in subsamples:
                for ff in filter_flip:
                    cgw = conv2d_grad_wrt_weights(input_val,
                                                  output_grad=filter_val,
                                                  filter_shape=fltr_shape,
                                                  input_shape=in_shape,
                                                  border_mode=bm,
                                                  subsample=ss,
                                                  filter_flip=ff)

                    conv_func = T.nnet.conv.conv2d(input_val,
                                                   filters=w,
                                                   border_mode=bm,
                                                   filter_shape=fltr_shape,
                                                   image_shape=in_shape,
                                                   subsample=ss)

                    f_prime = theano.grad(conv_func.sum(), w)
                    f = theano.function([w], f_prime)
                    T.allclose(cgw, f(filter_val))
