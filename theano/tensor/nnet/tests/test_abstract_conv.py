from __future__ import absolute_import, print_function, division
import unittest
import numpy
import numpy as np
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises, assert_true

import theano
from theano import tensor
from theano.gof.opt import check_stack_trace
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import (corr, corr3d, conv2d_transpose,
                                abstract_conv as conv)
from theano.tensor.nnet.abstract_conv import (get_conv_output_shape,
                                              get_conv_gradweights_shape,
                                              get_conv_gradinputs_shape,
                                              check_conv_gradinputs_shape,
                                              assert_conv_shape,
                                              assert_shape)
from theano.tensor.nnet.abstract_conv import AbstractConv2d
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs
from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradWeights
from theano.tensor.nnet.abstract_conv import bilinear_kernel_1D
from theano.tensor.nnet.abstract_conv import bilinear_kernel_2D
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
from theano.tensor.nnet.conv import ConvOp
from theano.tensor.nnet.corr import (CorrMM, CorrMM_gradWeights,
                                     CorrMM_gradInputs)
from theano.tensor.nnet.corr3d import (Corr3dMM, Corr3dMM_gradWeights,
                                       Corr3dMM_gradInputs)
from theano.tensor.nnet.Conv3D import Conv3D
from theano.tensor.nnet.ConvGrad3D import ConvGrad3D
from theano.tensor.nnet.ConvTransp3D import ConvTransp3D


def conv2d_corr(inputs, filters, border_mode="valid",
                subsample=(1, 1), conv_mode='conv',
                filter_dilation=(1, 1)):
    if conv_mode == 'conv':
        filters = filters[:, :, ::-1, ::-1]
    return corr.CorrMM(border_mode,
                       subsample,
                       filter_dilation)(inputs, filters)


def conv2d_corr_gw(inputs, topgrad, filters_shape,
                   border_mode="valid", subsample=(1, 1),
                   conv_mode='conv', filter_dilation=(1, 1)):
    rval = corr.CorrMM_gradWeights(border_mode,
                                   subsample,
                                   filter_dilation)(inputs, topgrad,
                                                    filters_shape[2:])
    if conv_mode == 'conv':
        rval = rval[:, :, ::-1, ::-1]
    return rval


def conv2d_corr_gi(filters, topgrad, inputs_shape,
                   border_mode="valid", subsample=(1, 1),
                   conv_mode='conv', filter_dilation=(1, 1)):
    if conv_mode == 'conv':
        filters = filters[:, :, ::-1, ::-1]
    return corr.CorrMM_gradInputs(border_mode,
                                  subsample,
                                  filter_dilation)(filters,
                                                   topgrad,
                                                   inputs_shape[2:])


def conv3d_corr(inputs, filters, border_mode="valid",
                subsample=(1, 1, 1), conv_mode='conv',
                filter_dilation=(1, 1, 1)):
    if conv_mode == 'conv':
        filters = filters[:, :, ::-1, ::-1, ::-1]
    return corr3d.Corr3dMM(border_mode,
                           subsample,
                           filter_dilation)(inputs, filters)


def conv3d_corr_gw(inputs, topgrad, filters_shape,
                   border_mode="valid", subsample=(1, 1, 1),
                   conv_mode='conv', filter_dilation=(1, 1, 1)):
    rval = corr3d.Corr3dMM_gradWeights(border_mode,
                                       subsample,
                                       filter_dilation)(inputs, topgrad,
                                                        filters_shape[2:])
    if conv_mode == 'conv':
        rval = rval[:, :, ::-1, ::-1, ::-1]
    return rval


def conv3d_corr_gi(filters, topgrad, inputs_shape,
                   border_mode="valid", subsample=(1, 1, 1),
                   conv_mode='conv', filter_dilation=(1, 1, 1)):
    if conv_mode == 'conv':
        filters = filters[:, :, ::-1, ::-1, ::-1]
    return corr3d.Corr3dMM_gradInputs(border_mode,
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

    def test_basic_3d(self):
        image_shape, kernel_shape = (3, 2, 12, 9, 7), (4, 2, 5, 6, 4)
        sub_sample = (1, 2, 1)
        filter_dilation = (2, 1, 1)
        test1_params = get_conv_output_shape(
            image_shape, kernel_shape, 'valid', sub_sample, filter_dilation)
        test2_params = get_conv_output_shape(
            image_shape, kernel_shape, 'half', sub_sample, filter_dilation)
        test3_params = get_conv_output_shape(
            image_shape, kernel_shape, 'full', sub_sample, filter_dilation)
        test4_params = get_conv_output_shape(
            image_shape, kernel_shape, (1, 2, 3), sub_sample, filter_dilation)

        self.assertTrue(test1_params == (3, 4, 4, 2, 4))
        self.assertTrue(test2_params == (3, 4, 12, 5, 8))
        self.assertTrue(test3_params == (3, 4, 20, 7, 10))
        self.assertTrue(test4_params == (3, 4, 6, 4, 10))


class TestConvGradInputsShape(unittest.TestCase):
    def test_check_shape(self):
        for i in range(1, 20):
            for k in range(1, 10):
                for b in ('valid', 'half', 'full', (0, 2)):
                    for s in (1, 2, 3):
                        for d in (1, 2, 3):
                            image_shape = (59, 61, i, i)
                            kernel_shape = (67, 61, k, k)

                            # compute the output that these inputs and parameters would produce
                            computed_shape = get_conv_output_shape(
                                image_shape, kernel_shape, b, (s, s), (d, d))
                            # this should be accepted
                            self.assertTrue(check_conv_gradinputs_shape(
                                image_shape, kernel_shape, computed_shape, b, (s, s), (d, d)))

                            # one or more None should also be accepted
                            trial_shape = (None, None, computed_shape[2], None)
                            self.assertTrue(check_conv_gradinputs_shape(
                                image_shape, kernel_shape, trial_shape, b, (s, s), (d, d)))

                            # the batch size and number of filters are important
                            trial_shape = (1, 1, computed_shape[2], computed_shape[3])
                            self.assertFalse(check_conv_gradinputs_shape(
                                image_shape, kernel_shape, trial_shape, b, (s, s), (d, d)))

                            # outputs that are too large or too small should be rejected
                            for o in (-3, -2, -1, 1, 2, 3):
                                trial_shape = (computed_shape[0], computed_shape[1],
                                               computed_shape[2] + o, computed_shape[3] + o)
                                self.assertFalse(check_conv_gradinputs_shape(
                                    image_shape, kernel_shape, trial_shape, b, (s, s), (d, d)))

    def test_get_shape(self):
        for i in range(1, 20):
            for k in range(1, 10):
                for b in ('valid', 'half', 'full', (0, 2)):
                    for d in (1, 2, 3):
                        image_shape = (59, 61, i, i)
                        kernel_shape = (67, 61, k, k)

                        # compute the output that these inputs and parameters would produce
                        output_shape = get_conv_output_shape(
                            image_shape, kernel_shape, b, (1, 1), (d, d))

                        # compute the image_shape given this output_shape
                        computed_image_shape = get_conv_gradinputs_shape(
                            kernel_shape, output_shape, b, (1, 1), (d, d))
                        self.assertEqual(computed_image_shape, image_shape)

                        # if subsample > 1, the shape should be None
                        computed_image_shape = get_conv_gradinputs_shape(
                            kernel_shape, output_shape, b, (2, 3), (d, d))
                        image_shape_with_None = image_shape[:2] + (None, None)
                        self.assertEqual(computed_image_shape, image_shape_with_None)

                        # compute the kernel_shape given this output_shape
                        computed_kernel_shape = get_conv_gradweights_shape(
                            image_shape, output_shape, b, (1, 1), (d, d))

                        # if border_mode == 'half', the shape should be None
                        if b == 'half':
                            kernel_shape_with_None = kernel_shape[:2] + (None, None)
                            self.assertEqual(computed_kernel_shape, kernel_shape_with_None)
                        else:
                            self.assertEqual(computed_kernel_shape, kernel_shape)

                        # if subsample > 1, the shape should be None
                        computed_kernel_shape = get_conv_gradweights_shape(
                            kernel_shape, output_shape, b, (2, 3), (d, d))
                        kernel_shape_with_None = kernel_shape[:2] + (None, None)
                        self.assertEqual(computed_kernel_shape, kernel_shape_with_None)


class TestAssertConvShape(unittest.TestCase):
    def test_basic(self):
        shape = tuple(tensor.iscalar() for i in range(4))
        f = theano.function(shape, assert_conv_shape(shape))

        self.assertEqual([1, 2, 3, 4], f(1, 2, 3, 4))
        self.assertEqual([0, 0, 1, 1], f(0, 0, 1, 1))
        assert_raises(AssertionError, f, 3, 3, 3, 0)
        assert_raises(AssertionError, f, 3, 3, 0, 3)
        assert_raises(AssertionError, f, 3, 3, -1, 3)
        assert_raises(AssertionError, f, 3, -1, 3, 3)
        assert_raises(AssertionError, f, -1, 3, 3, 3)


class TestAssertShape(unittest.TestCase):
    def test_basic(self):
        x = tensor.tensor4()
        s1 = tensor.iscalar()
        s2 = tensor.iscalar()
        expected_shape = [None, s1, s2, None]
        f = theano.function([x, s1, s2], assert_shape(x, expected_shape))

        v = numpy.zeros((3, 5, 7, 11), dtype='float32')
        self.assertEqual(0, numpy.sum(f(v, 5, 7)))

        assert_raises(AssertionError, f, v, 5, 0)
        assert_raises(AssertionError, f, v, 5, 9)
        assert_raises(AssertionError, f, v, 0, 7)
        assert_raises(AssertionError, f, v, 7, 7)

    def test_shape_check_conv2d(self):
        input = tensor.tensor4()
        filters = tensor.tensor4()

        out = conv.conv2d(input, filters,
                          input_shape=(3, 5, 7, 11),
                          filter_shape=(7, 5, 3, 3))
        f = theano.function([input, filters], out)
        # mismatched input_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 5, 9, 11), dtype='float32'),
                      numpy.zeros((7, 5, 3, 3), dtype='float32'))
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 5, 7, 11), dtype='float32'),
                      numpy.zeros((7, 5, 2, 2), dtype='float32'))

    def test_shape_check_conv3d(self):
        input = tensor.tensor5()
        filters = tensor.tensor5()

        out = conv.conv3d(input, filters,
                          input_shape=(3, 5, 7, 11, 13),
                          filter_shape=(7, 5, 3, 3, 3))
        f = theano.function([input, filters], out)
        # mismatched input_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 5, 9, 11, 13), dtype='float32'),
                      numpy.zeros((7, 5, 3, 3, 3), dtype='float32'))
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 5, 7, 11, 13), dtype='float32'),
                      numpy.zeros((7, 5, 2, 2, 2), dtype='float32'))

    def test_shape_check_conv2d_grad_wrt_inputs(self):
        output_grad = tensor.tensor4()
        filters = tensor.tensor4()

        out = conv.conv2d_grad_wrt_inputs(output_grad, filters,
                                          input_shape=(None, None, 7, 11),
                                          filter_shape=(7, 5, 3, 3))
        f = theano.function([output_grad, filters], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 6, 5, 9), dtype='float32'),
                      numpy.zeros((7, 6, 3, 3), dtype='float32'))

    def test_shape_check_conv3d_grad_wrt_inputs(self):
        output_grad = tensor.tensor5()
        filters = tensor.tensor5()

        out = conv.conv3d_grad_wrt_inputs(output_grad, filters,
                                          input_shape=(None, None, 7, 11, 13),
                                          filter_shape=(7, 5, 3, 3, 3))
        f = theano.function([output_grad, filters], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 6, 5, 9, 11), dtype='float32'),
                      numpy.zeros((7, 6, 3, 3, 3), dtype='float32'))

    def test_shape_check_conv2d_grad_wrt_weights(self):
        input = tensor.tensor4()
        output_grad = tensor.tensor4()

        out = conv.conv2d_grad_wrt_weights(input, output_grad,
                                           filter_shape=(None, None, 3, 3),
                                           input_shape=(3, 5, 7, 11))
        f = theano.function([input, output_grad], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 6, 7, 11), dtype='float32'),
                      numpy.zeros((3, 7, 5, 9), dtype='float32'))

    def test_shape_check_conv3d_grad_wrt_weights(self):
        input = tensor.tensor5()
        output_grad = tensor.tensor5()

        out = conv.conv3d_grad_wrt_weights(input, output_grad,
                                           filter_shape=(None, None, 3, 3, 3),
                                           input_shape=(3, 5, 7, 11, 13))
        f = theano.function([input, output_grad], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      numpy.zeros((3, 6, 7, 11, 13), dtype='float32'),
                      numpy.zeros((3, 7, 5, 9, 11), dtype='float32'))


class BaseTestConv(object):
    def get_output_shape(self, inputs_shape, filters_shape,
                         subsample, border_mode, filter_dilation):
        dil_filters = tuple((s - 1) * d + 1 for s, d in zip(filters_shape[2:],
                                                            filter_dilation))
        if border_mode == "valid":
            border_mode = (0,) * (len(inputs_shape) - 2)
        if border_mode == "half":
            border_mode = tuple(d // 2 for d in dil_filters)
        if border_mode == "full":
            border_mode = tuple(d - 1 for d in dil_filters)
        batch_size = inputs_shape[0]
        num_filters = filters_shape[0]
        return ((batch_size, num_filters,) +
                tuple(None if i is None or k is None
                      else ((i + 2 * pad - ((k - 1) * fd + 1)) // d + 1)
                      for i, k, d, pad, fd in zip(inputs_shape[2:],
                                                  filters_shape[2:],
                                                  subsample, border_mode,
                                                  filter_dilation)))

    def run_fwd(self, inputs_shape, filters_shape,
                conv_fn, conv_op, ref,
                subsample=None, verify_grad=True, mode=None,
                border_mode='valid', filter_flip=True,
                provide_shape=False, target_op=None,
                check_trace=False, filter_dilation=None):
        if subsample is None:
            subsample = (1,) * (len(inputs_shape) - 2)
        if filter_dilation is None:
            filter_dilation = (1,) * (len(inputs_shape) - 2)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        # scale down values to prevent rounding errors
        inputs_val /= 10
        filters_val /= 10

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
        c = conv_fn(inputs, filters,
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
        if verify_grad and inputs_val.size > 0 and filters_val.size > 0 and res.size > 0:
            utt.verify_grad(conv_op(border_mode=border_mode,
                                    imshp=imshp, kshp=kshp,
                                    subsample=subsample,
                                    filter_dilation=filter_dilation),
                            [inputs_val, filters_val],
                            mode=mode)

    def run_gradweight(self, inputs_shape, filters_shape, output_shape,
                       gradWeights_fn, ref, subsample=None,
                       filter_flip=True, verify_grad=True, mode=None,
                       border_mode='valid', provide_shape=False,
                       target_op=None, check_trace=False,
                       filter_dilation=None):
        if subsample is None:
            subsample = (1,) * (len(inputs_shape) - 2)
        if filter_dilation is None:
            filter_dilation = (1,) * (len(inputs_shape) - 2)

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
        c = gradWeights_fn(border_mode=border_mode,
                           filter_flip=filter_flip,
                           subsample=subsample,
                           imshp=imshp, kshp=kshp,
                           filter_dilation=filter_dilation)
        c = c(inputs, output, filters_shape[2:])
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

        def abstract_conv_gradweight(inputs_val, output_val):
            conv_op = gradWeights_fn(border_mode=border_mode,
                                     subsample=subsample,
                                     filter_dilation=filter_dilation)
            return conv_op(inputs_val, output_val, filters_shape[2:])

        if verify_grad and inputs_val.size > 0 and output_val.size > 0 and res.size > 0:
            utt.verify_grad(abstract_conv_gradweight,
                            [inputs_val, output_val],
                            mode=mode, eps=1)

    def run_gradinput(self, inputs_shape, filters_shape, output_shape,
                      gradInputs_fn, ref,
                      subsample=None, filter_flip=True,
                      verify_grad=True, mode=None, border_mode='valid',
                      provide_shape=False, target_op=None,
                      check_trace=False, filter_dilation=None):
        if subsample is None:
            subsample = (1,) * (len(inputs_shape) - 2)
        if filter_dilation is None:
            filter_dilation = (1,) * (len(inputs_shape) - 2)

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
        c = gradInputs_fn(border_mode=border_mode,
                          subsample=subsample,
                          filter_flip=filter_flip,
                          imshp=imshp, kshp=kshp,
                          filter_dilation=filter_dilation)
        c = c(filters, output, inputs_shape[2:])
        f = theano.function([], c, mode=mode)

        # ref is set to None for the inconsistent-shape tests.
        # The reference function also raises an exception, which would
        # mask the exception generated by the target implementation.
        if ref is not None:
            c_ref = ref(filters, output, inputs_shape,
                        border_mode=border_mode, subsample=subsample,
                        conv_mode=conv_mode, filter_dilation=filter_dilation)
            f_ref = theano.function([], c_ref, mode='FAST_RUN')

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])
            if check_trace:
                assert_true(check_stack_trace(f, ops_to_check=target_op))

        res = numpy.array(f())

        if ref is not None:
            res_ref = numpy.array(f_ref())
            utt.assert_allclose(res_ref, res)

        def abstract_conv_gradinputs(filters_val, output_val):
            conv_op = gradInputs_fn(border_mode=border_mode,
                                    subsample=subsample,
                                    filter_dilation=filter_dilation)
            return conv_op(filters_val, output_val, inputs_shape[2:])

        if verify_grad and filters_val.size > 0 and output_val.size > 0 and res.size > 0:
            utt.verify_grad(abstract_conv_gradinputs,
                            [filters_val, output_val],
                            mode=mode, eps=1)

    def test_all(self):
        if type(self) is BaseTestConv:
            raise SkipTest("base class")
        ds = self.default_subsamples
        db = self.default_border_mode
        dflip = self.default_filter_flip
        dprovide_shape = self.default_provide_shape
        for (i, f) in zip(self.inputs_shapes, self.filters_shapes):
            for provide_shape in self.provide_shape:
                yield (self.tcase, i, f, ds, db, dflip, provide_shape)
            if min(i) > 0 and min(f) > 0:
                for fd in self.filters_dilations:
                    for s in self.subsamples:
                        for b in self.border_modes:
                            yield (self.tcase, i, f, s, b, dflip,
                                   dprovide_shape, fd)
                for flip in self.filter_flip:
                    yield (self.tcase, i, f, ds, db, flip, dprovide_shape)


class BaseTestConv2d(BaseTestConv):
    @classmethod
    def setup_class(cls):
        # This tests can run even when theano.config.blas.ldflags is empty.
        cls.inputs_shapes = [(8, 1, 6, 6), (8, 1, 8, 8), (2, 1, 7, 7),
                             (6, 1, 10, 11), (2, 1, 6, 5), (1, 5, 9, 9),
                             (0, 1, 6, 6), (1, 0, 6, 6), (1, 1, 6, 6)]
        cls.filters_shapes = [(5, 1, 2, 2), (4, 1, 3, 3), (2, 1, 3, 3),
                              (1, 1, 2, 3), (4, 1, 1, 3), (4, 5, 3, 2),
                              (1, 1, 2, 2), (1, 0, 2, 2), (0, 1, 2, 2)]
        cls.subsamples = [(1, 1), (2, 2), (2, 4)]
        cls.default_subsamples = (1, 1)
        cls.filters_dilations = [(1, 1), (1, 2), (2, 1)]
        cls.default_filters_dilations = (1, 1)
        cls.border_modes = ["valid", "half", "full", (0, 0), (1, 1), (5, 5), (5, 2)]
        cls.default_border_mode = (0, 0)
        cls.filter_flip = [True, False]
        cls.default_filter_flip = True
        cls.provide_shape = [True, False]
        cls.default_provide_shape = True
        cls.shared = staticmethod(theano.compile.shared)

    def test_gradinput_arbitrary_output_shapes(self):
        # this computes the grad wrt inputs for an output shape
        # that the forward convolution would not produce
        input_shape = (2, 1, 7, 7)
        filter_shape = (2, 1, 3, 3)
        for output_shape in [(2, 2, 8, 8), (2, 2, 9, 9), (2, 2, 12, 12)]:
            for border_mode in ["valid", "half", "full"]:
                computed_shape = get_conv_output_shape(
                    input_shape, filter_shape, border_mode, self.default_subsamples, self.default_filters_dilations)
                # is this a valid combination?
                if tuple(computed_shape) == output_shape:
                    yield (self.tcase_gi,
                           input_shape,
                           filter_shape,
                           output_shape,
                           self.default_subsamples,
                           border_mode,
                           True,
                           True,
                           self.default_filters_dilations,
                           False)
                else:
                    # expect an error
                    yield (self.tcase_gi,
                           input_shape,
                           filter_shape,
                           output_shape,
                           self.default_subsamples,
                           border_mode,
                           True,
                           True,
                           self.default_filters_dilations,
                           True)

    def test_gradinput_impossible_output_shapes(self):
        def run_for_output_offsets(image_shape, kernel_shape, s, border_mode, d):
            # outputs that are too large or too small should be rejected
            for o in (-3, -1, 1, 2):
                output_shape = (1, 1, computed_shape[2] + o, computed_shape[3] + o)
                # expect an error
                self.tcase_gi(image_shape, kernel_shape, output_shape,
                              (s, s), border_mode, True, True, (d, d), True)

        for (i, k) in ((1, 1), (1, 2), (2, 1), (4, 2), (4, 3), (7, 3), (9, 5)):
            for border_mode in ('valid', 'half', 'full', (0, 2)):
                for (s, d) in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)):
                    image_shape = (1, 1, i, i)
                    kernel_shape = (1, 1, k, k)

                    # compute the output that these inputs and parameters would produce
                    computed_shape = get_conv_output_shape(
                        image_shape, kernel_shape, border_mode, (s, s), (d, d))

                    yield (run_for_output_offsets,
                           image_shape, kernel_shape, s, border_mode, d)

    def run_fwd(self, inputs_shape, filters_shape,
                conv_fn=conv.conv2d, conv_op=conv.AbstractConv2d,
                ref=conv2d_corr, **kwargs):
        super(BaseTestConv2d, self).run_fwd(
            inputs_shape=inputs_shape,
            filters_shape=filters_shape,
            conv_fn=conv_fn,
            conv_op=conv_op,
            ref=ref, **kwargs)

    def run_gradweight(self, inputs_shape, filters_shape, output_shape,
                       gradWeights_fn=conv.AbstractConv2d_gradWeights,
                       ref=conv2d_corr_gw, **kwargs):
        super(BaseTestConv2d, self).run_gradweight(
            inputs_shape=inputs_shape,
            filters_shape=filters_shape,
            output_shape=output_shape,
            gradWeights_fn=gradWeights_fn,
            ref=ref, **kwargs)

    def run_gradinput(self, inputs_shape, filters_shape, output_shape,
                      gradInputs_fn=conv.AbstractConv2d_gradInputs,
                      ref=conv2d_corr_gi, **kwargs):
        super(BaseTestConv2d, self).run_gradinput(
            inputs_shape=inputs_shape,
            filters_shape=filters_shape,
            output_shape=output_shape,
            gradInputs_fn=gradInputs_fn,
            ref=ref, **kwargs)


class TestCorrConv2d(BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        # This tests can run even when theano.config.blas.ldflags is empty.
        BaseTestConv2d.setup_class()

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        o = self.get_output_shape(i, f, s, b, fd)
        # This tests can run even when theano.config.blas.ldflags is empty.
        if (not theano.config.cxx or
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

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False):
        # This tests can run even when theano.config.blas.ldflags is empty.
        if (not theano.config.cxx or
                theano.config.mode == "FAST_COMPILE"):
            raise SkipTest("Need blas to test conv2d")
        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s, verify_grad=True,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip, target_op=CorrMM_gradInputs,
                               check_trace=True, filter_dilation=fd)
        else:
            assert_raises(ValueError,
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s, verify_grad=False,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip, target_op=CorrMM_gradInputs,
                          ref=None, check_trace=True, filter_dilation=fd)


class TestAbstractConvNoOptim(BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        # This tests can run even when theano.config.blas.ldflags is empty.
        BaseTestConv2d.setup_class()
        cls.inputs_shapes = [(8, 1, 6, 6)]
        cls.filters_shapes = [(5, 1, 2, 2)]
        cls.subsamples = [(1, 1), (2, 2)]
        cls.filters_dilations = [(1, 1), (1, 2), (2, 1)]
        cls.border_modes = ["valid", "half", "full"]
        cls.filter_flip = [True]
        cls.provide_shape = [False]

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        o = self.get_output_shape(i, f, s, b, fd)
        mode = theano.Mode(optimizer=None)

        if not theano.config.cxx:
            raise SkipTest("Need cxx to test conv2d")

        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, provide_shape=provide_shape,
                     border_mode=b, filter_flip=flip,
                     target_op=None, check_trace=True,
                     filter_dilation=fd, mode=mode)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s, verify_grad=True,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip, target_op=None,
                            check_trace=True, filter_dilation=fd,
                            mode=mode)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s, verify_grad=True,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip, target_op=None,
                           check_trace=True, filter_dilation=fd,
                           mode=mode)

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False):
        mode = theano.Mode(optimizer=None)
        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s, verify_grad=True,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip, target_op=None,
                               check_trace=True, filter_dilation=fd,
                               mode=mode)
        else:
            assert_raises(ValueError,
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s, verify_grad=False,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip, target_op=None,
                          check_trace=True, filter_dilation=fd,
                          ref=None, mode=mode)


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
        if not theano.config.cxx:
            raise SkipTest("Need cxx to test conv2d")

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
            # This test can run even when theano.config.blas.ldflags is empty.
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
            # This test can run even when theano.config.blas.ldflags is empty.
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
            # This test can run even when theano.config.blas.ldflags is empty.
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

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False):
        if fd != (1, 1):
            raise SkipTest("No dilation implementation for basic cpu ConvOp.")
        mode = self.mode

        if not flip:
            return
        if b not in ((0, 0), 'valid', 'full'):
            return
        if (not provide_shape) and (s != (1, 1)) and (b == 'full'):
            return
        if ((s[0] not in (1, 2)) or (s[1] not in (1, 2))) and (b == 'full'):
            return

        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=False, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=(ConvOp, ConvTransp3D),
                               check_trace=True,
                               filter_dilation=fd)
        else:
            # we do not check for inconsistent shapes,
            # because this older implementation does not check that
            raise SkipTest('Inconsistent shapes are not tested for old cpu ConvOp.')


class BaseTestConv3d(BaseTestConv):
    @classmethod
    def setup_class(cls):
        # This tests can run even when theano.config.blas.ldflags is empty.
        cls.inputs_shapes = [(2, 1, 5, 5, 5), (1, 2, 7, 5, 6),
                             (0, 1, 5, 5, 5), (1, 0, 5, 5, 5), (1, 1, 5, 5, 5)]
        cls.filters_shapes = [(2, 1, 2, 2, 2), (1, 2, 2, 1, 3),
                              (1, 1, 2, 2, 2), (1, 0, 2, 2, 2), (0, 1, 2, 2, 2)]
        cls.subsamples = [(1, 1, 1), (2, 2, 2), (1, 2, 3)]
        cls.default_subsamples = (1, 1, 1)
        cls.filters_dilations = [(1, 1, 1), (1, 2, 1), (2, 1, 2)]
        cls.default_filters_dilations = (1, 1, 1)
        cls.border_modes = ["valid", "half", "full", (0, 0, 0), (2, 2, 3)]
        cls.default_border_mode = (0, 0, 0)
        cls.filter_flip = [True, False]
        cls.default_filter_flip = True
        cls.provide_shape = [True, False]
        cls.default_provide_shape = True
        cls.shared = staticmethod(theano.compile.shared)

    def test_gradinput_arbitrary_output_shapes(self):
        # this computes the grad wrt inputs for an output shape
        # that the forward convolution would not produce
        input_shape = (2, 1, 7, 7, 7)
        filter_shape = (1, 1, 3, 3, 3)
        for output_shape in [(2, 1, 8, 8, 8), (2, 1, 9, 9, 9), (2, 1, 12, 12, 12)]:
            for border_mode in ["valid", "half", "full"]:
                # compute the output that these inputs and parameters would produce
                computed_shape = get_conv_output_shape(
                    input_shape, filter_shape, border_mode, self.default_subsamples, self.default_filters_dilations)
                # is this a valid combination?
                if tuple(computed_shape) == output_shape:
                    yield (self.tcase_gi,
                           input_shape,
                           filter_shape,
                           output_shape,
                           self.default_subsamples,
                           border_mode,
                           True,
                           True,
                           self.default_filters_dilations,
                           False)
                else:
                    # expect an error
                    yield (self.tcase_gi,
                           input_shape,
                           filter_shape,
                           output_shape,
                           self.default_subsamples,
                           border_mode,
                           True,
                           True,
                           self.default_filters_dilations,
                           True)

    def test_gradinput_impossible_output_shapes(self):
        def run_for_output_offsets(image_shape, kernel_shape, s, border_mode, d):
            # outputs that are too large or too small should be rejected
            for o in (-3, -1, 1, 2):
                output_shape = (1, 1, computed_shape[2] + o,
                                computed_shape[3] + o, computed_shape[4] + o)
                # expect an error
                self.tcase_gi(image_shape, kernel_shape, output_shape,
                              (s, s), border_mode, True, True, (d, d), True)

        for (i, k) in ((1, 1), (1, 2), (2, 1), (4, 2), (4, 3), (7, 3), (9, 5)):
            for border_mode in ('valid', 'half', 'full', (0, 2, 1)):
                for (s, d) in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)):
                    image_shape = (1, 1, i, i, i)
                    kernel_shape = (1, 1, k, k, k)

                    # compute the output that these inputs and parameters would produce
                    computed_shape = get_conv_output_shape(
                        image_shape, kernel_shape, border_mode, (s, s, s), (d, d, d))

                    yield (run_for_output_offsets,
                           image_shape, kernel_shape, s, border_mode, d)

    def run_fwd(self, inputs_shape, filters_shape,
                conv_fn=conv.conv3d, conv_op=conv.AbstractConv3d,
                ref=conv3d_corr, **kwargs):
        super(BaseTestConv3d, self).run_fwd(
            inputs_shape=inputs_shape,
            filters_shape=filters_shape,
            conv_fn=conv_fn,
            conv_op=conv_op,
            ref=ref, **kwargs)

    def run_gradweight(self, inputs_shape, filters_shape, output_shape,
                       gradWeights_fn=conv.AbstractConv3d_gradWeights,
                       ref=conv3d_corr_gw, **kwargs):
        super(BaseTestConv3d, self).run_gradweight(
            inputs_shape=inputs_shape,
            filters_shape=filters_shape,
            output_shape=output_shape,
            gradWeights_fn=gradWeights_fn,
            ref=ref, **kwargs)

    def run_gradinput(self, inputs_shape, filters_shape, output_shape,
                      gradInputs_fn=conv.AbstractConv3d_gradInputs,
                      ref=conv3d_corr_gi, **kwargs):
        super(BaseTestConv3d, self).run_gradinput(
            inputs_shape=inputs_shape,
            filters_shape=filters_shape,
            output_shape=output_shape,
            gradInputs_fn=gradInputs_fn,
            ref=ref, **kwargs)


class TestCorrConv3d(BaseTestConv3d):
    @classmethod
    def setup_class(cls):
        # This tests can run even when theano.config.blas.ldflags is empty.
        BaseTestConv3d.setup_class()

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        o = self.get_output_shape(i, f, s, b, fd)
        # This test can run even when theano.config.blas.ldflags is empty.
        if (not theano.config.cxx or
                theano.config.mode == "FAST_COMPILE"):
            raise SkipTest("Need blas to test conv3d")
        self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                     verify_grad=True, provide_shape=provide_shape,
                     border_mode=b, filter_flip=flip,
                     target_op=Corr3dMM, check_trace=True,
                     filter_dilation=fd)
        self.run_gradweight(inputs_shape=i, filters_shape=f,
                            output_shape=o, subsample=s, verify_grad=True,
                            provide_shape=provide_shape, border_mode=b,
                            filter_flip=flip, target_op=Corr3dMM_gradWeights,
                            check_trace=True, filter_dilation=fd)
        self.run_gradinput(inputs_shape=i, filters_shape=f,
                           output_shape=o, subsample=s, verify_grad=True,
                           provide_shape=provide_shape, border_mode=b,
                           filter_flip=flip, target_op=Corr3dMM_gradInputs,
                           check_trace=True, filter_dilation=fd)

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False):
        # This test can run even when theano.config.blas.ldflags is empty.
        if (not theano.config.cxx or
                theano.config.mode == "FAST_COMPILE"):
            raise SkipTest("Need blas to test conv3d")
        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s, verify_grad=True,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip, target_op=Corr3dMM_gradInputs,
                               check_trace=True, filter_dilation=fd)
        else:
            assert_raises(ValueError,
                          self.run_gradinput,
                          inputs_shape=i, filters_shape=f,
                          output_shape=o, subsample=s, verify_grad=False,
                          provide_shape=provide_shape, border_mode=b,
                          filter_flip=flip, target_op=Corr3dMM_gradInputs,
                          ref=None, check_trace=True, filter_dilation=fd)


class TestCpuConv3d(BaseTestConv3d):
    @classmethod
    def setup(cls):
        BaseTestConv3d.setup_class()
        cls.mode = theano.compile.mode.get_default_mode().excluding('conv_gemm')
        cls.opt_err = theano.config.on_opt_error
        theano.config.on_opt_error = 'ignore'

    @classmethod
    def tearDown(cls):
        theano.config.on_opt_error = cls.opt_err

    def tcase(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        if fd != (1, 1, 1):
            raise SkipTest("No dilation implementation for basic cpu Conv3D.")
        if not theano.config.cxx:
            raise SkipTest("Need cxx to test conv2d")
        if min(i) == 0 or min(f) == 0:
            raise SkipTest('Not tested for old cpu Conv3D.')

        mode = self.mode
        o = self.get_output_shape(i, f, s, b, fd)
        fwd_OK = True
        gradweight_OK = True
        gradinput_OK = True

        if b not in ((0, 0, 0), 'valid'):
            fwd_OK = False
            gradweight_OK = False
            gradinput_OK = False

        if fwd_OK:
            # This test can run even when theano.config.blas.ldflags is empty.
            self.run_fwd(inputs_shape=i, filters_shape=f,
                         subsample=s, verify_grad=(gradweight_OK and gradinput_OK),
                         mode=mode, provide_shape=provide_shape,
                         border_mode=b, filter_flip=flip, target_op=Conv3D,
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
            # This test can run even when theano.config.blas.ldflags is empty.
            self.run_gradweight(inputs_shape=i, filters_shape=f,
                                output_shape=o, subsample=s,
                                verify_grad=False, mode=mode,
                                provide_shape=provide_shape, border_mode=b,
                                filter_flip=flip,
                                target_op=ConvGrad3D,
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
            # This test can run even when theano.config.blas.ldflags is empty.
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=False, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=ConvTransp3D,
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

    def tcase_gi(self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False):
        if fd != (1, 1, 1):
            raise SkipTest("No dilation implementation for basic cpu Conv3D.")
        mode = self.mode
        if min(i) == 0 or min(f) == 0 or min(o) == 0:
            raise SkipTest('Not tested for old cpu Conv3D.')

        if b not in ((0, 0, 0), 'valid'):
            return

        if not expect_error:
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=False, mode=mode,
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=ConvTransp3D,
                               check_trace=True,
                               filter_dilation=fd)
        else:
            # we do not check for inconsistent shapes,
            # because this older implementation does not check that
            raise SkipTest('Inconsistent shapes are not tested for old cpu Conv3D.')


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
    # If theano.config.blas.ldflags is empty, Theano will use
    # a NumPy C implementation of [sd]gemm_.
    compile_mode = theano.compile.mode.get_default_mode()
    if theano.config.mode == "FAST_COMPILE":
        compile_mode = compile_mode.excluding("conv_gemm")
        compile_mode = compile_mode.excluding('AbstractConvCheck')
    elif not theano.config.cxx:
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


class TestConv2dTranspose(unittest.TestCase):
    mode = None

    def test_interface(self):
        """Test conv2d_transpose wrapper.

        This method tests that the order of the filter's
        axes expected by the function produces the correct
        output shape.

        """
        mode = self.mode
        if theano.config.mode == "FAST_COMPILE":
            mode = theano.compile.get_mode(
                mode).excluding("conv_gemm").excluding("AbstractConvCheck")

        output = theano.function(
            inputs=[],
            outputs=conv2d_transpose(input=tensor.ones((2, 2, 4, 4)),
                                     filters=tensor.ones((2, 1, 4, 4)),
                                     output_shape=(2, 1, 10, 10),
                                     input_dilation=(2, 2)),
            mode=mode)()
        expected_output = numpy.array(
            [[[[2, 2, 4, 4, 4, 4, 4, 4, 2, 2],
               [2, 2, 4, 4, 4, 4, 4, 4, 2, 2],
               [4, 4, 8, 8, 8, 8, 8, 8, 4, 4],
               [4, 4, 8, 8, 8, 8, 8, 8, 4, 4],
               [4, 4, 8, 8, 8, 8, 8, 8, 4, 4],
               [4, 4, 8, 8, 8, 8, 8, 8, 4, 4],
               [4, 4, 8, 8, 8, 8, 8, 8, 4, 4],
               [4, 4, 8, 8, 8, 8, 8, 8, 4, 4],
               [2, 2, 4, 4, 4, 4, 4, 4, 2, 2],
               [2, 2, 4, 4, 4, 4, 4, 4, 2, 2]]]] * 2)
        numpy.testing.assert_equal(output, expected_output)
