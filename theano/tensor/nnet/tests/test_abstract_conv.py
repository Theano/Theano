from __future__ import absolute_import, print_function, division
import unittest
import numpy as np
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises, assert_true

import theano
from theano import tensor
from theano import change_flags
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
from theano.tensor.nnet.abstract_conv import separable_conv2d, separable_conv3d
from theano.tensor.nnet.abstract_conv import causal_conv1d
from theano.tensor.nnet.corr import (CorrMM, CorrMM_gradWeights,
                                     CorrMM_gradInputs)
from theano.tensor.nnet.corr3d import (Corr3dMM, Corr3dMM_gradWeights,
                                       Corr3dMM_gradInputs)


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
    @change_flags([("conv.assert_shape", True)])
    def test_basic(self):
        x = tensor.tensor4()
        s1 = tensor.iscalar()
        s2 = tensor.iscalar()
        expected_shape = [None, s1, s2, None]
        f = theano.function([x, s1, s2], assert_shape(x, expected_shape))

        v = np.zeros((3, 5, 7, 11), dtype='float32')
        self.assertEqual(0, np.sum(f(v, 5, 7)))

        assert_raises(AssertionError, f, v, 5, 0)
        assert_raises(AssertionError, f, v, 5, 9)
        assert_raises(AssertionError, f, v, 0, 7)
        assert_raises(AssertionError, f, v, 7, 7)

    @change_flags([("conv.assert_shape", True)])
    def test_shape_check_conv2d(self):
        input = tensor.tensor4()
        filters = tensor.tensor4()

        out = conv.conv2d(input, filters,
                          input_shape=(3, 5, 7, 11),
                          filter_shape=(7, 5, 3, 3))
        f = theano.function([input, filters], out)
        # mismatched input_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 5, 9, 11), dtype='float32'),
                      np.zeros((7, 5, 3, 3), dtype='float32'))
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 5, 7, 11), dtype='float32'),
                      np.zeros((7, 5, 2, 2), dtype='float32'))

    @change_flags([("conv.assert_shape", True)])
    def test_shape_check_conv3d(self):
        if theano.config.cxx == "":
            raise SkipTest("test needs cxx")
        input = tensor.tensor5()
        filters = tensor.tensor5()

        out = conv.conv3d(input, filters,
                          input_shape=(3, 5, 7, 11, 13),
                          filter_shape=(7, 5, 3, 3, 3))
        f = theano.function([input, filters], out)
        # mismatched input_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 5, 9, 11, 13), dtype='float32'),
                      np.zeros((7, 5, 3, 3, 3), dtype='float32'))
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 5, 7, 11, 13), dtype='float32'),
                      np.zeros((7, 5, 2, 2, 2), dtype='float32'))

    @change_flags([("conv.assert_shape", True)])
    def test_shape_check_conv2d_grad_wrt_inputs(self):
        output_grad = tensor.tensor4()
        filters = tensor.tensor4()

        out = conv.conv2d_grad_wrt_inputs(output_grad, filters,
                                          input_shape=(None, None, 7, 11),
                                          filter_shape=(7, 5, 3, 3))
        f = theano.function([output_grad, filters], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 6, 5, 9), dtype='float32'),
                      np.zeros((7, 6, 3, 3), dtype='float32'))

    @change_flags([("conv.assert_shape", True)])
    def test_shape_check_conv3d_grad_wrt_inputs(self):
        if theano.config.cxx == "":
            raise SkipTest("test needs cxx")
        output_grad = tensor.tensor5()
        filters = tensor.tensor5()

        out = conv.conv3d_grad_wrt_inputs(output_grad, filters,
                                          input_shape=(None, None, 7, 11, 13),
                                          filter_shape=(7, 5, 3, 3, 3))
        f = theano.function([output_grad, filters], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 6, 5, 9, 11), dtype='float32'),
                      np.zeros((7, 6, 3, 3, 3), dtype='float32'))

    @change_flags([("conv.assert_shape", True)])
    def test_shape_check_conv2d_grad_wrt_weights(self):
        input = tensor.tensor4()
        output_grad = tensor.tensor4()

        out = conv.conv2d_grad_wrt_weights(input, output_grad,
                                           filter_shape=(None, None, 3, 3),
                                           input_shape=(3, 5, 7, 11))
        f = theano.function([input, output_grad], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 6, 7, 11), dtype='float32'),
                      np.zeros((3, 7, 5, 9), dtype='float32'))

    @change_flags([("conv.assert_shape", True)])
    def test_shape_check_conv3d_grad_wrt_weights(self):
        if theano.config.cxx == "":
            raise SkipTest("test needs cxx")
        input = tensor.tensor5()
        output_grad = tensor.tensor5()

        out = conv.conv3d_grad_wrt_weights(input, output_grad,
                                           filter_shape=(None, None, 3, 3, 3),
                                           input_shape=(3, 5, 7, 11, 13))
        f = theano.function([input, output_grad], out)
        # mismatched filter_shape
        assert_raises(AssertionError, f,
                      np.zeros((3, 6, 7, 11, 13), dtype='float32'),
                      np.zeros((3, 7, 5, 9, 11), dtype='float32'))


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

        inputs_val = np.random.random(inputs_shape).astype('float32')
        filters_val = np.random.random(filters_shape).astype('float32')

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

        res_ref = np.array(f_ref())
        res = np.array(f())
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

        inputs_val = np.random.random(inputs_shape).astype('float32')
        output_val = np.random.random(output_shape).astype('float32')

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

        res_ref = np.array(f_ref())
        res = np.array(f())
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

        output_val = np.random.random(output_shape).astype('float32')
        filters_val = np.random.random(filters_shape).astype('float32')
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

        res = np.array(f())

        if ref is not None:
            res_ref = np.array(f_ref())
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
        if not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("SciPy needed")

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

        if not theano.config.cxx:
            raise SkipTest("Need cxx to test conv2d")

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

        self.constant_tensor = np.zeros((3, 5, 7, 11), dtype='float32')

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
        """
        Gets numerical 1D kernel for bilinear upsampling
        """
        return np.array(list(range(1, ratio + 1)) +
                        list(range(ratio - 1, 0, -1)))

    def numerical_kernel_2D(self, ratio):
        """
        Gets numerical 2D kernel for bilinear upsampling
        """
        return np.array([i * j for i in self.numerical_kernel_1D(ratio) for j
                         in self.numerical_kernel_1D(ratio)]).\
            reshape(2 * ratio - 1, 2 * ratio - 1)

    def test_bilinear_kernel_2D(self):
        # Test 2D kernels used in bilinear upsampling
        #
        # This method tests the correctness of the
        # 2D kernel values used in bilinear upsampling
        # for some upsampling ratios.

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
        # Test 1D kernels used in bilinear upsampling
        #
        # This method tests the correctness of the
        # 1D kernel values used in bilinear upsampling
        # for some upsampling ratios.

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
        """
        Compute upsampling multiplier

        This method computes the multipliers of an array
        that will be upsampled using bilinear interpolation.

        Parameters
        ----------
        ratio: int
            the ratio by which the array will be upsampled.

        Returns
        -------
        1D numpy array
            The multipliers that can be used in bilinear interpolation
            to upsample an array.

        int
            The size of the multipliers array
        """
        kern = np.arange(ratio + 1)
        return kern, kern.shape[0]

    def get_upsampled_twobytwo_mat(self, two_by_two, ratio):
        """
        Upsample 4D array with two rows and two columns

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
        # Test bilinear upsampling using 1D kernels
        #
        # This method tests the bilinear_upsampling method
        # when using 1D kernels for some upsampling ratios.

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
        #
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
        # Compare 1D and 2D upsampling
        #
        # This method verifies the bilinear upsampling done by using
        # 1D and 2D kernels will generate the same result.

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

    def test_fractional_bilinear_upsampling(self):
        """Test bilinear upsampling with nonsimilar fractional
        row and col ratios
        """
        input_x = np.array([[[1, 2], [3, 4]],
                            [[5, 6], [7, 8]],
                            [[9, 10], [11, 12]]],
                           ndmin=4).astype(theano.config.floatX)
        up_x = bilinear_upsampling(input=input_x,
                                   frac_ratio=((7, 4), (5, 3)),
                                   use_1D_kernel=False)
        num_up_x = np.array(
            [[[[1., 1.2, 1.8, 2.],
              [1.28571429, 1.48571429, 2.08571429, 2.28571429],
              [2.42857143, 2.62857143, 3.22857143, 3.42857143],
              [3., 3.2, 3.8, 4.]],
             [[5., 5.2, 5.8, 6.],
              [5.28571429, 5.48571429, 6.08571429, 6.28571429],
              [6.42857143, 6.62857143, 7.22857143, 7.42857143],
              [7., 7.2, 7.8, 8.]],
             [[9., 9.2, 9.8, 10.],
              [9.28571429, 9.48571429, 10.08571429, 10.28571429],
              [10.42857143, 10.62857143, 11.22857143, 11.42857143],
              [11., 11.2, 11.8, 12.]]]]
            ).astype(theano.config.floatX)
        f_up_x = theano.function([], up_x, mode=self.compile_mode)
        utt.assert_allclose(f_up_x(), num_up_x, rtol=1e-6)

    def test_fractional_bilinear_upsampling_shape(self):
        x = np.random.rand(1, 1, 200, 200).astype(theano.config.floatX)
        resize = (24, 20)
        z = bilinear_upsampling(tensor.as_tensor_variable(x), frac_ratio=resize, use_1D_kernel=False)
        out = theano.function([], z.shape, mode='FAST_RUN')()
        utt.assert_allclose(out, (1, 1, 240, 240))


class TestConv2dTranspose(unittest.TestCase):
    mode = None

    def test_interface(self):
        # Test conv2d_transpose wrapper.
        #
        # This method tests that the order of the filter's
        # axes expected by the function produces the correct
        # output shape.
        if theano.config.cxx == "":
            raise SkipTest("test needs cxx")

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
        expected_output = np.array(
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
        np.testing.assert_equal(output, expected_output)


class TestConv2dGrads(unittest.TestCase):

    def setUp(self):

        if (not theano.config.cxx or
                theano.config.mode == "FAST_COMPILE"):
            raise SkipTest("Need blas to test conv2d")

        self.random_stream = np.random.RandomState(utt.fetch_seed())

        self.inputs_shapes = [(8, 1, 12, 12), (1, 1, 5, 5), (1, 1, 5, 6), (1, 1, 6, 6)]
        self.filters_shapes = [(5, 1, 2, 2), (1, 1, 3, 3)]

        self.subsamples = [(1, 1), (2, 2)]
        self.border_modes = ["valid", "full"]
        self.filter_flip = [True, False]

        self.output_grad = theano.tensor.tensor4()
        self.output_grad_wrt = theano.tensor.tensor4()

        self.x = theano.tensor.tensor4('x', theano.config.floatX)  # inputs
        self.w = theano.tensor.tensor4('w', theano.config.floatX)  # filter weights

    def test_conv2d_grad_wrt_inputs(self):
        # Compares calculated abstract grads wrt inputs with the fwd grads
        # This method checks the outputs of conv2_grad_wrt_inputs against
        # the outputs of T.nnet.conv forward grads to make sure the
        # results are the same.

        for (in_shape, fltr_shape) in zip(self.inputs_shapes, self.filters_shapes):
            for bm in self.border_modes:
                for ss in self.subsamples:
                    for ff in self.filter_flip:
                        input_val = self.random_stream.random_sample(in_shape).astype(theano.config.floatX)
                        filter_val = self.random_stream.random_sample(fltr_shape).astype(theano.config.floatX)
                        out_grad_shape = theano.tensor.nnet.abstract_conv.get_conv_output_shape(image_shape=in_shape,
                                                                                                kernel_shape=fltr_shape,
                                                                                                border_mode=bm,
                                                                                                subsample=ss)
                        out_grad_val = self.random_stream.random_sample(out_grad_shape).astype(theano.config.floatX)
                        conv_out = theano.tensor.nnet.conv2d(self.x,
                                                             filters=self.w,
                                                             border_mode=bm,
                                                             subsample=ss,
                                                             input_shape=in_shape,
                                                             filter_shape=fltr_shape,
                                                             filter_flip=ff
                                                             )
                        conv_grad = theano.grad(conv_out.sum(), wrt=self.x, known_grads={conv_out: self.output_grad})
                        f_old = theano.function([self.x, self.w, self.output_grad], conv_grad)

                        conv_wrt_i_out = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_inputs(output_grad=self.output_grad_wrt,
                                                                                                 filters=self.w,
                                                                                                 border_mode=bm,
                                                                                                 subsample=ss,
                                                                                                 input_shape=in_shape,
                                                                                                 filter_shape=fltr_shape,
                                                                                                 filter_flip=ff
                                                                                                 )
                        f_new = theano.function([self.w, self.output_grad_wrt], conv_wrt_i_out)

                        # check that they're equal
                        utt.assert_allclose(f_new(filter_val, out_grad_val), f_old(input_val, filter_val, out_grad_val))

    def test_conv2d_grad_wrt_weights(self):
        # Compares calculated abstract grads wrt weights with the fwd grads
        # This method checks the outputs of conv2_grad_wrt_weights against
        # the outputs of T.nnet.conv forward grads to make sure the
        # results are the same.

        for (in_shape, fltr_shape) in zip(self.inputs_shapes, self.filters_shapes):
            for bm in self.border_modes:
                for ss in self.subsamples:
                    for ff in self.filter_flip:
                        input_val = self.random_stream.random_sample(in_shape).astype(theano.config.floatX)
                        filter_val = self.random_stream.random_sample(fltr_shape).astype(theano.config.floatX)
                        out_grad_shape = theano.tensor.nnet.abstract_conv.get_conv_output_shape(image_shape=in_shape,
                                                                                                kernel_shape=fltr_shape,
                                                                                                border_mode=bm,
                                                                                                subsample=ss)
                        out_grad_val = self.random_stream.random_sample(out_grad_shape).astype(theano.config.floatX)
                        conv_out = theano.tensor.nnet.conv2d(self.x,
                                                             filters=self.w,
                                                             border_mode=bm,
                                                             subsample=ss,
                                                             input_shape=in_shape,
                                                             filter_shape=fltr_shape,
                                                             filter_flip=ff
                                                             )
                        conv_grad = theano.grad(conv_out.sum(), wrt=self.w, known_grads={conv_out: self.output_grad})
                        f_old = theano.function([self.x, self.w, self.output_grad], conv_grad)

                        conv_wrt_w_out = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_weights(self.x,
                                                                                                  output_grad=self.output_grad_wrt,
                                                                                                  border_mode=bm,
                                                                                                  subsample=ss,
                                                                                                  input_shape=in_shape,
                                                                                                  filter_shape=fltr_shape,
                                                                                                  filter_flip=ff
                                                                                                  )
                        f_new = theano.function([self.x, self.output_grad_wrt], conv_wrt_w_out)
                        utt.assert_allclose(f_new(input_val, out_grad_val), f_old(input_val, filter_val, out_grad_val))


class Grouped_conv_noOptim(unittest.TestCase):
    conv = theano.tensor.nnet.abstract_conv.AbstractConv2d
    conv_gradw = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights
    conv_gradi = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs
    conv_op = theano.tensor.nnet.abstract_conv.AbstractConv2d
    conv_gradw_op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights
    conv_gradi_op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs
    mode = theano.Mode(optimizer=None)
    is_dnn = False

    def setUp(self):
        self.num_groups = [3, 2, 4, 4]
        self.border_mode = 'valid'
        self.subsample = (1, 1)
        self.img_shape = [(5, 6, 5, 5), (4, 4, 7, 5), (3, 8, 5, 3), (2, 4, 7, 7)]
        self.kern_shape = [(6, 2, 3, 3), (6, 2, 5, 3), (4, 2, 3, 3), (4, 1, 3, 5)]
        self.top_shape = [(5, 6, 3, 3), (4, 6, 3, 3), (3, 4, 3, 1), (2, 4, 5, 3)]
        self.filter_dilation = (1, 1)
        self.ref_mode = 'FAST_RUN'
        self.convdim = 2
        self.corr_fwd = conv2d_corr
        self.corr_gradw = conv2d_corr_gw
        self.corr_gradi = conv2d_corr_gi
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("CorrMM needs cxx and SciPy")

    def test_fwd(self):
        if self.convdim == 2:
            img_sym = theano.tensor.tensor4('img')
            kern_sym = theano.tensor.tensor4('kern')
        else:
            img_sym = theano.tensor.tensor5('img')
            kern_sym = theano.tensor.tensor5('kern')

        for imshp, kshp, groups in zip(self.img_shape, self.kern_shape, self.num_groups):
            img = np.random.random(imshp).astype(theano.config.floatX)
            kern = np.random.random(kshp).astype(theano.config.floatX)
            split_imgs = np.split(img, groups, axis=1)
            split_kern = np.split(kern, groups, axis=0)

            grouped_conv_op = self.conv(border_mode=self.border_mode,
                                        subsample=self.subsample,
                                        filter_dilation=self.filter_dilation,
                                        num_groups=groups)
            grouped_conv_output = grouped_conv_op(img_sym, kern_sym)

            grouped_func = theano.function([img_sym, kern_sym], grouped_conv_output, mode=self.mode)
            assert any([isinstance(node.op, self.conv_op)
                       for node in grouped_func.maker.fgraph.toposort()])
            grouped_output = grouped_func(img, kern)

            ref_conv_op = self.corr_fwd(img_sym,
                                        kern_sym,
                                        border_mode=self.border_mode,
                                        subsample=self.subsample,
                                        filter_dilation=self.filter_dilation)
            ref_func = theano.function([img_sym, kern_sym], ref_conv_op,
                                       mode=self.ref_mode)
            ref_concat_output = [ref_func(img_arr, kern_arr)
                                 for img_arr, kern_arr in zip(split_imgs, split_kern)]
            ref_concat_output = np.concatenate(ref_concat_output, axis=1)

            utt.assert_allclose(grouped_output, ref_concat_output)

            utt.verify_grad(grouped_conv_op,
                            [img, kern],
                            mode=self.mode,
                            eps=1)

    def test_gradweights(self):
        if self.convdim == 2:
            img_sym = theano.tensor.tensor4('img')
            top_sym = theano.tensor.tensor4('kern')
        else:
            img_sym = theano.tensor.tensor5('img')
            top_sym = theano.tensor.tensor5('kern')
        for imshp, kshp, tshp, groups in zip(self.img_shape, self.kern_shape, self.top_shape, self.num_groups):
            img = np.random.random(imshp).astype(theano.config.floatX)
            top = np.random.random(tshp).astype(theano.config.floatX)
            split_imgs = np.split(img, groups, axis=1)
            split_top = np.split(top, groups, axis=1)

            grouped_convgrad_op = self.conv_gradw(border_mode=self.border_mode,
                                                  subsample=self.subsample,
                                                  filter_dilation=self.filter_dilation,
                                                  num_groups=groups)
            grouped_conv_output = grouped_convgrad_op(img_sym,
                                                      top_sym,
                                                      tensor.as_tensor_variable(
                                                          kshp[-self.convdim:]))
            grouped_func = theano.function([img_sym, top_sym], grouped_conv_output, mode=self.mode)
            assert any([isinstance(node.op, self.conv_gradw_op)
                       for node in grouped_func.maker.fgraph.toposort()])
            grouped_output = grouped_func(img, top)

            ref_conv_op = self.corr_gradw(img_sym,
                                          top_sym,
                                          kshp,
                                          border_mode=self.border_mode,
                                          subsample=self.subsample,
                                          filter_dilation=self.filter_dilation)
            ref_func = theano.function([img_sym, top_sym], ref_conv_op,
                                       mode=self.ref_mode)
            ref_concat_output = [ref_func(img_arr, top_arr)
                                 for img_arr, top_arr in zip(split_imgs, split_top)]
            ref_concat_output = np.concatenate(ref_concat_output, axis=0)

            utt.assert_allclose(grouped_output, ref_concat_output)

            def conv_gradweight(inputs_val, output_val):
                return grouped_convgrad_op(inputs_val, output_val,
                                           tensor.as_tensor_variable(
                                               kshp[-self.convdim:]))

            utt.verify_grad(conv_gradweight,
                            [img, top],
                            mode=self.mode, eps=1)

    def test_gradinputs(self):
        if self.convdim == 2:
            kern_sym = theano.tensor.tensor4('kern')
            top_sym = theano.tensor.tensor4('top')
        else:
            kern_sym = theano.tensor.tensor5('kern')
            top_sym = theano.tensor.tensor5('top')
        for imshp, kshp, tshp, groups in zip(self.img_shape, self.kern_shape, self.top_shape, self.num_groups):
            kern = np.random.random(kshp).astype(theano.config.floatX)
            top = np.random.random(tshp).astype(theano.config.floatX)
            split_kerns = np.split(kern, groups, axis=0)
            split_top = np.split(top, groups, axis=1)

            grouped_convgrad_op = self.conv_gradi(border_mode=self.border_mode,
                                                  subsample=self.subsample,
                                                  filter_dilation=self.filter_dilation,
                                                  num_groups=groups)
            grouped_conv_output = grouped_convgrad_op(kern_sym,
                                                      top_sym,
                                                      tensor.as_tensor_variable(
                                                          imshp[-self.convdim:]))
            grouped_func = theano.function([kern_sym, top_sym], grouped_conv_output, mode=self.mode)
            assert any([isinstance(node.op, self.conv_gradi_op)
                       for node in grouped_func.maker.fgraph.toposort()])
            grouped_output = grouped_func(kern, top)

            ref_conv_op = self.corr_gradi(kern_sym,
                                          top_sym,
                                          imshp,
                                          border_mode=self.border_mode,
                                          subsample=self.subsample,
                                          filter_dilation=self.filter_dilation)
            ref_func = theano.function([kern_sym, top_sym], ref_conv_op,
                                       mode=self.ref_mode)
            ref_concat_output = [ref_func(kern_arr, top_arr)
                                 for kern_arr, top_arr in zip(split_kerns, split_top)]
            ref_concat_output = np.concatenate(ref_concat_output, axis=1)

            utt.assert_allclose(grouped_output, ref_concat_output)

            def conv_gradinputs(filters_val, output_val):
                return grouped_convgrad_op(filters_val, output_val,
                                           tensor.as_tensor_variable(
                                               imshp[-self.convdim:]))

            utt.verify_grad(conv_gradinputs,
                            [kern, top],
                            mode=self.mode, eps=1)


class Grouped_conv3d_noOptim(Grouped_conv_noOptim):
    conv = theano.tensor.nnet.abstract_conv.AbstractConv3d
    conv_gradw = theano.tensor.nnet.abstract_conv.AbstractConv3d_gradWeights
    conv_gradi = theano.tensor.nnet.abstract_conv.AbstractConv3d_gradInputs
    conv_op = theano.tensor.nnet.abstract_conv.AbstractConv3d
    conv_gradw_op = theano.tensor.nnet.abstract_conv.AbstractConv3d_gradWeights
    conv_gradi_op = theano.tensor.nnet.abstract_conv.AbstractConv3d_gradInputs
    mode = theano.Mode(optimizer=None)

    def setUp(self):
        self.num_groups = [3, 2, 4, 4]
        self.border_mode = 'valid'
        self.subsample = (1, 1, 1)
        self.img_shape = [(2, 6, 5, 5, 5), (1, 4, 7, 5, 7), (1, 8, 5, 3, 5), (2, 4, 7, 7, 7)]
        self.kern_shape = [(3, 2, 3, 3, 3), (6, 2, 5, 3, 5), (4, 2, 3, 3, 3), (4, 1, 3, 5, 3)]
        self.top_shape = [(2, 3, 3, 3, 3), (1, 6, 3, 3, 3), (1, 4, 3, 1, 3), (2, 4, 5, 3, 5)]
        self.filter_dilation = (1, 1, 1)
        self.ref_mode = 'FAST_RUN'
        self.convdim = 3
        self.corr_fwd = conv3d_corr
        self.corr_gradw = conv3d_corr_gw
        self.corr_gradi = conv3d_corr_gi
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("CorrMM needs cxx")


class Separable_conv(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[[[1, 2, 3, 4, 5], [3, 2, 1, 4, 5], [3, 3, 1, 3, 6], [5, 3, 2, 1, 1], [4, 7, 1, 2, 1]],
                            [[3, 3, 1, 2, 6], [6, 5, 4, 3, 1], [3, 4, 5, 2, 3], [6, 4, 1, 3, 4], [2, 3, 4, 2, 5]]]]).astype(theano.config.floatX)

        self.depthwise_filter = np.array([[[[3, 2, 1], [5, 3, 2], [6, 4, 2]]], [[[5, 5, 2], [3, 7, 4], [3, 5, 4]]],
                                          [[[7, 4, 7], [5, 3, 3], [1, 3, 1]]], [[[4, 4, 4], [2, 4, 6], [0, 0, 7]]]]).astype(theano.config.floatX)

        self.pointwise_filter = np.array([[[[4]], [[1]], [[3]], [[5]]], [[[2]], [[1]], [[2]], [[8]]]]).astype(theano.config.floatX)

        self.precomp_output_valid = np.array([[[[1385, 1333, 1339], [1382, 1243, 1291], [1303, 1120, 1228]],
                                               [[1532, 1410, 1259], [1522, 1346, 1314], [1379, 1192, 1286]]]]).astype(theano.config.floatX)

        self.precomp_output_full = np.array([[[[140, 266, 343, 206, 59],
                                              [395, 697, 979, 585, 245],
                                              [429, 863, 1385, 919, 453],
                                              [243, 499, 864, 627, 371],
                                              [90, 183, 291, 254, 202]],

                                             [[149, 289, 359, 213, 58],
                                              [400, 750, 1076, 662, 266],
                                              [387, 854, 1532, 1091, 540],
                                              [174, 411, 971, 786, 518],
                                              [51, 110, 286, 299, 298]]]]).astype(theano.config.floatX)

    def test_interface2d(self):
        if theano.config.cxx == "":
            raise SkipTest("test needs cxx")
        x_sym = theano.tensor.tensor4('x')
        dfilter_sym = theano.tensor.tensor4('d')
        pfilter_sym = theano.tensor.tensor4('p')

        sep_op = separable_conv2d(x_sym, dfilter_sym, pfilter_sym, self.x.shape[1])
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')

        # test for square matrix
        top = fun(self.x, self.depthwise_filter, self.pointwise_filter)
        utt.assert_allclose(top, self.precomp_output_valid)

        # test for non-square matrix
        top = fun(self.x[:, :, :3, :], self.depthwise_filter, self.pointwise_filter)
        utt.assert_allclose(top, self.precomp_output_valid[:, :, :1, :])

        # test if it infers shape
        sep_op = separable_conv2d(x_sym,
                                  dfilter_sym,
                                  pfilter_sym,
                                  self.x.shape[1],
                                  input_shape=self.x.shape,
                                  depthwise_filter_shape=self.depthwise_filter.shape,
                                  pointwise_filter_shape=self.pointwise_filter.shape)
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')
        top = fun(self.x, self.depthwise_filter, self.pointwise_filter)
        utt.assert_allclose(top, self.precomp_output_valid)

        # test non-default subsample
        sep_op = separable_conv2d(x_sym,
                                  dfilter_sym,
                                  pfilter_sym,
                                  self.x.shape[1],
                                  subsample=(2, 2))
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')
        top = fun(self.x, self.depthwise_filter, self.pointwise_filter)
        utt.assert_allclose(top, np.delete(np.delete(self.precomp_output_valid, 1, axis=3), 1, axis=2))

        # test non-default border_mode
        sep_op = separable_conv2d(x_sym, dfilter_sym, pfilter_sym, self.x.shape[1], border_mode='full')
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')
        top = fun(self.x[:, :, :3, :3], self.depthwise_filter, self.pointwise_filter)
        utt.assert_allclose(top, self.precomp_output_full)

    def test_interface3d(self):
        if theano.config.cxx == "":
            raise SkipTest("test needs cxx")
        # Expand the filter along the depth
        x = np.tile(np.expand_dims(self.x, axis=2), (1, 1, 5, 1, 1))
        depthwise_filter = np.tile(np.expand_dims(self.depthwise_filter, axis=2), (1, 1, 3, 1, 1))
        pointwise_filter = np.expand_dims(self.pointwise_filter, axis=2)
        precomp_output = np.tile(np.expand_dims(self.precomp_output_valid, axis=2), (1, 1, 3, 1, 1)) * 3

        x_sym = theano.tensor.tensor5('x')
        dfilter_sym = theano.tensor.tensor5('d')
        pfilter_sym = theano.tensor.tensor5('p')

        sep_op = separable_conv3d(x_sym, dfilter_sym, pfilter_sym, x.shape[1])
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')

        # test for square matrix
        top = fun(x, depthwise_filter, pointwise_filter)
        utt.assert_allclose(top, precomp_output)
        # test for non-square matrix
        top = fun(x[:, :, :3, :, :3], depthwise_filter, pointwise_filter)
        utt.assert_allclose(top, precomp_output[:, :, :1, :, :1])
        # test if it infers shape
        sep_op = separable_conv3d(x_sym,
                                  dfilter_sym,
                                  pfilter_sym,
                                  x.shape[1],
                                  input_shape=x.shape,
                                  depthwise_filter_shape=depthwise_filter.shape,
                                  pointwise_filter_shape=pointwise_filter.shape)
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')
        top = fun(x, depthwise_filter, pointwise_filter)
        utt.assert_allclose(top, precomp_output)

        # test non-default subsample
        sep_op = separable_conv3d(x_sym,
                                  dfilter_sym,
                                  pfilter_sym,
                                  x.shape[1],
                                  subsample=(2, 2, 2))
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')
        top = fun(x, depthwise_filter, pointwise_filter)
        utt.assert_allclose(top, np.delete(np.delete(
            np.delete(precomp_output, 1, axis=4), 1, axis=3), 1, axis=2))
        # test non-default border_mode
        precomp_output = np.tile(np.expand_dims(self.precomp_output_full, axis=2),
                                 (1, 1, 5, 1, 1)) * np.array([[[[[1]], [[2]], [[3]], [[2]], [[1]]]]])

        sep_op = separable_conv3d(x_sym, dfilter_sym, pfilter_sym, x.shape[1], border_mode='full')
        fun = theano.function([x_sym, dfilter_sym, pfilter_sym], sep_op, mode='FAST_RUN')
        top = fun(x[:, :, :3, :3, :3], depthwise_filter, pointwise_filter)
        utt.assert_allclose(top, precomp_output)


class TestUnsharedConv(unittest.TestCase):
    conv2d = theano.tensor.nnet.abstract_conv.AbstractConv2d
    conv2d_gradw = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights
    conv2d_gradi = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs
    conv2d_op = theano.tensor.nnet.abstract_conv.AbstractConv2d
    conv2d_gradw_op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights
    conv2d_gradi_op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs

    mode = theano.compile.mode.Mode(optimizer='None')

    def setUp(self):
        self.img_shape = [(2, 2, 4, 4), (3, 2, 4, 2), (3, 3, 5, 3), (3, 4, 4, 4)]
        self.kern_shape = [(2, 2, 2, 2, 3, 3), (2, 4, 2, 2, 4, 2), (3, 2, 1, 1, 3, 3), (4, 3, 3, 2, 4, 2)]
        self.topgrad_shape = [(2, 2, 2, 2), (3, 2, 4, 2), (3, 3, 2, 1), (3, 4, 3, 3)]
        self.border_mode = ['valid', 'full', 'valid', 'full']
        self.subsample = [(1, 1), (2, 2), (2, 1), (3, 2)]
        self.filter_dilation = (1, 1)
        self.num_groups = [1, 1, 3, 2]

        # self.verify_flags = np.random.choice([True, False], 4, [0.5, 0.5])
        # Above line can be used instead if speed is a concern
        self.verify_flags = [True] * 4

        self.ref_mode = 'FAST_RUN'
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("CorrMM needs cxx or SciPy")

    def test_fwd(self):
        tensor6 = theano.tensor.TensorType(theano.config.floatX, (False,) * 6)
        img_sym = theano.tensor.tensor4('img')
        kern_sym = tensor6('kern')
        ref_kern_sym = theano.tensor.tensor4('ref_kern')

        for imshp, kshp, mode, sub, groups, verify in zip(self.img_shape, self.kern_shape, self.border_mode,
                                                          self.subsample, self.num_groups, self.verify_flags):
            img = np.random.random(imshp).astype(theano.config.floatX)
            kern = np.random.random(kshp).astype(theano.config.floatX)

            unshared_conv_op = self.conv2d(border_mode=mode, subsample=sub,
                                           filter_dilation=self.filter_dilation,
                                           num_groups=groups, unshared=True)
            unshared_out_sym = unshared_conv_op(img_sym, kern_sym)
            unshared_func = theano.function([img_sym, kern_sym], unshared_out_sym, mode=self.mode)
            assert any([isinstance(node.op, self.conv2d_op)
                        for node in unshared_func.maker.fgraph.toposort()])
            unshared_output = unshared_func(img, kern)

            single_kshp = kshp[:1] + kshp[3:]

            ref_conv_op = self.conv2d(border_mode=mode, subsample=sub,
                                      filter_dilation=self.filter_dilation,
                                      num_groups=groups, unshared=False)
            ref_out_sym = ref_conv_op(img_sym, ref_kern_sym)
            ref_func = theano.function([img_sym, ref_kern_sym], ref_out_sym, mode=self.mode)

            for i in range(0, kshp[1]):
                for j in range(0, kshp[2]):
                    single_kern = kern[:, i, j, ...].reshape(single_kshp)
                    ref_val = ref_func(img, single_kern)
                    utt.assert_allclose(ref_val[:, :, i, j], unshared_output[:, :, i, j])

            if verify:
                utt.verify_grad(unshared_conv_op, [img, kern], mode=self.mode, eps=1)

    def test_gradweight(self):
        img_sym = theano.tensor.tensor4('img')
        top_sym = theano.tensor.tensor4('top')

        for imshp, kshp, topshp, mode, sub, groups, verify in zip(self.img_shape, self.kern_shape, self.topgrad_shape,
                                                                  self.border_mode, self.subsample, self.num_groups,
                                                                  self.verify_flags):
            img = np.random.random(imshp).astype(theano.config.floatX)
            top = np.random.random(topshp).astype(theano.config.floatX)

            unshared_conv_op = self.conv2d_gradw(border_mode=mode, subsample=sub,
                                                 filter_dilation=self.filter_dilation,
                                                 num_groups=groups, unshared=True)
            unshared_out_sym = unshared_conv_op(img_sym, top_sym, tensor.as_tensor_variable(kshp[-2:]))
            unshared_func = theano.function([img_sym, top_sym], unshared_out_sym, mode=self.mode)
            assert any([isinstance(node.op, self.conv2d_gradw_op)
                        for node in unshared_func.maker.fgraph.toposort()])
            unshared_output = unshared_func(img, top)

            single_kshp = kshp[:1] + kshp[3:]

            ref_conv_op = self.conv2d_gradw(border_mode=mode, subsample=sub,
                                            filter_dilation=self.filter_dilation,
                                            num_groups=groups, unshared=False)
            ref_out_sym = ref_conv_op(img_sym, top_sym, tensor.as_tensor_variable(single_kshp[-2:]))
            ref_func = theano.function([img_sym, top_sym], ref_out_sym, mode=self.mode)

            for i in range(0, topshp[2]):
                for j in range(0, topshp[3]):
                    top_single = np.zeros_like(top)
                    top_single[:, :, i, j] = top[:, :, i, j]
                    ref_output = ref_func(img, top_single)
                    utt.assert_allclose(unshared_output[:, i, j, ...], ref_output)

            def conv_gradweight(inputs_val, output_val):
                return unshared_conv_op(inputs_val, output_val, tensor.as_tensor_variable(kshp[-2:]))

            if verify:
                utt.verify_grad(conv_gradweight, [img, top], mode=self.mode, eps=1)

    def test_gradinput(self):
        tensor6 = theano.tensor.TensorType(theano.config.floatX, (False,) * 6)
        kern_sym = tensor6('kern')
        top_sym = theano.tensor.tensor4('top')
        ref_kern_sym = theano.tensor.tensor4('ref_kern')

        for imshp, kshp, topshp, mode, sub, groups, verify in zip(self.img_shape, self.kern_shape, self.topgrad_shape,
                                                                  self.border_mode, self.subsample, self.num_groups,
                                                                  self.verify_flags):
            single_kshp = kshp[:1] + kshp[3:]

            kern = np.random.random(kshp).astype(theano.config.floatX)
            top = np.random.random(topshp).astype(theano.config.floatX)

            unshared_conv_op = self.conv2d_gradi(border_mode=mode, subsample=sub,
                                                 filter_dilation=self.filter_dilation,
                                                 num_groups=groups, unshared=True)
            unshared_out_sym = unshared_conv_op(kern_sym, top_sym, tensor.as_tensor_variable(imshp[-2:]))
            unshared_func = theano.function([kern_sym, top_sym], unshared_out_sym, mode=self.mode)
            assert any([isinstance(node.op, self.conv2d_gradi_op)
                        for node in unshared_func.maker.fgraph.toposort()])
            unshared_output = unshared_func(kern, top)

            ref_conv_op = self.conv2d_gradi(border_mode=mode, subsample=sub,
                                            filter_dilation=self.filter_dilation,
                                            num_groups=groups, unshared=False)
            ref_out_sym = ref_conv_op(ref_kern_sym, top_sym, tensor.as_tensor_variable(imshp[-2:]))
            ref_func = theano.function([ref_kern_sym, top_sym], ref_out_sym, mode=self.mode)

            ref_output = np.zeros(imshp)

            for i in range(0, topshp[2]):
                for j in range(0, topshp[3]):
                    single_kern = kern[:, i, j, ...].reshape(single_kshp)
                    top_single = np.zeros_like(top)
                    top_single[:, :, i, j] = top[:, :, i, j]
                    ref_output += ref_func(single_kern, top_single)

            utt.assert_allclose(ref_output, unshared_output)

            def conv_gradinputs(filters_val, output_val):
                return unshared_conv_op(filters_val, output_val, tensor.as_tensor_variable(imshp[-2:]))

            if verify:
                utt.verify_grad(conv_gradinputs, [kern, top], mode=self.mode, eps=1)


class TestAsymmetricPadding(unittest.TestCase):
    conv2d = theano.tensor.nnet.abstract_conv.AbstractConv2d
    conv2d_gradw = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights
    conv2d_gradi = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs
    conv2d_op = theano.tensor.nnet.abstract_conv.AbstractConv2d
    conv2d_gradw_op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights
    conv2d_gradi_op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs

    mode = theano.compile.mode.Mode(optimizer='None')

    img_shape = [(2, 2, 4, 4), (3, 2, 4, 2), (3, 3, 5, 3)]
    kern_shape = [(4, 2, 2, 2), (2, 2, 4, 2), (2, 3, 3, 3)]
    topgrad_shape = [(2, 4, 6, 6), (3, 2, 3, 4), (3, 2, 6, 1)]
    border_mode = [((1, 2), (2, 1)), ((1, 1), (0, 3)), ((2, 1), (0, 0))]

    def test_fwd(self):
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("SciPy and cxx needed")
        img_sym = theano.tensor.tensor4('img')
        kern_sym = theano.tensor.tensor4('kern')

        for imshp, kshp, pad in zip(self.img_shape, self.kern_shape, self.border_mode):
            img = np.random.random(imshp).astype(theano.config.floatX)
            kern = np.random.random(kshp).astype(theano.config.floatX)

            asymmetric_conv_op = self.conv2d(border_mode=pad, subsample=(1, 1),
                                             filter_dilation=(1, 1))
            asymmetric_out_sym = asymmetric_conv_op(img_sym, kern_sym)
            asymmetric_func = theano.function([img_sym, kern_sym], asymmetric_out_sym, mode=self.mode)
            assert any([isinstance(node.op, self.conv2d_op)
                        for node in asymmetric_func.maker.fgraph.toposort()])
            asymmetric_output = asymmetric_func(img, kern)

            ref_conv_op = self.conv2d(border_mode="valid", subsample=(1, 1),
                                      filter_dilation=(1, 1))
            ref_out_sym = ref_conv_op(img_sym, kern_sym)
            ref_func = theano.function([img_sym, kern_sym], ref_out_sym, mode=self.mode)

            exp_imshp = (imshp[0], imshp[1],
                         imshp[2] + pad[0][0] + pad[0][1],
                         imshp[3] + pad[1][0] + pad[1][1])

            exp_img = np.zeros(exp_imshp, dtype=theano.config.floatX)
            exp_img[:, :, pad[0][0]:imshp[2] + pad[0][0],
                    pad[1][0]:imshp[3] + pad[1][0]] = img
            ref_output = ref_func(exp_img, kern)

            utt.assert_allclose(asymmetric_output, ref_output)

            utt.verify_grad(asymmetric_conv_op, [img, kern], mode=self.mode, eps=1)

    def test_gradweight(self):
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("SciPy and cxx needed")

        img_sym = theano.tensor.tensor4('img')
        top_sym = theano.tensor.tensor4('top')

        for imshp, kshp, topshp, pad in zip(self.img_shape, self.kern_shape, self.topgrad_shape, self.border_mode):
            img = np.random.random(imshp).astype(theano.config.floatX)
            top = np.random.random(topshp).astype(theano.config.floatX)

            asymmetric_conv_op = self.conv2d_gradw(border_mode=pad, subsample=(1, 1),
                                                   filter_dilation=(1, 1))
            asymmetric_out_sym = asymmetric_conv_op(img_sym, top_sym, kshp[-2:])
            asymmetric_func = theano.function([img_sym, top_sym], asymmetric_out_sym, mode=self.mode)
            assert any([isinstance(node.op, self.conv2d_gradw_op)
                        for node in asymmetric_func.maker.fgraph.toposort()])
            asymmetric_output = asymmetric_func(img, top)

            ref_conv_op = self.conv2d_gradw(border_mode="valid", subsample=(1, 1),
                                            filter_dilation=(1, 1))
            ref_out_sym = ref_conv_op(img_sym, top_sym, kshp[-2:])
            ref_func = theano.function([img_sym, top_sym], ref_out_sym, mode=self.mode)

            exp_imshp = (imshp[0], imshp[1],
                         imshp[2] + pad[0][0] + pad[0][1],
                         imshp[3] + pad[1][0] + pad[1][1])

            exp_img = np.zeros(exp_imshp, dtype=theano.config.floatX)
            exp_img[:, :, pad[0][0]:imshp[2] + pad[0][0],
                    pad[1][0]:imshp[3] + pad[1][0]] = img
            ref_output = ref_func(exp_img, top)

            utt.assert_allclose(asymmetric_output, ref_output)

            def conv_gradweight(inputs_val, output_val):
                return asymmetric_conv_op(inputs_val, output_val, tensor.as_tensor_variable(kshp[-2:]))

            utt.verify_grad(conv_gradweight, [img, top], mode=self.mode, eps=1)

    def test_gradinput(self):
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("test needs cxx and SciPy")
        kern_sym = theano.tensor.tensor4('kern')
        top_sym = theano.tensor.tensor4('top')

        for imshp, kshp, topshp, pad in zip(self.img_shape, self.kern_shape, self.topgrad_shape, self.border_mode):
            kern = np.random.random(kshp).astype(theano.config.floatX)
            top = np.random.random(topshp).astype(theano.config.floatX)

            asymmetric_conv_op = self.conv2d_gradi(border_mode=pad, subsample=(1, 1),
                                                   filter_dilation=(1, 1))
            asymmetric_out_sym = asymmetric_conv_op(kern_sym, top_sym, imshp[-2:])
            asymmetric_func = theano.function([kern_sym, top_sym], asymmetric_out_sym, mode=self.mode)
            assert any([isinstance(node.op, self.conv2d_gradi_op)
                        for node in asymmetric_func.maker.fgraph.toposort()])
            asymmetric_output = asymmetric_func(kern, top)

            ref_conv_op = self.conv2d_gradi(border_mode="valid", subsample=(1, 1),
                                            filter_dilation=(1, 1))
            exp_imshp = [imshp[2] + pad[0][0] + pad[0][1],
                         imshp[3] + pad[1][0] + pad[1][1]]
            ref_out_sym = ref_conv_op(kern_sym, top_sym, exp_imshp)
            ref_func = theano.function([kern_sym, top_sym], ref_out_sym, mode=self.mode)

            ref_output = ref_func(kern, top)

            ref_output = ref_output[:, :, pad[0][0]:imshp[2] + pad[0][0],
                                    pad[1][0]:imshp[3] + pad[1][0]]

            utt.assert_allclose(asymmetric_output, ref_output)

            def conv_gradinputs(filters_val, output_val):
                return asymmetric_conv_op(filters_val, output_val, tensor.as_tensor_variable(imshp[-2:]))

            utt.verify_grad(conv_gradinputs, [kern, top], mode=self.mode, eps=1)


class TestCausalConv(unittest.TestCase):
    mode = theano.compile.mode.Mode(optimizer='None')

    img = np.array([[[2, 4, 9, 5, 8], [0, 0, 4, 0, 5]],
                    [[2, 5, 8, 5, 5], [1, 3, 0, 7, 9]],
                    [[7, 0, 7, 1, 0], [0, 1, 4, 7, 2]]]).astype(theano.config.floatX)
    kern = np.array([[[5, 3, 1], [3, 1, 0]],
                     [[6, 4, 9], [2, 2, 7]]]).astype(theano.config.floatX)
    dilation = 2
    precomp_top = np.array([[[10, 20, 63, 37, 88], [12, 24, 70, 46, 120]],
                            [[13, 34, 47, 64, 78], [14, 36, 58, 70, 105]],
                            [[35, 3, 68, 27, 38], [42, 2, 78, 22, 103]]]).astype(theano.config.floatX)

    def test_interface(self):
        img_sym = theano.tensor.tensor3('img')
        kern_sym = theano.tensor.tensor3('kern')
        if theano.config.cxx == "" or not theano.tensor.nnet.abstract_conv.imported_scipy_signal:
            raise SkipTest("SciPy and cxx needed")
        sym_out = causal_conv1d(img_sym, kern_sym, self.kern.shape, filter_dilation=self.dilation)

        causal_func = theano.function([img_sym, kern_sym], sym_out, mode=self.mode)

        output = causal_func(self.img, self.kern)

        utt.assert_allclose(output, self.precomp_top)

        def causal_conv_fn(inputs_val, filters_val):
            return causal_conv1d(inputs_val, filters_val, self.kern.shape, filter_dilation=1)

        utt.verify_grad(causal_conv_fn, [self.img, self.kern], mode=self.mode, eps=1)
