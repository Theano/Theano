from __future__ import absolute_import, print_function, division

from itertools import product, chain

import numpy as np

import theano
import theano.tests.unittest_tools as utt
from theano.configdefaults import (SUPPORTED_DNN_CONV_ALGO_FWD, SUPPORTED_DNN_CONV_ALGO_BWD_FILTER, SUPPORTED_DNN_CONV_ALGO_BWD_DATA)
from .config import mode_with_gpu, ref_cast
from .. import dnn


SUPPORTED_DNN_CONV_PRECISION = ('float16', 'float32', 'float64')


def get_available_precisions():
    # Starting from floatX and up to max supported precision (float64).
    return SUPPORTED_DNN_CONV_PRECISION[SUPPORTED_DNN_CONV_PRECISION.index(theano.config.floatX):]


class BaseTestDnnConv(object):
    functions_checked_for_fwd = False
    functions_checked_for_gradinput = False
    functions_checked_for_gradweights = False

    # Abstract functions.

    def get_gpu_conv_function(self):
        raise NotImplementedError

    def get_cpu_conv_class(self):
        raise NotImplementedError

    def get_cases(self):
        # Should return an iterable of test cases. Each test case is a tuple (or list) with following syntax:
        # ( (input shape, filter shape, subsample, dilation), border mode, convolution mode )
        raise NotImplementedError

    def run_conv3d_fwd(self, algo, precision, inputs_shape, filters_shape, subsample, dilation, border_mode, conv_mode):
        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)

        # Scale down the input values to prevent very large absolute errors
        # due to float rounding
        inputs_val /= 10
        filters_val /= 10

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

        # Compile a theano function for the cuDNN implementation
        conv = self.get_gpu_conv_function()(img=inputs, kerns=filters,
                                            border_mode=border_mode, subsample=subsample,
                                            dilation=dilation,
                                            conv_mode=conv_mode,
                                            algo=algo,
                                            precision=precision)
        f = theano.function([], conv, mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        conv_ref = self.get_cpu_conv_class()(border_mode=border_mode,
                                             subsample=subsample,
                                             filter_dilation=dilation,
                                             )(ref_cast(inputs), flipped_filters)
        f_ref = theano.function([], conv_ref, mode="FAST_RUN")

        if not self.functions_checked_for_fwd:
            self.functions_checked_for_fwd = True
            assert any(isinstance(node.op, dnn.GpuDnnConv) for node in f.maker.fgraph.apply_nodes)
            assert not any(isinstance(node.op, (dnn.GpuDnnConvGradI, dnn.GpuDnnConvGradW))
                           for node in f.maker.fgraph.apply_nodes)

            assert not any(isinstance(node.op, (dnn.GpuDnnConv, dnn.GpuDnnConvGradW, dnn.GpuDnnConvGradI))
                           for node in f_ref.maker.fgraph.apply_nodes)

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        # raise rtol to make the test pass with more seed.
        rtol = None
        # Raise tolerance for float16
        if theano.config.floatX == 'float16':
            rtol = 6e-2
        utt.assert_allclose(res_ref, res, rtol=rtol)

    def run_conv3d_gradinput(self, algo, precision, inputs_shape, filters_shape, subsample, dilation, border_mode, conv_mode):

        theano.config.dnn.conv.algo_bwd_data = algo
        theano.config.dnn.conv.precision = precision

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

        # Compile a theano function for the cuDNN implementation
        conv = self.get_gpu_conv_function()(img=inputs, kerns=filters,
                                            border_mode=border_mode,
                                            subsample=subsample,
                                            dilation=dilation,
                                            conv_mode=conv_mode)

        grad_i = theano.tensor.grad(conv.sum(), [inputs])

        f = theano.function([], grad_i, mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        conv_ref = self.get_cpu_conv_class()(border_mode=border_mode,
                                             subsample=subsample,
                                             filter_dilation=dilation,
                                             )(ref_cast(inputs), flipped_filters)
        grad_i_ref, = theano.tensor.grad(conv_ref.sum(), [inputs])
        f_ref = theano.function([], grad_i_ref, mode="FAST_RUN")

        if not self.functions_checked_for_gradinput:
            self.functions_checked_for_gradinput = True
            assert any(isinstance(node.op, dnn.GpuDnnConvGradI) for node in f.maker.fgraph.apply_nodes)
            assert not any(isinstance(node.op, (dnn.GpuDnnConv, dnn.GpuDnnConvGradW))
                           for node in f.maker.fgraph.apply_nodes)

            assert not any(isinstance(node.op, (dnn.GpuDnnConv, dnn.GpuDnnConvGradW, dnn.GpuDnnConvGradI))
                           for node in f_ref.maker.fgraph.apply_nodes)

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        # Needed for big size for some seed
        # raise rtol to make the test pass with more seed.
        rtol = None
        # Raise tolerance for float16
        if theano.config.floatX == 'float16':
            rtol = 5e-2
        utt.assert_allclose(res_ref, res, rtol=rtol)

    def run_conv3d_gradweights(self, algo, precision, inputs_shape, filters_shape, subsample, dilation, border_mode, conv_mode):

        theano.config.dnn.conv.algo_bwd_filter = algo
        theano.config.dnn.conv.precision = precision

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

        # Compile a theano function for the cuDNN implementation
        conv = self.get_gpu_conv_function()(img=inputs, kerns=filters,
                                            border_mode=border_mode,
                                            subsample=subsample,
                                            dilation=dilation,
                                            conv_mode=conv_mode)

        grad_w, = theano.tensor.grad(conv.sum(), [filters])

        f = theano.function([], grad_w, mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        conv_ref = self.get_cpu_conv_class()(border_mode=border_mode,
                                             subsample=subsample,
                                             filter_dilation=dilation,
                                             )(ref_cast(inputs), flipped_filters)
        grad_w_ref, = theano.tensor.grad(conv_ref.sum(), [filters])
        f_ref = theano.function([], grad_w_ref, mode="FAST_RUN")

        if not self.functions_checked_for_gradweights:
            self.functions_checked_for_gradweights = True
            assert any(isinstance(node.op, dnn.GpuDnnConvGradW) for node in f.maker.fgraph.apply_nodes)
            assert not any(isinstance(node.op, (dnn.GpuDnnConv, dnn.GpuDnnConvGradI))
                           for node in f.maker.fgraph.apply_nodes)

            assert not any(isinstance(node.op, (dnn.GpuDnnConv, dnn.GpuDnnConvGradW, dnn.GpuDnnConvGradI))
                           for node in f_ref.maker.fgraph.apply_nodes)

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        # Needed for big size for some seed
        # raise rtol to make the test pass with more seed.
        rtol = None
        # Raise tolerance for float16
        if theano.config.floatX == 'float16':
            rtol = 5e-2
        utt.assert_allclose(res_ref, res, rtol=rtol)

    def test_fwd(self):
        for precision in get_available_precisions():
            for algo in SUPPORTED_DNN_CONV_ALGO_FWD:
                for (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode in self.get_cases():
                    yield (self.run_conv3d_fwd, algo, precision, inputs_shape, filters_shape,
                           subsample, dilation, border_mode, conv_mode)

    def test_gradinput(self):
        for precision in get_available_precisions():
            for algo in SUPPORTED_DNN_CONV_ALGO_BWD_DATA:
                for (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode in self.get_cases():
                    yield (self.run_conv3d_gradinput, algo, precision, inputs_shape, filters_shape,
                           subsample, dilation, border_mode, conv_mode)

    def test_gradweights(self):
        for precision in get_available_precisions():
            for algo in SUPPORTED_DNN_CONV_ALGO_BWD_FILTER:
                for (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode in self.get_cases():
                    yield (self.run_conv3d_gradweights, algo, precision, inputs_shape, filters_shape,
                           subsample, dilation, border_mode, conv_mode)


class TestDnnConv2D(BaseTestDnnConv):
    def get_gpu_conv_function(self):
        return dnn.dnn_conv

    def get_cpu_conv_class(self):
        return theano.tensor.nnet.corr.CorrMM

    def get_cases(self):
        # Inspired from:
        # - theano.tensor.nnet.tests.test_abstract_conv.BaseTestConv2d#setup_class
        # - theano.tensor.nnet.tests.test_abstract_conv.BaseTestConv#test_all

        inputs_shapes = [(8, 1, 6, 6), (8, 1, 8, 8), (2, 1, 7, 7),
                         (6, 1, 10, 11), (2, 1, 6, 5), (1, 5, 9, 9),
                         (0, 1, 6, 6), (1, 0, 6, 6), (1, 1, 6, 6)]
        filters_shapes = [(5, 1, 2, 2), (4, 1, 3, 3), (2, 1, 3, 3),
                          (1, 1, 2, 3), (4, 1, 1, 3), (4, 5, 3, 2),
                          (1, 1, 2, 2), (1, 0, 2, 2), (0, 1, 2, 2)]
        subsamples = [(1, 1), (2, 2), (2, 4)]
        dilations = [(1, 1), (1, 2), (2, 1)]
        default_subsample = (1, 1)
        default_dilation = (1, 1)
        border_modes = ["valid", "half", "full", (0, 0), (1, 1), (5, 5), (5, 2)]
        conv_modes = ['conv', 'cross']

        assert len(inputs_shapes) == len(filters_shapes)

        iterables = []

        for input_shape, filter_shape in zip(inputs_shapes, filters_shapes):
            if 0 not in input_shape and 0 not in filter_shape:
                local_subsamples = subsamples
                local_dilations = dilations
            else:
                local_subsamples = [default_subsample]
                local_dilations = [default_dilation]
            iterables += [product(product([input_shape], [filter_shape], local_subsamples, local_dilations),
                                  border_modes,
                                  conv_modes)]

        return chain(*iterables)


class TestDnnConv3D(BaseTestDnnConv):
    def get_gpu_conv_function(self):
        return dnn.dnn_conv3d

    def get_cpu_conv_class(self):
        return theano.tensor.nnet.corr3d.Corr3dMM

    def get_cases(self):
        # small case for quick test.
        input_shape = (128, 3, 5, 5, 5)
        filter_shape = (64, 3, 1, 2, 4)
        subsample = (1, 1, 1)
        dilation = (1, 1, 1)
        border_mode = 'valid'
        conv_mode = 'conv'
        return (((input_shape, filter_shape, subsample, dilation), border_mode, conv_mode),)

    def get_cases_real(self):
        # Copy of: theano.gpuarray.tests.test_dnn.get_conv3d_test_cases

        # Every element of test_shapes follows the format
        # [input_shape, filter_shape, subsample, dilation]
        test_shapes = [[(128, 3, 5, 5, 5), (64, 3, 1, 2, 4), (1, 1, 1), (1, 1, 1)],
                       [(8, 4, 20, 12, 15), (5, 4, 6, 12, 4), (2, 2, 2), (1, 1, 1)],
                       [(8, 1, 20, 12, 15), (5, 1, 6, 12, 4), (3, 3, 3), (1, 1, 1)],
                       [(8, 1, 20, 12, 15), (5, 1, 6, 12, 4), (3, 2, 1), (1, 1, 1)],
                       # Test with 1x1x1 filters
                       [(8, 1, 10, 10, 10), (10, 1, 1, 1, 1), (1, 1, 1), (1, 1, 1)],
                       # Test with dimensions larger than 1024 (thread block dim)
                       [(1025, 1, 2, 3, 4), (5, 1, 1, 2, 3), (1, 1, 1), (1, 1, 1)],
                       [(8, 1, 2, 3, 4), (1025, 1, 1, 2, 3), (1, 1, 1), (1, 1, 1)],
                       [(8, 1025, 2, 3, 4), (5, 1025, 1, 1, 2), (1, 1, 1), (1, 1, 1)],
                       [(8, 1, 1030, 3, 4), (5, 1, 1025, 1, 1), (1, 1, 1), (1, 1, 1)],
                       [(8, 1, 2, 1030, 4), (5, 1, 2, 1025, 1), (1, 1, 1), (1, 1, 1)],
                       [(8, 1, 2, 3, 1030), (5, 1, 1, 2, 1025), (1, 1, 1), (1, 1, 1)],
                       # The equivalent of this caused a crash with conv2d
                       [(1, 1, 1, 44800, 1), (6, 1, 1, 1, 1), (1, 1, 1), (1, 1, 1)]]

        # With border mode 'full', test with kernel bigger than image in some/all
        # dimensions
        test_shapes_full = [[(6, 2, 2, 2, 2), (4, 2, 3, 1, 1), (1, 1, 1), (1, 1, 1)],
                            [(6, 2, 2, 2, 2), (4, 2, 1, 3, 1), (1, 1, 1), (1, 1, 1)],
                            [(6, 2, 2, 2, 2), (4, 2, 1, 1, 3), (1, 1, 1), (1, 1, 1)],
                            [(6, 2, 2, 2, 2), (4, 2, 5, 5, 5), (1, 1, 1), (1, 1, 1)]]

        if dnn.version() >= 6000:
            test_shapes.extend([
                [(8, 1, 20, 12, 15), (5, 1, 6, 3, 4), (1, 1, 2), (3, 2, 1)],
                [(8, 1, 20, 12, 15), (5, 1, 6, 3, 4), (2, 2, 1), (1, 2, 3)]])
            test_shapes_full.append(
                [(6, 2, 2, 2, 2), (4, 2, 5, 5, 5), (1, 1, 1), (3, 2, 1)])

        border_modes = ['valid', 'full', 'half', (1, 2, 3), (3, 2, 1), 1, 2]
        conv_modes = ['conv', 'cross']

        itt = chain(product(test_shapes, border_modes, conv_modes),
                    product(test_shapes_full, ['full'], conv_modes))

        return itt
