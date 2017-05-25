from __future__ import absolute_import, print_function, division

from itertools import product, chain

import numpy as np

import theano
import theano.tests.unittest_tools as utt
from theano.compile.ops import shape_i_op
from theano.configdefaults import (SUPPORTED_DNN_CONV_ALGO_FWD, SUPPORTED_DNN_CONV3D_ALGO_FWD,
                                   SUPPORTED_DNN_CONV_ALGO_BWD_FILTER, SUPPORTED_DNN_CONV3D_ALGO_BWD_FILTER,
                                   SUPPORTED_DNN_CONV_ALGO_BWD_DATA, SUPPORTED_DNN_CONV3D_ALGO_BWD_DATA)
from theano.tensor.nnet.abstract_conv import get_conv_output_shape, assert_conv_shape
from theano.tensor.opt import Assert
from .config import mode_with_gpu, ref_cast
from ..basic_ops import infer_context_name, as_gpuarray_variable, gpu_contiguous, GpuAllocEmpty
from ..dnn import (GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI, version, get_precision)

PRECISIONS = ('float16', 'float32', 'float64')


def get_available_precisions():
    # Starting from floatX up to max supported precision (float64).
    return PRECISIONS[PRECISIONS.index(theano.config.floatX):]


def array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation):
    # Return an random array with inferred convolution output shape.
    out_shp = get_conv_output_shape(inputs_shape, filters_shape,
                                    border_mode,
                                    subsample,
                                    filter_dilation=dilation)
    out_shp = assert_conv_shape(out_shp)
    return np.random.random(out_shp).astype(theano.config.floatX)


# We provide a special implementation of dnn_conv, dnn_gradweight and dnn_gradinput
# that take algo, alpha, beta and out as parameters.

def dnn_conv(img, kerns, alpha=1, beta=0, out=None, border_mode='valid', subsample=(1, 1), dilation=(1, 1),
             conv_mode='conv', algo=None, precision=None):
    # Establish dtype in which to perform the computation of the convolution
    precision = get_precision(precision, [img, kerns])

    ctx_name = infer_context_name(img, kerns)

    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                          conv_mode=conv_mode, precision=precision)(kerns.shape)
    desc_op = desc.owner.op
    # We can use Shape_i and bypass the infer_shape here as this is on
    # the input of node and it will always be present.
    ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
    kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
    out_shp = get_conv_output_shape(ishape, kshape,
                                    desc_op.border_mode,
                                    desc_op.subsample,
                                    filter_dilation=dilation)
    out_shp = assert_conv_shape(out_shp)
    if beta != 0:
        assert out is not None
        out = as_gpuarray_variable(out, ctx_name)
        out = gpu_contiguous(out)
        check = Assert('GpuDnnConv: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, out_shp)))
    else:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
    return GpuDnnConv(algo=algo)(img, kerns, real_out, desc, alpha, beta)


def dnn_gradweight(img, topgrad, kerns_shp, alpha=1, beta=0, out=None, border_mode='valid', subsample=(1, 1),
                   dilation=(1, 1), conv_mode='conv', algo=None, precision=None):
    ctx_name = infer_context_name(img, topgrad)

    img = as_gpuarray_variable(img, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)

    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)

    kerns_shp = theano.tensor.as_tensor_variable(kerns_shp)
    precision = get_precision(precision, [img, topgrad])

    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                          conv_mode=conv_mode, precision=precision)(kerns_shp)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*kerns_shp)
    else:
        assert out is not None
        out = as_gpuarray_variable(out, ctx_name)
        out = gpu_contiguous(out)
        check = Assert('GpuDnnConvGradW: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, kerns_shp)))

    return GpuDnnConvGradW(algo=algo)(img, topgrad, real_out, desc, alpha, beta)


def dnn_gradinput(kerns, topgrad, img_shp, alpha=1, beta=0, out=None, border_mode='valid', subsample=(1, 1),
                  dilation=(1, 1), conv_mode='conv', algo=None, precision=None):
    ctx_name = infer_context_name(kerns, topgrad)

    kerns = as_gpuarray_variable(kerns, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)

    kerns = gpu_contiguous(kerns)
    topgrad = gpu_contiguous(topgrad)

    img_shp = theano.tensor.as_tensor_variable(img_shp)
    precision = get_precision(precision, [kerns, topgrad])

    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                          conv_mode=conv_mode, precision=precision)(kerns.shape)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=kerns.dtype, context_name=ctx_name)(*img_shp)
    else:
        assert out is not None
        out = as_gpuarray_variable(out, ctx_name)
        out = gpu_contiguous(out)
        check = Assert('GpuDnnConvGradI: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, img_shp)))

    return GpuDnnConvGradI(algo=algo)(kerns, topgrad, real_out, desc, alpha, beta)


class BaseTestDnnConv(object):
    _functions_checked_for_fwd = False
    _functions_checked_for_gradinput = False
    _functions_checked_for_gradweight = False

    # Abstract attributes.

    fwd_algorithms = None
    bwd_filter_algorithms = None
    bwd_data_algorithms = None

    cpu_conv_class = None
    cpu_gradinput_class = None
    cpu_gradweight_class = None

    # Abstract methods.

    def get_cases(self):
        # Should return an iterable of test cases. Each test case is a tuple (or list) with following syntax:
        # ( (input shape, filter shape, subsample, dilation), border mode, convolution mode, alpha, beta )
        raise NotImplementedError

    # Run methods.

    def run_conv_fwd(self, algo, precision, parameters):
        (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)

        # Scale down the input values to prevent very large absolute errors
        # due to float rounding
        inputs_val /= 10
        filters_val /= 10

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

        out = None if beta == 0 else array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample,
                                                            dilation)
        # Compile a theano function for the cuDNN implementation
        conv = dnn_conv(img=inputs, kerns=filters, alpha=alpha, beta=beta, out=out, border_mode=border_mode,
                        subsample=subsample, dilation=dilation, conv_mode=conv_mode, algo=algo, precision=precision)
        f = theano.function([], conv, mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            if inputs.ndim == 5:
                flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
            else:
                flipped_filters = filters[:, :, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        conv_ref = self.cpu_conv_class(border_mode=border_mode,
                                       subsample=subsample,
                                       filter_dilation=dilation)(ref_cast(inputs), flipped_filters)
        f_ref = theano.function([], conv_ref, mode="FAST_RUN")

        if not self._functions_checked_for_fwd:
            self._functions_checked_for_fwd = True
            assert any(isinstance(node.op, GpuDnnConv) for node in f.maker.fgraph.apply_nodes)
            assert not any(isinstance(node.op, (GpuDnnConvGradI, GpuDnnConvGradW))
                           for node in f.maker.fgraph.apply_nodes)

            assert not any(isinstance(node.op, (GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI))
                           for node in f_ref.maker.fgraph.apply_nodes)

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()

        # Raise tolerance for float16
        rtol = 6e-2 if theano.config.floatX == 'float16' else None
        if beta == 0:
            utt.assert_allclose(alpha * res_ref, res, rtol=rtol)
        else:
            print('(conv: beta not null) ', end='')
            utt.assert_allclose(alpha * res_ref + beta * out, res, rtol=rtol)

    def run_conv_gradinput(self, algo, precision, parameters):
        (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)
        topgrad_val = array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation)

        filters = theano.shared(filters_val)
        topgrad = theano.shared(topgrad_val)

        # Compile a theano function for the cuDNN implementation
        grad_i = dnn_gradinput(filters, topgrad, inputs_shape, alpha=alpha, beta=beta, out=inputs_val,
                               border_mode=border_mode, subsample=subsample, dilation=dilation, conv_mode=conv_mode,
                               algo=algo, precision=precision)

        f = theano.function([], grad_i, mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            if filters.ndim == 5:
                flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
            else:
                flipped_filters = filters[:, :, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        grad_i_ref = self.cpu_gradinput_class(border_mode=border_mode,
                                              subsample=subsample,
                                              filter_dilation=dilation
                                              )(ref_cast(flipped_filters), ref_cast(topgrad), inputs_shape[2:])
        f_ref = theano.function([], grad_i_ref, mode="FAST_RUN")

        if not self._functions_checked_for_gradinput:
            self._functions_checked_for_gradinput = True
            assert any(isinstance(node.op, GpuDnnConvGradI) for node in f.maker.fgraph.apply_nodes)
            assert not any(isinstance(node.op, (GpuDnnConv, GpuDnnConvGradW))
                           for node in f.maker.fgraph.apply_nodes)

            assert not any(isinstance(node.op, (GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI))
                           for node in f_ref.maker.fgraph.apply_nodes)

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        # Needed for big size for some seed
        # raise rtol to make the test pass with more seed.

        # Raise tolerance for float16
        rtol = 5e-2 if theano.config.floatX == 'float16' else None
        utt.assert_allclose(alpha * res_ref + beta * inputs_val, res, rtol=rtol)

    def run_conv_gradweight(self, algo, precision, parameters):
        (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)
        topgrad_val = array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation)

        inputs = theano.shared(inputs_val)
        topgrad = theano.shared(topgrad_val)

        # Compile a theano function for the cuDNN implementation
        grad_w = dnn_gradweight(inputs, topgrad, filters_shape, alpha=alpha, beta=beta, out=filters_val,
                                border_mode=border_mode, subsample=subsample, dilation=dilation, conv_mode=conv_mode,
                                algo=algo, precision=precision)

        f = theano.function([], grad_w, mode=mode_with_gpu)

        # Compile a theano function for the reference implementation
        grad_w_ref = self.cpu_gradweight_class(border_mode=border_mode,
                                               subsample=subsample,
                                               filter_dilation=dilation)(ref_cast(inputs), ref_cast(topgrad),
                                                                         filters_shape[2:])
        if conv_mode == 'conv':
            grad_w_ref = grad_w_ref[:, :, ::-1, ::-1, ::-1]
        f_ref = theano.function([], grad_w_ref, mode="FAST_RUN")

        if not self._functions_checked_for_gradweight:
            self._functions_checked_for_gradweight = True
            assert any(isinstance(node.op, GpuDnnConvGradW) for node in f.maker.fgraph.apply_nodes)
            assert not any(isinstance(node.op, (GpuDnnConv, GpuDnnConvGradI))
                           for node in f.maker.fgraph.apply_nodes)

            assert not any(isinstance(node.op, (GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI))
                           for node in f_ref.maker.fgraph.apply_nodes)

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        # Needed for big size for some seed
        # raise rtol to make the test pass with more seed.

        # Raise tolerance for float16
        rtol = 5e-2 if theano.config.floatX == 'float16' else None
        utt.assert_allclose(alpha * res_ref + beta * filters_val, res, rtol=rtol)

    # Iterable test methods.

    def test_fwd(self):
        for precision in get_available_precisions():
            for algo in self.fwd_algorithms:
                for parameters in self.get_cases():
                    yield (self.run_conv_fwd, algo, precision, parameters)

    def test_gradinput(self):
        for precision in get_available_precisions():
            for algo in self.bwd_data_algorithms:
                for parameters in self.get_cases():
                    yield (self.run_conv_gradinput, algo, precision, parameters)

    def test_gradweight(self):
        for precision in get_available_precisions():
            for algo in self.bwd_filter_algorithms:
                for parameters in self.get_cases():
                    yield (self.run_conv_gradweight, algo, precision, parameters)


class TestDnnConv2D(BaseTestDnnConv):
    fwd_algorithms = SUPPORTED_DNN_CONV_ALGO_FWD
    bwd_filter_algorithms = SUPPORTED_DNN_CONV_ALGO_BWD_FILTER
    bwd_data_algorithms = SUPPORTED_DNN_CONV_ALGO_BWD_DATA

    cpu_conv_class = theano.tensor.nnet.corr.CorrMM
    cpu_gradinput_class = theano.tensor.nnet.corr.CorrMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr.CorrMM_gradWeights

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
                                  conv_modes, [1], [0])]

        return chain(*iterables)


class TestDnnConv3D(BaseTestDnnConv):
    fwd_algorithms = SUPPORTED_DNN_CONV3D_ALGO_FWD
    bwd_filter_algorithms = SUPPORTED_DNN_CONV3D_ALGO_BWD_FILTER
    bwd_data_algorithms = SUPPORTED_DNN_CONV3D_ALGO_BWD_DATA

    cpu_conv_class = theano.tensor.nnet.corr3d.Corr3dMM
    cpu_gradinput_class = theano.tensor.nnet.corr3d.Corr3dMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr3d.Corr3dMM_gradWeights

    def get_cases(self):
        # small case for quick test.
        input_shape = (128, 3, 5, 5, 5)
        filter_shape = (64, 3, 1, 2, 4)
        subsample = (1, 1, 1)
        dilation = (1, 1, 1)
        border_mode = 'valid'
        conv_mode = 'conv'
        return (((input_shape, filter_shape, subsample, dilation), border_mode, conv_mode, 2.1, -5.7),)

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

        if version() >= 6000:
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
