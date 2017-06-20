#!/usr/bin/env python

# Without args, this script executes all its tests like `nosetests -vs`
# python check_dnn.py       # nosetests mode.

# You can pass args for nosetests as long as your first arg is not in `help, infos, fwd, bwd-filter, bwd-data`.
# python check_dnn.py -xvs  # nosetests: verbose mode, capture output, exit at first error.

# Else, this script uses its own args and can be used to run a specific test case.
# python check_dnn.py help   # Print help for script mode.
# python check_dnn.py infos  # Print infos about algorithms and number of test cases.
# python check_dnn.py {fwd|bwd-filter|bwd-data} {2d|3d} -a <algo> -i <inputShape> -f <filterShape> ...

from __future__ import absolute_import, print_function, division

import argparse
import sys
from itertools import product, chain

import nose
import numpy as np
from nose.plugins.skip import SkipTest

import theano
import theano.tests.unittest_tools as utt
from theano.compile.ops import shape_i_op
from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME
from theano.gpuarray import cudnn_defs
from theano.gpuarray.basic_ops import infer_context_name, as_gpuarray_variable, gpu_contiguous, GpuAllocEmpty
from theano.gpuarray.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI, version, get_precision
from theano.gpuarray.tests.config import mode_with_gpu, ref_cast
from theano.tensor.nnet.abstract_conv import get_conv_output_shape, assert_conv_shape
from theano.tensor.opt import Assert

cudnn = cudnn_defs.get_definitions(version(raises=False))


def ifilter(function, sequence):
    # For compatibility with Python 3.
    return (element for element in sequence if function(element))


class DnnCase:
    """
    Help class to generate special test cases quickly.

    """

    def __init__(self,
                 type, inputs_shape, filters_shape,
                 algo=None, dtype=None, precision=None,
                 subsample=None, dilation=None, border_mode='valid',
                 conv_mode='conv', alpha=1, beta=0,
                 should_fail=False):
        assert type in ('fwd', 'bwd-filter', 'bwd-data')
        assert len(inputs_shape) == len(filters_shape) > 2
        ndim = len(inputs_shape) - 2
        if dtype is None:
            dtype = theano.config.floatX
        if precision is None:
            precision = theano.config.floatX
        if subsample is None:
            subsample = (1,) * ndim
        if dilation is None:
            dilation = (1,) * ndim
        assert dtype in ('float16', 'float32', 'float64')
        assert precision in ('float16', 'float32', 'float64')
        assert len(subsample) == len(dilation) == ndim
        assert border_mode in ('valid', 'full', 'half') or len(border_mode) == ndim
        assert conv_mode in ('conv', 'cross')
        assert alpha != 0

        self.type = type
        self.ndim = ndim
        self.algo = algo
        self.inputs_shape = inputs_shape
        self.filters_shape = filters_shape
        self.dtype = dtype
        self.precision = precision
        self.subsample = subsample
        self.dilation = dilation
        self.border_mode = border_mode
        self.conv_mode = conv_mode
        self.alpha = alpha
        self.beta = beta
        self.should_fail = bool(should_fail)

    def is_fwd(self):
        return self.type == 'fwd'

    def is_bwd_filter(self):
        return self.type == 'bwd-filter'

    def is_bwd_data(self):
        return self.type == 'bwd-data'

    def get_case(self):
        return (self.algo, self.dtype, self.precision,
                (self.inputs_shape, self.filters_shape,
                 self.subsample, self.dilation, self.border_mode,
                 self.conv_mode, self.alpha, self.beta))

    @staticmethod
    def fwd(*args, **kwargs):
        return DnnCase('fwd', *args, **kwargs)

    @staticmethod
    def bwd_filter(*args, **kwargs):
        return DnnCase('bwd-filter', *args, **kwargs)

    @staticmethod
    def bwd_data(*args, **kwargs):
        return DnnCase('bwd-data', *args, **kwargs)


class DnnCaseGenerator:
    """
    Main class used to generate test cases.

    """

    def as_tuple_of_tuples(self, iterable):
        return tuple(tuple(sequence) for sequence in iterable)

    def __init__(self,
                 ndim=2, alpha=2, beta=-3, batch_size=2, input_channels=3, inputs_sizes=None, output_channels=2,
                 filters_sizes=None, borders=None, subsamples=None, dilations=None):
        self.ndim = int(ndim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)

        assert self.ndim >= 2
        assert self.alpha != 0
        assert self.batch_size > 0
        assert self.input_channels > 0
        assert self.output_channels > 0

        if inputs_sizes is None:
            inputs_sizes = ((5,) * self.ndim,
                            (300, 5) + (2,) * (self.ndim - 2))
        if filters_sizes is None:
            filters_sizes = ((4,) * self.ndim,
                             (40, 4) + (2,) * (self.ndim - 2))
        if borders is None:
            borders = ((1,) * self.ndim,
                       tuple(range(1, self.ndim + 1)))
        if subsamples is None:
            subsamples = ((1,) * self.ndim,
                          tuple(range(1, self.ndim + 1)))
        if dilations is None:
            dilations = ((1,) * self.ndim,)
            if cudnn.version >= 6:
                dilations += (tuple(range(1, self.ndim + 1)),)

        for sequence_list in (inputs_sizes, filters_sizes, borders, subsamples, dilations):
            assert (isinstance(sequence_list, (tuple, list)) and
                    all(isinstance(sequence, (tuple, list)) and len(sequence) == self.ndim
                        for sequence in sequence_list)), sequence_list

        self.inputs_sizes = self.as_tuple_of_tuples(inputs_sizes)
        self.filters_sizes = self.as_tuple_of_tuples(filters_sizes)
        self.borders = self.as_tuple_of_tuples(borders)
        self.subsamples = self.as_tuple_of_tuples(subsamples)
        self.dilations = self.as_tuple_of_tuples(dilations)

    @staticmethod
    def get_if_valid_conv_output_shape(case_tuple):
        out_shp = get_conv_output_shape(case_tuple[0],  # input shape
                                        case_tuple[1],  # filter shape
                                        case_tuple[4],  # border mode
                                        case_tuple[2],  # subsample
                                        case_tuple[3]  # dilation
                                        )
        try:
            return assert_conv_shape(out_shp)
        except ValueError:
            return False

    def get_cases(self):
        # Generate an iterator of tuples with format:
        # (input shape, filter shape, subsample, dilation, border mode, convolution mode, alpha, beta)
        all_batch_sizes = (self.batch_size,)
        all_input_channels = (self.input_channels,)
        all_input_sizes = self.inputs_sizes
        all_output_channels = (self.output_channels,)
        all_filter_sizes = self.filters_sizes
        all_subsamples = self.subsamples
        all_dilations = self.dilations
        all_border_modes = ('valid', 'full', 'half') + self.borders
        all_conv_modes = ('conv', 'cross')
        all_alphas = (self.alpha,)
        all_betas = (0,) if self.beta == 0 else (0, self.beta)

        all_input_shapes = ((bs, ic) + ins
                            for bs in all_batch_sizes for ic in all_input_channels for ins in all_input_sizes)
        all_filter_shapes = ((oc, ic) + fis
                             for oc in all_output_channels for ic in all_input_channels for fis in all_filter_sizes)
        return ifilter(DnnCaseGenerator.get_if_valid_conv_output_shape,
                       product(all_input_shapes, all_filter_shapes, all_subsamples, all_dilations,
                               all_border_modes, all_conv_modes, all_alphas, all_betas))


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
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
    else:
        assert out is not None
        out = as_gpuarray_variable(out, ctx_name)
        out = gpu_contiguous(out)
        check = Assert('GpuDnnConv: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, out_shp)))
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


def check_fwd_dtype_config_support(dtype, precision):
    inputs_shape = (1, 1, 3, 3)
    filters_shape = (1, 1, 2, 2)
    inputs = np.zeros(inputs_shape, dtype=dtype)
    filters = np.zeros(filters_shape, dtype=dtype)
    inputs = theano.shared(inputs)
    filters = theano.shared(filters)
    conv = dnn_conv(inputs, filters, precision=precision)
    f = theano.function([], conv, mode=mode_with_gpu)
    try:
        f()
    except RuntimeError as e:
        assert 'CUDNN_STATUS_ARCH_MISMATCH' in e.message
        return False
    return True


def test_fwd_true_half_config_support():
    # For cuDNN V5.1 and V6.0:
    # "TRUE_HALF_CONFIG is only supported on architectures with true fp16 support (compute capability 5.3 and 6.0)"
    if not check_fwd_dtype_config_support('float16', 'float16'):
        raise SkipTest('FWD: TRUE_HALF_CONFIG not supported on this GPU.')


class BaseTestDnnConv(object):
    """
    Base class for exhaustive tests. Use its subclasses
    to run actual tests.
    """

    # Abstract attributes.

    ndim = 2

    fwd_algorithms = None
    bwd_filter_algorithms = None
    bwd_data_algorithms = None

    cpu_conv_class = None
    cpu_gradinput_class = None
    cpu_gradweight_class = None

    special_cases = []  # List of DnnCases.

    # Utility methods.

    def get_cases(self):
        # Return an iterable of test cases. Each test case is a tuple (or list) with following syntax:
        # (input shape, filter shape, subsample, dilation, border mode, convolution mode, alpha, beta)
        generator = DnnCaseGenerator(ndim=self.ndim)
        return generator.get_cases()

    def array_like_conv_output(self, inputs_shape, filters_shape, border_mode, subsample, dilation, dtype):
        # Return an random array with inferred convolution output shape.
        out_shp = get_conv_output_shape(inputs_shape, filters_shape, border_mode, subsample, dilation)
        out_shp = assert_conv_shape(out_shp)
        return np.random.random(out_shp).astype(dtype)

    def run_conv_fwd(self, algo, dtype, precision, parameters):
        inputs_shape, filters_shape, subsample, dilation, border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(dtype)
        filters_val = np.random.random(filters_shape).astype(dtype)

        # Scale down the input values to prevent very large absolute errors
        # due to float rounding
        inputs_val /= 10
        filters_val /= 10

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

        if beta == 0:
            out = None
        else:
            out = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation, dtype)
            out /= 10
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

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        if algo in cudnn.deterministic_fwd_algorithms:
            res2 = f()
            utt.assert_allclose(res, res2)

        # Raise tolerance for float16
        rtol = 6e-2 if dtype == 'float16' else None
        if beta == 0:
            utt.assert_allclose(alpha * res_ref, res, rtol=rtol)
        else:
            utt.assert_allclose(alpha * res_ref + beta * out, res, rtol=rtol)

    def run_conv_gradinput(self, algo, dtype, precision, parameters):
        inputs_shape, filters_shape, subsample, dilation, border_mode, conv_mode, alpha, beta = parameters

        if beta == 0:
            inputs_val = None
        else:
            inputs_val = np.random.random(inputs_shape).astype(dtype)
            inputs_val /= 10
        filters_val = np.random.random(filters_shape).astype(dtype)
        topgrad_val = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation, dtype)

        # Scale down the input values to prevent absolute errors in utt.assert_allclose.
        filters_val /= 10
        topgrad_val /= 10

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

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        if algo in cudnn.deterministic_bwd_data_algorithms:
            res2 = f()
            utt.assert_allclose(res, res2)

        # Raise tolerance for float16
        rtol = 5e-2 if dtype == 'float16' else None
        if beta == 0:
            utt.assert_allclose(alpha * res_ref, res, rtol=rtol)
        else:
            utt.assert_allclose(alpha * res_ref + beta * inputs_val, res, rtol=rtol)

    def run_conv_gradweight(self, algo, dtype, precision, parameters):
        inputs_shape, filters_shape, subsample, dilation, border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(dtype)
        if beta == 0:
            filters_val = None
        else:
            filters_val = np.random.random(filters_shape).astype(dtype)
            filters_val /= 10
        topgrad_val = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation, dtype)

        # Scale down the input values to prevent absolute errors in utt.assert_allclose.
        inputs_val /= 10
        topgrad_val /= 10

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
            if inputs.ndim == 5:
                grad_w_ref = grad_w_ref[:, :, ::-1, ::-1, ::-1]
            else:
                grad_w_ref = grad_w_ref[:, :, ::-1, ::-1]
        f_ref = theano.function([], grad_w_ref, mode="FAST_RUN")

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        if algo in cudnn.deterministic_bwd_filter_algorithms:
            res2 = f()
            utt.assert_allclose(res, res2)

        # Raise tolerance for float16
        rtol = 5e-2 if dtype == 'float16' else None
        if beta == 0:
            utt.assert_allclose(alpha * res_ref, res, rtol=rtol)
        else:
            utt.assert_allclose(alpha * res_ref + beta * filters_val, res, rtol=rtol)

    def get_expected_tcount(self):
        """
        Utility function to get expected test count
        without actually run nosetests.
        """
        len_cases = sum(1 for case in self.get_cases())
        count_contexts = 0
        for dtype, precision in cudnn.get_fwd_dtype_configs(check_runtime=check_fwd_dtype_config_support):
            algos = (algo for algo in self.fwd_algorithms
                     if cudnn.fwd_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            count_contexts += sum(1 for algo in algos) + len(SUPPORTED_DNN_CONV_ALGO_RUNTIME)
        for dtype, precision in cudnn.get_bwd_data_dtype_configs():
            algos = (algo for algo in self.bwd_data_algorithms
                     if cudnn.bwd_data_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            count_contexts += sum(1 for algo in algos) + len(SUPPORTED_DNN_CONV_ALGO_RUNTIME)
        for dtype, precision in cudnn.get_bwd_filter_dtype_configs():
            algos = (algo for algo in self.bwd_filter_algorithms
                     if cudnn.bwd_filter_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            count_contexts += sum(1 for algo in algos) + len(SUPPORTED_DNN_CONV_ALGO_RUNTIME)
        return len(self.special_cases) + len_cases * count_contexts

    def should_fail(self, callable, *args):
        try:
            print('(should fail)', file=sys.stderr, end=' ')
            callable(*args)
        except Exception:
            pass
        else:
            raise AssertionError('Should fail', callable.__name__, *args)

    # Iterable test methods.

    def test_fwd(self):
        for dtype, precision in cudnn.get_fwd_dtype_configs(check_runtime=check_fwd_dtype_config_support):
            algos = (algo for algo in self.fwd_algorithms
                     if cudnn.fwd_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in chain(algos, SUPPORTED_DNN_CONV_ALGO_RUNTIME):
                for parameters in self.get_cases():
                    yield (self.run_conv_fwd, algo, dtype, precision, parameters)
        for dnn_case in self.special_cases:
            if dnn_case.is_fwd():
                if dnn_case.should_fail:
                    yield (self.should_fail, self.run_conv_fwd,) + dnn_case.get_case()
                else:
                    yield (self.run_conv_fwd,) + dnn_case.get_case()

    def test_gradinput(self):
        for dtype, precision in cudnn.get_bwd_data_dtype_configs():
            algos = (algo for algo in self.bwd_data_algorithms
                     if cudnn.bwd_data_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in chain(algos, SUPPORTED_DNN_CONV_ALGO_RUNTIME):
                for parameters in self.get_cases():
                    yield (self.run_conv_gradinput, algo, dtype, precision, parameters)
        for dnn_case in self.special_cases:
            if dnn_case.is_bwd_data():
                if dnn_case.should_fail:
                    yield (self.should_fail, self.run_conv_gradinput,) + dnn_case.get_case()
                else:
                    yield (self.run_conv_gradinput,) + dnn_case.get_case()

    def test_gradweight(self):
        for dtype, precision in cudnn.get_bwd_filter_dtype_configs():
            algos = (algo for algo in self.bwd_filter_algorithms
                     if cudnn.bwd_filter_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in chain(algos, SUPPORTED_DNN_CONV_ALGO_RUNTIME):
                for parameters in self.get_cases():
                    yield (self.run_conv_gradweight, algo, dtype, precision, parameters)
        for dnn_case in self.special_cases:
            if dnn_case.is_bwd_filter():
                if dnn_case.should_fail:
                    yield (self.should_fail, self.run_conv_gradweight,) + dnn_case.get_case()
                else:
                    yield (self.run_conv_gradweight,) + dnn_case.get_case()


class TestDnnConv2D(BaseTestDnnConv):
    ndim = 2

    fwd_algorithms = cudnn.cudnnConvolutionFwdAlgo_t.get_aliases()
    bwd_filter_algorithms = cudnn.cudnnConvolutionBwdFilterAlgo_t.get_aliases()
    bwd_data_algorithms = cudnn.cudnnConvolutionBwdDataAlgo_t.get_aliases()

    cpu_conv_class = theano.tensor.nnet.corr.CorrMM
    cpu_gradinput_class = theano.tensor.nnet.corr.CorrMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr.CorrMM_gradWeights

    special_cases = [DnnCase.bwd_filter(algo='deterministic', dtype='float32', precision='float32',
                                        inputs_shape=(1, 1, 541211, 10), filters_shape=(50, 1, 3, 10),
                                        border_mode=(1, 0), should_fail=True)]


class TestDnnConv3D(BaseTestDnnConv):
    ndim = 3

    fwd_algorithms = cudnn.conv3d_fwd_algorithms
    bwd_filter_algorithms = cudnn.conv3d_bwd_filter_algorithms
    bwd_data_algorithms = cudnn.conv3d_bwd_data_algorithms

    cpu_conv_class = theano.tensor.nnet.corr3d.Corr3dMM
    cpu_gradinput_class = theano.tensor.nnet.corr3d.Corr3dMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr3d.Corr3dMM_gradWeights


class CheckDnn():
    """
    Utility functions for scripting and infos printing.
    """

    @staticmethod
    def dtype_config_to_str(dtype_config):
        dtype, precision = dtype_config
        if dtype == precision == 'float16':
            return 'TRUE_HALF_CONFIG'
        if dtype == 'float16' and precision == 'float32':
            return 'PSEUDO_HALF_CONFIG'
        if dtype == precision == 'float32':
            return 'FLOAT_CONFIG'
        if dtype == precision == 'float64':
            return 'DOUBLE_CONFIG'
        raise ValueError

    @staticmethod
    def print_infos():
        # Print infos about tests and cuDNN supported algorithms and configurations.
        test_2d = TestDnnConv2D()
        test_3d = TestDnnConv3D()
        print()
        print('Available data type configurations:',
              ', '.join(CheckDnn.dtype_config_to_str(d) for d in cudnn.get_supported_dtype_configs()))
        print()
        print('2D algorithms:')
        print('FWD        :', ', '.join(test_2d.fwd_algorithms))
        print('BWD FILTER :', ', '.join(test_2d.bwd_filter_algorithms))
        print('BWD DATA   :', ', '.join(test_2d.bwd_data_algorithms))
        print()
        print('3D algorithms:')
        print('FWD        :', ', '.join(test_3d.fwd_algorithms))
        print('BWD FILTER :', ', '.join(test_3d.bwd_filter_algorithms))
        print('BWD DATA   :', ', '.join(test_3d.bwd_data_algorithms))
        print()
        count_tests_2d = test_2d.get_expected_tcount()
        count_tests_3d = test_3d.get_expected_tcount()
        print(count_tests_2d, 'conv2D test cases.')
        print(count_tests_3d, 'conv3D test cases.')
        print(count_tests_2d + count_tests_3d, 'total conv test cases.')
        print()

    class TupleAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            values = tuple(int(v) for v in values.split(','))
            setattr(namespace, self.dest, values)

    class BorderAction(TupleAction):
        def __call__(self, parser, namespace, values, option_string=None):
            if values not in ('valid', 'full', 'half'):
                super(CheckDnn.BorderAction, self).__call__(parser, namespace, values, option_string)
            else:
                setattr(namespace, self.dest, values)


if __name__ == '__main__':

    computations = FWD, BWD_FILTER, BWD_DATA = ('fwd', 'bwd-filter', 'bwd-data')

    # We remove programe name from args.
    args = sys.argv[1:]

    if len(args) == 0 or args[0] not in computations + ('help', 'infos'):
        # We run all tests with nosetests.
        module_name = sys.modules[__name__].__file__
        if len(args) == 0:
            # No args given: run nosetests -vs
            args = ['--verbose', '--nocapture']
        # Else, use given args.
        argv = [sys.argv[0], module_name] + args

        CheckDnn.print_infos()
        nose.main(argv=argv)
    elif len(args) == 1 and args[0] == 'infos':
        CheckDnn.print_infos()
    else:
        # User wants to run a specific test.

        dimensions = ('2D', '2d', '3D', '3d')
        algorithms = (tuple(sorted(list(set(cudnn.cudnnConvolutionFwdAlgo_t.get_aliases() +
                                            cudnn.cudnnConvolutionBwdFilterAlgo_t.get_aliases() +
                                            cudnn.cudnnConvolutionBwdDataAlgo_t.get_aliases())))) +
                      SUPPORTED_DNN_CONV_ALGO_RUNTIME)
        types = ('float16', 'float32', 'float64')

        parser = argparse.ArgumentParser()

        parser.add_argument('computation', choices=computations,
                            help='Computation to run.')
        parser.add_argument('ndim', choices=dimensions,
                            help='Number od dimensions ("2D" or "3D", case ignored).')

        parser.add_argument('-a', '--algo', choices=algorithms, required=True,
                            help='Algorithm to use for computation.')
        parser.add_argument('-i', '--input-shape', action=CheckDnn.TupleAction, required=True,
                            help='Input shape. Comma-separated list of integers (no spaces).')
        parser.add_argument('-f', '--filter-shape', action=CheckDnn.TupleAction, required=True,
                            help='Filter shape. Comma-separated list of integers (no spaces).')

        parser.add_argument('-t', '--dtype', choices=types, default=theano.config.floatX,
                            help='Data type (default theano floatX).')
        parser.add_argument('-p', '--precision', choices=types, default=theano.config.floatX,
                            help='Precision (default theano floatX).')
        parser.add_argument('-s', '--subsample', action=CheckDnn.TupleAction,
                            help='Subsample. Comma-separated list of integers (no spaces).')
        parser.add_argument('-d', '--dilation', action=CheckDnn.TupleAction,
                            help='Dilation. Comma-separated list of integers (no spaces).')
        parser.add_argument('-b', '--border-mode', default='valid', action=CheckDnn.BorderAction,
                            help='Border mode. "valid" (default), "full", "half" '
                                 'or a comma-separated list of integers (no spaces).')
        parser.add_argument('-c', '--conv-mode', choices=('conv', 'cross'), default='conv',
                            help='Conv mode (default: conv).')
        parser.add_argument('-A', '--alpha', type=float, default=1,
                            help="alpha (floating), must not be zero. Default 1.")
        parser.add_argument('-B', '--beta', type=float, default=0,
                            help='beta (floating). Default 0.')

        parser.add_argument('--print-infos', action='store_true', default=False,
                            help='Print some infos before testing.')

        if len(args) == 1 and args[0] == 'help':
            parser.parse_args(['-h'])
            exit(0)
        args = parser.parse_args(args)

        test = args.computation
        ndim = int(args.ndim[0])
        if ndim == 2:
            tests = TestDnnConv2D()
        if ndim == 3:
            tests = TestDnnConv3D()
        if args.subsample is None:
            args.subsample = (1,) * ndim
        if args.dilation is None:
            args.dilation = (1,) * ndim
        if not (ndim == len(args.input_shape[2:]) == len(args.filter_shape[2:]) == len(args.subsample) == len(
                args.dilation)):
            raise ValueError('Expected parameters sized for %d dimensions.' % ndim)
        if isinstance(args.border_mode, tuple) and ndim != len(args.border_mode):
            raise ValueError('Expected borders sized for %d dimensions.' % ndim)
        if args.alpha == 0:
            raise ValueError('Nothing could be computed if alpha is 0.')

        if (args.dtype, args.precision) not in cudnn.get_supported_dtype_configs():
            raise ValueError('Unsupported data type configuration %s %s.' % (args.dtype, args.precision))
        if args.algo not in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
            check_config = False
            if test == FWD:
                check_config = cudnn.fwd_algo_supports_dtype_config(args.algo, args.dtype, args.precision, ndim)
            if test == BWD_FILTER:
                check_config = cudnn.bwd_filter_algo_supports_dtype_config(args.algo, args.dtype, args.precision, ndim)
            if test == BWD_DATA:
                check_config = cudnn.bwd_data_algo_supports_dtype_config(args.algo, args.dtype, args.precision, ndim)
            if not check_config:
                raise ValueError('%s computation does not support configuration (%s, %s) for algo %s.' % (
                    test, args.dtype, args.precision, args.algo))
        algo = args.algo
        dtype = args.dtype
        precision = args.precision
        parameters = (
            args.input_shape, args.filter_shape, args.subsample, args.dilation, args.border_mode, args.conv_mode,
            args.alpha, args.beta)
        if args.print_infos:
            CheckDnn.print_infos()
        print('======================')
        print('Running %s %s %s %s %s' % (test, algo, dtype, precision, str(parameters)))
        if test == FWD:
            tests.run_conv_fwd(algo, dtype, precision, parameters)
        if test == BWD_FILTER:
            tests.run_conv_gradweight(algo, dtype, precision, parameters)
        if test == BWD_DATA:
            tests.run_conv_gradinput(algo, dtype, precision, parameters)
        print('... OK')
