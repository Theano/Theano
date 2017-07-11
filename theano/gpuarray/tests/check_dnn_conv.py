#!/usr/bin/env python

# Without args, this script executes all its tests like `nosetests -vs`
# python check_dnn_conv.py

# If there is only one arg `infos`, this script prints some infos about
# supported algorithms and data type configurations for current GPU and cuDNN version.
# python check_dnn_conv.py infos

# If there is only one arg `list`, this script prints all tests without running them.
# python check_dnn_conv.py list

# Else, any arg will be directly passed to nosetests.
# python check_dnn_conv.py -xvs  # nosetests: verbose mode, capture output, exit at first error.

from __future__ import absolute_import, print_function, division

import sys
from itertools import product

import nose
import numpy as np
from nose.plugins.skip import SkipTest

import theano
import theano.tests.unittest_tools as utt
from theano.compat import ifilter
from theano.compile.ops import shape_i_op
from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME
from theano.gpuarray import cudnn_defs
from theano.gpuarray.basic_ops import infer_context_name, as_gpuarray_variable, gpu_contiguous, GpuAllocEmpty
from theano.gpuarray.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI, version, get_precision
from theano.gpuarray.tests.config import mode_with_gpu, ref_cast
from theano.tensor.nnet.abstract_conv import get_conv_output_shape, assert_conv_shape
from theano.tensor.opt import Assert


# We provide a special implementation of dnn_conv, dnn_gradweight and dnn_gradinput
# that support alpha, beta and out as parameters.

def dnn_conv(img, kerns, alpha=1, beta=0, out=None, border_mode='valid', subsample=(1, 1), dilation=(1, 1),
             conv_mode='conv', algo=None, precision=None):
    ctx_name = infer_context_name(img, kerns)

    img = gpu_contiguous(as_gpuarray_variable(img, ctx_name))
    kerns = gpu_contiguous(as_gpuarray_variable(kerns, ctx_name))

    precision = get_precision(precision, [img, kerns])
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                          conv_mode=conv_mode, precision=precision)(kerns.shape)
    desc_op = desc.owner.op
    # We can use Shape_i and bypass the infer_shape here as this is on
    # the input of node and it will always be present.
    ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
    kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
    out_shp = get_conv_output_shape(ishape, kshape, desc_op.border_mode, desc_op.subsample, filter_dilation=dilation)
    out_shp = assert_conv_shape(out_shp)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
    else:
        assert out is not None
        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
        check = Assert('GpuDnnConv: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, out_shp)))
    return GpuDnnConv(algo=algo)(img, kerns, real_out, desc, alpha, beta)


def dnn_gradweight(img, topgrad, kerns_shp, alpha=1, beta=0, out=None, border_mode='valid', subsample=(1, 1),
                   dilation=(1, 1), conv_mode='conv', algo=None, precision=None):
    ctx_name = infer_context_name(img, topgrad)

    img = gpu_contiguous(as_gpuarray_variable(img, ctx_name))
    topgrad = gpu_contiguous(as_gpuarray_variable(topgrad, ctx_name))
    kerns_shp = theano.tensor.as_tensor_variable(kerns_shp)

    precision = get_precision(precision, [img, topgrad])
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                          conv_mode=conv_mode, precision=precision)(kerns_shp)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*kerns_shp)
    else:
        assert out is not None
        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
        check = Assert('GpuDnnConvGradW: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, kerns_shp)))
    return GpuDnnConvGradW(algo=algo)(img, topgrad, real_out, desc, alpha, beta)


def dnn_gradinput(kerns, topgrad, img_shp, alpha=1, beta=0, out=None, border_mode='valid', subsample=(1, 1),
                  dilation=(1, 1), conv_mode='conv', algo=None, precision=None):
    ctx_name = infer_context_name(kerns, topgrad)

    kerns = gpu_contiguous(as_gpuarray_variable(kerns, ctx_name))
    topgrad = gpu_contiguous(as_gpuarray_variable(topgrad, ctx_name))
    img_shp = theano.tensor.as_tensor_variable(img_shp)

    precision = get_precision(precision, [kerns, topgrad])
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                          conv_mode=conv_mode, precision=precision)(kerns.shape)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=kerns.dtype, context_name=ctx_name)(*img_shp)
    else:
        assert out is not None
        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
        check = Assert('GpuDnnConvGradI: qiven output (for beta not null) does not have expected shape')
        real_out = check(out, theano.tensor.all(theano.tensor.eq(out.shape, img_shp)))
    return GpuDnnConvGradI(algo=algo)(kerns, topgrad, real_out, desc, alpha, beta)


def check_dtype_config_support(dtype, precision):
    # We use FWD 2D to check it.
    # Based on documentation, algo small (CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
    # should support all configurations, for both v5.1 and v6.
    inputs = theano.shared(np.zeros((1, 1, 2, 2), dtype=dtype))
    filters = theano.shared(np.zeros((1, 1, 2, 2), dtype=dtype))
    conv = dnn_conv(inputs, filters, precision=precision, algo='small')
    f = theano.function([], conv, mode=mode_with_gpu)
    try:
        f()
    except RuntimeError as e:
        assert 'CUDNN_STATUS_ARCH_MISMATCH' in e.message
        return False
    return True


cudnn = cudnn_defs.get_definitions(version(raises=False))


class ConvCase:
    """
    Help class to describe a special test case quickly.
    This handles only 2D and 3D cases.
    """

    FWD, GRADINPUT, GRADWEIGHT = 0, 1, 2

    def __init__(self, type,
                 inputs_shape, filters_shape,
                 algo=None, dtype=None, precision=None,
                 subsample=None, dilation=None, border_mode='valid',
                 conv_mode='conv', alpha=1, beta=0,
                 should_fail=False):
        assert type in (ConvCase.FWD, ConvCase.GRADINPUT, ConvCase.GRADWEIGHT)
        assert len(inputs_shape) == len(filters_shape) in (4, 5)
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
        assert (border_mode in ('valid', 'full', 'half') or
                (isinstance(border_mode, (list, tuple)) and len(border_mode) == ndim))
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
        return self.type == ConvCase.FWD

    def is_bwd_filter(self):
        return self.type == ConvCase.GRADWEIGHT

    def is_bwd_data(self):
        return self.type == ConvCase.GRADINPUT

    def get_case(self):
        return (self.algo, self.dtype, self.precision,
                (self.inputs_shape, self.filters_shape,
                 self.subsample, self.dilation, self.border_mode,
                 self.conv_mode, self.alpha, self.beta))

    @staticmethod
    def fwd(*args, **kwargs):
        return ConvCase(ConvCase.FWD, *args, **kwargs)

    @staticmethod
    def bwd_filter(*args, **kwargs):
        return ConvCase(ConvCase.GRADWEIGHT, *args, **kwargs)

    @staticmethod
    def bwd_data(*args, **kwargs):
        return ConvCase(ConvCase.GRADINPUT, *args, **kwargs)


class ConvCaseGenerator:
    """
    Main class used to generate test cases.
    This handles only 2D and 3D cases.
    """

    def _as_tuple_of_tuples(self, iterable):
        return tuple(tuple(sequence) for sequence in iterable)

    def __init__(self, ndim,
                 alpha=2, beta=-3, batch_size=2, input_channels=3, inputs_sizes=None, output_channels=2,
                 filters_sizes=None, subsamples=None, dilations=None, borders=None,
                 with_border_valid=True, with_border_half=True, with_border_full=True):
        self.ndim = int(ndim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)

        assert self.ndim in (2, 3)
        assert self.alpha != 0
        assert self.batch_size > 0
        assert self.input_channels > 0
        assert self.output_channels > 0

        # NB: it is a bit arbitrary to choose default values for inputs sizes and filters sizes.
        # Here, we just put some values that may generate errors in some cases, but that should be OK for other cases.
        # For instance, input size 300 is > 256, that is a limit for certain algorithms (cf. documentation).
        # Filter size 40 is > 32 and > 16, that are limits for certain algorithms (cf. documentation).
        # We should either manually specify sizes, or give an appropriate filter to this generator
        # (see `self.get_cases()`) before testing values.

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
                        for sequence in sequence_list)), (self.ndim, sequence_list)

        self.auto_borders = tuple()
        if with_border_valid:
            self.auto_borders += ('valid',)
        if with_border_half:
            self.auto_borders += ('half',)
        if with_border_full:
            self.auto_borders += ('full',)

        self.inputs_sizes = self._as_tuple_of_tuples(inputs_sizes)
        self.filters_sizes = self._as_tuple_of_tuples(filters_sizes)
        self.borders = self._as_tuple_of_tuples(borders)
        self.subsamples = self._as_tuple_of_tuples(subsamples)
        self.dilations = self._as_tuple_of_tuples(dilations)

    @staticmethod
    def get_if_valid_conv_output_shape(case_tuple):
        # Filter function to keep only cases that produce valid convolution output shapes.
        out_shp = get_conv_output_shape(case_tuple[0],  # input shape
                                        case_tuple[1],  # filter shape
                                        case_tuple[4],  # border mode
                                        case_tuple[2],  # subsample
                                        case_tuple[3])  # dilation
        try:
            return assert_conv_shape(out_shp)
        except ValueError:
            return False

    def get_cases(self, filter=None):
        # Generate an iterator of tuples with format:
        # (input shape, filter shape, subsample, dilation, border mode, convolution mode, alpha, beta)
        # filter may be a callable that gets one tuple (with format specified above) and returns
        # a boolean, so that tuple is kept only if filter(tuple) is True.

        all_batch_sizes = (self.batch_size,)
        all_input_channels = (self.input_channels,)
        all_input_sizes = self.inputs_sizes
        all_output_channels = (self.output_channels,)
        all_filter_sizes = self.filters_sizes
        all_subsamples = self.subsamples
        all_dilations = self.dilations
        all_border_modes = self.auto_borders + self.borders
        all_conv_modes = ('conv', 'cross')
        all_alphas = (self.alpha,)
        all_betas = (0,) if self.beta == 0 else (0, self.beta)

        all_input_shapes = ((bs, ic) + ins
                            for bs in all_batch_sizes for ic in all_input_channels for ins in all_input_sizes)
        all_filter_shapes = ((oc, ic) + fis
                             for oc in all_output_channels for ic in all_input_channels for fis in all_filter_sizes)
        if callable(filter):
            def local_filter(case_tuple):
                return ConvCaseGenerator.get_if_valid_conv_output_shape(case_tuple) and filter(case_tuple)
        else:
            local_filter = ConvCaseGenerator.get_if_valid_conv_output_shape
        return ifilter(local_filter,
                       product(all_input_shapes, all_filter_shapes, all_subsamples, all_dilations,
                               all_border_modes, all_conv_modes, all_alphas, all_betas))


class CuDNNV51ConvCaseGenerator(object):
    """
    Helper class to generate specific test cases for every algorithm supported by cuDNN V5.1.
    Same class exists for cuDNN V6.0 (see below).
    This should help avoid test cases that are intended to fail according to cuDNN documentations.
    """
    NONE = 'none'
    FFT = 'fft'
    FFT_TILING = 'fft_tiling'
    WINOGRAD = 'winograd'
    WINOGRAD_NON_FUSED = 'winograd_non_fused'

    # Protected interface.

    def _dilations(self, ndim):
        return [(1,) * ndim]

    def _fwd_fft(self, ndim):
        inputs_sizes = [(10,) * ndim,
                        (248, 5) + (2,) * (ndim - 2)]
        filters_sizes = [tuple(range(9, 9 - ndim, -1))]
        subsamples = [(1,) * ndim]
        return ConvCaseGenerator(ndim=ndim,
                                 inputs_sizes=inputs_sizes,
                                 filters_sizes=filters_sizes,
                                 subsamples=subsamples,
                                 dilations=self._dilations(ndim))

    def _fwd_fft_tiling(self, ndim):
        if ndim == 2:
            filters_sizes = [(32, 5)]
        if ndim == 3:
            filters_sizes = [(16, 5, 5)]
        subsamples = [(1,) * ndim]
        return ConvCaseGenerator(ndim=ndim,
                                 filters_sizes=filters_sizes,
                                 subsamples=subsamples,
                                 dilations=self._dilations(ndim))

    def _fwd_winograd(self, ndim):
        filters_sizes = [(3,) * ndim]
        subsamples = [(1,) * ndim]
        return ConvCaseGenerator(ndim=ndim,
                                 filters_sizes=filters_sizes,
                                 subsamples=subsamples,
                                 dilations=self._dilations(ndim))

    def _fwd_winograd_non_fused(self, ndim):
        filters_sizes = [(3,) * ndim]
        if not check_dtype_config_support('float16', 'float16'):
            filters_sizes += [(5,) * ndim]
        subsamples = [(1,) * ndim]
        return ConvCaseGenerator(ndim=ndim,
                                 filters_sizes=filters_sizes,
                                 subsamples=subsamples,
                                 dilations=self._dilations(ndim))

    def _gw_fft(self, ndim):
        return self._fwd_fft(ndim)

    def _gw_winograd_non_fused(self, ndim):
        return self._fwd_winograd_non_fused(ndim)

    def _gi_fft(self, ndim):
        return self._fwd_fft(ndim)

    def _gi_fft_tiling(self, ndim):
        return self._fwd_fft_tiling(ndim)

    def _gi_winograd(self, ndim):
        return self._fwd_winograd(ndim)

    def _gi_winograd_non_fused(self, ndim):
        return self._fwd_winograd_non_fused(ndim)

    # Public interface.

    def fwd(self, algo, ndim):
        if algo == self.FFT:
            return self._fwd_fft(ndim)
        if algo == self.FFT_TILING:
            return self._fwd_fft_tiling(ndim)
        if algo == self.WINOGRAD:
            return self._fwd_winograd(ndim)
        if algo == self.WINOGRAD_NON_FUSED:
            return self._fwd_winograd_non_fused(ndim)
        return ConvCaseGenerator(ndim=ndim, dilations=self._dilations(ndim))

    def gw(self, algo, ndim):
        if algo == self.FFT:
            return self._gw_fft(ndim)
        if algo == self.WINOGRAD_NON_FUSED:
            return self._gw_winograd_non_fused(ndim)
        return ConvCaseGenerator(ndim=ndim, dilations=self._dilations(ndim))

    def gi(self, algo, ndim):
        if algo == self.FFT:
            return self._gi_fft(ndim)
        if algo == self.FFT_TILING:
            return self._gi_fft_tiling(ndim)
        if algo == self.WINOGRAD:
            return self._gi_winograd(ndim)
        if algo == self.WINOGRAD_NON_FUSED:
            return self._gi_winograd_non_fused(ndim)
        return ConvCaseGenerator(ndim=ndim, dilations=self._dilations(ndim))


class CuDNNV6ConvCaseGenerator(CuDNNV51ConvCaseGenerator):
    def _fwd_none(self, ndim):
        # All dilations allowed.
        return ConvCaseGenerator(ndim=ndim)

    def _fwd_fft_tiling(self, ndim):
        if ndim == 2:
            filters_sizes = [(32, 5), (256, 1), (10, 10), (5, 1)]
            subsamples = [(1, 1)]
            borders = [(1, 1), (2, 1)]
            return ConvCaseGenerator(ndim=ndim,
                                     filters_sizes=filters_sizes,
                                     subsamples=subsamples,
                                     borders=borders,
                                     dilations=self._dilations(ndim))
        if ndim == 3:
            return super(CuDNNV6ConvCaseGenerator, self)._fwd_fft_tiling(ndim)

    def _gw_none(self, ndim):
        return self._fwd_none(ndim)

    def _gw_fft_tiling(self, ndim):
        inputs_sizes = [(256, 1), (20, 1)]
        filters_sizes = [(3, 1), (10, 1)]
        subsamples = [(1,) * ndim]
        borders = [(1, 1), (2, 1)]
        return ConvCaseGenerator(ndim=ndim,
                                 inputs_sizes=inputs_sizes,
                                 filters_sizes=filters_sizes,
                                 subsamples=subsamples,
                                 borders=borders,
                                 dilations=self._dilations(ndim))

    def _gi_none(self, ndim):
        return self._fwd_none(ndim)

    def fwd(self, algo, ndim):
        if algo == self.NONE:
            return self._fwd_none(ndim)
        return super(CuDNNV6ConvCaseGenerator, self).fwd(algo, ndim)

    def gw(self, algo, ndim):
        if algo == self.NONE:
            return self._gw_none(ndim)
        return super(CuDNNV6ConvCaseGenerator, self).gw(algo, ndim)

    def gi(self, algo, ndim):
        if algo == self.NONE:
            return self._gi_none(ndim)
        return super(CuDNNV6ConvCaseGenerator, self).gi(algo, ndim)


cudnn_conv_case_generator = CuDNNV51ConvCaseGenerator() if cudnn.version < 6 else CuDNNV6ConvCaseGenerator()


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

    special_cases = []  # List of special ConvCases.

    # Utility methods.

    def __init__(self):
        utt.seed_rng(1234)
        self.dtype_configs = cudnn.get_supported_dtype_configs(check_dtype_config_support)

    def get_cases(self, filter=None):
        # Return an iterable of test cases. Each test case is a tuple (or list) with following syntax:
        # (input shape, filter shape, subsample, dilation, border mode, convolution mode, alpha, beta)
        return ConvCaseGenerator(ndim=self.ndim).get_cases(filter)

    def array_like_conv_output(self, inputs_shape, filters_shape, border_mode, subsample, dilation, dtype):
        # Return a random array with inferred convolution output shape.
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
        # filters flipped according to the width, height and time axis
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
        res = np.asarray(f())
        if algo in cudnn.deterministic_fwd_algorithms:
            utt.assert_allclose(res, np.asarray(f()))

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
        # filters flipped according to the width, height and time axis
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
        res = np.asarray(f())
        if algo in cudnn.deterministic_bwd_data_algorithms:
            utt.assert_allclose(res, np.asarray(f()))

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
        res = np.asarray(f())
        if algo in cudnn.deterministic_bwd_filter_algorithms:
            utt.assert_allclose(res, np.asarray(f()))

        # Raise tolerance for float16
        rtol = 5e-2 if dtype == 'float16' else None
        if beta == 0:
            utt.assert_allclose(alpha * res_ref, res, rtol=rtol)
        else:
            utt.assert_allclose(alpha * res_ref + beta * filters_val, res, rtol=rtol)

    def should_fail(self, callable, *args):
        try:
            print('(should fail)', file=sys.stderr, end=' ')
            callable(*args)
        except Exception:
            pass
        else:
            raise AssertionError('Should fail', callable.__name__, *args)

    def get_expected_tcount(self):
        """
        Utility function to get expected test count
        without actually run nosetests.
        """
        return (sum(1 for t in self.test_fwd()) +
                sum(1 for t in self.test_gradweight()) +
                sum(1 for t in self.test_gradinput()))

    # Iterable test methods.

    def test_fwd(self):
        for dtype, precision in self.dtype_configs:
            algos = (algo for algo in self.fwd_algorithms
                     if cudnn.fwd_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in algos:
                for parameters in cudnn_conv_case_generator.fwd(algo, self.ndim).get_cases():
                    yield (self.run_conv_fwd, algo, dtype, precision, parameters)
            for algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
                for parameters in self.get_cases():
                    yield (self.run_conv_fwd, algo, dtype, precision, parameters)
        for dnn_case in self.special_cases:
            if dnn_case.is_fwd():
                if dnn_case.should_fail:
                    yield (self.should_fail, self.run_conv_fwd,) + dnn_case.get_case()
                else:
                    yield (self.run_conv_fwd,) + dnn_case.get_case()

    def test_gradinput(self):
        for dtype, precision in self.dtype_configs:
            algos = (algo for algo in self.bwd_data_algorithms
                     if cudnn.bwd_data_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in algos:
                for parameters in cudnn_conv_case_generator.gi(algo, self.ndim).get_cases():
                    yield (self.run_conv_gradinput, algo, dtype, precision, parameters)
            for algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
                for parameters in self.get_cases():
                    yield (self.run_conv_gradinput, algo, dtype, precision, parameters)
        for dnn_case in self.special_cases:
            if dnn_case.is_bwd_data():
                if dnn_case.should_fail:
                    yield (self.should_fail, self.run_conv_gradinput,) + dnn_case.get_case()
                else:
                    yield (self.run_conv_gradinput,) + dnn_case.get_case()

    def test_gradweight(self):
        for dtype, precision in self.dtype_configs:
            algos = (algo for algo in self.bwd_filter_algorithms
                     if cudnn.bwd_filter_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in algos:
                for parameters in cudnn_conv_case_generator.gw(algo, self.ndim).get_cases():
                    yield (self.run_conv_gradweight, algo, dtype, precision, parameters)
            for algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
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

    special_cases = [ConvCase.bwd_filter(algo='deterministic', dtype='float32', precision='float32',
                                         inputs_shape=(1, 1, 541211, 10), filters_shape=(50, 1, 3, 10),
                                         border_mode=(1, 0), should_fail=True),
                     ConvCase.fwd(algo='small', dtype='float32', precision='float32',
                                  inputs_shape=(65536, 2, 2, 2), filters_shape=(1, 2, 2, 2)),
                     ConvCase.fwd(algo='small', dtype='float32', precision='float32',
                                  inputs_shape=(65537, 2, 2, 2), filters_shape=(1, 2, 2, 2),
                                  should_fail=(version(raises=False) <= 6020))]


class TestDnnConv3D(BaseTestDnnConv):
    ndim = 3

    fwd_algorithms = cudnn.conv3d_fwd_algorithms
    bwd_filter_algorithms = cudnn.conv3d_bwd_filter_algorithms
    bwd_data_algorithms = cudnn.conv3d_bwd_data_algorithms

    cpu_conv_class = theano.tensor.nnet.corr3d.Corr3dMM
    cpu_gradinput_class = theano.tensor.nnet.corr3d.Corr3dMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr3d.Corr3dMM_gradWeights

    special_cases = [ConvCase.fwd(algo='small', dtype='float32', precision='float32',
                                  inputs_shape=(65536, 2, 2, 2, 2), filters_shape=(1, 2, 2, 2, 2)),
                     ConvCase.fwd(algo='small', dtype='float32', precision='float32',
                                  inputs_shape=(65537, 2, 2, 2, 2), filters_shape=(1, 2, 2, 2, 2),
                                  should_fail=(version(raises=False) <= 6020))]


def test_true_half_config_support():
    # For cuDNN V5.1 and V6.0:
    # "TRUE_HALF_CONFIG is only supported on architectures with true fp16 support (compute capability 5.3 and 6.0)"
    if not check_dtype_config_support('float16', 'float16'):
        raise SkipTest('FWD: TRUE_HALF_CONFIG not supported on this GPU.')


class CheckDnn:
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
        raise ValueError('unknown data type configuration', dtype_config)

    @staticmethod
    def print_infos(count_tests=True):
        # Print infos about tests and cuDNN supported algorithms and configurations.
        test_2d = TestDnnConv2D()
        test_3d = TestDnnConv3D()
        print()
        print('Available data type configurations:',
              ', '.join(CheckDnn.dtype_config_to_str(d)
                        for d in cudnn.get_supported_dtype_configs(check_dtype_config_support)))
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
        if count_tests:
            count_tests_2d = test_2d.get_expected_tcount()
            count_tests_3d = test_3d.get_expected_tcount()
            print(count_tests_2d, 'conv2D test cases.')
            print(count_tests_3d, 'conv3D test cases.')
            print('1 supplementary test.')
            print(count_tests_2d + count_tests_3d + 1, 'total conv tests.')
            print()

    @staticmethod
    def print_tests():
        # Print test cases without running them.
        for test in (TestDnnConv2D(), TestDnnConv3D()):
            for tcase in test.test_fwd():
                print(*(x.__name__ if callable(x) else x for x in tcase))
            for tcase in test.test_gradinput():
                print(*(x.__name__ if callable(x) else x for x in tcase))
            for tcase in test.test_gradweight():
                print(*(x.__name__ if callable(x) else x for x in tcase))


if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) == 1 and args[0] in ('infos', 'list'):
        if args[0] == 'infos':
            CheckDnn.print_infos()
        if args[0] == 'list':
            CheckDnn.print_tests()
    else:
        # We run all tests with nosetests.
        module_name = sys.modules[__name__].__file__
        if len(args) == 0:
            # No args given: run nosetests -vs
            args = ['--verbose', '--nocapture']
        # Else, use given args.
        argv = [sys.argv[0], module_name] + args

        CheckDnn.print_infos()
        nose.main(argv=argv)
