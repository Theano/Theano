#!/usr/bin/env python
# You can pass nosetests args when running this script. Examples:
# python theano/gpuarray/tests/check_dnn.py       # Normal mode.
# python theano/gpuarray/tests/check_dnn.py -xvs  # Verbose mode, capture output, exit at first error.
from __future__ import absolute_import, print_function, division

from itertools import ifilter, product, chain

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


class DnnCaseGenerator:
    """
    Main class used to generate test cases.

    """

    def _sub_size(self, sub_size=None):
        return int(sub_size) if sub_size is not None else self.input_size // 3 + 1

    def _at_least_one(self, value):
        return (value,) if value == 1 else (1, value)

    def _shapes(self, size):
        # Shapes:
        # [1, 1, ...] (at least)
        # [size, size, ...]
        # [..., size + 2, size + 1, size]
        if size == 1:
            return ((1,) * self.ndim,
                    tuple(size + self.ndim - i - 1 for i in range(self.ndim)))
        return ((1,) * self.ndim,
                (size,) * self.ndim,
                tuple(size + self.ndim - i - 1 for i in range(self.ndim)))

    def __init__(self,
                 ndim=2, alpha=2, beta=-3, batch_size=2, input_channels=3, input_size=8, output_channels=2,
                 filter_size=None, border_size=None, subsample_size=None, dilation_size=None):
        self.ndim = int(ndim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.input_channels = int(input_channels)
        self.input_size = int(input_size)
        self.output_channels = int(output_channels)
        self.filter_size = self._sub_size(filter_size)
        self.border_size = self._sub_size(border_size)
        self.subsample_size = self._sub_size(subsample_size)
        self.dilation_size = self._sub_size(dilation_size)

        assert self.ndim >= 2
        assert self.alpha != 0
        assert self.batch_size > 0
        assert self.input_channels > 0
        assert self.input_size > 0
        assert self.output_channels > 0
        assert self.filter_size > 0
        assert self.border_size > 0
        assert self.subsample_size > 0
        assert self.dilation_size > 0

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
        all_input_sizes = self._shapes(self.input_size)
        all_output_channels = (self.output_channels,)
        all_filter_sizes = self._shapes(((self.filter_size - 1) * self.dilation_size + 1)
                                        if cudnn.version < 6
                                        else self.filter_size)
        all_subsamples = self._shapes(self.subsample_size)
        all_dilations = ((1,) * self.ndim,) if cudnn.version < 6 else self._shapes(self.dilation_size)
        all_border_modes = ('valid', 'full', 'half') + self._shapes(self.border_size)
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
            grad_w_ref = grad_w_ref[:, :, ::-1, ::-1, ::-1]
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
        return len_cases * count_contexts

    # Iterable test methods.

    def test_fwd(self):
        for dtype, precision in cudnn.get_fwd_dtype_configs(check_runtime=check_fwd_dtype_config_support):
            algos = (algo for algo in self.fwd_algorithms
                     if cudnn.fwd_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in chain(algos, SUPPORTED_DNN_CONV_ALGO_RUNTIME):
                for parameters in self.get_cases():
                    yield (self.run_conv_fwd, algo, dtype, precision, parameters)

    def test_gradinput(self):
        for dtype, precision in cudnn.get_bwd_data_dtype_configs():
            algos = (algo for algo in self.bwd_data_algorithms
                     if cudnn.bwd_data_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in chain(algos, SUPPORTED_DNN_CONV_ALGO_RUNTIME):
                for parameters in self.get_cases():
                    yield (self.run_conv_gradinput, algo, dtype, precision, parameters)

    def test_gradweight(self):
        for dtype, precision in cudnn.get_bwd_filter_dtype_configs():
            algos = (algo for algo in self.bwd_filter_algorithms
                     if cudnn.bwd_filter_algo_supports_dtype_config(algo, dtype, precision, self.ndim))
            for algo in chain(algos, SUPPORTED_DNN_CONV_ALGO_RUNTIME):
                for parameters in self.get_cases():
                    yield (self.run_conv_gradweight, algo, dtype, precision, parameters)


class TestDnnConv2D(BaseTestDnnConv):
    ndim = 2

    fwd_algorithms = cudnn.cudnnConvolutionFwdAlgo_t.get_aliases()
    bwd_filter_algorithms = cudnn.cudnnConvolutionBwdFilterAlgo_t.get_aliases()
    bwd_data_algorithms = cudnn.cudnnConvolutionBwdDataAlgo_t.get_aliases()

    cpu_conv_class = theano.tensor.nnet.corr.CorrMM
    cpu_gradinput_class = theano.tensor.nnet.corr.CorrMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr.CorrMM_gradWeights


class TestDnnConv3D(BaseTestDnnConv):
    ndim = 3

    fwd_algorithms = cudnn.conv3d_fwd_algorithms
    bwd_filter_algorithms = cudnn.conv3d_bwd_filter_algorithms
    bwd_data_algorithms = cudnn.conv3d_bwd_data_algorithms

    cpu_conv_class = theano.tensor.nnet.corr3d.Corr3dMM
    cpu_gradinput_class = theano.tensor.nnet.corr3d.Corr3dMM_gradInputs
    cpu_gradweight_class = theano.tensor.nnet.corr3d.Corr3dMM_gradWeights


if __name__ == '__main__':

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

    test_2d = TestDnnConv2D()
    test_3d = TestDnnConv3D()
    print()
    print('Available data type configurations     :',
          ', '.join(dtype_config_to_str(d) for d in cudnn.get_supported_dtype_configs()))
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
    nose.main(defaultTest='theano.gpuarray.tests.check_dnn')
