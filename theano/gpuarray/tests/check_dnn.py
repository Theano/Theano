from __future__ import absolute_import, print_function, division

from itertools import ifilter, product

import numpy as np

import theano
import theano.tests.unittest_tools as utt
from theano.compile.ops import shape_i_op
from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME
from theano.gof import COp, Apply, ParamsType
from theano.gof.type import CDataType
from theano.gpuarray import cudnn_defs
from theano.gpuarray.basic_ops import infer_context_name, as_gpuarray_variable, gpu_contiguous, GpuAllocEmpty
from theano.gpuarray.dnn import (GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI, version, get_precision,
                                 DnnBase, handle_type, DNN_CONV_ALGO_CHOOSE_ONCE, DNN_CONV_ALGO_CHOOSE_TIME)
from theano.gpuarray.tests.check_dnn_doc import check_fwd_algorithm
from theano.gpuarray.tests.config import mode_with_gpu, ref_cast
from theano.scalar import bool as bool_t
from theano.tensor.nnet.abstract_conv import get_conv_output_shape, assert_conv_shape
from theano.tensor.opt import Assert

cudnn = cudnn_defs.get_definitions(version(raises=False))

cudnnConvolutionFwdAlgo_t = cudnn.cudnnConvolutionFwdAlgo_t
cudnnConvolutionBwdFilterAlgo_t = cudnn.cudnnConvolutionBwdFilterAlgo_t
cudnnConvolutionBwdDataAlgo_t = cudnn.cudnnConvolutionBwdDataAlgo_t

AVAILABLE_PRECISIONS = cudnn.supported_precisions(theano.config.floatX)


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
        out_shp = get_conv_output_shape(case_tuple[0][0],  # input shape
                                        case_tuple[0][1],  # filter shape
                                        case_tuple[1],  # border mode
                                        case_tuple[0][2],  # subsample
                                        case_tuple[0][3]  # dilation
                                        )
        try:
            return assert_conv_shape(out_shp)
        except ValueError:
            return False

    def get_cases(self):
        # Generate an iterator of tuples with format:
        # ( (input shape, filter shape, subsample, dilation), border mode, convolution mode, alpha, beta )
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
                       product(product(all_input_shapes, all_filter_shapes, all_subsamples, all_dilations),
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


class BaseGpuDnnConvChooseAlgo(DnnBase):
    """
    This class and its subclasses allow to retrieve a cuDNN algorithm
    at runtime without any computation, given the user choose option
    (time_once, time_on_shape_change, guess_once or guess_on_shape_change).
    To help reduce whole test time, I suggest we use these classes when
    algo is one of choose options, as any chosen algorithm would have
    been tested by the other exhaustive tests.
    """

    _f16_ok = True
    check_input = False
    __props__ = ('choice',)
    params_type = ParamsType(choose_once=bool_t, choose_time=bool_t, handle=handle_type)

    # Abstract attributes.
    func_file = None
    func_name = None

    def __init__(self, choice):
        COp.__init__(self, ["../dnn_base.c", "../dnn_conv_base.c", self.func_file], self.func_name)
        assert choice in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        self.choice = choice
        self.choose_once = self.choice in DNN_CONV_ALGO_CHOOSE_ONCE
        self.choose_time = self.choice in DNN_CONV_ALGO_CHOOSE_TIME

    def dnn_context(self, node):
        return node.inputs[0].type.context_name

    def _prepare_inputs(self, i1, name_i1, i2, name_i2, output, desc):
        ctx_name = infer_context_name(i1, i2, output)
        i1 = as_gpuarray_variable(i1, ctx_name)
        i2 = as_gpuarray_variable(i2, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        if i1.type.ndim not in (4, 5):
            raise TypeError('%s must be 4D or 5D tensor' % name_i1)
        if i2.type.ndim not in (4, 5):
            raise TypeError('%s must be 4D or 5D tensor' % name_i2)
        if output.type.ndim not in (4, 5):
            raise TypeError('output must be 4D or 5D tensor')
        if i1.type.ndim != i2.type.ndim or i1.type.ndim != output.type.ndim:
            raise TypeError("The number of dimensions of %s, %s and output must match" % (name_i1, name_i2))
        if not isinstance(desc.type, CDataType) or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')
        return (i1, i2, output, desc)


class GpuDnnConvChooseFwdAlgo(BaseGpuDnnConvChooseAlgo):
    func_file = 'dnn_choose_fwd.c'
    func_name = 'APPLY_SPECIFIC(choose_fwd_algo)'

    def make_node(self, img, kern, output, desc):
        img, kern, output, desc = self._prepare_inputs(img, 'img', kern, 'kern', output, desc)
        return Apply(self, [img, kern, output, desc], [cudnn.cudnnConvolutionFwdAlgo_t()])


class GpuDnnConvChooseBwdFilterAlgo(BaseGpuDnnConvChooseAlgo):
    func_file = 'dnn_choose_gw.c'
    func_name = 'APPLY_SPECIFIC(choose_bwd_filter_algo)'

    def make_node(self, img, topgrad, output, desc):
        img, topgrad, output, desc = self._prepare_inputs(img, 'img', topgrad, 'topgrad', output, desc)
        return Apply(self, [img, topgrad, output, desc], [cudnn.cudnnConvolutionBwdFilterAlgo_t()])


class GpuDnnConvChooseBwdDataAlgo(BaseGpuDnnConvChooseAlgo):
    func_file = 'dnn_choose_gi.c'
    func_name = 'APPLY_SPECIFIC(choose_bwd_data_algo)'

    def make_node(self, kern, topgrad, output, desc):
        kern, topgrad, output, desc = self._prepare_inputs(kern, 'kern', topgrad, 'topgrad', output, desc)
        return Apply(self, [kern, topgrad, output, desc], [cudnn.cudnnConvolutionBwdDataAlgo_t()])


class BaseTestDnnConv(object):
    """
    Base class for exhaustive tests. Use its subclasses
    to run actual tests.
    """

    _functions_checked_for_fwd = False
    _functions_checked_for_gradinput = False
    _functions_checked_for_gradweight = False

    # Abstract attributes.

    ndim = 2

    fwd_algorithms = None
    bwd_filter_algorithms = None
    bwd_data_algorithms = None

    cpu_conv_class = None
    cpu_gradinput_class = None
    cpu_gradweight_class = None

    def get_cases(self):
        # Return an iterable of test cases. Each test case is a tuple (or list) with following syntax:
        # ( (input shape, filter shape, subsample, dilation), border mode, convolution mode, alpha, beta )
        generator = DnnCaseGenerator(ndim=self.ndim)
        return generator.get_cases()

    # Run and utility methods.

    def array_like_conv_output(self, inputs_shape, filters_shape, border_mode, subsample, dilation):
        # Return an random array with inferred convolution output shape.
        out_shp = get_conv_output_shape(inputs_shape, filters_shape, border_mode, subsample, dilation)
        out_shp = assert_conv_shape(out_shp)
        return np.random.random(out_shp).astype(theano.config.floatX)

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

        out = None if beta == 0 else self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample,
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
        if algo in cudnn.deterministic_fwd_algorithms:
            res2 = f()
            utt.assert_allclose(res, res2)

        # Raise tolerance for float16
        rtol = 6e-2 if theano.config.floatX == 'float16' else None
        if beta == 0:
            utt.assert_allclose(alpha * res_ref, res, rtol=rtol)
        else:
            # print('(conv: beta not null) ', end='')
            utt.assert_allclose(alpha * res_ref + beta * out, res, rtol=rtol)

    def run_conv_gradinput(self, algo, precision, parameters):
        (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)
        topgrad_val = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation)

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
        if algo in cudnn.deterministic_bwd_data_algorithms:
            res2 = f()
            utt.assert_allclose(res, res2)

        # Raise tolerance for float16
        rtol = 5e-2 if theano.config.floatX == 'float16' else None
        utt.assert_allclose(alpha * res_ref + beta * inputs_val, res, rtol=rtol)

    def run_conv_gradweight(self, algo, precision, parameters):
        (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)
        topgrad_val = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation)

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
        if algo in cudnn.deterministic_bwd_filter_algorithms:
            res2 = f()
            utt.assert_allclose(res, res2)

        # Raise tolerance for float16
        rtol = 5e-2 if theano.config.floatX == 'float16' else None
        utt.assert_allclose(alpha * res_ref + beta * filters_val, res, rtol=rtol)

    def run_choose_runtime_algos(self, algo, precision, parameters):
        (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters
        out_shp = assert_conv_shape(
            get_conv_output_shape(inputs_shape, filters_shape, border_mode, subsample, dilation))

        inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
        filters_val = np.random.random(filters_shape).astype(theano.config.floatX)
        topgrad_val = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation)

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)
        topgrad = theano.shared(topgrad_val)
        ctx_name = infer_context_name(inputs, topgrad)

        desc_filter = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                                     conv_mode=conv_mode, precision=precision)(filters_shape)

        array_like_filters = GpuAllocEmpty(dtype=inputs.dtype, context_name=ctx_name)(*filters_shape)
        array_like_inputs = GpuAllocEmpty(dtype=inputs.dtype, context_name=ctx_name)(*inputs_shape)
        array_like_conv_output = GpuAllocEmpty(dtype=inputs.dtype, context_name=ctx_name)(*out_shp)

        algo_filter = GpuDnnConvChooseBwdFilterAlgo(algo)(inputs, topgrad, array_like_filters, desc_filter)
        algo_input = GpuDnnConvChooseBwdDataAlgo(algo)(filters, topgrad, array_like_inputs, desc_filter)
        algo_conv = GpuDnnConvChooseFwdAlgo(algo)(inputs, filters, array_like_conv_output, desc_filter)
        f = theano.function([], [algo_filter, algo_input, algo_conv], mode=mode_with_gpu)

        # Just test that it runs.
        algo_filter_val, algo_input_val, algo_conv_val = f()
        # How to test if it "works" ?

    def get_expected_tcount(self):
        """
        Utility function to get expected test count
        without actually run nosetests.
        """
        len_cases = 0
        for c in self.get_cases():
            len_cases += 1
        print(len_cases, 'conv cases for %dD' % self.ndim)
        return len(AVAILABLE_PRECISIONS) * len_cases * len(self.fwd_algorithms +
                                                           self.bwd_data_algorithms +
                                                           self.bwd_filter_algorithms +
                                                           SUPPORTED_DNN_CONV_ALGO_RUNTIME)

    # Iterable test methods.

    def test_fwd(self):
        for precision, algo, parameters in product(AVAILABLE_PRECISIONS, self.fwd_algorithms, self.get_cases()):
            yield (self.run_conv_fwd, algo, precision, parameters)

    def test_gradinput(self):
        for precision, algo, parameters in product(AVAILABLE_PRECISIONS, self.bwd_data_algorithms, self.get_cases()):
            yield (self.run_conv_gradinput, algo, precision, parameters)

    def test_gradweight(self):
        for precision, algo, parameters in product(AVAILABLE_PRECISIONS, self.bwd_filter_algorithms, self.get_cases()):
            yield (self.run_conv_gradweight, algo, precision, parameters)

    def test_choose_runtime_algos(self):
        for precision, algo, parameters in product(AVAILABLE_PRECISIONS, SUPPORTED_DNN_CONV_ALGO_RUNTIME,
                                                   self.get_cases()):
            yield (self.run_choose_runtime_algos, algo, precision, parameters)

    def check_fwd_predictions(self):
        """
        Call this method to check if tests fail when they
        don't follow cuDNN V5.1 doc conditions for FWD algorithms.
        Script will exit as soon as there is a test that does not fail when expected.
        """

        print()
        print('TESTING FWD FAILURES PREDICTED FOR %dD' % self.ndim)
        count = 0
        for precision, algo, parameters in product(AVAILABLE_PRECISIONS, self.fwd_algorithms,
                                                   self.get_cases()):
            (inputs_shape, filters_shape, subsample, dilation), border_mode, conv_mode, alpha, beta = parameters

            inputs_val = np.random.random(inputs_shape).astype(theano.config.floatX)
            filters_val = np.random.random(filters_shape).astype(theano.config.floatX)
            # Scale down the input values to prevent very large absolute errors
            # due to float rounding
            inputs_val /= 10
            filters_val /= 10
            out = self.array_like_conv_output(inputs_shape, filters_shape, border_mode, subsample, dilation)
            desc_op = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
                                     conv_mode=conv_mode, precision=precision)
            should_compute = check_fwd_algorithm(inputs_val, filters_val, out, desc_op,
                                                 algo, precision, subsample, dilation)

            if not should_compute.ok:
                infos = ['ndim               : %s' % (len(inputs_shape) - 2),
                         'precision          : %s' % precision]
                infos += should_compute.messages
                try:
                    self.run_conv_fwd(algo, precision, parameters)
                except Exception as e:
                    print('(FAILS as expected)', algo, precision, parameters)
                    print(e.message.split('\n')[0])
                    for info in infos:
                        print(info)
                        # exit(0)
                else:
                    print('**SHOULD FAIL**|', algo, precision, parameters)
                    for info in infos:
                        print(info)
                    exit(-1)
            count += 1
            if count % 200 == 0:
                print(count, 'passed')
        print(count, 'finished')


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
    test_2d = TestDnnConv2D()
    test_3d = TestDnnConv3D()
    print('2D algorithms:')
    print('FWD       :', test_2d.fwd_algorithms)
    print('BWD FILTER:', test_2d.bwd_filter_algorithms)
    print('BWD DATA  :', test_2d.bwd_data_algorithms)
    print('3D algorithms:')
    print('FWD       :', test_3d.fwd_algorithms)
    print('BWD FILTER:', test_3d.bwd_filter_algorithms)
    print('BWD DATA  :', test_3d.bwd_data_algorithms)
    count_tests_2d = test_2d.get_expected_tcount()
    count_tests_3d = test_3d.get_expected_tcount()
    print(count_tests_2d, 'total cases for 2D.')
    print(count_tests_3d, 'total cases for 3D.')
    print(count_tests_2d + count_tests_3d, 'total cases.')
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        test_2d.check_fwd_predictions()
        test_3d.check_fwd_predictions()
