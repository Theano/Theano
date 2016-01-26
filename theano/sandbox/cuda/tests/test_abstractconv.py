import unittest
import numpy
import itertools

import theano
from theano import tensor
from theano.tests import unittest_tools as utt
import theano.tensor.nnet.abstract_conv as conv
from theano.sandbox.cuda import float32_shared_constructor as gpu_shared
from theano.compile import shared as cpu_shared
from theano.sandbox.cuda.dnn import (
    dnn_available, dnn_conv, dnn_gradweight, dnn_gradinput,
    GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI)
from theano.sandbox.cuda.blas import (
    GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs)
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda
if not cuda.cuda_available:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.get_default_mode().excluding('gpu')


class TestConv2d(unittest.TestCase):

    def setUp(self):

        super(TestConv2d, self).setUp()
        self.inputs_shapes = [(8, 1, 12, 12), (8, 1, 18, 18), (2, 1, 4, 4),
                              (6, 1, 10, 11), (2, 1, 6, 5), (1, 5, 9, 9)]
        self.filters_shapes = [(5, 1, 2, 2), (4, 1, 3, 3), (2, 1, 3, 3),
                               (1, 1, 2, 5), (4, 1, 2, 2), (4, 5, 2, 2)]
        self.subsamples = [(1, 1), (2, 2), (2, 4)]
        self.border_modes = ["valid", "full", "half",
                             (0, 0), (1, 1), (5, 5), (5, 2)]
        self.filter_flip = [True, False]

    def get_output_shape(self, inputs_shape, filters_shape,
                         subsample, border_mode):
        if border_mode == "valid":
            border_mode = (0, 0)
        elif border_mode == "full":
            border_mode = (filters_shape[2] - 1, filters_shape[3] - 1)
        elif border_mode == "half":
            border_mode = (filters_shape[2] // 2, filters_shape[3] // 2)
        batch_size = inputs_shape[0]
        num_filters = filters_shape[0]
        return (batch_size, num_filters,) \
            + tuple(None if i is None or k is None
                    else ((i + 2 * pad - k) // d + 1)
                    for i, k, d, pad in zip(inputs_shape[2:], filters_shape[2:],
                                            subsample, border_mode))

    def run_fwd(self, inputs_shape, filters_shape, ref=dnn_conv,
                subsample=(1, 1), verify_grad=True, mode=mode_without_gpu,
                border_mode='valid', filter_flip=True, device='cpu', provide_shape=False,
                target_op=None):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')
        if device == 'gpu':
            inputs = gpu_shared(inputs_val)
            filters = gpu_shared(filters_val)
        else:
            inputs = theano.tensor.as_tensor_variable(cpu_shared(inputs_val))
            filters = theano.tensor.as_tensor_variable(cpu_shared(filters_val))
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
                    conv_mode=conv_mode)
        c = conv.conv2d(inputs, filters,
                        border_mode=border_mode,
                        subsample=subsample,
                        filter_flip=filter_flip,
                        input_shape=imshp,
                        filter_shape=kshp)
        f_ref = theano.function([], c_ref, mode=mode)
        f = theano.function([], c, mode)

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])

        res_ref = numpy.array(f_ref())
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode="valid", imshp=imshp, kshp=kshp,
                                                subsample=subsample),
                            [inputs_val, filters_val],
                            mode=mode)

    def run_gradweight(self, inputs_shape, filters_shape, output_shape,
                       ref=dnn_gradweight, subsample=(1, 1), filter_flip=True,
                       verify_grad=True, mode=mode_without_gpu, border_mode='valid',
                       device='cpu', provide_shape=False, target_op=None):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        output_val = numpy.random.random(output_shape).astype('float32')
        if device == 'gpu':
            inputs = gpu_shared(inputs_val)
            output = gpu_shared(output_val)
        else:
            inputs = theano.tensor.as_tensor_variable(cpu_shared(inputs_val))
            output = theano.tensor.as_tensor_variable(cpu_shared(output_val))
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
                                            imshp=imshp, kshp=kshp)
        c = c(inputs, output, filters_shape[-2:])
        c_ref = ref(inputs, output,
                    filters_shape,
                    border_mode=border_mode,
                    subsample=subsample,
                    conv_mode=conv_mode)
        f = theano.function([], c, mode)
        f_ref = theano.function([], c_ref, mode)

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])

        res_ref = numpy.array(f_ref())
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)

        def abstract_conv2d_gradweight(inputs_val, output_val):
            conv_op = conv.AbstractConv2d_gradWeights(border_mode=border_mode, subsample=subsample)
            return conv_op(inputs_val, output_val, filters_shape[-2:])

        if verify_grad:
            utt.verify_grad(abstract_conv2d_gradweight, [inputs_val, output_val],
                            mode=mode, eps=1)

    def run_gradinput(self, inputs_shape, filters_shape,
                      output_shape, ref=dnn_gradinput,
                      subsample=(1, 1), filter_flip=True,
                      verify_grad=True, mode=mode_without_gpu,
                      border_mode='valid', device='cpu', provide_shape=False,
                      target_op=None):

        output_val = numpy.random.random(output_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')
        if device == 'gpu':
            output = gpu_shared(output_val)
            filters = gpu_shared(filters_val)
        else:
            output = theano.tensor.as_tensor_variable(cpu_shared(output_val))
            filters = theano.tensor.as_tensor_variable(cpu_shared(filters_val))
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
                                           imshp=imshp, kshp=kshp)
        c = c(filters, output, inputs_shape[-2:])
        c_ref = ref(filters, output, inputs_shape,
                    border_mode=border_mode, subsample=subsample,
                    conv_mode=conv_mode)
        f = theano.function([], c, mode)
        f_ref = theano.function([], c_ref, mode)

        if target_op is not None:
            assert any([isinstance(n.op, target_op) for n
                        in f.maker.fgraph.toposort()])

        res_ref = numpy.array(f_ref())
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)

        def abstract_conv2d_gradinputs(filters_val, output_val):
            conv_op = conv.AbstractConv2d_gradInputs(border_mode=border_mode, subsample=subsample)
            return conv_op(filters_val, output_val, inputs_shape[-2:])
        if verify_grad:
            utt.verify_grad(abstract_conv2d_gradinputs, [filters_val, output_val],
                            mode=mode, eps=1)

    def test_dnn_conv(self):
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        mode = mode_with_gpu
        # provide_shape is not used by the CuDNN impementation
        provide_shape = False
        for (i, f), s, b, flip in itertools.product(
                zip(self.inputs_shapes, self.filters_shapes),
                self.subsamples,
                self.border_modes,
                self.filter_flip):
            o = self.get_output_shape(i, f, s, b)
            self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                         verify_grad=True, mode=mode, device='gpu',
                         provide_shape=provide_shape, border_mode=b,
                         filter_flip=flip, target_op=GpuDnnConv)
            self.run_gradweight(inputs_shape=i, filters_shape=f,
                                output_shape=o, subsample=s,
                                verify_grad=True, mode=mode, device='gpu',
                                provide_shape=provide_shape, border_mode=b,
                                filter_flip=flip, target_op=GpuDnnConvGradW)
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=True, mode=mode, device='gpu',
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip, target_op=GpuDnnConvGradI)

    def test_gpucorrmm_conv(self):
        if not dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)

        mode = mode_with_gpu.excluding('cudnn')
        for (i, f), s, b, flip, provide_shape in itertools.product(
                zip(self.inputs_shapes, self.filters_shapes),
                self.subsamples,
                self.border_modes,
                self.filter_flip,
                [False, True]):

            o = self.get_output_shape(i, f, s, b)
            self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                         verify_grad=True, mode=mode, device='gpu',
                         provide_shape=provide_shape, border_mode=b,
                         filter_flip=flip,
                         target_op=(GpuCorrMM,
                                    GpuCorrMM_gradWeights,
                                    GpuCorrMM_gradInputs))
            self.run_gradweight(inputs_shape=i, filters_shape=f,
                                output_shape=o, subsample=s,
                                verify_grad=True, mode=mode, device='gpu',
                                provide_shape=provide_shape, border_mode=b,
                                filter_flip=flip,
                                target_op=GpuCorrMM_gradWeights)
            self.run_gradinput(inputs_shape=i, filters_shape=f,
                               output_shape=o, subsample=s,
                               verify_grad=True, mode=mode, device='gpu',
                               provide_shape=provide_shape, border_mode=b,
                               filter_flip=flip,
                               target_op=GpuCorrMM_gradInputs)

    def test_grad_types(self):
        # This function simply tests the behaviour of the AbstractConv
        # Ops, not their optimizations
        cpu_input = tensor.ftensor4()
        cpu_filters = tensor.ftensor4()
        cpu_topgrad = tensor.ftensor4()
        gpu_input = cuda.ftensor4()
        gpu_filters = cuda.ftensor4()
        gpu_topgrad = cuda.ftensor4()

        out_shape = tensor.lvector()

        # Check the gradient of the forward conv2d
        for input, filters in itertools.product(
                (cpu_input, gpu_input),
                (cpu_filters, gpu_filters)):
            output = conv.conv2d(input, filters)
            grad_input, grad_filters = theano.grad(output.sum(),
                                                   wrt=(input, filters))
            assert grad_input.type == input.type, (
                grad_input, grad_input.type, input, input.type)
            assert grad_filters.type == filters.type, (
                grad_filters, grad_filters.type, filters, filters.type)

        # Check the gradient of gradweight
        for input, topgrad in itertools.product(
                (cpu_input, gpu_input),
                (cpu_topgrad, gpu_topgrad)):
            grad_filters = conv.AbstractConv2d_gradWeights()(
                input, topgrad, out_shape)
            grad_input, grad_topgrad = theano.grad(grad_filters.sum(),
                                                   wrt=(input, topgrad))

            assert grad_input.type == input.type, (
                grad_input, grad_input.type, input, input.type)
            assert grad_topgrad.type == topgrad.type, (
                grad_topgrad, grad_topgrad.type, topgrad, topgrad.type)

        # Check the gradient of gradinputs
        for filters, topgrad in itertools.product(
                (cpu_filters, gpu_filters),
                (cpu_topgrad, gpu_topgrad)):
            grad_input = conv.AbstractConv2d_gradInputs()(
                filters, topgrad, out_shape)
            grad_filters, grad_topgrad = theano.grad(grad_input.sum(),
                                                     wrt=(filters, topgrad))

            assert grad_filters.type == filters.type, (
                grad_filters, grad_filters.type, filters, filters.type)
            assert grad_topgrad.type == topgrad.type, (
                grad_topgrad, grad_topgrad.type, topgrad, topgrad.type)
