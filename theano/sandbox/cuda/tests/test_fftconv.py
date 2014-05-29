import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

from theano.sandbox.cuda import float32_shared_constructor as shared

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestConv2dFFT(unittest.TestCase):
    def run_conv(self, inputs_shape, filters_shape, pad=False, **other_args):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv_ref = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                                  **other_args)
        conv_fft = theano.sandbox.cuda.fftconv.conv2d_fft(inputs, filters,
                                                          pad_last_dim=pad,
                                                          **other_args)

        f_ref = theano.function([], conv_ref)
        f_fft = theano.function([], conv_fft, mode=mode_with_gpu)

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)

    def test_valid(self):
        self.run_conv(inputs_shape=(5, 3, 7, 6),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='valid')
        self.run_conv(inputs_shape=(5, 3, 7, 7),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='valid', pad=True)

    def test_full(self):
        self.run_conv(inputs_shape=(5, 3, 7, 6),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='full')
        self.run_conv(inputs_shape=(5, 3, 7, 7),
                      filters_shape=(2, 3, 3, 3),
                      border_mode='full', pad=True)

    def test_opt_valid(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters)

        mode = mode_with_gpu.including('conv_fft_valid')

        f_ref = theano.function([], conv)
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2


        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)

    def test_opt_full(self):
        inputs_shape = (5, 3, 7, 6)
        filters_shape = (2, 3, 3, 3)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.conv.conv2d(inputs, filters,
                                              border_mode='full')

        mode = mode_with_gpu.including('conv_fft_full')

        f_ref = theano.function([], conv)
        f_fft = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f_fft.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.fftconv.CuFFTOp)
                   for n in topo) == 2

        res_ref = f_ref()
        res_fft = f_fft()

        utt.assert_allclose(res_ref, res_fft)






class TestConv3dFFT(unittest.TestCase):

    @staticmethod
    def perform_conv2d_fft(inputs, filters, border_mode, function_mode):

        assert(border_mode in ['valid', 'full'])
        # function_mode is just mode_with_gpu from the environment

        if inputs.shape[-1] % 2 == 1:
            pad_last_dim = True
        else:
            pad_last_dim = False

        sym_inputs  = theano.tensor.tensor4()
        sym_filters = theano.tensor.tensor4()

        sym_outputs = theano.sandbox.cuda.fftconv.conv2d_fft(sym_inputs, sym_filters, image_shape=inputs.shape, filter_shape=filters.shape, border_mode=border_mode, pad_last_dim=pad_last_dim)
        #f = theano.function([sym_inputs, sym_filters], sym_outputs, mode=function_mode)
        f = theano.function([sym_inputs, sym_filters], sym_outputs)
        outputs_on_gpu = f(inputs, filters)
        outputs = numpy.array(outputs_on_gpu)

        return outputs

    @staticmethod
    def perform_conv3d_through_multiple_conv2d_fft(inputs, filters, border_mode, function_mode):

        assert(border_mode in ['valid', 'full'])
        # function_mode is just mode_with_gpu from the environment

        (nbr_images, nbr_channels, image_height,  image_width,  image_duration)  = inputs.shape
        (nbr_filters,           _, filter_height, filter_width, filter_duration) = filters.shape

        if border_mode == 'valid':
            outputs = numpy.zeros( (nbr_images, nbr_filters,
                                    image_height - filter_height + 1,
                                    image_width - filter_width + 1,
                                    image_duration - filter_duration + 1), dtype=numpy.float32 )

            for t in range(image_duration - filter_duration + 1):
                for sub_t in range(filter_duration):
                    #print "(t, sub_t) is (%d, %d),     (t + sub_t, filter_duration - 1 -sub_t) is (%d, %d)" % (t, sub_t, t + sub_t, filter_duration - 1 -sub_t)
                    outputs[:,:,:,:,t] = outputs[:,:,:,:,t] + TestConv3dFFT.perform_conv2d_fft(inputs[:,:,:,:,t + sub_t].copy(), filters[:,:,:,:, filter_duration - 1 - sub_t].copy(), border_mode, function_mode)

            return outputs

        elif border_mode == 'full':

            # pad in time, and then rely on the proper 2d convolution to work out the padding in the height and width
            padded_inputs = numpy.zeros( (nbr_images, nbr_channels,
                                          image_height + 2 * (filter_height - 1),
                                          image_width + 2 * (filter_width - 1),
                                          image_duration + 2 * (filter_duration - 1) ), dtype=numpy.float32)
            padded_inputs[:,:,filter_height-1:filter_height-1+image_height,filter_width-1:filter_width-1+image_width,filter_duration-1:filter_duration-1+image_duration] = inputs.copy()

            return TestConv3dFFT.perform_conv3d_through_multiple_conv2d_fft(padded_inputs, filters, border_mode='valid', function_mode=function_mode)

    @staticmethod
    def perform_fftconv3d(inputs, filters, border_mode, function_mode):

        assert(border_mode in ['valid', 'full'])

        if inputs.shape[-1] % 2 == 1:
            pad_last_dim = True
        else:
            pad_last_dim = False

        tensor5 = theano.tensor.TensorType('float32', (False,)*5)

        sym_inputs  = tensor5()
        sym_filters = tensor5()

        sym_outputs = theano.sandbox.cuda.fftconv.conv3d_fft(sym_inputs, sym_filters, image_shape=inputs.shape, filter_shape=filters.shape, border_mode=border_mode, pad_last_dim=pad_last_dim)
        #f = theano.function([sym_inputs, sym_filters], sym_outputs, mode=mode_with_gpu)
        f = theano.function([sym_inputs, sym_filters], sym_outputs)
        outputs_on_gpu = f(inputs, filters)
        outputs = numpy.array(outputs_on_gpu)

        return outputs


    def run_conv(self, inputs_shape, filters_shape, border_mode):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        res_ref = TestConv3dFFT.perform_conv3d_through_multiple_conv2d_fft(inputs_val, filters_val, border_mode, mode_with_gpu)
        res_fft = TestConv3dFFT.perform_fftconv3d(inputs_val, filters_val, border_mode, mode_with_gpu)

        utt.assert_allclose(res_ref, res_fft)

    def test_valid(self):

        for offset1 in range(2):
            for offset2 in range(2):
                for offset3 in range(2):
                    self.run_conv(inputs_shape=(5, 3, 5 + offset1, 6 + offset2, 4 + offset3),
                                  filters_shape=(2, 3, 3 + offset1, 3 + offset2, 2 + offset3),
                                  border_mode='valid')

    def test_full(self):

        for offset1 in range(2):
            for offset2 in range(2):
                for offset3 in range(2):
                    self.run_conv(inputs_shape=(5, 3, 5 + offset1, 6 + offset2, 4 + offset3),
                                  filters_shape=(2, 3, 3 + offset1, 3 + offset2, 3 + offset3),
                                  border_mode='full')







