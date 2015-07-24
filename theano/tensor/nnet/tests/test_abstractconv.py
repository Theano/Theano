import unittest
import numpy
import copy

import theano
from theano.tests import unittest_tools as utt

from nose.plugins.skip import SkipTest
import theano.tensor.nnet.conv as conv_ref
import theano.tensor.nnet.abstract_conv2d as conv

from theano.sandbox.cuda import float32_shared_constructor as shared
from theano.sandbox.cuda.tests.test_conv_cuda_ndarray import py_conv
#from theano.sandbox.cuda.dnn import dnn_available


if theano.config.mode == 'FAST_COMPILE':
    #mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    #mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')


class TestConv2d(unittest.TestCase):

    def run_conv(self,
                 inputs_shape,
                 filters_shape,
                 subsample=(1, 1),
                 verify_grad=True,
                 mode=mode_without_gpu):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        ### FIXME (CPU vs GPU)
        inputs = theano.tensor.shared(inputs_val)
        filters = theano.tensor.shared(filters_val)


        c_ref = conv_ref.conv2d(inputs, filters,
                                border_mode="valid",
                                subsample=subsample)

        c = conv.conv2d(inputs, filters,
                        border_mode="valid", subsample=subsample)


        f_ref = theano.function([], c_ref, mode=mode)
        f = theano.function([], c, mode)

        res_ref = f_ref()
        res = f()
        print res_ref.shape, res.shape
        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
                                                imshp=inputs_shape,
                                                kshp=filters_shape,
                                                bsize=inputs_shape[0],
                                                subsample=subsample),
                            [inputs_val, filters_val])


    def run_gradweight(self,
                       inputs_shape,
                       filters_shape,
                       output_shape,
                       subsample=(1, 1),
                       verify_grad=True,
                       mode=mode_without_gpu,
                       device='gpu',
                       provide_shape = False):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        output_val = numpy.random.random(output_shape).astype('float32')

        if device == 'gpu':
            inputs = shared(inputs_val)
            filters = shared(filters_val)
        else:
            inputs = theano.tensor.shared(inputs_val)
            output = theano.tensor.shared(output_val)

        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None,
            kshp = None

        c = conv.AbstractConv2d_gradWeights(border_mode="valid",
                                            subsample=subsample,
                                            imshp = imshp, kshp = kshp)
        c = c(inputs, output, filters_shape)
        f = theano.function([], c, mode)
        res_ref = py_conv(inputs_val.transpose((1, 0, 2, 3)),
                          output_val.transpose((1, 0, 2, 3)),
                          'valid', subsample).transpose((1, 0, 2, 3))
        print res_ref.shape, numpy.array(f()).shape
        res = numpy.array(f())
        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
                                                subsample=subsample),
                            [inputs_val, filters_val])


    def run_gradinput(self,
                      inputs_shape,
                      filters_shape,
                      subsample=(1, 1),
                      verify_grad=True,
                      mode=mode_without_gpu):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
        c = conv.AbstractConv2d_gradInputs(border_mode="valid",
                                           subsample=subsample)
        c = c(filters, inputs, inputs_shape)
        f = theano.function([], c, mode)
        res_ref = py_conv(inputs_val, filters_val, 'full', subsample)
        res = numpy.array(f()) #.transpose((1, 0, 2, 3))
        print "2, ", res_ref.shape, res.shape

        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
                                                subsample=subsample),
                            [inputs_val, filters_val])



    # def test_corrmm(self):
    #     mode = mode_with_gpu
    #     mode = mode.excluding('cudnn')
    #     self.run_conv(inputs_shape=(16, 1, 2, 2),
    #                   filters_shape=(10, 1, 2, 2),
    #                   verify_grad=False, mode=mode)
    #     self.run_gradweight(inputs_shape=(16, 1, 2, 2),
    #                         filters_shape=(10, 1, 2, 2),
    #                         verify_grad=False, mode=mode)
    #     self.run_gradinput(inputs_shape=(1, 1, 2, 2),
    #                        filters_shape=(10, 1, 2, 2),
    #                        verify_grad=False, mode=mode)


    #def test_cpu(self):
        #self.run_conv(inputs_shape=(16, 1, 2, 2),
        #              filters_shape=(10, 1, 2, 2),
        #              verify_grad=False,
        #              mode=mode_without_gpu)
        # self.run_gradinput(inputs_shape=(1, 1, 2, 2),
        #                    filters_shape=(10, 1, 2, 2),
        #                    verify_grad=False, mode=mode_without_gpu)

        # mode = mode_without_gpu
        # self.run_conv(inputs_shape=(16, 1, 2, 2),
        #               filters_shape=(10, 1, 2, 2),
        #               verify_grad=False, mode=mode)
        # self.run_gradweight(inputs_shape=(16, 1, 2, 2),
        #                     filters_shape=(10, 1, 2, 2),
        #                     verify_grad=False, mode=mode)
        # self.run_gradinput(inputs_shape=(1, 1, 2, 2),
        #                    filters_shape=(10, 1, 2, 2),
        #                    verify_grad=False, mode=mode)



        # # self.run_conv(inputs_shape=(16, 1, 8, 8),
        # #               filters_shape=(10, 1, 4, 4),
        # #                subsample=(2, 2),
        # #               verify_grad=False,mode=mode)
        # # self.run_conv(inputs_shape=(16, 1, 2, 2),
        # #               filters_shape=(10, 1, 2, 2),
        # #               verify_grad=True,mode=mode)
        # # self.run_conv(inputs_shape=(16, 1, 8, 8),
        # #               filters_shape=(10, 1, 2, 2),
        # #               subsample=(2, 2),
        # #               verify_grad=True,mode=mode)

    def test_cpu_grad_weight(self):
        self.run_gradweight(inputs_shape=(16, 1, 2, 2),
                            filters_shape=(10, 1, 2, 2),
                            output_shape=(16, 10, 1, 1),
                            verify_grad=False, mode=mode_without_gpu, device='cpu')
        self.run_gradweight(inputs_shape=(16, 1, 2, 2),
                            filters_shape=(10, 1, 2, 2),
                            output_shape=(16, 10, 1, 1),
                            verify_grad=False,
                            mode=mode_without_gpu, device='cpu',
                            provide_shape=True)


