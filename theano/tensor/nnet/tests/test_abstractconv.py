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


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

from theano.sandbox.cuda.dnn import dnn_available

class TestConv2d(unittest.TestCase):

    def run_conv(self,
                 inputs_shape,
                 filters_shape,
                 subsample=(1, 1),
                 verify_grad=True,
                 mode=mode_with_gpu):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        c_ref = conv_ref.conv2d(inputs, filters,
                                border_mode="valid",
                                subsample=subsample)

        c = conv.conv2d(inputs, filters,
                        border_mode="valid", subsample=subsample)


        f_ref = theano.function([], c_ref, mode=mode_with_gpu)
        f = theano.function([], c, mode)

        res_ref = f_ref()
        res = f()
        print res_ref.shape, res.shape
        utt.assert_allclose(res_ref, res)
        # if verify_grad:
        #     utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
        #                                         subsample=subsample),
        #                     [inputs_val, filters_val])


    def run_gradweight(self,
                       inputs_shape,
                       filters_shape,
                       subsample=(1, 1),
                       verify_grad=True,
                       mode=mode_with_gpu):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val.transpose((1, 0, 2, 3)))
        filters = shared(filters_val.transpose((1, 0, 2, 3))[:,:,::-1,::-1])

        c = conv.AbstractConv2d_gradWeights(border_mode="valid",
                                            subsample=subsample)
        c = c(inputs, filters)
        f = theano.function([], c, mode)
        res_ref = py_conv(inputs_val, filters_val, 'valid', subsample)
        res = numpy.array(f()).transpose((1, 0, 2, 3))
        utt.assert_allclose(res_ref, res)
        # if verify_grad:
        #     utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
        #                                         subsample=subsample),
        #                     [inputs_val, filters_val])


    def run_gradinput(self,
                      inputs_shape,
                      filters_shape,
                      subsample=(1, 1),
                      verify_grad=True,
                      mode=mode_with_gpu):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val.transpose(1, 0, 2, 3))
        c = conv.AbstractConv2d_gradInputs(border_mode="valid",
                                           subsample=subsample)
        c = c(filters, inputs)
        f = theano.function([], c, mode)
        res_ref = py_conv(inputs_val, filters_val, 'full', subsample)
        res = numpy.array(f()) #.transpose((1, 0, 2, 3))
        print "2, ", res_ref.shape, res.shape
        utt.assert_allclose(res_ref, res)
        # if verify_grad:
        #     utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
        #                                         subsample=subsample),
        #                     [inputs_val, filters_val])




    def test_valid(self):
        mode = mode_with_gpu
       # if dnn_available():
            # self.run_conv(inputs_shape=(16, 1, 2, 2),
            #               filters_shape=(10, 1, 2, 2),
            #               verify_grad=False)
            # # self.run_conv(inputs_shape=(16, 1, 8, 8),
            # #               filters_shape=(10, 1, 2, 2),
            # #               subsample=(2, 2),
            # #               verify_grad=False)
            # self.run_conv(inputs_shape=(16, 1, 2, 2),
            #               filters_shape=(10, 1, 2, 2),
            #               verify_grad=True)
            # # self.run_conv(inputs_shape=(16, 1, 8, 8),
            # #               filters_shape=(10, 1, 2, 2),
            # #               subsample=(2, 2),
            # #               verify_grad=True)

        mode = mode.excluding('cudnn')
        self.run_conv(inputs_shape=(16, 1, 2, 2),
                      filters_shape=(10, 1, 2, 2),
                      verify_grad=False, mode=mode)
        self.run_gradweight(inputs_shape=(16, 1, 2, 2),
                            filters_shape=(10, 1, 2, 2),
                            verify_grad=False, mode=mode)
        self.run_gradinput(inputs_shape=(1, 1, 2, 2),
                           filters_shape=(10, 1, 2, 2),
                           verify_grad=False, mode=mode)



        # self.run_conv(inputs_shape=(16, 1, 8, 8),
        #               filters_shape=(10, 1, 4, 4),
        #                subsample=(2, 2),
        #               verify_grad=False,mode=mode)
        # self.run_conv(inputs_shape=(16, 1, 2, 2),
        #               filters_shape=(10, 1, 2, 2),
        #               verify_grad=True,mode=mode)
        # self.run_conv(inputs_shape=(16, 1, 8, 8),
        #               filters_shape=(10, 1, 2, 2),
        #               subsample=(2, 2),
        #               verify_grad=True,mode=mode)


