import unittest
import numpy
import copy

import theano
from theano.tests import unittest_tools as utt

from nose.plugins.skip import SkipTest
import theano.tensor.nnet.conv as conv_ref
import theano.tensor.nnet.abstract_conv2d as conv

from theano.sandbox.cuda import float32_shared_constructor as gpu_shared
from theano.compile import shared as cpu_shared

from theano.sandbox.cuda.tests.test_conv_cuda_ndarray import py_conv
#from theano.sandbox.cuda.dnn import dnn_available


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.get_default_mode().excluding('gpu')


class TestConv2d(unittest.TestCase):

    def run_fwd(self,
                inputs_shape,
                filters_shape,
                subsample=(1, 1),
                verify_grad=True,
                mode=mode_without_gpu,
                border_mode='valid',
                device='gpu',
                provide_shape=False):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        if device == 'gpu':
            inputs = gpu_shared(inputs_val)
            filters = gpu_shared(filters_val)
        else:
            inputs = cpu_shared(inputs_val)
            filters = cpu_shared(filters_val)
        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None
            kshp = None

        c_ref = conv_ref.conv2d(inputs, filters,
                                border_mode=border_mode,
                                subsample=subsample)
        c = conv.conv2d(inputs, filters,
                        border_mode=border_mode, subsample=subsample)

        f_ref = theano.function([], c_ref, mode=mode)
        f = theano.function([], c, mode)

        res_ref = f_ref()
        res = f()
        print res_ref.shape, res.shape
        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
                                                imshp=imshp,
                                                kshp=kshp,
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
                       border_mode='valid',
                       device='gpu',
                       provide_shape = False):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        output_val = numpy.random.random(output_shape).astype('float32')

        if device == 'gpu':
            inputs = gpu_shared(inputs_val)
            output = gpu_shared(output_val)
        else:
            inputs = cpu_shared(inputs_val)
            output = cpu_shared(output_val)

        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None
            kshp = None

        c = conv.AbstractConv2d_gradWeights(border_mode=border_mode,
                                            subsample=subsample,
                                            imshp = imshp, kshp = kshp)
        c = c(inputs, output, filters_shape)
        f = theano.function([], c, mode)
        res_ref = py_conv(inputs_val.transpose((1, 0, 2, 3)),
                          output_val.transpose((1, 0, 2, 3))[:, :, ::-1, ::-1],
                          'valid', subsample).transpose((1, 0, 2, 3))
        res = numpy.array(f())
        print res_ref.shape, res.shape

        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d(border_mode="valid",
                                                subsample=subsample),
                            [inputs_val, filters_val])


    def run_gradinput(self,
                      inputs_shape,
                      filters_shape,
                      output_shape,
                      subsample=(1, 1),
                      verify_grad=True,
                      mode=mode_without_gpu,
                      border_mode='valid',
                      device='gpu',
                      provide_shape = False):

        output_val = numpy.random.random(output_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')


        if device == 'gpu':
            output = gpu_shared(output_val)
            filters = gpu_shared(filters_val)
        else:
            output = cpu_shared(output_val)
            filters = cpu_shared(filters_val)
        if provide_shape:
            imshp = inputs_shape
            kshp = filters_shape
        else:
            imshp = None
            kshp = None

        c = conv.AbstractConv2d_gradInputs(border_mode="valid",
                                           subsample=subsample,
                                           imshp = imshp, kshp = kshp)
        c = c(filters, output, inputs_shape)
        f = theano.function([], c, mode)
        res_ref = py_conv(output_val,
                          filters_val.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1],
                          'full', subsample)
        print filters_val.shape, output_val.shape, inputs_shape
        res = numpy.array(f())
        print "2, ", res_ref.shape, res.shape

        utt.assert_allclose(res_ref, res)
        if verify_grad:
            utt.verify_grad(conv.AbstractConv2d_gradInputs(border_mode=border_mode,
                                                           subsample=subsample),
                            [filters_val, output_val,
                             numpy.array(inputs_shape).astype('float32')])



    #def test_corrmm(self):
    #    mode = mode_with_gpu
    #    mode = mode.excluding('cudnn')
    #    self.run_fwd(inputs_shape=(16, 1, 2, 2),
    #                 filters_shape=(10, 1, 2, 2),
    #                 verify_grad=False, mode=mode)
    #     self.run_gradweight(inputs_shape=(16, 1, 2, 2),
    #                         filters_shape=(10, 1, 2, 2),
    #                         verify_grad=False, mode=mode)
    #     self.run_gradinput(inputs_shape=(1, 1, 2, 2),
    #                        filters_shape=(10, 1, 2, 2),
    #                        verify_grad=False, mode=mode)



    def test_cpu_conv(self):

        inputs_shapes =  [(16, 1, 2, 2), (16, 1, 8, 8), (16, 1, 4, 4)]
        filters_shapes = [(10, 1, 2, 2), (10, 1, 2, 2), (10, 1, 2, 2),]
        output_shapes =  [(16, 10, 1, 1), (16, 10, 7, 7), (16, 10, 3, 3)]
        subsamples =     [(1, 1), (1, 1), (1, 1)]

        border_mode= 'valid'
        for i, f, o, s in zip(inputs_shapes[0:1], filters_shapes[0:1], output_shapes[0:1], subsamples[0:1]):
            for provide_shape in [True]:
                self.run_fwd(inputs_shape=i, filters_shape=f, subsample=s,
                             verify_grad=True, mode=mode_without_gpu, device='cpu',
                             provide_shape=provide_shape, border_mode=border_mode)
        return
        ### No reference implementation of full available yet
        border_mode= 'full'
        provide_shape = True
        self.run_gradweight(inputs_shape=(16, 1, 2, 2),
                            filters_shape=(10, 1, 2, 2),
                            output_shape=(16, 10, 3, 3),
                            subsample=(1, 1),
                            verify_grad=True, mode=mode_without_gpu, device='cpu',
                            provide_shape=provide_shape, border_mode=border_mode)




    def test_cpu_grad_weight(self):

        ### FIXME subsample
        inputs_shapes =  [(16, 1, 2, 2), (16, 1, 8, 8), (16, 1, 4, 4)]
        filters_shapes = [(10, 1, 2, 2), (10, 1, 2, 2), (10, 1, 2, 2),]
        output_shapes =  [(16, 10, 1, 1), (16, 10, 7, 7), (16, 10, 3, 3)]
        subsamples =     [(1, 1), (1, 1), (1, 1)]

        border_mode = 'valid'
        for i, f, o, s in zip(inputs_shapes[:], filters_shapes[:], output_shapes[:], subsamples[:]):
            for provide_shape in [False, True]:
                self.run_gradweight(inputs_shape=i, filters_shape=f,
                                    output_shape=o, subsample=s,
                                    verify_grad=False, mode=mode_without_gpu, device='cpu',
                                    provide_shape=provide_shape, border_mode=border_mode)
        return
        ### No reference implementation of full available yet
        border_mode= 'full'
        provide_shape = True
        self.run_gradweight(inputs_shape=(16, 1, 2, 2),
                            filters_shape=(10, 1, 2, 2),
                            output_shape=(16, 10, 3, 3),
                            subsample=(1, 1),
                            verify_grad=True, mode=mode_without_gpu, device='cpu',
                            provide_shape=provide_shape, border_mode=border_mode)


    def test_cpu_grad_input(self):

        ### FIXME subsample
        inputs_shapes =  [(16, 1, 2, 2), (16, 1, 8, 8), (16, 1, 4, 4)]
        filters_shapes = [(10, 1, 2, 2), (10, 1, 2, 2), (10, 1, 2, 2),]
        output_shapes =  [(16, 10, 1, 1), (16, 10, 7, 7), (16, 10, 3, 3)]
        subsamples =     [(1, 1), (1, 1), (1, 1)]

        border_mode= 'valid'
        for i, f, o, s in zip(inputs_shapes[:], filters_shapes[:], output_shapes[:], subsamples[:]):
            for provide_shape in [True, False]:
                self.run_gradinput(inputs_shape=i, filters_shape=f,
                                   output_shape=o, subsample=s,
                                   verify_grad=False, mode=mode_without_gpu, device='cpu',
                                   provide_shape=provide_shape, border_mode=border_mode)
        return
        ### No reference implementation of full available yet
        border_mode= 'full'
        provide_shape = True
        self.run_gradweight(inputs_shape=(16, 1, 2, 2),
                            filters_shape=(10, 1, 2, 2),
                            output_shape=(16, 10, 3, 3),
                            subsample=(1, 1),
                            verify_grad=False, mode=mode_without_gpu, device='cpu',
                            provide_shape=provide_shape, border_mode=border_mode)


